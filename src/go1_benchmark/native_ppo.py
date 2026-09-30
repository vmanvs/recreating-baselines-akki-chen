"""Run a frozen Brax PPO policy on native MuJoCo without retraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def checkpoint_digest(path):
    digest = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def policy_observation(model, data, environment, previous_action):
    """Pinned Playground actor input with observation noise disabled.

    Joystick.step builds the next observation BEFORE updating last_act. Thus
    the supplied previous_action is two control decisions behind the upcoming
    decision. Preserve that timing rather than silently changing the policy.
    """
    gravity = data.site("imu").xmat.reshape(3, 3).T @ np.array([0, 0, -1])
    state = np.concatenate(
        (
            data.sensor("local_linvel").data,
            data.sensor("gyro").data,
            gravity,
            data.qpos[7:] - model.key("home").qpos[7:],
            data.qvel[6:],
            previous_action,
            environment["command_m_s_rad_s"],
        )
    )
    if "gait" in environment:
        phase = 2 * np.pi * data.time / environment["gait"]["period_s"]
        state = np.concatenate((state, [np.sin(phase), np.cos(phase)]))
    return state.astype(np.float32)


class NativePPO:
    def __init__(self, root: Path, model, common_environment):
        root = root.resolve()
        import jax
        import jax.numpy as jp
        from brax.training import networks as brax_networks
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import checkpoint, networks

        self.manifest = json.loads((root / "manifest.json").read_text())
        self.environment = self.manifest["environment"]
        for key in ("command_m_s_rad_s", "simulation_dt_s", "control_dt_s", "kp", "kd"):
            if self.environment[key] != common_environment[key]:
                raise ValueError(f"PPO training and evaluation disagree on {key}")
        candidates = [
            p
            for p in (root / "checkpoints").iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        self.checkpoint = max(candidates, key=lambda p: int(p.name))
        cfg = json.loads((self.checkpoint / "ppo_network_config.json").read_text())
        kwargs = cfg["network_factory_kwargs"]
        if kwargs["policy_obs_key"] != "state" or cfg["action_size"] != model.nu:
            raise ValueError(
                "This adapter requires the saved state actor and 12 actions"
            )
        sizes = {
            key: tuple(value["shape"]) for key, value in cfg["observation_size"].items()
        }
        self.state_size = sizes["state"][0]
        kwargs["activation"] = brax_networks.ACTIVATION[kwargs["activation"]]
        for key in (
            "policy_network_kernel_init_fn",
            "value_network_kernel_init_fn",
            "mean_kernel_init_fn",
        ):
            if kwargs.get(key) is not None:
                kwargs[key] = brax_networks.KERNEL_INITIALIZER[kwargs[key]]
        network = networks.make_ppo_networks(
            sizes,
            model.nu,
            preprocess_observations_fn=running_statistics.normalize
            if cfg["normalize_observations"]
            else lambda obs, params: obs,
            **kwargs,
        )
        params = checkpoint.load(str(self.checkpoint))
        infer = networks.make_inference_fn(network)(params, deterministic=True)

        # Normalization expects both saved keys, although the actor uses state
        # only and the privileged critic is not evaluated.
        @jax.jit
        def actor(state):
            return infer(
                {
                    "state": state,
                    "privileged_state": jp.zeros(
                        sizes["privileged_state"], dtype=jp.float32
                    ),
                },
                jax.random.PRNGKey(0),
            )[0]

        self.actor = actor
        self.jp = jp
        self.model = model
        self.reset()
        warm = self.actor(jp.zeros(self.state_size, dtype=jp.float32))
        warm.block_until_ready()
        self.provenance = {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": checkpoint_digest(self.checkpoint),
            "training_seed": self.manifest["seed"],
            "profile": self.manifest["profile"],
            "training_manifest": self.manifest,
            "inference_devices": [str(d) for d in jax.devices()],
            "observation_noise_level": 0,
            "action_history": "Preserves pinned Joystick observation-before-last_act-update timing",
        }

    def reset(self):
        self.last_action = np.zeros(12, dtype=np.float32)
        self.observation_last_action = self.last_action.copy()

    def action(self, data):
        obs = policy_observation(
            self.model, data, self.environment, self.observation_last_action
        )
        if obs.shape != (self.state_size,):
            raise ValueError(
                f"Actor shape mismatch: {obs.shape} != {(self.state_size,)}"
            )
        raw = np.asarray(self.actor(self.jp.asarray(obs)))
        self.observation_last_action = self.last_action.copy()
        self.last_action = raw.copy()
        target = (
            self.model.key("home").qpos[7:] + self.environment["action_scale_rad"] * raw
        )
        return target, raw
