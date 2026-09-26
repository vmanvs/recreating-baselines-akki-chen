"""MuJoCo Playground Go1 environment with a fixed velocity command.

Training uses Playground's feet-only collision model for accelerator throughput.
Evaluation can select the full-collision model so a non-foot ground contact can
be counted as failure. A separate gait-aware profile adds phase observations
and rewards; the original baseline retains its existing observation dimensions.
"""

from __future__ import annotations

from typing import Any


def _imports() -> dict[str, Any]:
    """Import optional accelerator dependencies only when this backend is used."""
    try:
        import jax
        import jax.numpy as jp
        from mujoco_playground._src.locomotion.go1 import base as go1_base
        from mujoco_playground._src.locomotion.go1 import go1_constants as consts
        from mujoco_playground._src.locomotion.go1 import joystick
    except ImportError as exc:  # pragma: no cover - exercised on Kaggle.
        raise RuntimeError(
            "The RL backend is not installed. Run `uv sync --extra rl` in a "
            "CUDA-enabled Kaggle notebook."
        ) from exc
    return {
        "jax": jax,
        "jp": jp,
        "go1_base": go1_base,
        "consts": consts,
        "joystick": joystick,
    }


def make_fixed_velocity_env(
    *,
    command: tuple[float, float, float] = (0.5, 0.0, 0.0),
    impl: str = "jax",
    full_collisions: bool = False,
    noise_level: float = 1.0,
    randomize_initial_state: bool = True,
    settings: dict[str, Any] | None = None,
):
    """Create the public Go1 task with a command that never changes.

    The returned class is defined lazily so importing metric-only modules does
    not require JAX, MuJoCo, or Playground.
    """
    modules = _imports()
    jp = modules["jp"]
    from mujoco import mjx
    from mujoco_playground._src import mjx_env

    mjx_env.ensure_menagerie_exists()

    go1_base = modules["go1_base"]
    consts = modules["consts"]
    joystick = modules["joystick"]
    fixed_command = jp.asarray(command)
    gait = settings.get("gait") if settings else None
    if gait is not None:
        order = tuple(gait["swing_order"])
        if set(order) != {"FR", "FL", "RR", "RL"} or len(order) != 4:
            raise ValueError("gait.swing_order must contain each foot exactly once")
        if float(gait["period_s"]) <= 0:
            raise ValueError("gait.period_s must be positive")
        if not 0 < float(gait["swing_fraction"]) <= 0.25:
            raise ValueError("gait.swing_fraction must be in (0, 0.25]")
        if float(gait["target_swing_height_m"]) <= 0:
            raise ValueError("gait.target_swing_height_m must be positive")
        scales = gait["reward_scales"]
        expected = {
            "walk_contact_match",
            "walk_swing_height",
            "walk_multi_swing",
            "walk_no_support",
        }
        if set(scales) != expected:
            raise ValueError(f"gait.reward_scales must contain {sorted(expected)}")
        offsets = jp.asarray(
            [order.index(foot) / 4.0 for foot in ("FR", "FL", "RR", "RL")]
        )

    class FixedVelocityGo1(joystick.Joystick):
        def __init__(self):
            config = joystick.default_config()
            self._gait = gait
            if gait is not None:
                self._gait_offsets = offsets
                for name, weight in scales.items():
                    config.reward_config.scales[name] = float(weight)
            for source, target in {
                "simulation_dt_s": "sim_dt",
                "control_dt_s": "ctrl_dt",
                "episode_length_steps": "episode_length",
                "action_scale_rad": "action_scale",
                "kp": "Kp",
                "kd": "Kd",
            }.items():
                if settings and source in settings:
                    config[target] = settings[source]
            config.impl = impl
            config.noise_config.level = noise_level
            config.pert_config.enable = False
            if full_collisions:
                # Joystick's public task selector uses feet-only XMLs. Calling
                # the base initializer directly preserves Joystick's reward and
                # observation implementation while enabling body collisions.
                go1_base.Go1Env.__init__(
                    self,
                    xml_path=consts.FULL_COLLISIONS_FLAT_TERRAIN_XML.as_posix(),
                    config=config,
                )
                self._post_init()
            else:
                super().__init__(task="flat_terrain", config=config)

        def _desired_swing(self, data):
            phase = jp.mod(data.time / self._gait["period_s"], 1.0)
            foot_phase = jp.mod(phase - self._gait_offsets, 1.0)
            return foot_phase < self._gait["swing_fraction"]

        def _get_obs(self, data, info):
            obs = super()._get_obs(data, info)
            if self._gait is None:
                return obs
            phase = 2.0 * jp.pi * data.time / self._gait["period_s"]
            phase_obs = jp.asarray([jp.sin(phase), jp.cos(phase)])
            return {
                "state": jp.hstack((obs["state"], phase_obs)),
                "privileged_state": jp.hstack(
                    (obs["privileged_state"], phase_obs)
                ),
            }

        def _get_reward(
            self, data, action, info, metrics, done, first_contact, contact
        ):
            rewards = super()._get_reward(
                data, action, info, metrics, done, first_contact, contact
            )
            if self._gait is None:
                return rewards
            desired_swing = self._desired_swing(data)
            foot_height = data.site_xpos[self._feet_site_id, 2]
            rewards.update(
                walk_contact_match=jp.mean(contact == ~desired_swing),
                walk_swing_height=jp.sum(
                    jp.where(
                        desired_swing,
                        jp.clip(
                            foot_height / self._gait["target_swing_height_m"],
                            0.0,
                            1.0,
                        ),
                        0.0,
                    )
                ),
                walk_multi_swing=jp.maximum(jp.sum(~contact) - 1, 0),
                walk_no_support=(~jp.any(contact)).astype(jp.float32),
            )
            return rewards

        def reset(self, rng):
            state = super().reset(rng)
            info = dict(state.info)
            if not randomize_initial_state:
                data = state.data.replace(
                    qpos=self._init_q,
                    qvel=jp.zeros(self.mjx_model.nv),
                    ctrl=self._init_q[7:],
                )
                data = mjx.forward(self.mjx_model, data)
                state = state.replace(data=data)
            info["command"] = fixed_command
            # Also override sample_command below; the large counter makes fixed
            # command intent visible in saved states and avoids needless draws.
            info["steps_until_next_cmd"] = jp.asarray(2**30, dtype=jp.int32)
            obs = self._get_obs(state.data, info)
            return state.replace(info=info, obs=obs)

        def sample_command(self, rng, previous_command):
            del rng, previous_command
            return fixed_command

    return FixedVelocityGo1()
