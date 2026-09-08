"""Evaluate a trained Go1 PPO checkpoint under a deterministic force pulse."""

from __future__ import annotations

import argparse
import functools
import os
from math import cos, pi, sin
from pathlib import Path

from go1_benchmark.fixed_velocity_env import make_fixed_velocity_env
from go1_benchmark.metrics import (
    cost_of_transport,
    disturbance_impulse,
    recovery_time,
)
from go1_benchmark.train_ppo import DEFAULT_CONFIG, _load_config, _write_json


def _latest_checkpoint(path: Path) -> Path:
    if not path.is_dir():
        return path
    numbered = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    if numbered:
        return max(numbered, key=lambda child: int(child.name))
    return path


def evaluate(args: argparse.Namespace) -> Path:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper
    from mujoco_playground.config import locomotion_params

    document, profile = _load_config(args.config, args.profile)
    env_cfg = document["environment"]
    command = tuple(float(value) for value in env_cfg["command_m_s_rad_s"])
    env = make_fixed_velocity_env(
        command=command,
        impl=env_cfg["implementation"],
        full_collisions=True,
        noise_level=args.noise_level,
        randomize_initial_state=False,
    )

    ppo_cfg = locomotion_params.brax_ppo_config(
        "Go1JoystickFlatTerrain", env_cfg["implementation"]
    )
    for key in ("learning_rate", "clipping_epsilon", "num_envs", "num_evals"):
        if key in profile:
            ppo_cfg[key] = profile[key]
    ppo_cfg.num_timesteps = 0

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_cfg.network_factory.to_dict()
    )
    training_params = ppo_cfg.to_dict()
    training_params.pop("network_factory")
    training_params.pop("num_eval_envs", None)
    training_params["num_evals"] = 1

    checkpoint = _latest_checkpoint(args.checkpoint.resolve())
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        network_factory=network_factory,
        seed=args.seed,
        restore_checkpoint_path=str(checkpoint),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=1,
        **training_params,
    )
    inference = jax.jit(make_inference_fn(params, deterministic=True))

    torso_id = env.mj_model.body("trunk").id
    robot_mass_kg = float(env.mj_model.body_subtreemass[torso_id])
    floor_id = env.mj_model.geom("floor").id
    foot_ids = jp.asarray(
        [env.mj_model.geom(name).id for name in ("FR", "FL", "RR", "RL")]
    )
    angle_rad = args.direction_deg * pi / 180.0
    push_direction = jp.asarray([cos(angle_rad), sin(angle_rad), 0.0])
    dt_s = float(env.dt)
    steps = round(args.duration_s / dt_s)

    def has_nonfoot_ground_contact(data):
        contact = data.contact
        # Contact arrays have a static padded size under JIT. Masking by ncon
        # prevents unused slots (often zero-filled) from becoming false falls.
        active = jp.arange(contact.dist.shape[0]) < data.ncon
        active &= contact.dist <= 0.0
        floor_is_first = contact.geom1 == floor_id
        floor_is_second = contact.geom2 == floor_id
        other = jp.where(floor_is_first, contact.geom2, contact.geom1)
        involves_floor = floor_is_first | floor_is_second
        is_foot = jp.any(other[:, None] == foot_ids[None, :], axis=1)
        return jp.any(active & involves_floor & ~is_foot)

    def scan_step(carry, step_index):
        state, rng = carry
        time_s = step_index * dt_s
        push_on = (time_s >= args.push_start_s) & (
            time_s < args.push_start_s + args.push_duration_s
        )
        force = jp.where(push_on, args.force_n * push_direction, jp.zeros(3))
        xfrc = jp.zeros_like(state.data.xfrc_applied).at[torso_id, :3].set(force)
        state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
        rng, action_rng = jax.random.split(rng)
        action, _ = inference(state.obs, action_rng)
        state = env.step(state, action)
        sample = {
            "time_s": state.data.time,
            "qpos": state.data.qpos,
            "qvel": state.data.qvel,
            "actuator_force": state.data.actuator_force,
            "global_linvel": env.get_global_linvel(state.data),
            "local_linvel": env.get_local_linvel(state.data),
            "xfrc_applied": state.data.xfrc_applied[torso_id, :3],
            "reward": state.reward,
            "terminated": state.done,
            "nonfoot_ground_contact": has_nonfoot_ground_contact(state.data),
        }
        return (state, rng), sample

    rng = jax.random.PRNGKey(args.seed)
    state = jax.jit(env.reset)(rng)
    (_, _), trajectory = jax.jit(
        lambda initial, key: jax.lax.scan(scan_step, (initial, key), jp.arange(steps))
    )(state, jax.random.PRNGKey(args.seed + 1))
    trajectory_np = jax.tree.map(lambda value: np.asarray(value), trajectory)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trajectory.npz", **trajectory_np)

    qpos = trajectory_np["qpos"]
    qvel = trajectory_np["qvel"][:, 6:]
    torques = trajectory_np["actuator_force"]
    forward_distance = float(qpos[-1, 0] - qpos[0, 0])
    velocity_error = np.linalg.norm(
        trajectory_np["local_linvel"][:, :2] - np.asarray(command[:2]), axis=1
    )
    try:
        cot = cost_of_transport(
            torques.tolist(),
            qvel.tolist(),
            dt_s,
            robot_mass_kg,
            forward_distance,
        )
    except ValueError:
        cot = None
    recovered_s = recovery_time(
        trajectory_np["time_s"].tolist(),
        velocity_error.tolist(),
        args.push_start_s + args.push_duration_s,
        tolerance=args.recovery_tolerance_m_s,
        dwell_s=args.recovery_dwell_s,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "command_m_s_rad_s": command,
        "duration_s": args.duration_s,
        "push": {
            "force_n": args.force_n,
            "direction_deg": args.direction_deg,
            "start_s": args.push_start_s,
            "duration_s": args.push_duration_s,
            "impulse_n_s": disturbance_impulse(
                args.force_n, args.push_duration_s
            ),
        },
        "failure": bool(np.any(trajectory_np["nonfoot_ground_contact"])),
        "terminated": bool(np.any(trajectory_np["terminated"])),
        "recovery_time_s": recovered_s,
        "forward_distance_m": forward_distance,
        "velocity_rmse_m_s": float(np.sqrt(np.mean(velocity_error**2))),
        "positive_work_cot": cot,
        "robot_mass_kg": robot_mass_kg,
        "control_dt_s": dt_s,
        "full_collision_evaluation": True,
        "observation_noise_level": args.noise_level,
        "upstream": document["upstream"],
    }
    _write_json(output / "summary.json", summary)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("paper_budget_proxy", "playground_reference"),
        default="paper_budget_proxy",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--force-n", type=float, default=150.0)
    parser.add_argument("--direction-deg", type=float, default=0.0)
    parser.add_argument("--push-start-s", type=float, default=5.0)
    parser.add_argument("--push-duration-s", type=float, default=0.2)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--recovery-tolerance-m-s", type=float, default=0.1)
    parser.add_argument("--recovery-dwell-s", type=float, default=0.5)
    return parser


def main() -> None:
    output = evaluate(build_parser().parse_args())
    print(f"Evaluation artifacts written to {output}")


if __name__ == "__main__":
    main()
