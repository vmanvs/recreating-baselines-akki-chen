"""Evaluate a trained Go1 PPO checkpoint under a deterministic force pulse."""

from __future__ import annotations

import argparse
import json
import os
from math import cos, pi, sin
from pathlib import Path

from go1_benchmark.fixed_velocity_env import make_fixed_velocity_env
from go1_benchmark.metrics import (
    cost_of_transport,
    disturbance_impulse,
    recovery_time,
)
from go1_benchmark.train_ppo import _load_config, _write_json


def _latest_checkpoint(path: Path) -> Path:
    if not path.is_dir():
        return path
    numbered = [
        child for child in path.iterdir() if child.is_dir() and child.name.isdigit()
    ]
    if numbered:
        return max(numbered, key=lambda child: int(child.name))
    return path


def evaluate(args: argparse.Namespace) -> Path:
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    checkpoint = _latest_checkpoint(args.checkpoint.resolve())
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    manifest_path = next(
        (
            parent / "manifest.json"
            for parent in checkpoint.parents
            if (parent / "manifest.json").exists()
        ),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(
            "Keep manifest.json and experiment.json with checkpoints."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    document, _ = _load_config(
        manifest_path.parent / "experiment.json", manifest["profile"]
    )
    env_cfg = document["environment"]
    command = tuple(float(value) for value in env_cfg["command_m_s_rad_s"])
    env = make_fixed_velocity_env(
        command=command,
        impl=env_cfg["implementation"],
        full_collisions=True,
        noise_level=args.noise_level,
        randomize_initial_state=False,
        settings=env_cfg,
    )

    # Reconstruct the trained network and its normalizer from Brax's own saved
    # checkpoint metadata, without initializing a batched training environment.
    inference = jax.jit(ppo_checkpoint.load_policy(str(checkpoint), deterministic=True))

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
    if steps < 1 or not 0 <= args.warmup_s < args.duration_s:
        raise ValueError("Duration must be positive and longer than warmup.")
    if args.force_n and (
        args.push_start_s < 0
        or args.push_duration_s <= 0
        or args.push_start_s + args.push_duration_s > args.duration_s
    ):
        raise ValueError("The complete force pulse must lie within the rollout.")

    def has_nonfoot_ground_contact(data):
        contact = data.contact
        # MJX JAX allocates potential contact slots statically. Test geometry
        # identities and penetration rather than treating every slot as contact.
        active = jp.arange(contact.dist.shape[0]) < data.ncon
        active &= contact.dist <= 0.0
        floor_is_first = contact.geom1 == floor_id
        floor_is_second = contact.geom2 == floor_id
        other = jp.where(floor_is_first, contact.geom2, contact.geom1)
        involves_floor = floor_is_first | floor_is_second
        involves_floor &= (other >= 0) & (other != floor_id)
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
            "action": action,
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
    initial_x = float(state.data.qpos[0])
    (_, _), trajectory = jax.jit(
        lambda initial, key: jax.lax.scan(scan_step, (initial, key), jp.arange(steps))
    )(state, jax.random.PRNGKey(args.seed + 1))
    trajectory_np = jax.tree.map(lambda value: np.asarray(value), trajectory)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty evaluation output directory.")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trajectory.npz", **trajectory_np)

    qpos = trajectory_np["qpos"]
    qvel = trajectory_np["qvel"][:, 6:]
    torques = trajectory_np["actuator_force"]
    forward_distance = float(qpos[-1, 0] - initial_x)
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
    steady = trajectory_np["time_s"] >= args.warmup_s
    steady_rmse = float(np.sqrt(np.mean(velocity_error[steady] ** 2)))
    finite = all(np.isfinite(value).all() for value in trajectory_np.values())
    contact_failure = bool(np.any(trajectory_np["nonfoot_ground_contact"]))
    terminated = bool(np.any(trajectory_np["terminated"]))
    nominal_pass = (
        finite
        and not contact_failure
        and not terminated
        and steady_rmse <= args.max_rmse_m_s
        and forward_distance > 0
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
            "impulse_n_s": disturbance_impulse(args.force_n, args.push_duration_s),
        },
        "failure": contact_failure,
        "terminated": terminated,
        "finite": bool(finite),
        "recovery_time_s": recovered_s if args.force_n else None,
        "steady_planar_rmse_m_s": steady_rmse,
        "steady_mean_forward_velocity_m_s": float(
            np.mean(trajectory_np["local_linvel"][steady, 0])
        ),
        "nominal_pass": bool(nominal_pass) if not args.force_n else None,
        "verification": {
            "warmup_s": args.warmup_s,
            "max_planar_rmse_m_s": args.max_rmse_m_s,
            "scope": "Single canonical-start rollout; not multi-seed validation",
        },
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
    print(json.dumps(summary, indent=2), flush=True)
    if args.require_pass and (args.force_n or not nominal_pass):
        raise SystemExit("Nominal verification failed or was requested for a push run.")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--force-n", type=float, default=0.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--max-rmse-m-s", type=float, default=0.15)
    parser.add_argument("--require-pass", action="store_true")
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
