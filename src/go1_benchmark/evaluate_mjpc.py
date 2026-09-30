"""Evaluate actual MJPC predictive sampling on the PPO full-collision Go1 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

from go1_benchmark.metrics import cost_of_transport, recovery_time
from go1_benchmark.mjpc_controller import MJPCController
from go1_benchmark.mjpc_model import (
    FEET,
    MJPC_REVISION,
    TERMS,
    calf_clearance,
    foot_normal_forces,
    make_model,
    validate_config,
)


def pulse_impulse(peak_n, duration_s, shape):
    """Analytical area of the commanded pulse, distinct from delivered area."""
    if not math.isfinite(peak_n) or peak_n < 0:
        raise ValueError("Peak force must be finite and nonnegative")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Push duration must be finite and positive")
    if shape not in ("rectangular", "triangular"):
        raise ValueError("Unknown push shape")
    return peak_n * duration_s * (0.5 if shape == "triangular" else 1.0)


def force_at(time_s, peak_n, start_s, duration_s, shape):
    if duration_s <= 0:
        raise ValueError("Push duration must be positive")
    phase = (time_s - start_s) / duration_s
    if not 0 <= phase < 1:
        return 0.0
    if shape == "rectangular":
        return peak_n
    if shape == "triangular":
        return peak_n * (1 - abs(2 * phase - 1))
    raise ValueError("Unknown push shape")


def write_json(path, value):
    path.write_text(
        json.dumps(json_finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def json_finite(value):
    """Preserve failure reports even when the saved trajectory contains NaNs."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    return value


def walking_check(summary):
    """A gait check, separate from nominal speed tracking and planner timing."""
    if not summary.get("nominal_pass"):
        return False
    gait = summary["gait"]
    return bool(
        gait["steady_contact_phase_match_fraction"] >= 0.85
        and gait["steady_exactly_one_swing_fraction"] >= 0.65
        and gait["steady_all_feet_airborne_fraction"] <= 0.01
    )


def evaluate(args):
    import mujoco
    import numpy as np

    config = json.loads(args.config.read_text())
    validate_config(config)
    values = [
        args.duration_s,
        args.warmup_s,
        args.force_n,
        args.direction_deg,
        args.push_start_s,
        args.push_duration_s,
        args.initial_joint_noise_rad,
    ]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("All rollout arguments must be finite")
    if not 0 <= args.warmup_s < args.duration_s or args.initial_joint_noise_rad < 0:
        raise ValueError(
            "Duration must exceed nonnegative warmup; noise must be nonnegative"
        )
    if args.force_n < 0 or args.push_duration_s <= 0:
        raise ValueError("Use nonnegative force and positive push duration")
    if args.force_n and not (
        0 <= args.push_start_s
        and args.push_start_s + args.push_duration_s <= args.duration_s
    ):
        raise ValueError("The full pulse must fit in the rollout")
    env = config["environment"]
    dt, ctrl_dt = env["simulation_dt_s"], env["control_dt_s"]
    steps = round(args.duration_s / ctrl_dt)
    if steps < 1 or not math.isclose(steps * ctrl_dt, args.duration_s):
        raise ValueError(
            "Duration must be a positive integer number of control intervals"
        )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    if not args.library.is_file():
        raise FileNotFoundError(
            "MJPC library missing. Run python scripts/build_mjpc.py first."
        )
    model = make_model(config, args.menagerie_root)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.mjb"
    mujoco.mj_saveModel(model, str(model_path))
    write_json(output / "experiment.json", config)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    rng = np.random.default_rng(args.seed)
    data.qpos[7:] += rng.normal(0, args.initial_joint_noise_rad, model.nu)
    data.ctrl[:] = data.qpos[7:]
    mujoco.mj_forward(model, data)
    initial_x = float(data.qpos[0])
    torso = model.body("trunk").id
    mass = float(model.body_subtreemass[torso])
    floor = model.geom("floor").id
    feet = [model.geom(f).id for f in FEET]
    decimation = round(ctrl_dt / dt)
    direction = np.array(
        [
            math.cos(math.radians(args.direction_deg)),
            math.sin(math.radians(args.direction_deg)),
            0,
        ]
    )
    rows = []
    work_physics = 0.0
    measured_impulse = 0.0
    started = time.perf_counter()
    nonfoot_geometries = {}
    first_nonfoot_time = []

    def contacts(record=False):
        foot = np.zeros(4, dtype=bool)
        nonfoot = False
        touched = set()
        for c in data.contact:
            if c.dist > 0 or floor not in c.geom:
                continue
            other = int(c.geom[1] if c.geom[0] == floor else c.geom[0])
            if other in feet:
                foot[feet.index(other)] = True
            elif other >= 0 and other != floor:
                nonfoot = True
                touched.add(model.geom(other).name or f"geom-{other}")
        if record:
            for name in touched:
                nonfoot_geometries[name] = nonfoot_geometries.get(name, 0) + 1
            if nonfoot and not first_nonfoot_time:
                first_nonfoot_time.append(float(data.time))
        return foot, nonfoot

    write_json(
        output / "manifest.json",
        {
            "controller": "upstream MJPC SamplingPlanner with independent Go1 task",
            "mjpc_revision": MJPC_REVISION,
            "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "task_residual_dimensions": TERMS,
            "task_weights": config["task"]["weights"],
            "platform": platform.platform(),
            "seed": args.seed,
            "seed_scope": "Initial joint jitter only; upstream absl::BitGen sampling is unseeded",
            "initial_joint_noise_rad": args.initial_joint_noise_rad,
            "command_m_s_rad_s": env["command_m_s_rad_s"],
            "comparison_scope": "Matches PPO model/PD settings, but native MuJoCo and MJX need cross-backend validation. Gait contact filtering differs.",
            "energy_sampling": "Both 50 Hz endpoint estimate (PPO-compatible) and physics-rate estimate saved",
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
        },
    )
    error = None
    with MJPCController(args.library, model_path, config) as controller:
        for step in range(steps):
            tick = time.perf_counter()
            try:
                action, predicted_cost = controller.action(data)
            except RuntimeError as exc:
                error = str(exc)
                break
            planning_s = time.perf_counter() - tick
            data.ctrl[:] = action
            interval_contact = False
            for _ in range(decimation):
                # Midpoint discretization preserves triangular pulse area well.
                force = force_at(
                    data.time + dt / 2,
                    args.force_n,
                    args.push_start_s,
                    args.push_duration_s,
                    args.push_shape,
                )
                data.xfrc_applied[torso, :3] = force * direction
                measured_impulse += force * dt
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                work_physics += float(
                    np.maximum(data.actuator_force * data.qvel[6:], 0).sum() * dt
                )
                interval_contact |= contacts(record=True)[1]
            foot, nonfoot = contacts()
            local = data.sensor("local_linvel").data.copy()
            global_vel = data.sensor("global_linvel").data.copy()
            residual = controller.residual(data)
            term_costs = []
            offset = 0
            for name, dim in TERMS.items():
                term_costs.append(
                    0.5
                    * config["task"]["weights"].get(name, 0)
                    * float(
                        residual[offset : offset + dim]
                        @ residual[offset : offset + dim]
                    )
                )
                offset += dim
            rows.append(
                {
                    "time_s": float(data.time),
                    "qpos": data.qpos.copy(),
                    "qvel": data.qvel.copy(),
                    "action": action,
                    "actuator_force": data.actuator_force.copy(),
                    "local_linvel": local,
                    "global_linvel": global_vel,
                    "foot_contact": foot,
                    "foot_position_m": np.array(
                        [data.site(f).xpos.copy() for f in FEET]
                    ),
                    "foot_normal_force_n": foot_normal_forces(model, data),
                    "task_residual": residual,
                    "task_term_cost": np.array(term_costs),
                    "calf_clearance_m": calf_clearance(model, data),
                    "nonfoot_ground_contact": nonfoot,
                    "nonfoot_ground_contact_any_substep": interval_contact,
                    "xfrc_applied": data.xfrc_applied[torso, :3].copy(),
                    "planning_time_s": planning_s,
                    "predicted_cost": predicted_cost,
                }
            )
            if (step + 1) % max(1, round(1 / ctrl_dt)) == 0 or step + 1 == steps:
                print(
                    f"MJPC steps={step + 1:,}/{steps:,} sim={data.time:.2f}s/{args.duration_s:.2f}s wall={time.perf_counter() - started:.1f}s vx={local[0]:.3f} plan_ms={planning_s * 1000:.1f}",
                    flush=True,
                )
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                error = "Nonfinite plant state"
                break
    if not rows:
        write_json(output / "error.json", {"error": error or "No samples recorded"})
        raise RuntimeError(error or "No rollout samples")
    trajectory = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(output / "trajectory.npz", **trajectory)
    steady = trajectory["time_s"] >= args.warmup_s
    velocity_error = np.linalg.norm(
        trajectory["local_linvel"][:, :2] - np.array(env["command_m_s_rad_s"][:2]),
        axis=1,
    )
    distance = float(data.qpos[0] - initial_x)
    finite = all(np.isfinite(x).all() for x in trajectory.values())
    rmse = (
        float(np.sqrt(np.mean(velocity_error[steady] ** 2))) if steady.any() else None
    )
    contact_failure = bool(trajectory["nonfoot_ground_contact_any_substep"].any())
    warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    cot = (
        cost_of_transport(
            trajectory["actuator_force"].tolist(),
            trajectory["qvel"][:, 6:].tolist(),
            ctrl_dt,
            mass,
            distance,
        )
        if finite and distance > 0
        else None
    )
    recovered = (
        recovery_time(
            trajectory["time_s"].tolist(),
            velocity_error.tolist(),
            args.push_start_s + args.push_duration_s,
            tolerance=0.1,
            dwell_s=0.5,
        )
        if args.force_n
        else None
    )
    contact = trajectory["foot_contact"][steady]
    calf = trajectory["calf_clearance_m"][steady]
    phases = (
        trajectory["time_s"][:, None] / config["task"]["gait_period_s"]
        - np.array([config["task"]["swing_order"].index(f) / 4 for f in FEET])
    ) % 1
    expected_contact = phases >= config["task"]["swing_fraction"]
    summary = {
        "controller": "MJPC predictive sampling",
        "duration_s": float(data.time),
        "requested_duration_s": args.duration_s,
        "complete": len(rows) == steps,
        "seed": args.seed,
        "seed_scope": "Initial joint jitter only; planner is unseeded",
        "command_m_s_rad_s": env["command_m_s_rad_s"],
        "control_dt_s": ctrl_dt,
        "simulation_dt_s": dt,
        "robot_mass_kg": mass,
        "full_collision_evaluation": True,
        "failure": contact_failure,
        "failure_sampling": "Checked every physics step",
        "first_nonfoot_contact_time_s": first_nonfoot_time[0]
        if first_nonfoot_time
        else None,
        "nonfoot_contact_physics_steps_by_geometry": nonfoot_geometries,
        "sampled_nonfoot_contact_fraction": float(
            trajectory["nonfoot_ground_contact"].mean()
        ),
        "physics_interval_nonfoot_contact_fraction": float(
            trajectory["nonfoot_ground_contact_any_substep"].mean()
        ),
        "finite": finite,
        "warnings": warnings,
        "controller_error": error,
        "forward_distance_m": distance,
        "steady_planar_rmse_m_s": rmse,
        "steady_mean_forward_velocity_m_s": float(
            trajectory["local_linvel"][steady, 0].mean()
        )
        if steady.any()
        else None,
        "positive_work_cot": cot,
        "positive_work_cot_physics": work_physics / (mass * 9.81 * distance)
        if finite and distance > 0
        else None,
        "recovery_time_s": recovered,
        "nominal_pass": bool(
            finite
            and not contact_failure
            and not error
            and not warnings
            and len(rows) == steps
            and rmse is not None
            and rmse <= 0.15
            and distance > 0
        )
        if not args.force_n
        else None,
        "push": {
            "force_n": args.force_n,
            "direction_deg": args.direction_deg,
            "start_s": args.push_start_s,
            "duration_s": args.push_duration_s,
            "shape": args.push_shape,
            "impulse_n_s": args.force_n
            * args.push_duration_s
            * (0.5 if args.push_shape == "triangular" else 1),
            "measured_impulse_n_s": measured_impulse,
        },
        "planning": {
            "mean_ms": float(trajectory["planning_time_s"].mean() * 1000),
            "p95_ms": float(np.quantile(trajectory["planning_time_s"], 0.95) * 1000),
            "fraction_exceeding_control_budget": float(
                (trajectory["planning_time_s"] > ctrl_dt).mean()
            ),
            "execution": "Synchronous offline; over-budget actions are not delayed in simulation",
        },
        "gait": {
            "foot_order": FEET,
            "contact_method": "Instantaneous penetrating foot-floor contact; not PPO's filtered last_contact",
            "steady_exactly_one_swing_fraction": float(
                (np.sum(~contact, axis=1) == 1).mean()
            )
            if steady.any()
            else None,
            "steady_all_feet_airborne_fraction": float(
                (~np.any(contact, axis=1)).mean()
            )
            if steady.any()
            else None,
            "steady_contact_phase_match_fraction": float(
                (contact == expected_contact[steady]).mean()
            )
            if steady.any()
            else None,
        },
        "task_cost": {
            "term_order": list(TERMS),
            "steady_mean_weighted_terms": dict(
                zip(TERMS, trajectory["task_term_cost"][steady].mean(axis=0).tolist())
            )
            if steady.any()
            else None,
        },
        "calf_clearance": {
            "steady_minimum_m_by_leg": dict(
                zip(FEET, [float(x) for x in calf.min(axis=0)])
            )
            if steady.any()
            else None
        },
        "verification": {
            "warmup_s": args.warmup_s,
            "scope": "Single rollout, not multi-seed validation",
        },
        "mjpc_revision": MJPC_REVISION,
    }
    # Keep the earlier speed/contact check separate from the stricter gait check.
    summary["walking_criteria"] = {
        "minimum_phase_match": 0.85,
        "minimum_exactly_one_swing_fraction": 0.65,
        "maximum_all_feet_airborne_fraction": 0.01,
    }
    summary["walking_pass"] = (
        walking_check(summary) if not args.force_n and steady.any() else None
    )
    summary["planner_budget_pass"] = bool(
        summary["planning"]["fraction_exceeding_control_budget"] == 0
    )
    write_json(output / "summary.json", summary)
    print(json.dumps(json_finite(summary), indent=2, allow_nan=False), flush=True)
    if error or not finite or warnings or len(rows) != steps:
        raise RuntimeError("MJPC rollout incomplete or invalid; inspect saved summary")
    if args.require_pass and (args.force_n or not summary["nominal_pass"]):
        raise SystemExit("Nominal verification failed or was requested for a push run")
    if args.require_walking and not summary["walking_pass"]:
        raise SystemExit("Walking verification failed; inspect the gait metrics")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mjpc_go1.json"))
    parser.add_argument(
        "--library", type=Path, default=Path(".build/mjpc/libgo1_mjpc.so")
    )
    parser.add_argument("--menagerie-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=50)
    parser.add_argument("--warmup-s", type=float, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-joint-noise-rad", type=float, default=0)
    parser.add_argument("--force-n", type=float, default=0)
    parser.add_argument("--direction-deg", type=float, default=0)
    parser.add_argument("--push-start-s", type=float, default=5)
    parser.add_argument("--push-duration-s", type=float, default=0.2)
    parser.add_argument(
        "--push-shape", choices=("rectangular", "triangular"), default="rectangular"
    )
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-walking", action="store_true")
    return parser


def main():
    output = evaluate(build_parser().parse_args())
    print(f"MJPC evaluation saved to {output}")


if __name__ == "__main__":
    main()
