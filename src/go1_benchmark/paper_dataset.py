"""Frozen, serial native MuJoCo comparison with resumable trial outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from go1_benchmark.evaluate_mjpc import force_at, write_json
from go1_benchmark.metrics import recovery_time
from go1_benchmark.mjpc_controller import MJPCController
from go1_benchmark.mjpc_model import (
    FEET,
    MJPC_REVISION,
    calf_clearance,
    foot_normal_forces,
    make_model,
)
from go1_benchmark.native_ppo import NativePPO, checkpoint_digest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trials(protocol):
    for controller in protocol["controllers"]:
        for seed in protocol["initial_state_seeds"]:
            yield {
                "id": f"{controller}-nominal-{seed}",
                "controller": controller,
                "seed": seed,
                "duration_s": protocol["nominal_duration_s"],
                "force_n": 0,
                "direction_deg": 0,
            }
        if controller in protocol["push_controllers"]:
            for force in protocol["push_forces_n"]:
                for direction in protocol["push_directions_deg"]:
                    for seed in protocol["push_seeds"]:
                        yield {
                            "id": f"{controller}-push-{force}-{direction}-{seed}",
                            "controller": controller,
                            "seed": seed,
                            "duration_s": protocol["push_duration_total_s"],
                            "force_n": force,
                            "direction_deg": direction,
                        }


def run_trial(model, policy, config, protocol, trial, output):
    import mujoco

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "trial.json", trial)
    # Each replay is self-contained and identical across controller types.
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    write_json(output / "experiment.json", config)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[7:] += np.random.default_rng(trial["seed"]).normal(
        0, protocol["initial_joint_noise_rad"], 12
    )
    data.ctrl[:] = data.qpos[7:]
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    env = config["environment"]
    dt, ctrl_dt = env["simulation_dt_s"], env["control_dt_s"]
    steps = round(trial["duration_s"] / ctrl_dt)
    torso = model.body("trunk").id
    floor = model.geom("floor").id
    feet = [model.geom(f).id for f in FEET]
    mass = float(model.body_subtreemass[torso])
    direction = np.array(
        [
            math.cos(math.radians(trial["direction_deg"])),
            math.sin(math.radians(trial["direction_deg"])),
            0,
        ]
    )
    rows, physics = [], []
    first_failure, failure_reason = None, None
    touched_geometries = {}
    started = time.perf_counter()
    controller_error = None
    work = 0.0
    measured_impulse = 0.0

    def contacts():
        foot = np.zeros(4, dtype=bool)
        other_geoms = set()
        for contact in data.contact:
            if contact.dist > 0 or floor not in contact.geom:
                continue
            other = int(
                contact.geom[1] if contact.geom[0] == floor else contact.geom[0]
            )
            if other in feet:
                foot[feet.index(other)] = True
            elif other >= 0 and other != floor:
                other_geoms.add(model.geom(other).name or f"geom-{other}")
        return foot, other_geoms

    for step in range(steps):
        interval_start = float(data.time)
        tick = time.perf_counter()
        try:
            action, _auxiliary = policy.action(data)
            action = np.asarray(action, dtype=float)
            if not np.isfinite(action).all():
                raise RuntimeError("Nonfinite controller action")
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            controller_error = f"{type(exc).__name__}: {exc}"
            first_failure, failure_reason = float(data.time), "controller_error"
            break
        decision_time = time.perf_counter() - tick
        data.ctrl[:] = action
        interval_nonfoot = False
        for _ in range(round(ctrl_dt / dt)):
            force = force_at(
                data.time + dt / 2,
                trial["force_n"],
                protocol["push_start_s"],
                protocol["pulse_duration_s"],
                protocol["pulse_shape"],
            )
            data.xfrc_applied[torso, :3] = force * direction
            measured_impulse += force * dt
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            foot, other = contacts()
            interval_nonfoot |= bool(other)
            for geom in other:
                touched_geometries[geom] = touched_geometries.get(geom, 0) + 1
            power = float(np.maximum(data.actuator_force * data.qvel[6:], 0).sum())
            work += power * dt
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            up_z = float(data.body("trunk").xmat.reshape(3, 3)[2, 2])
            warnings = any(w.number for w in data.warning)
            physics.append(
                {
                    "time_s": float(data.time),
                    "qpos": data.qpos.copy(),
                    "qvel": data.qvel.copy(),
                    "actuator_force": data.actuator_force.copy(),
                    "positive_power_w": power,
                    "foot_contact": foot,
                    "nonfoot_ground_contact": bool(other),
                    "xfrc_applied": data.xfrc_applied[torso, :3].copy(),
                    "calf_clearance_m": calf_clearance(model, data),
                }
            )
            if other or up_z < 0 or not finite or warnings:
                first_failure = float(data.time)
                failure_reason = (
                    "nonfoot_ground_contact"
                    if other
                    else "inverted_torso"
                    if up_z < 0
                    else "nonfinite_state"
                    if not finite
                    else "mujoco_warning"
                )
                break
        foot, _ = contacts()
        rows.append(
            {
                "time_s": float(data.time),
                "interval_dt_s": float(data.time) - interval_start,
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "action": action.copy(),
                "actuator_force": data.actuator_force.copy(),
                "local_linvel": data.sensor("local_linvel").data.copy(),
                "global_linvel": data.sensor("global_linvel").data.copy(),
                "foot_contact": foot,
                "foot_normal_force_n": foot_normal_forces(model, data),
                "foot_position_m": np.array([data.site(f).xpos.copy() for f in FEET]),
                "calf_clearance_m": calf_clearance(model, data),
                "nonfoot_ground_contact_any_substep": interval_nonfoot,
                "xfrc_applied": data.xfrc_applied[torso, :3].copy(),
                "planning_time_s": decision_time,
                "up_z": float(data.body("trunk").xmat.reshape(3, 3)[2, 2]),
            }
        )
        if (step + 1) % 50 == 0 or first_failure is not None or step + 1 == steps:
            print(
                f"{trial['id']} sim={data.time:.2f}/{trial['duration_s']}s "
                f"wall={time.perf_counter() - started:.1f}s vx={rows[-1]['local_linvel'][0]:.3f} "
                f"decision_ms={decision_time * 1000:.1f} failure={failure_reason}",
                flush=True,
            )
        if first_failure is not None:
            break
    arrays = (
        {key: np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}
    )
    substeps = (
        {key: np.asarray([row[key] for row in physics]) for key in physics[0]}
        if physics
        else {}
    )
    np.savez_compressed(output / "trajectory.npz", **arrays)
    np.savez_compressed(output / "physics.npz", **substeps)
    times = arrays.get("time_s", np.array([]))
    error = (
        np.linalg.norm(
            arrays["local_linvel"][:, :2] - env["command_m_s_rad_s"][:2], axis=1
        )
        if rows
        else np.array([])
    )
    steady = times >= protocol["warmup_s"]
    prepush = steady & (times < protocol["push_start_s"])
    post = times >= protocol["push_start_s"] + protocol["pulse_duration_s"]
    rmse = lambda mask: (
        float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None
    )
    distance = float(data.qpos[0] - initial_qpos[0])
    complete = first_failure is None and len(rows) == steps
    finite = bool(all(np.isfinite(v).all() for v in arrays.values())) and bool(rows)
    recovery = (
        recovery_time(
            times.tolist(),
            error.tolist(),
            protocol["push_start_s"] + protocol["pulse_duration_s"],
            protocol["recovery_tolerance_m_s"],
            protocol["recovery_dwell_s"],
        )
        if trial["force_n"] and rows
        else None
    )
    if first_failure is not None:
        recovery = None
    nominal_pass = (
        complete
        and finite
        and rmse(steady) is not None
        and rmse(steady) <= protocol["max_nominal_rmse_m_s"]
        and distance > 0
    )
    prepush_qualified = bool(
        rmse(prepush) is not None
        and rmse(prepush) <= protocol["max_nominal_rmse_m_s"]
        and (first_failure is None or first_failure >= protocol["push_start_s"])
    )
    contact = arrays["foot_contact"][steady] if rows else np.empty((0, 4), dtype=bool)
    task = config["task"]
    phase = (
        times[steady, None] / task["gait_period_s"]
        - np.array([task["swing_order"].index(f) / 4 for f in FEET])
    ) % 1
    gait = {
        "phase_match": float(np.mean(contact == (phase >= task["swing_fraction"])))
        if steady.any()
        else None,
        "one_swing_fraction": float(np.mean(np.sum(~contact, axis=1) == 1))
        if steady.any()
        else None,
        "all_air_fraction": float(np.mean(~np.any(contact, axis=1)))
        if steady.any()
        else None,
    }
    endpoint_work = (
        float(
            np.sum(
                np.maximum(arrays["actuator_force"] * arrays["qvel"][:, 6:], 0).sum(
                    axis=1
                )
                * arrays["interval_dt_s"]
            )
        )
        if rows
        else 0
    )
    decision = arrays.get("planning_time_s", np.array([]))
    summary = {
        **trial,
        "complete": complete,
        "finite": finite,
        "failure_reason": failure_reason,
        "first_failure_time_s": first_failure,
        "observed_duration_s": float(data.time),
        "nominal_pass": bool(nominal_pass) if not trial["force_n"] else None,
        "prepush_qualified": prepush_qualified if trial["force_n"] else None,
        "push_success": bool(
            complete and finite and prepush_qualified and recovery is not None
        )
        if trial["force_n"]
        else None,
        "recovery_time_s": recovery,
        "steady_planar_rmse_m_s": rmse(steady),
        "prepush_planar_rmse_m_s": rmse(prepush),
        "postpush_planar_rmse_m_s": rmse(post),
        "steady_mean_forward_velocity_m_s": float(
            arrays["local_linvel"][steady, 0].mean()
        )
        if steady.any()
        else None,
        "forward_distance_m": distance,
        "lateral_displacement_m": float(data.qpos[1] - initial_qpos[1]),
        "positive_work_j_physics": work,
        "positive_work_j_control": endpoint_work,
        "positive_work_cot_physics": work / (mass * 9.81 * distance)
        if finite and distance > 0
        else None,
        "positive_work_cot_control": endpoint_work / (mass * 9.81 * distance)
        if finite and distance > 0
        else None,
        "robot_mass_kg": mass,
        "gait": gait,
        "minimum_calf_clearance_m": float(arrays["calf_clearance_m"][steady].min())
        if steady.any()
        else None,
        "nonfoot_physics_steps_by_geometry": touched_geometries,
        "decision_mean_ms": float(decision.mean() * 1000) if rows else None,
        "decision_p95_ms": float(np.quantile(decision, 0.95) * 1000) if rows else None,
        "decision_deadline_miss_fraction": float((decision > ctrl_dt).mean())
        if rows
        else None,
        "wall_time_s": time.perf_counter() - started,
        "measured_impulse_n_s": measured_impulse,
        "intended_impulse_n_s": trial["force_n"] * protocol["pulse_duration_s"],
        "warnings": {
            str(i): int(w.number) for i, w in enumerate(data.warning) if w.number
        },
        "controller_error": controller_error,
        "initial_qpos": initial_qpos.tolist(),
        "model_sha256": sha256(output / "model.mjb"),
    }
    write_json(output / "summary.json", summary)
    return summary


def collect(args):
    import mujoco

    root, output = args.root.resolve(), args.output.resolve()
    protocol = json.loads(args.protocol.read_text())
    config = json.loads((root / "configs/mjpc_go1.json").read_text())
    sources = [
        root / "src/go1_benchmark" / name
        for name in (
            "paper_dataset.py",
            "native_ppo.py",
            "mjpc_model.py",
            "mjpc_controller.py",
            "metrics.py",
            "evaluate_mjpc.py",
        )
    ]
    model = make_model(config)
    policies = {}
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "protocol_sha256": sha256(args.protocol),
        "mjpc_config_sha256": sha256(root / "configs/mjpc_go1.json"),
        "library_sha256": sha256(args.library),
        "mjpc_revision": MJPC_REVISION,
        "sources": {p.name: sha256(p) for p in sources},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("mujoco", "numpy", "jax", "jaxlib", "brax", "flax", "playground")
        },
        "policies": {},
    }
    for label, directory in protocol["policy_directories"].items():
        policies[label] = NativePPO(root / directory, model, config["environment"])
        provenance["policies"][label] = policies[label].provenance
    plan = list(trials(protocol))
    provenance["planned_trials"] = plan
    if output.exists():
        old = json.loads((output / "dataset-manifest.json").read_text())
        for key in (
            "protocol_sha256",
            "mjpc_config_sha256",
            "library_sha256",
            "sources",
        ):
            if old[key] != provenance[key]:
                raise ValueError(f"Cannot resume after changing {key}")
        for label in policies:
            if (
                old["policies"][label]["checkpoint_sha256"]
                != provenance["policies"][label]["checkpoint_sha256"]
            ):
                raise ValueError("Cannot resume after changing a policy")
    else:
        output.mkdir(parents=True)
        (output / "runs").mkdir()
        (output / "frozen-source").mkdir()
        for p in sources:
            (output / "frozen-source" / p.name).write_bytes(p.read_bytes())
        write_json(output / "protocol.json", protocol)
        write_json(output / "mjpc-config.json", config)
        write_json(output / "dataset-manifest.json", provenance)
    summaries = []
    for index, trial in enumerate(plan):
        run = output / "runs" / trial["id"]
        if (run / "summary.json").exists():
            summaries.append(json.loads((run / "summary.json").read_text()))
            continue
        if run.exists():
            raise RuntimeError(
                f"Incomplete trial retained at {run}; inspect it before retrying"
            )
        if sha256(args.library) != provenance["library_sha256"] or any(
            sha256(p) != provenance["sources"][p.name] for p in sources
        ):
            raise RuntimeError("Library or evaluator changed during collection")
        print(f"START trial={index + 1}/{len(plan)} {trial['id']}", flush=True)
        if trial["controller"] == "mjpc":
            model_file = output / "planner-model.mjb"
            mujoco.mj_saveModel(model, str(model_file))
            with MJPCController(args.library, model_file, config) as controller:
                result = run_trial(model, controller, config, protocol, trial, run)
        else:
            controller = policies[trial["controller"]]
            controller.reset()
            result = run_trial(model, controller, config, protocol, trial, run)
        summaries.append(result)
        write_json(output / "trial-index.json", summaries)
        write_json(
            output / "collection-status.json",
            {
                "completed": len(summaries),
                "planned": len(plan),
                "last_trial": trial["id"],
                "complete": len(summaries) == len(plan),
            },
        )
        print(
            f"END trial={index + 1}/{len(plan)} failure={result['failure_reason']} "
            f"nominal_pass={result['nominal_pass']} push_success={result['push_success']}",
            flush=True,
        )
    if len(summaries) != len(plan):
        raise RuntimeError("Missing planned trials")
    for label, policy in policies.items():
        if (
            checkpoint_digest(policy.checkpoint)
            != provenance["policies"][label]["checkpoint_sha256"]
        ):
            raise RuntimeError("Checkpoint changed during evaluation")
    print(f"COMPLETE {len(plan)} trials saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
