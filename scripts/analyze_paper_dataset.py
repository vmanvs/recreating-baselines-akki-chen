"""Audit the complete paper dataset and export tables and scientific figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from go1_benchmark.native_ppo import checkpoint_digest

LABELS = {
    "ppo_reference": "PPO reference",
    "ppo_gait": "PPO gait",
    "ppo_calf": "PPO gait + calf",
    "mjpc": "MJPC",
}
COLORS = {
    "ppo_reference": "#767676",
    "ppo_gait": "#b76527",
    "ppo_calf": "#0072b2",
    "mjpc": "#009e73",
}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes, n):
    z = 1.959963984540054
    rate = successes / n
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, center - radius), min(1, center + radius)


def moments(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def analyze(dataset, output):
    manifest = json.loads((dataset / "dataset-manifest.json").read_text())
    protocol = manifest["protocol"]
    index = json.loads((dataset / "trial-index.json").read_text())
    expected = {trial["id"] for trial in manifest["planned_trials"]}
    if {trial["id"] for trial in index} != expected or len(index) != len(expected):
        raise ValueError("Dataset is incomplete or contains duplicate trial IDs")
    if output.exists():
        raise FileExistsError("Use a new analysis output directory")
    output.mkdir(parents=True)
    flat, initial, model_hashes, artifacts, derived = [], {}, set(), {}, {}
    joint_work_rows = []
    for label, policy in manifest["policies"].items():
        checkpoint = (
            dataset
            / "frozen-policies"
            / label
            / "checkpoints"
            / Path(policy["checkpoint"]).name
        )
        if (
            not checkpoint.is_dir()
            or checkpoint_digest(checkpoint) != policy["checkpoint_sha256"]
        ):
            raise ValueError(f"Archived policy checkpoint hash mismatch: {label}")
        archived_manifest = json.loads(
            (checkpoint.parent.parent / "manifest.json").read_text()
        )
        if archived_manifest != policy["training_manifest"]:
            raise ValueError(f"Archived training manifest mismatch: {label}")
    if (
        hashlib.sha256((dataset / "libgo1_mjpc.so").read_bytes()).hexdigest()
        != manifest["library_sha256"]
    ):
        raise ValueError("Archived MJPC library hash mismatch")
    for name, digest in manifest["sources"].items():
        if (
            hashlib.sha256((dataset / "frozen-source" / name).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError(f"Frozen source hash mismatch: {name}")
    for trial in index:
        run = dataset / "runs" / trial["id"]
        summary = json.loads((run / "summary.json").read_text())
        if trial != summary:
            raise ValueError(f"Index differs from saved summary: {trial['id']}")
        model_digest = hashlib.sha256((run / "model.mjb").read_bytes()).hexdigest()
        if model_digest != trial["model_sha256"]:
            raise ValueError("Model hash mismatch")
        model_hashes.add(model_digest)
        if trial["seed"] in initial:
            np.testing.assert_array_equal(initial[trial["seed"]], trial["initial_qpos"])
        initial[trial["seed"]] = trial["initial_qpos"]
        with (
            np.load(run / "trajectory.npz") as data,
            np.load(run / "physics.npz") as physics,
        ):
            if not len(data["time_s"]) or not len(physics["time_s"]):
                raise ValueError("Empty trial requires separate investigation")
            np.testing.assert_allclose(np.diff(physics["time_s"]), 0.004, atol=1e-10)
            np.testing.assert_allclose(
                data["time_s"][-1], trial["observed_duration_s"], atol=1e-10
            )
            work = float(
                np.sum(
                    np.maximum(physics["actuator_force"] * physics["qvel"][:, 6:], 0)
                )
                * 0.004
            )
            np.testing.assert_allclose(
                work, trial["positive_work_j_physics"], rtol=1e-10, atol=1e-9
            )
            impulse = float(
                np.linalg.norm(physics["xfrc_applied"], axis=1).sum() * 0.004
            )
            np.testing.assert_allclose(
                impulse, trial["measured_impulse_n_s"], atol=1e-8
            )
            if trial["complete"]:
                np.testing.assert_allclose(
                    trial["observed_duration_s"], trial["duration_s"], atol=1e-8
                )
                np.testing.assert_allclose(
                    impulse, trial["intended_impulse_n_s"], atol=1e-8
                )
            if (
                trial["failure_reason"] == "nonfoot_ground_contact"
                and not physics["nonfoot_ground_contact"].any()
            ):
                raise ValueError("Failure flag unsupported by physics trajectory")
            steady = data["time_s"] >= protocol["warmup_s"]
            world_error = data["global_linvel"][:, :2] - [0.5, 0]
            w, x, y, z = data["qpos"][-1, 3:7]
            derived[trial["id"]] = {
                "steady_world_planar_rmse_m_s": float(
                    np.sqrt(np.mean(np.sum(world_error[steady] ** 2, axis=1)))
                )
                if steady.any()
                else None,
                "final_heading_deg": float(
                    np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                ),
                "steady_lateral_position_rms_m": float(
                    np.sqrt(np.mean(data["qpos"][steady, 1] ** 2))
                )
                if steady.any()
                else None,
                "minimum_calf_clearance_m_physics": float(
                    physics["calf_clearance_m"].min()
                ),
            }
            joint_work = (
                np.maximum(physics["actuator_force"] * physics["qvel"][:, 6:], 0).sum(
                    axis=0
                )
                * 0.004
            )
            for joint, energy in zip(
                (
                    f"{foot}_{joint}"
                    for foot in ("FR", "FL", "RR", "RL")
                    for joint in ("hip", "thigh", "calf")
                ),
                joint_work,
            ):
                joint_work_rows.append(
                    {
                        "trial_id": trial["id"],
                        "controller": trial["controller"],
                        "joint": joint,
                        "positive_work_j": float(energy),
                        "fraction_of_trial_work": float(energy / work)
                        if work > 0
                        else None,
                        "observed_duration_s": trial["observed_duration_s"],
                        "complete": trial["complete"],
                    }
                )
        for name in (
            "summary.json",
            "trajectory.npz",
            "physics.npz",
            "model.mjb",
            "trial.json",
            "experiment.json",
        ):
            path = run / name
            artifacts[path.relative_to(dataset).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        row = {
            key: value
            for key, value in trial.items()
            if not isinstance(value, (dict, list))
        }
        row.update({f"gait_{key}": value for key, value in trial["gait"].items()})
        row.update(derived[trial["id"]])
        row["failure_geometries"] = ";".join(trial["nonfoot_physics_steps_by_geometry"])
        row["full_pulse_delivered"] = bool(
            trial["force_n"]
            and np.isclose(trial["measured_impulse_n_s"], trial["intended_impulse_n_s"])
        )
        flat.append(row)
    if len(model_hashes) != 1:
        raise ValueError("Controllers did not share the same compiled plant")
    write_csv(output / "trials.csv", flat)
    write_csv(output / "joint-work.csv", joint_work_rows)
    policy_rows = []
    learning_rows = []
    for label, policy in manifest["policies"].items():
        training = policy["training_manifest"]
        log = dataset / "frozen-policies" / label / "progress.jsonl"
        if log.is_file():
            for line in log.read_text().splitlines():
                event = json.loads(line)
                if event["step"] <= int(Path(policy["checkpoint"]).name):
                    learning_rows.append(
                        {
                            "controller": label,
                            "step": event["step"],
                            "training_eval_reward": event["metrics"].get(
                                "eval/episode_reward"
                            ),
                            "training_eval_reward_std": event["metrics"].get(
                                "eval/episode_reward_std"
                            ),
                            "training_eval_episode_length": event["metrics"].get(
                                "eval/avg_episode_length"
                            ),
                            "wall_time_s": event["wall_time_s"],
                        }
                    )
        policy_rows.append(
            {
                "controller": label,
                "profile": policy["profile"],
                "training_seed": policy["training_seed"],
                "checkpoint_step": int(Path(policy["checkpoint"]).name),
                "requested_training_timesteps": training["ppo"]["num_timesteps"],
                "training_collisions": training["environment"]["training_collisions"],
                "checkpoint_sha256": policy["checkpoint_sha256"],
                "architecture": json.dumps(training["ppo"]["network_factory"]),
                "training_environment": json.dumps(
                    training["environment"], sort_keys=True
                ),
            }
        )
    write_csv(output / "policy-summary.csv", policy_rows)
    write_csv(output / "training-curves.csv", learning_rows)
    (output / "artifact-sha256.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    nominal, push = [], []
    for controller in protocol["controllers"]:
        group = [
            trial
            for trial in index
            if trial["controller"] == controller and not trial["force_n"]
        ]
        survivors = [trial for trial in group if trial["complete"]]
        passed = sum(bool(trial["nominal_pass"]) for trial in group)
        low, high = wilson(passed, len(group))
        row = {
            "controller": controller,
            "trials": len(group),
            "completed": len(survivors),
            "nominal_passes": passed,
            "pass_rate": passed / len(group),
            "wilson_low": low,
            "wilson_high": high,
            "failure_reasons": json.dumps(
                dict(
                    Counter(
                        trial["failure_reason"]
                        for trial in group
                        if trial["failure_reason"]
                    )
                )
            ),
            "survivor_metric_scope": "Completed 30 s trials only; no imputation for early failures",
        }
        for metric in (
            "steady_mean_forward_velocity_m_s",
            "steady_planar_rmse_m_s",
            "positive_work_cot_physics",
            "positive_work_cot_control",
            "decision_mean_ms",
            "decision_p95_ms",
            "minimum_calf_clearance_m",
            "steady_world_planar_rmse_m_s",
            "steady_lateral_position_rms_m",
            "final_heading_deg",
        ):
            for key, value in moments(
                [
                    derived[trial["id"]].get(metric, trial.get(metric))
                    for trial in survivors
                ]
            ).items():
                row[f"{metric}_{key}"] = value
        for metric in ("phase_match", "one_swing_fraction", "all_air_fraction"):
            for key, value in moments(
                [trial["gait"][metric] for trial in survivors]
            ).items():
                row[f"gait_{metric}_{key}"] = value
        nominal.append(row)
    for controller in protocol["push_controllers"]:
        for direction in protocol["push_directions_deg"]:
            for force in protocol["push_forces_n"]:
                group = [
                    trial
                    for trial in index
                    if trial["controller"] == controller
                    and trial["direction_deg"] == direction
                    and trial["force_n"] == force
                ]
                successes = sum(bool(trial["push_success"]) for trial in group)
                low, high = wilson(successes, len(group))
                row = {
                    "controller": controller,
                    "direction_deg": direction,
                    "force_n": force,
                    "impulse_n_s": force * protocol["pulse_duration_s"],
                    "trials": len(group),
                    "successes": successes,
                    "success_rate": successes / len(group),
                    "wilson_low": low,
                    "wilson_high": high,
                    "prepush_qualified": sum(
                        bool(trial["prepush_qualified"]) for trial in group
                    ),
                    "failed_before_push": sum(
                        trial["first_failure_time_s"] is not None
                        and trial["first_failure_time_s"] < protocol["push_start_s"]
                        for trial in group
                    ),
                    "completed": sum(trial["complete"] for trial in group),
                    "full_pulse_delivered": sum(
                        np.isclose(
                            trial["measured_impulse_n_s"], trial["intended_impulse_n_s"]
                        )
                        for trial in group
                    ),
                    "recovery_scope": "Successful trials only; unrecovered trials remain failures",
                }
                row.update(
                    {
                        f"recovery_time_s_{key}": value
                        for key, value in moments(
                            [
                                trial["recovery_time_s"]
                                for trial in group
                                if trial["push_success"]
                            ]
                        ).items()
                    }
                )
                row["full_pulse_delivered"] = int(row["full_pulse_delivered"])
                row["physical_completion_rate"] = row["completed"] / len(group)
                (
                    row["physical_completion_wilson_low"],
                    row["physical_completion_wilson_high"],
                ) = wilson(row["completed"], len(group))
                push.append(row)
    write_csv(output / "nominal-summary.csv", nominal)
    write_csv(output / "push-summary.csv", push)
    (output / "analysis.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "trial_count": len(index),
                "shared_model_sha256": next(iter(model_hashes)),
                "nominal": nominal,
                "push": push,
            },
            indent=2,
        )
        + "\n"
    )
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=220)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    grid = np.linspace(0, 30, 751)
    for controller in protocol["controllers"]:
        group = [
            trial
            for trial in index
            if trial["controller"] == controller and not trial["force_n"]
        ]
        failure_times = [
            trial["first_failure_time_s"]
            if trial["first_failure_time_s"] is not None
            else np.inf
            for trial in group
        ]
        survival = np.mean(np.array(failure_times)[:, None] > grid[None, :], axis=0)
        ax.step(
            grid,
            survival,
            where="post",
            label=LABELS[controller],
            color=COLORS[controller],
        )
    ax.set(
        xlabel="Simulation time (s)",
        ylabel="Fraction without physical failure",
        ylim=(-0.03, 1.04),
        xlim=(0, 30),
    )
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "nominal-survival")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    for controller, ax in zip(protocol["controllers"], axes.flat):
        trial = next(
            trial
            for trial in index
            if trial["controller"] == controller
            and trial["seed"] == 100
            and not trial["force_n"]
        )
        with np.load(dataset / "runs" / trial["id"] / "trajectory.npz") as data:
            ax.plot(
                data["time_s"],
                data["local_linvel"][:, 0],
                color=COLORS[controller],
                lw=0.9,
            )
        ax.axhline(0.5, color="black", ls="--", lw=0.8)
        if trial["first_failure_time_s"] is not None:
            ax.axvline(trial["first_failure_time_s"], color="red", lw=1)
        ax.set(
            title=f"{LABELS[controller]} · initial seed 100",
            ylabel="Forward speed (m/s)",
            ylim=(-0.1, 0.85),
            xlim=(0, 30),
        )
    for ax in axes[1]:
        ax.set_xlabel("Simulation time (s)")
    save(fig, "nominal-speed-traces")
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for controller in protocol["controllers"]:
        trial = next(
            trial
            for trial in index
            if trial["controller"] == controller
            and trial["seed"] == 100
            and not trial["force_n"]
        )
        with np.load(dataset / "runs" / trial["id"] / "trajectory.npz") as data:
            ax.plot(
                data["qpos"][:, 0],
                data["qpos"][:, 1],
                label=LABELS[controller],
                color=COLORS[controller],
            )
            if not trial["complete"]:
                ax.scatter(
                    data["qpos"][-1, 0],
                    data["qpos"][-1, 1],
                    color=COLORS[controller],
                    marker="x",
                    s=60,
                )
    ax.axhline(0, color="black", ls="--", lw=0.8, label="Straight path reference")
    ax.set(
        xlabel="World forward position (m)",
        ylabel="World lateral position (m)",
        title="Initial seed 100 · crosses mark physical failure",
    )
    ax.legend(fontsize=8)
    save(fig, "nominal-paths")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    for direction, ax in zip(protocol["push_directions_deg"], axes):
        for controller in protocol["push_controllers"]:
            group = [
                row
                for row in push
                if row["controller"] == controller and row["direction_deg"] == direction
            ]
            x = np.array([row["impulse_n_s"] for row in group])
            y = np.array([row["success_rate"] for row in group])
            low = np.array([row["wilson_low"] for row in group])
            high = np.array([row["wilson_high"] for row in group])
            shift = -0.3 if controller == "ppo_calf" else 0.3
            ax.errorbar(
                x + shift,
                y,
                yerr=np.stack((np.maximum(y - low, 0), np.maximum(high - y, 0))),
                marker="o",
                capsize=3,
                color=COLORS[controller],
                label=LABELS[controller],
            )
            for xx, row in zip(x + shift, group):
                ax.annotate(
                    f"{row['successes']}/{row['trials']}",
                    (xx, row["success_rate"]),
                    xytext=(-10 if controller == "ppo_calf" else 10, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        ax.set(
            title=f"World lateral direction {direction}°",
            xlabel="Rectangular pulse impulse (N s)",
            ylim=(-0.05, 1.18),
            xticks=[10, 20, 30],
        )
    axes[0].set_ylabel("Strict success fraction\nWilson 95% interval", fontsize=10)
    axes[1].legend(fontsize=8, loc="center right")
    save(fig, "push-success")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    for direction, ax in zip(protocol["push_directions_deg"], axes):
        for controller in protocol["push_controllers"]:
            group = [
                row
                for row in push
                if row["controller"] == controller and row["direction_deg"] == direction
            ]
            x = np.array([row["impulse_n_s"] for row in group])
            y = np.array([row["physical_completion_rate"] for row in group])
            low = np.array([row["physical_completion_wilson_low"] for row in group])
            high = np.array([row["physical_completion_wilson_high"] for row in group])
            shift = -0.3 if controller == "ppo_calf" else 0.3
            ax.errorbar(
                x + shift,
                y,
                yerr=np.stack((np.maximum(y - low, 0), np.maximum(high - y, 0))),
                marker="o",
                capsize=3,
                color=COLORS[controller],
                label=LABELS[controller],
            )
            for xx, row in zip(x + shift, group):
                ax.annotate(
                    f"{row['completed']}/{row['trials']}",
                    (xx, row["physical_completion_rate"]),
                    xytext=(-10 if controller == "ppo_calf" else 10, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        ax.set(
            title=f"World lateral direction {direction}°",
            xlabel="Rectangular pulse impulse (N s)",
            ylim=(-0.05, 1.18),
            xticks=[10, 20, 30],
        )
    axes[0].set_ylabel("Physical completion fraction\nWilson 95% interval", fontsize=10)
    axes[1].legend(fontsize=8, loc="center right")
    save(fig, "push-physical-survival")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, controller in enumerate(protocol["controllers"]):
        group = [
            trial
            for trial in index
            if trial["controller"] == controller and not trial["force_n"]
        ]
        for j, trial in enumerate(group):
            ax.scatter(
                i + (j - 2) * 0.045,
                trial["decision_mean_ms"],
                color=COLORS[controller],
                s=24,
            )
    ax.axhline(20, color="red", ls="--", lw=1, label="20 ms control budget")
    ax.set(
        yscale="log",
        ylabel="Mean decision wall time per trial (ms)",
        xticks=range(4),
        xticklabels=[LABELS[c] for c in protocol["controllers"]],
    )
    ax.legend(fontsize=8)
    save(fig, "decision-latency")
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7), sharex=True)
    for controller, ax in zip(protocol["controllers"], axes):
        trial = next(
            trial
            for trial in index
            if trial["controller"] == controller
            and trial["seed"] == 100
            and not trial["force_n"]
        )
        with np.load(dataset / "runs" / trial["id"] / "trajectory.npz") as data:
            t = data["time_s"]
            keep = t <= 6
            ax.imshow(
                data["foot_contact"][keep].T,
                interpolation="nearest",
                aspect="auto",
                origin="upper",
                cmap="Blues",
                vmin=0,
                vmax=1,
                extent=(0, t[keep][-1], 3.5, -0.5),
            )
        ax.set(
            title=LABELS[controller],
            yticks=range(4),
            yticklabels=["FR", "FL", "RR", "RL"],
            xlim=(0, 6),
        )
        if trial["first_failure_time_s"] is not None:
            ax.axvline(trial["first_failure_time_s"], color="red", lw=1)
    axes[-1].set_xlabel(
        "Simulation time (s) · blue = instantaneous foot contact; blank = trial ended"
    )
    save(fig, "foot-contact-rasters")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5))
    metrics = [
        ("steady_planar_rmse_m_s", "Planar tracking RMSE (m/s)"),
        ("positive_work_cot_physics", "Positive mechanical work CoT (250 Hz)"),
    ]
    for ax, (metric, label) in zip(axes, metrics):
        for i, controller in enumerate(protocol["controllers"]):
            group = [
                trial
                for trial in index
                if trial["controller"] == controller
                and not trial["force_n"]
                and trial["complete"]
            ]
            for j, trial in enumerate(group):
                ax.scatter(
                    i + (j - 2) * 0.045, trial[metric], color=COLORS[controller], s=24
                )
            if not group:
                ax.text(
                    i,
                    0.05,
                    "No completed\ntrials",
                    ha="center",
                    va="bottom",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                )
        ax.set(
            ylabel=label,
            xticks=range(4),
            xticklabels=["PPO ref.", "PPO gait", "PPO calf", "MJPC"],
        )
        ax.set_title("Completed nominal trials only", fontsize=10)
    save(fig, "tracking-and-energy")
    sampling_rows = []
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for position, controller in enumerate(("ppo_calf", "mjpc")):
        group = [
            trial
            for trial in index
            if trial["controller"] == controller
            and not trial["force_n"]
            and trial["complete"]
        ]
        for j, trial in enumerate(group):
            fine, coarse = (
                trial["positive_work_cot_physics"],
                trial["positive_work_cot_control"],
            )
            difference = 100 * (coarse / fine - 1)
            sampling_rows.append(
                {
                    "trial_id": trial["id"],
                    "controller": controller,
                    "cot_50hz": coarse,
                    "cot_250hz": fine,
                    "relative_difference_percent": difference,
                }
            )
            offset = (j - (len(group) - 1) / 2) * 0.035
            ax.plot(
                [position - 0.12 + offset, position + 0.12 + offset],
                [coarse, fine],
                color=COLORS[controller],
                marker="o",
                lw=0.9,
                ms=3,
            )
    ax.set(
        xticks=[-0.12, 0.12, 0.88, 1.12],
        xticklabels=["PPO\n50 Hz", "PPO\n250 Hz", "MJPC\n50 Hz", "MJPC\n250 Hz"],
        ylabel="Positive mechanical work CoT",
        title="Completed nominal trials · paired endpoint estimates",
    )
    save(fig, "cot-sampling-sensitivity")
    write_csv(output / "cot-sampling-sensitivity.csv", sampling_rows)
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 6), sharex=True)
    for column, controller in enumerate(("ppo_calf", "mjpc")):
        trial = next(
            trial
            for trial in index
            if trial["controller"] == controller
            and trial["seed"] == 100
            and not trial["force_n"]
        )
        with np.load(dataset / "runs" / trial["id"] / "physics.npz") as data:
            mask = (data["time_s"] >= 2) & (data["time_s"] <= 4)
            for joint, name in enumerate(("FR hip", "FR thigh", "FR calf")):
                ax = axes[joint, column]
                ax.plot(
                    data["time_s"][mask],
                    data["actuator_force"][mask, joint],
                    color=COLORS[controller],
                    lw=0.85,
                )
                ax.set(ylabel=f"{name} torque (N m)", xlim=(2, 4))
                if joint == 0:
                    ax.set_title(LABELS[controller])
                if joint == 2:
                    ax.set_xlabel("Simulation time (s)")
    fig.suptitle(
        "Seed 100 · illustrative shared 2–4 s window before first failure", fontsize=10
    )
    save(fig, "front-right-joint-torques")
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 7), sharex=True)
    for row, force in enumerate(protocol["push_forces_n"]):
        for column, direction in enumerate(protocol["push_directions_deg"]):
            ax = axes[row, column]
            for controller in protocol["push_controllers"]:
                trial = next(
                    trial
                    for trial in index
                    if trial["controller"] == controller
                    and trial["seed"] == 100
                    and trial["force_n"] == force
                    and trial["direction_deg"] == direction
                )
                with np.load(dataset / "runs" / trial["id"] / "trajectory.npz") as data:
                    ax.plot(
                        data["time_s"],
                        data["global_linvel"][:, 1],
                        color=COLORS[controller],
                        lw=1,
                        label=LABELS[controller],
                    )
                    if trial["first_failure_time_s"] is not None:
                        ax.scatter(
                            data["time_s"][-1],
                            data["global_linvel"][-1, 1],
                            marker="x",
                            color=COLORS[controller],
                            s=35,
                        )
            ax.axvspan(5, 5.2, color="red", alpha=0.12)
            ax.axhline(0, color="black", lw=0.6, ls="--")
            ax.set(
                title=f"{force} N · world {direction}°",
                ylabel="World lateral speed (m/s)",
                xlim=(0, 12),
            )
            if row == 2:
                ax.set_xlabel("Simulation time (s)")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "Push responses · seed 100 · crosses mark physical failure", fontsize=10
    )
    save(fig, "push-response-traces")
    if learning_rows:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, controller in zip(axes, protocol["policy_directories"]):
            group = [row for row in learning_rows if row["controller"] == controller]
            if group:
                ax.plot(
                    [row["step"] / 1e6 for row in group],
                    [row["training_eval_reward"] for row in group],
                    color=COLORS[controller],
                    marker="o",
                    ms=3,
                )
            ax.set(
                title=LABELS[controller],
                xlabel="Training steps (million)",
                ylabel="Own training evaluation return",
            )
        fig.suptitle(
            "Saved training logs · different rewards; return scales are not comparable",
            fontsize=10,
        )
        save(fig, "training-curves")
    lines = [
        "# Paper dataset results, 30 September 2026",
        "",
        f"All {len(index)} frozen trials were collected and audited.",
        "",
        "The plant, physical checks and measurements are shared native MuJoCo. Policies are frozen. Trials stop at the first physical failure. All failed trials remain in the dataset.",
        "",
        "| Controller | Nominal pass | Completed | Survivor speed (m/s) | Survivor RMSE (m/s) | Survivor mechanical CoT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda value: "N/A" if value is None else f"{value:.4f}"
    for row in nominal:
        lines.append(
            f"| {LABELS[row['controller']]} | {row['nominal_passes']}/{row['trials']} | {row['completed']}/{row['trials']} | {fmt(row['steady_mean_forward_velocity_m_s_mean'])} | {fmt(row['steady_planar_rmse_m_s_mean'])} | {fmt(row['positive_work_cot_physics_mean'])} |"
        )
    lines += [
        "",
        "Speed, RMSE and CoT in this table describe completed nominal trials only. Early failures have no steady-state estimate; their time to failure and failure geometry are in trials.csv. CoT integrates positive actuator work at 250 Hz and divides by mass, gravity and net forward displacement. It is not electrical energy.",
        "",
        "| Controller | Direction | Force (N) | Impulse (N s) | Strict success | Physically completed | Prepush qualified | Full pulse delivered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in push:
        lines.append(
            f"| {LABELS[row['controller']]} | {row['direction_deg']}° | {row['force_n']} | {row['impulse_n_s']:.0f} | {row['successes']}/{row['trials']} | {row['completed']}/{row['trials']} | {row['prepush_qualified']}/{row['trials']} | {row['full_pulse_delivered']}/{row['trials']} |"
        )
    lines += [
        "",
        "Success requires prepush qualification, completing the trial without failure, and returning below 0.15 m/s planar error for at least 0.5 s. This tolerance is prespecified and differs from earlier pilot evaluations. Prepush failures do not establish a disturbance threshold. Push directions are world-frame ±Y; headings may drift.",
        "Physical completion is reported separately from strict success. A contact-free trial may fail prepush tracking qualification or the sustained recovery criterion. The physical-completion plot does not certify recovered command tracking.",
        "",
        "## Interpretation limits",
        "",
        "Only one trained seed per PPO variant is available. The nominal comparison is descriptive, not a causal reward ablation. Five initial poses and three push trials per cell give wide intervals. Wilson intervals describe these finite trial outcomes; they do not establish broad generalization. Upstream MJPC sampling is unseeded. The disturbance grid is coarse and does not identify a maximum robust force.",
        "",
        "All pushes start at 5 s, so gait-aware controllers are tested at one scheduled onset phase. Results cannot establish robustness across other push phases or onset times. The reference PPO is the public Playground-based policy, not the original authors' SB3 controller. policy-summary.csv retains actual checkpoint steps, requested training budgets, architecture and training settings; these variants are selected frozen policies rather than a budget-controlled causal comparison.",
        "",
        "All controllers execute synchronously offline. Latency includes observation/host handling and blocked actor inference or MJPC planning, excludes policy compilation, and is measured on this machine. No delayed-action real-time experiment was performed. MJPC receives exact full state; PPO receives its saved observation with noise disabled. The common native backend is an evaluation transfer from MJX training. No terrain or physical-robot tests were run.",
        "",
        "The laptop temporarily ran on battery during MJPC 50 N -Y push trials. Reported clock dropped to 1.7 GHz and measured planning rose near 500–600 ms, then returned near 230 ms after AC power was connected. Raw timings retain this transition and hardware diagnostic records. The latency figure uses nominal trials; timing is not an isolated fixed-power hardware benchmark.",
        "",
        "## Files",
        "",
        "- trials.csv: all outcomes, including early failures",
        "- joint-work.csv: per-joint positive work for every trial; durations and completion flags retained",
        "- policy-summary.csv: evaluated checkpoint identities and training settings",
        "- cot-sampling-sensitivity.csv: paired 50 Hz and 250 Hz endpoint estimates for completed nominal trials",
        "- training-curves.csv: existing training evaluation logs through the selected checkpoints; these are not held-out benchmark results",
        "- nominal-summary.csv: nominal counts, Wilson intervals, and survivor metrics with sample counts",
        "- push-summary.csv: disturbance outcomes, qualification and delivery counts",
        "- analysis.json: machine-readable aggregates",
        "- artifact-sha256.json: raw artifact integrity inventory",
        "- PNG and PDF figures: survival, speed, paths, pushes, latency, gait contacts, joint torques, tracking and energy",
        "",
        "World-frame tracking error, final heading and lateral drift are retained in trials.csv and nominal-summary.csv. Nominal success uses the frozen body-frame criterion; it does not certify negligible straight-line drift. The illustrative joint-torque plot uses the shared 2–4 s interval of seed 100, before its first failure, and is not a controlled joint-effort experiment.",
        "The CoT sampling figure compares two endpoint quadratures of the same trajectories. The 250 Hz estimate is the declared metric, not a demonstrated continuous-time ground truth. No physics time-step convergence test was performed.",
        "",
        f"Shared model SHA256: `{next(iter(model_hashes))}`",
        f"Protocol SHA256: `{manifest['protocol_sha256']}`",
        "",
    ]
    (output / "paper-data-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {"trial_count": len(index), "nominal": nominal, "push": push}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.dataset.resolve(), args.output.resolve())
