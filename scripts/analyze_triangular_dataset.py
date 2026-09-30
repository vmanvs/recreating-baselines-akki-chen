"""Audit triangular native trials and compare with the frozen rectangular set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from go1_benchmark.evaluate_mjpc import force_at, pulse_impulse, write_json
from go1_benchmark.native_ppo import checkpoint_digest
from go1_benchmark.paper_dataset import sha256


def interval(successes, n):
    z = 1.959963984540054
    rate = successes / n
    divisor = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / divisor
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / divisor
    return max(0, center - half), min(1, center + half)


def audit(dataset):
    manifest = json.loads((dataset / "dataset-manifest.json").read_text())
    protocol = manifest["protocol"]
    config = json.loads((dataset / "mjpc-config.json").read_text())
    index = json.loads((dataset / "trial-index.json").read_text())
    plan = {t["id"]: t for t in manifest["planned_trials"]}
    if len(index) != len(plan) or {t["id"] for t in index} != set(plan):
        raise ValueError("Incomplete or duplicate dataset")
    for name, digest in manifest["sources"].items():
        assert sha256(dataset / "frozen-source" / name) == digest
    assert sha256(dataset / "libgo1_mjpc.so") == manifest["library_sha256"]
    for label, identity in manifest["policies"].items():
        checkpoint = (
            dataset
            / "frozen-policies"
            / label
            / "checkpoints"
            / Path(identity["checkpoint"]).name
        )
        assert checkpoint_digest(checkpoint) == identity["checkpoint_sha256"]
        assert (
            json.loads((checkpoint.parent.parent / "manifest.json").read_text())
            == identity["training_manifest"]
        )
    poses, models, artifacts = {}, set(), {}
    dt = config["environment"]["simulation_dt_s"]
    for trial in index:
        run = dataset / "runs" / trial["id"]
        assert json.loads((run / "summary.json").read_text()) == trial
        assert all(trial[k] == v for k, v in plan[trial["id"]].items())
        digest = sha256(run / "model.mjb")
        assert digest == trial["model_sha256"]
        models.add(digest)
        if trial["seed"] in poses:
            np.testing.assert_array_equal(poses[trial["seed"]], trial["initial_qpos"])
        poses[trial["seed"]] = trial["initial_qpos"]
        with (
            np.load(run / "physics.npz") as physics,
            np.load(run / "trajectory.npz") as trace,
        ):
            assert len(physics["time_s"]) and len(trace["time_s"])
            np.testing.assert_allclose(np.diff(physics["time_s"]), dt, atol=1e-10)
            measured = np.linalg.norm(physics["xfrc_applied"], axis=1)
            expected = np.array(
                [
                    force_at(
                        t - dt / 2,
                        trial["force_n"],
                        protocol["push_start_s"],
                        protocol["pulse_duration_s"],
                        protocol["pulse_shape"],
                    )
                    for t in physics["time_s"]
                ]
            )
            np.testing.assert_allclose(measured, expected, atol=1e-7)
            impulse = measured.sum() * dt
            np.testing.assert_allclose(
                impulse, trial["measured_impulse_n_s"], atol=1e-8
            )
            intended = pulse_impulse(
                trial["force_n"], protocol["pulse_duration_s"], protocol["pulse_shape"]
            )
            np.testing.assert_allclose(
                intended, trial["intended_impulse_n_s"], atol=1e-8
            )
            if trial["complete"]:
                np.testing.assert_allclose(impulse, intended, atol=1e-8)
                np.testing.assert_allclose(
                    physics["time_s"][-1], trial["duration_s"], atol=1e-8
                )
            if trial["failure_reason"] == "nonfoot_ground_contact":
                assert physics["nonfoot_ground_contact"].any()
            if trial["failure_reason"] == "inverted_torso":
                assert trace["up_z"][-1] < 0
            work = (
                np.maximum(physics["actuator_force"] * physics["qvel"][:, 6:], 0).sum()
                * dt
            )
            np.testing.assert_allclose(
                work, trial["positive_work_j_physics"], rtol=1e-10
            )
        for p in run.iterdir():
            if p.is_file():
                artifacts[p.relative_to(dataset).as_posix()] = sha256(p)
    assert len(models) == 1
    return manifest, index, poses, next(iter(models)), artifacts


def analyze(args):
    triangle, tri_index, tri_poses, tri_model, artifacts = audit(args.dataset)
    rectangle, rect_index, rect_poses, rect_model, _ = audit(args.rectangular)
    assert tri_model == rect_model
    for seed, pose in tri_poses.items():
        np.testing.assert_array_equal(pose, rect_poses[seed])
    for key in ("library_sha256", "mjpc_config_sha256", "mjpc_revision"):
        assert triangle[key] == rectangle[key]
    assert (
        triangle["policies"]["ppo_calf"]["checkpoint_sha256"]
        == rectangle["policies"]["ppo_calf"]["checkpoint_sha256"]
    )
    for key in (
        "push_seeds",
        "initial_joint_noise_rad",
        "push_duration_total_s",
        "warmup_s",
        "push_forces_n",
        "push_directions_deg",
        "push_start_s",
        "pulse_duration_s",
        "max_nominal_rmse_m_s",
        "recovery_tolerance_m_s",
        "recovery_dwell_s",
    ):
        assert triangle["protocol"][key] == rectangle["protocol"][key]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for shape, manifest, index in (
        ("rectangular", rectangle, rect_index),
        ("triangular", triangle, tri_index),
    ):
        p = manifest["protocol"]
        for controller in p["push_controllers"]:
            for direction in p["push_directions_deg"]:
                for force in p["push_forces_n"]:
                    group = [
                        t
                        for t in index
                        if t["controller"] == controller
                        and t["force_n"] == force
                        and t["direction_deg"] == direction
                    ]
                    successes = sum(bool(t["push_success"]) for t in group)
                    completed = sum(bool(t["complete"]) for t in group)
                    rows.append(
                        {
                            "shape": shape,
                            "controller": controller,
                            "direction_deg": direction,
                            "force_n": force,
                            "impulse_n_s": pulse_impulse(
                                force, p["pulse_duration_s"], shape
                            ),
                            "trials": len(group),
                            "strict_successes": successes,
                            "physical_completions": completed,
                            "strict_wilson95": interval(successes, len(group)),
                            "physical_wilson95": interval(completed, len(group)),
                            "prepush_qualified": sum(
                                bool(t["prepush_qualified"]) for t in group
                            ),
                            "full_pulse_delivered": sum(
                                np.isclose(
                                    t["measured_impulse_n_s"], t["intended_impulse_n_s"]
                                )
                                for t in group
                            ),
                            "failure_reasons": dict(
                                Counter(
                                    t["failure_reason"]
                                    for t in group
                                    if t["failure_reason"]
                                )
                            ),
                            "successful_recovery_times_s": [
                                t["recovery_time_s"] for t in group if t["push_success"]
                            ],
                        }
                    )
    # Convert NumPy scalar count to JSON native types.
    for row in rows:
        row["full_pulse_delivered"] = int(row["full_pulse_delivered"])
    write_json(args.output / "push-comparison.json", rows)
    write_json(args.output / "artifact-sha256.json", artifacts)
    write_json(
        args.output / "audit.json",
        {
            "passed": True,
            "triangular_trials": len(tri_index),
            "shared_model_sha256": tri_model,
            "matched_initial_poses": True,
            "matched_checkpoint_config_library": True,
            "physics_pulse_verified": True,
        },
    )
    labels = {"ppo_calf": "PPO gait + calf", "mjpc": "MJPC"}
    colors = {"ppo_calf": "#0072b2", "mjpc": "#009e73"}
    for axis_key, axis_label, filename in (
        ("force_n", "Peak force (N)", "push-outcomes-by-peak"),
        ("impulse_n_s", "Commanded impulse (N s)", "push-outcomes-by-impulse"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
        for col, direction in enumerate((90, 270)):
            for r, metric in enumerate(("strict_successes", "physical_completions")):
                ax = axes[r, col]
                for controller, label in labels.items():
                    for shape in ("rectangular", "triangular"):
                        group = [
                            v
                            for v in rows
                            if v["controller"] == controller
                            and v["shape"] == shape
                            and v["direction_deg"] == direction
                        ]
                        x = np.array([v[axis_key] for v in group], dtype=float)
                        y = np.array([v[metric] / v["trials"] for v in group])
                        bounds = np.array(
                            [
                                v["strict_wilson95" if r == 0 else "physical_wilson95"]
                                for v in group
                            ]
                        )
                        offset = (-1.5 if controller == "ppo_calf" else 1.5) * (
                            1 if axis_key == "force_n" else 0.1
                        )
                        offset += (-0.4 if shape == "rectangular" else 0.4) * (
                            1 if axis_key == "force_n" else 0.1
                        )
                        ax.errorbar(
                            x + offset,
                            y,
                            # Endpoint Wilson bounds can differ from 0/1 by a
                            # rounding ulp. Preserve bounds, clamp display lengths.
                            yerr=np.maximum([y - bounds[:, 0], bounds[:, 1] - y], 0),
                            color=colors[controller],
                            marker="o" if shape == "rectangular" else "s",
                            linestyle="--" if shape == "rectangular" else "-",
                            linewidth=1,
                            capsize=2,
                            label=f"{label}, {shape}",
                        )
                ax.set_ylim(-0.05, 1.05)
                ax.grid(alpha=0.2)
                ax.set_title(
                    f"{'+' if direction == 90 else '-'}Y / {'strict success' if r == 0 else 'physical completion'}",
                    fontsize=10,
                )
                ax.set_xlabel(axis_label)
                if col == 0:
                    ax.set_ylabel("Fraction (Wilson 95% interval)")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(args.output / f"{filename}.{ext}", dpi=180)
        plt.close(fig)
    lines = [
        "# Triangular pulse follow-up",
        "",
        "All 36 triangular trials were retained and audited against saved native MuJoCo physics. The original rectangular dataset is unchanged.",
        "",
        "Only the commanded pulse shape changed: 0.2 s symmetric triangles at 50/100/150 N deliver 5/10/15 N s rather than rectangular 10/20/30 N s. The policy, planner configuration, library, full-collision plant, initial poses, failure checks and tracking criteria match the original tests.",
        "",
        "| Pulse | Controller | Direction | Peak N | Impulse N s | Strict success | Physical completion | Prepush qualified | Full pulse |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['shape']} | {labels[row['controller']]} | {row['direction_deg']} | {row['force_n']} | {row['impulse_n_s']:g} | {row['strict_successes']}/{row['trials']} | {row['physical_completions']}/{row['trials']} | {row['prepush_qualified']}/{row['trials']} | {row['full_pulse_delivered']}/{row['trials']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Triangles reduce total impulse at the same peak, so any improved completion is a response to a different disturbance, not an improved or retrained controller. Equal impulse also does not imply equal peak force or force history. Three initial-condition trials per cell and one trained PPO seed are insufficient to establish a maximum robust force.",
        "",
        "The upstream MJPC sampler is unseeded. Matched poses do not guarantee paired stochastic planning trajectories. Any MJPC difference includes sampling variability, so this is a descriptive follow-up, not an isolated causal shape experiment.",
        "",
        "Triangular trials ran in two isolated CPU processes with unchanged four-thread MJPC settings and 4 ms physics checks. Decision wall times are resource-contended and must not replace the original serial latency measurements. No GPU backend, training, reward changes or controller tuning were used.",
        "",
        "Trials remain 12 s with pushes at 5 s, rather than the original paper's declared 50 s. Existing 30 s nominal tests remain in the original dataset. Gait onset phase, terrain and training-seed generalization were not expanded.",
        "",
        "Files: push-comparison.json, audit.json, artifact-sha256.json, and peak/impulse PNG/PDF outcome figures. Raw trajectories, physics, summaries, worker placement, frozen policies, source and library remain in the dataset directory.",
        "",
    ]
    (args.output / "triangular-report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    for controller in labels:
        group = [t for t in tri_index if t["controller"] == controller]
        print(
            f"{controller}: strict={sum(bool(t['push_success']) for t in group)}/{len(group)} physical={sum(bool(t['complete']) for t in group)}/{len(group)}",
            flush=True,
        )
    print(f"AUDIT PASSED; report {args.output / 'triangular-report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rectangular", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
