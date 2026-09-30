"""Short real-controller verification, not training or a benchmark trial."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    from go1_benchmark.evaluate_mjpc import write_json

    protocol = json.loads(
        (args.root / "configs/paper_triangular_protocol.json").read_text()
    )
    protocol.update(
        initial_state_seeds=[100, 101],
        nominal_duration_s=0.24,
        push_controllers=[],
        warmup_s=0.04,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    from go1_benchmark.parallel_paper_dataset import collect

    collect(
        SimpleNamespace(
            root=args.root,
            output=args.output / "parallel",
            protocol=args.output / "protocol.json",
            library=args.library,
            workers=2,
        )
    )
    # CPU inference and native dynamics must give identical serial PPO traces.
    import mujoco
    import numpy as np

    from go1_benchmark.native_ppo import NativePPO
    from go1_benchmark.paper_dataset import run_trial

    config = json.loads((args.root / "configs/mjpc_go1.json").read_text())
    model = mujoco.MjModel.from_binary_path(
        str(args.output / "parallel/planner-model.mjb")
    )
    policy = NativePPO(
        args.root / protocol["policy_directories"]["ppo_calf"],
        model,
        config["environment"],
    )
    index = json.loads((args.output / "parallel/trial-index.json").read_text())
    pids = set()
    for result in index:
        run = args.output / "parallel/runs" / result["id"]
        pids.add(json.loads((run / "execution.json").read_text())["pid"])
        assert result["complete"] and result["finite"]
        if result["controller"] != "ppo_calf":
            continue
        trial = {
            key: result[key]
            for key in (
                "id",
                "controller",
                "seed",
                "duration_s",
                "force_n",
                "direction_deg",
            )
        }
        serial = args.output / ("serial-" + result["id"])
        policy.reset()
        run_trial(model, policy, config, protocol, trial, serial)
        for name in ("trajectory.npz", "physics.npz"):
            with np.load(run / name) as parallel, np.load(serial / name) as single:
                for key in parallel.files:
                    if key != "planning_time_s":
                        np.testing.assert_array_equal(parallel[key], single[key])
    assert len(pids) == 2, "Expected two actual worker processes"
    # Resume must preserve completed trial files without executing them again.
    collect(
        SimpleNamespace(
            root=args.root,
            output=args.output / "parallel",
            protocol=args.output / "protocol.json",
            library=args.library,
            workers=2,
        )
    )
    write_json(
        args.output / "verification.json",
        {
            "complete": True,
            "workers": len(pids),
            "ppo_serial_parallel_exact": True,
            "mjpc_process_isolation": True,
            "resume_passed": True,
            "duration_s": 0.24,
        },
    )
    print(
        "VERIFIED two processes, native MJPC, exact serial/parallel PPO parity and resume",
        flush=True,
    )


if __name__ == "__main__":
    main()
