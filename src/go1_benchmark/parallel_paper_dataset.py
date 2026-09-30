"""Process-isolated CPU collection using the unchanged native trial physics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

_worker = None


def initialize_worker(root, output, config, protocol, affinities):
    """Set placement before importing NumPy or initializing JAX thread pools."""
    global _worker
    os.environ["JAX_PLATFORMS"] = "cpu"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    if hasattr(os, "sched_setaffinity"):
        # A queue assigns each spawned process one disjoint CPU set.
        os.sched_setaffinity(0, affinities.get())
    import mujoco

    from go1_benchmark.mjpc_model import make_model

    model = mujoco.MjModel.from_binary_path(str(Path(output) / "planner-model.mjb"))
    # Use the same construction path to ensure pinned model assets are available.
    if model.nu != make_model(config).nu:
        raise RuntimeError("Worker model mismatch")
    _worker = (Path(root), Path(output), config, protocol, model, {})


def execute_trial(trial):
    from go1_benchmark.mjpc_controller import MJPCController
    from go1_benchmark.native_ppo import NativePPO, checkpoint_digest
    from go1_benchmark.paper_dataset import run_trial

    root, output, config, protocol, model, actors = _worker
    manifest = json.loads((output / "dataset-manifest.json").read_text())
    from go1_benchmark.paper_dataset import sha256

    if (
        any(
            sha256(root / "src/go1_benchmark" / name) != digest
            for name, digest in manifest["sources"].items()
        )
        or sha256(output / "libgo1_mjpc.so") != manifest["library_sha256"]
    ):
        raise RuntimeError("Evaluator or frozen library changed during collection")
    run = output / "runs" / trial["id"]
    if trial["controller"] == "mjpc":
        with MJPCController(
            output / "libgo1_mjpc.so", output / "planner-model.mjb", config
        ) as controller:
            result = run_trial(model, controller, config, protocol, trial, run)
        devices = ["native MuJoCo / CPU MJPC"]
    else:
        label = trial["controller"]
        if label not in actors:
            actors[label] = NativePPO(
                output / "frozen-policies" / label, model, config["environment"]
            )
        controller = actors[label]
        if (
            checkpoint_digest(controller.checkpoint)
            != manifest["policies"][label]["checkpoint_sha256"]
        ):
            raise RuntimeError("Frozen checkpoint changed")
        import jax

        if any(device.platform != "cpu" for device in jax.devices()):
            raise RuntimeError("This collector requires CPU JAX")
        devices = controller.provenance["inference_devices"]
        controller.reset()
        result = run_trial(model, controller, config, protocol, trial, run)
    from go1_benchmark.evaluate_mjpc import write_json

    write_json(
        run / "execution.json",
        {
            "pid": os.getpid(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "devices": devices,
            "timing_scope": "Parallel CPU, resource-contended; not serial latency",
        },
    )
    return result


def collect(args):
    # Parent must not initialize JAX, and child processes always use spawn.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import mujoco

    from go1_benchmark.evaluate_mjpc import pulse_impulse, write_json
    from go1_benchmark.mjpc_model import MJPC_REVISION, make_model
    from go1_benchmark.native_ppo import checkpoint_digest
    from go1_benchmark.paper_dataset import sha256, trials

    root, output = args.root.resolve(), args.output.resolve()
    protocol = json.loads(args.protocol.read_text())
    config = json.loads((root / "configs/mjpc_go1.json").read_text())
    pulse_impulse(0, protocol["pulse_duration_s"], protocol["pulse_shape"])
    cpus = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    if not 1 <= args.workers <= len(cpus):
        raise ValueError("Worker count exceeds available affinity CPUs")
    affinity_sets = [cpus[i :: args.workers] for i in range(args.workers)]
    if any(len(c) < config["planner"]["threads"] for c in affinity_sets):
        raise ValueError("Too few affinity CPUs per worker for unchanged MJPC threads")
    sources = [
        root / "src/go1_benchmark" / name
        for name in (
            "parallel_paper_dataset.py",
            "paper_dataset.py",
            "native_ppo.py",
            "mjpc_model.py",
            "mjpc_controller.py",
            "metrics.py",
            "evaluate_mjpc.py",
        )
    ]
    plan = list(trials(protocol))
    if not plan or len({t["id"] for t in plan}) != len(plan):
        raise ValueError("Empty or duplicate trial plan")
    policies = {}
    for label, directory in protocol["policy_directories"].items():
        policy_root = root / directory
        training = json.loads((policy_root / "manifest.json").read_text())
        checkpoint = max(
            (
                p
                for p in (policy_root / "checkpoints").iterdir()
                if p.is_dir() and p.name.isdigit()
            ),
            key=lambda p: int(p.name),
        )
        policies[label] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest(checkpoint),
            "training_manifest": training,
            "training_manifest_sha256": sha256(policy_root / "manifest.json"),
            "training_seed": training["seed"],
            "profile": training["profile"],
        }
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "protocol_sha256": sha256(args.protocol),
        "mjpc_config_sha256": sha256(root / "configs/mjpc_go1.json"),
        "library_sha256": sha256(args.library),
        "mjpc_revision": MJPC_REVISION,
        "sources": {p.name: sha256(p) for p in sources},
        "policies": policies,
        "planned_trials": plan,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("mujoco", "numpy", "jax", "jaxlib", "brax", "flax", "playground")
        },
        "execution": {
            "mode": "parallel CPU spawn",
            "workers": args.workers,
            "affinity_sets": affinity_sets,
            "planner_threads": config["planner"]["threads"],
            "timing_scope": "Contended parallel timing, not serial realtime benchmark",
        },
    }
    if output.exists():
        old = json.loads((output / "dataset-manifest.json").read_text())
        for key in (
            "protocol_sha256",
            "mjpc_config_sha256",
            "library_sha256",
            "sources",
            "policies",
            "execution",
            "packages",
        ):
            if old[key] != manifest[key]:
                raise ValueError(f"Cannot resume after changing {key}")
    else:
        output.mkdir(parents=True)
        (output / "runs").mkdir()
        (output / "frozen-source").mkdir()
        for p in sources:
            shutil.copy2(p, output / "frozen-source" / p.name)
        shutil.copy2(args.library, output / "libgo1_mjpc.so")
        for label, identity in policies.items():
            dest = output / "frozen-policies" / label
            checkpoint = Path(identity["checkpoint"])
            shutil.copytree(checkpoint, dest / "checkpoints" / checkpoint.name)
            for name in ("manifest.json", "experiment.json", "progress.jsonl"):
                source = checkpoint.parent.parent / name
                if source.exists():
                    shutil.copy2(source, dest / name)
        mujoco.mj_saveModel(make_model(config), str(output / "planner-model.mjb"))
        write_json(output / "protocol.json", protocol)
        write_json(output / "mjpc-config.json", config)
        write_json(output / "dataset-manifest.json", manifest)
    # Detect damage to frozen dependencies before starting or resuming.
    if sha256(output / "libgo1_mjpc.so") != manifest["library_sha256"]:
        raise RuntimeError("Frozen library mismatch")
    for p in sources:
        if sha256(output / "frozen-source" / p.name) != manifest["sources"][p.name]:
            raise RuntimeError("Frozen evaluator mismatch")
    results, pending = {}, []
    for trial in plan:
        run = output / "runs" / trial["id"]
        if (run / "summary.json").exists():
            if not (run / "execution.json").exists():
                raise RuntimeError(f"Incomplete execution record retained: {run}")
            result = json.loads((run / "summary.json").read_text())
            if any(result[k] != v for k, v in trial.items()):
                raise RuntimeError("Saved trial identity differs")
            results[trial["id"]] = result
        elif run.exists():
            raise RuntimeError(
                f"Incomplete trial retained: {run}; inspect before retrying"
            )
        else:
            pending.append(trial)

    def save_status():
        ordered = [results[t["id"]] for t in plan if t["id"] in results]
        write_json(output / "trial-index.json", ordered)
        write_json(
            output / "collection-status.json",
            {
                "completed": len(ordered),
                "planned": len(plan),
                "complete": len(ordered) == len(plan),
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )

    save_status()
    context = mp.get_context("spawn")
    queue = context.Queue()
    for affinity in affinity_sets:
        queue.put(affinity)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(str(root), str(output), config, protocol, queue),
    ) as pool:
        futures = {pool.submit(execute_trial, t): t for t in pending}
        for future in as_completed(futures):
            trial = futures[future]
            result = future.result()
            results[trial["id"]] = result
            save_status()
            print(
                f"DONE {len(results)}/{len(plan)} {trial['id']} success={result['push_success']} failure={result['failure_reason']}",
                flush=True,
            )
    if any(sha256(p) != manifest["sources"][p.name] for p in sources):
        raise RuntimeError("Evaluator changed during collection")
    print(f"COMPLETE {len(results)} trials: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
