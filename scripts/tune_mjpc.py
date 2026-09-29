"""Sequential MJPC development repeats, separate from held-out paper data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--library", type=Path, default=Path(".build/mjpc/libgo1_mjpc.so")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=12)
    parser.add_argument("--warmup-s", type=float, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--initial-joint-noise-rad", type=float, default=0.01)
    parser.add_argument("--require-walking", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    configs = [p.resolve() for p in args.configs]
    if len({p.stem for p in configs}) != len(configs):
        parser.error("Config names must be unique")
    root = Path(__file__).resolve().parents[1]
    output, library = args.output.resolve(), args.library.resolve()
    from go1_benchmark.mjpc_model import validate_config

    contents = {p: p.read_text() for p in configs}
    for raw in contents.values():
        validate_config(json.loads(raw))
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    report = {
        "scope": "Development validation, not held-out paper data",
        "planner_rng": "Unseeded upstream absl::BitGen; repeat seeds control initial jitter only",
        "library_sha256": library_hash,
        "runs": [],
    }
    for config in configs:
        frozen = output / config.name
        frozen.write_text(contents[config])
        for repeat in range(args.repeats):
            if hashlib.sha256(library.read_bytes()).hexdigest() != library_hash:
                raise RuntimeError(
                    "Library changed during validation; do not rebuild concurrently"
                )
            run = output / f"{config.stem}-repeat-{repeat:02d}"
            command = [
                sys.executable,
                "-m",
                "go1_benchmark.evaluate_mjpc",
                "--config",
                str(frozen),
                "--library",
                str(library),
                "--output",
                str(run),
                "--duration-s",
                str(args.duration_s),
                "--warmup-s",
                str(args.warmup_s),
                "--seed",
                str(repeat),
                "--initial-joint-noise-rad",
                str(args.initial_joint_noise_rad),
            ]
            if args.require_walking:
                command.append("--require-walking")
            print(f"BEGIN {config.stem} repeat={repeat}", flush=True)
            with (output / f"{run.name}.log").open("w") as log:  # noqa: SIM117
                with subprocess.Popen(
                    command,
                    cwd=root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ) as proc:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith("MJPC steps="):
                            print(line.rstrip(), flush=True)
                    return_code = proc.wait()
            summary_path = run / "summary.json"
            summary = (
                json.loads(summary_path.read_text()) if summary_path.exists() else None
            )
            record = {
                "config": str(config),
                "repeat": repeat,
                "config_sha256": hashlib.sha256(contents[config].encode()).hexdigest(),
                "command": command,
                "exit_code": return_code,
                "summary": summary,
            }
            report["runs"].append(record)
            (output / "development-report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "repeat": repeat,
                        "nominal_pass": summary.get("nominal_pass")
                        if summary
                        else None,
                        "walking_pass": summary.get("walking_pass")
                        if summary
                        else None,
                    }
                ),
                flush=True,
            )
    if any(run["exit_code"] for run in report["runs"]):
        raise SystemExit("Some rollouts errored; inspect development-report.json")
    print(f"Development report: {output / 'development-report.json'}")


if __name__ == "__main__":
    main()
