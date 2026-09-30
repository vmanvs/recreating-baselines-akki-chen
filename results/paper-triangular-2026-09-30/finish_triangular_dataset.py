"""One-shot post-processing for an already running triangular collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rectangular", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collector-pid", type=int, required=True)
    args = parser.parse_args()
    print(
        f"Waiting for existing collection PID {args.collector_pid}; no new trials will be started",
        flush=True,
    )
    while True:
        status = json.loads((args.dataset / "collection-status.json").read_text())
        alive = process_alive(args.collector_pid)
        # Wait for the parent to exit, including its final source-identity check.
        if not alive:
            if not status["complete"]:
                raise RuntimeError(
                    "Collector exited before completion; saved trials retained"
                )
            break
        time.sleep(5)
    # A changed source after final validation must never be silently accepted.
    manifest = json.loads((args.dataset / "dataset-manifest.json").read_text())
    source_root = Path(__file__).resolve().parents[1] / "src/go1_benchmark"
    for name, digest in manifest["sources"].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                "Evaluator changed during the batch; investigate before analysis"
            )
    from analyze_triangular_dataset import analyze

    analyze(args)
    # Preserve the exact post-processing code with its generated report.
    for name in ("analyze_triangular_dataset.py", "finish_triangular_dataset.py"):
        shutil.copy2(Path(__file__).parent / name, args.output / name)
    archive = args.dataset.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError("Refusing to overwrite an existing raw-data archive")
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as bundle:
        for directory, prefix in (
            (args.dataset, args.dataset.name),
            (args.output, "results/" + args.output.name),
        ):
            for item in sorted(directory.rglob("*")):
                if item.is_file():
                    bundle.write(
                        item, prefix + "/" + item.relative_to(directory).as_posix()
                    )
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"Archive CRC failed: {bad}")
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    checksum = f"{digest}  {archive.name}\n"
    archive.with_suffix(".zip.sha256").write_text(checksum)
    (args.output / "archive-checksum.sha256").write_text(checksum)
    print(f"COMPLETE post-processing; archive={archive}; sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
