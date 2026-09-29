"""Build the headless native MJPC bridge using this Python's MuJoCo library."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path(".build/mjpc"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--mjpc-source", type=Path, help="Optional checkout at the pinned revision"
    )
    args = parser.parse_args()
    if os.name == "nt":
        raise SystemExit("Build and run in WSL Ubuntu or Linux, not native Windows.")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for tool in ("cmake", "ninja", "c++", "git"):
        if not shutil.which(tool):
            raise SystemExit(
                f"Missing {tool}. Install build-essential cmake ninja-build git."
            )
    import mujoco

    root = Path(__file__).resolve().parents[1]
    package = Path(mujoco.__file__).resolve().parent
    libraries = list(package.glob("libmujoco.so.*"))
    if len(libraries) != 1:
        raise SystemExit("Expected one libmujoco.so in the active MuJoCo Python wheel")
    build = args.build_dir.resolve()
    command = [
        "cmake",
        "-S",
        str(root / "mjpc"),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DMUJOCO_INCLUDE_DIR={package / 'include'}",
        f"-DMUJOCO_LIBRARY={libraries[0]}",
    ]
    if args.mjpc_source:
        from go1_benchmark.mjpc_model import MJPC_REVISION

        source = args.mjpc_source.resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != MJPC_REVISION:
            raise SystemExit(f"MJPC source must be at {MJPC_REVISION}, not {revision}")
        subprocess.run(
            ["git", "-C", str(source), "diff", "--exit-code", revision, "--"],
            check=True,
        )
        command.append(f"-DFETCHCONTENT_SOURCE_DIR_MJPC_SOURCE={source}")
    subprocess.run(command, check=True)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "go1_mjpc",
            "--parallel",
            str(args.jobs),
        ],
        check=True,
    )
    (build / "build-manifest.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "mujoco": mujoco.__version__,
                "mujoco_library": str(libraries[0]),
                "command": command,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"MJPC built: {build / 'libgo1_mjpc.so'}")


if __name__ == "__main__":
    main()
