"""Train the fixed-velocity Go1 PPO baseline on MuJoCo Playground."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from go1_benchmark.fixed_velocity_env import make_fixed_velocity_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ppo_fixed_velocity.json"


def _load_config(path: Path, profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        selected = document["profiles"][profile]
    except KeyError as exc:
        available = ", ".join(document.get("profiles", {}))
        raise ValueError(f"Unknown profile {profile!r}; choose one of: {available}") from exc
    return document, selected


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def train(args: argparse.Namespace) -> Path:
    # Set accelerator behavior before importing JAX.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper
    from mujoco_playground.config import locomotion_params

    document, profile = _load_config(args.config, args.profile)
    env_cfg = document["environment"]
    command = tuple(float(value) for value in env_cfg["command_m_s_rad_s"])
    env = make_fixed_velocity_env(
        command=command,
        impl=env_cfg["implementation"],
        full_collisions=False,
        noise_level=float(env_cfg["observation_noise_level"]),
    )

    ppo_cfg = locomotion_params.brax_ppo_config(
        "Go1JoystickFlatTerrain", env_cfg["implementation"]
    )
    for key in (
        "num_timesteps",
        "learning_rate",
        "clipping_epsilon",
        "num_envs",
        "num_evals",
    ):
        if key in profile:
            ppo_cfg[key] = profile[key]

    output = args.output.resolve()
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"

    manifest = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "profile": args.profile,
        "seed": args.seed,
        "command_m_s_rad_s": command,
        "project_git_revision": _git_revision(PROJECT_ROOT),
        "upstream": document["upstream"],
        "environment": env_cfg,
        "ppo": ppo_cfg.to_dict(),
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    _write_json(output / "manifest.json", manifest)

    network_config = ppo_cfg.network_factory.to_dict()
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_config
    )
    training_params = ppo_cfg.to_dict()
    training_params.pop("network_factory")
    num_eval_envs = int(profile.get("num_eval_envs", args.num_eval_envs))
    training_params.pop("num_eval_envs", None)

    def progress(step: int, metrics: dict[str, Any]) -> None:
        record = {
            "step": int(step),
            "wall_time_s": time.monotonic() - started,
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        reward = record["metrics"].get("eval/episode_reward")
        print(f"step={step} eval_reward={reward} wall_time_s={record['wall_time_s']:.1f}")

    started = time.monotonic()
    make_inference_fn, _, metrics = ppo.train(
        environment=env,
        progress_fn=progress,
        network_factory=network_factory,
        seed=args.seed,
        restore_checkpoint_path=(
            str(args.resume.resolve()) if args.resume is not None else None
        ),
        save_checkpoint_path=str(checkpoint_dir),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        **training_params,
    )
    del make_inference_fn
    _write_json(
        output / "final_metrics.json",
        {key: float(value) for key, value in metrics.items()},
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Experiment JSON config"
    )
    parser.add_argument(
        "--profile",
        choices=("paper_budget_proxy", "playground_reference"),
        default="paper_budget_proxy",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = train(args)
    print(f"Training artifacts written to {output}")


if __name__ == "__main__":
    main()
