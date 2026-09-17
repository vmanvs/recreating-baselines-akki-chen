# Go1 simulation and PPO training

The `master` branch is the canonical workspace for simulations and controller
development. The repository includes the earlier simulation scripts and an
independent PPO training pipeline built on MuJoCo Playground and Brax.

See [Train and verify a Go1 PPO policy](docs/train-from-scratch.md) for installation,
fresh training, checkpoint verification, disturbance evaluation, and continuation.

## PPO files

- `src/go1_benchmark/train_ppo.py`: trains a neural policy from random weights.
- `src/go1_benchmark/fixed_velocity_env.py`: Go1 environment commanded at 0.5 m/s.
- `src/go1_benchmark/evaluate_ppo.py`: restores and evaluates a trained checkpoint.
- `configs/ppo_fixed_velocity.json`: environment and training profiles.
- `pyproject.toml` and `uv.lock`: package definition and locked dependencies.

For GPU training on this Windows machine, use Ubuntu under WSL2 with a separate
Linux environment. Native Windows JAX supports CPU execution, not NVIDIA CUDA.
The guide includes WSL installation, GPU checks, lower-memory training settings,
and native PowerShell CPU commands. Do not share a virtual environment between
Windows and Linux.
Then use `uv run --no-sync go1-train-ppo` and
`uv run --no-sync go1-evaluate-ppo` with the arguments in the guide.

The existing top-level `controllers.py` and `run_go1_simulation.py` remain the
earlier simulation implementation. Their `RLController` is a hand-written gait,
not the learned PPO policy. The PPO commands above use the new package. They do
not silently replace that controller in the legacy runner.

Imported source data live in `mpc-and-rl-benchmark`; previous local trajectories
live in `sim_results`. New PPO runs should use separate output directories.
This is an independent reconstruction of the benchmark associated with
[DOI 10.1109/ACCESS.2025.3582523](https://doi.org/10.1109/ACCESS.2025.3582523).
The authors' unavailable controller implementation is not claimed to be reproduced exactly.
