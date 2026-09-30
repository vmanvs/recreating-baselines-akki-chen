# Survival and recovery of learned and model predictive Go1 locomotion controllers

This repository holds the code, frozen protocols, results and paper source for an
independent simulation comparison of learned (PPO) and model predictive (MJPC)
controllers for the Unitree Go1 quadruped in MuJoCo. Three saved PPO policies and
one upstream MJPC predictive-sampling controller are evaluated in a shared
full-collision plant. There are 92 trials in total: 20 nominal, 36 rectangular-pulse
and 36 triangular-pulse.

This is an independent reconstruction motivated by
[Akki and Chen, IEEE Access, 2025](https://doi.org/10.1109/ACCESS.2025.3582523).
It does not claim to reproduce the authors' controller implementation exactly.

## Paper

The LaTeX source is in [`paper/`](paper). It uses the IEEE conference template and
the vector figures in `paper/figures/`. The compiled PDF is not committed. To build it:

```bash
cd paper
latexmk -pdf main.tex
```

`latexmk` needs Perl. Without Perl, run `pdflatex main.tex` twice instead.

## Data

The raw trial data are published as the
[`paper-data-2026-09-30` release](https://github.com/vmanvs/recreating-baselines-akki-chen/releases/tag/paper-data-2026-09-30).

| Archive | Size | Contents |
|---|---|---|
| `paper-2026-09-30.zip` | 1.1 GB | Main dataset: 20 nominal and 36 rectangular-push trials, frozen policies and sources, analysis, replay videos |
| `paper-triangular-2026-09-30.zip` | 0.7 GB | Follow-up: 36 triangular-push trials, frozen policies and sources |

Each trial folder under `runs/` contains `trial.json`, `experiment.json`, the
compiled `model.mjb`, `summary.json`, the 50 Hz `trajectory.npz` and the 250 Hz
`physics.npz`. The dataset-level `trial-index.json` summarizes every trial.
`dataset-manifest.json` records the source, checkpoint, model and library
identities.

Verify a download against the checksums tracked in this repository. Place both
archives in the repository root, then run:

```bash
sha256sum -c results/paper-2026-09-30/archive-checksum.sha256
sha256sum -c results/paper-triangular-2026-09-30/archive-checksum.sha256
```

On Windows PowerShell, `Get-FileHash <archive> -Algorithm SHA256` gives the same hash.

Compact reports, tables and figures are tracked in git:

- [Main results report](results/paper-2026-09-30/paper-data-report.md)
- [Triangular follow-up report](results/paper-triangular-2026-09-30/triangular-report.md)
- [Main protocol and reproduction commands](docs/paper-dataset.md)
- [Triangular protocol](docs/paper-triangular-tests.md)

Main results: the calf-clearance PPO passed 5/5 nominal trials and MJPC passed 2/5.
Under rectangular pushes, the final PPO completed 0/18 trials and MJPC 6/18.
Under triangular pushes at the same peak forces, the counts were 6/18 and 12/18.
Only 5 PPO and 4 MJPC triangular trials met the strict tracking-and-recovery
criterion. The triangular pulses carry half the rectangular impulse. These
outcomes describe the fixed selected controllers. They are not a multi-seed
causal ablation or a demonstration of real-time or physical-robot performance.

## Repository layout

| Path | Contents |
|---|---|
| `src/go1_benchmark/` | PPO training and evaluation, native MJPC interface, dataset collection and metrics |
| `mjpc/` | C++ Go1 task and bridge to the upstream MJPC planner |
| `configs/` | Training, planner and protocol settings |
| `scripts/` | Build, analysis, rendering and verification tools |
| `tests/` | Unit and native implementation checks |
| `ppo-*/` | Training manifests and logs for the three PPO policies (weights are in the data release) |
| `results/` | Tracked reports, tables and figures for both datasets |
| `paper/` | Paper LaTeX source and figures |
| `docs/` | Setup, training, MJPC and dataset guides |
| `mpc-and-rl-benchmark/` | Data and figures imported from the motivating benchmark |
| `controllers.py`, `run_go1_simulation.py` | Earlier simulation implementation (see below) |

## PPO training

See [Train and verify a Go1 PPO policy](docs/train-from-scratch.md) for installation,
fresh training, checkpoint verification, disturbance evaluation and continuation.

- `src/go1_benchmark/train_ppo.py`: trains a neural policy from random weights.
- `src/go1_benchmark/fixed_velocity_env.py`: Go1 environment commanded at 0.5 m/s.
- `src/go1_benchmark/evaluate_ppo.py`: restores and evaluates a trained checkpoint.
- `configs/ppo_fixed_velocity.json`: environment and training profiles.
- `pyproject.toml` and `uv.lock`: package definition and locked dependencies.

For GPU training on Windows, use Ubuntu under WSL2 with a separate Linux
environment. Native Windows JAX supports CPU execution, not NVIDIA CUDA. The
guide includes WSL installation, GPU checks, lower-memory training settings and
native PowerShell CPU commands. Do not share a virtual environment between
Windows and Linux. Then use `uv run --no-sync go1-train-ppo` and
`uv run --no-sync go1-evaluate-ppo` with the arguments in the guide.

## MJPC baseline

See [Build and evaluate MJPC](docs/mjpc.md). The `mjpc/` C++ task and the
`go1-evaluate-mjpc` command use upstream MJPC's predictive-sampling planner, not
the older `MPCController` in `controllers.py`. They use the same full-collision
model and actuator settings as the PPO evaluation. Planner and task settings are
chosen independently; they are not recovered author settings. Planning exceeds
the 20 ms control budget on the development machine, and all evaluations run
synchronously offline. See [Tuning outcomes and limits](docs/mjpc-tuning.md).
Build and run on Linux or WSL.

## Earlier simulation code

The top-level `controllers.py` and `run_go1_simulation.py` are an earlier
simulation implementation. Their `RLController` is a hand-written gait, not the
learned PPO policy, and they are not used for the paper results. Previous local
trajectories live in `sim_results/`.

## License

Code, configuration and documentation are released under the [MIT License](LICENSE).
The released datasets, trained policies and the contents of `results/` are released
under [CC BY 4.0](LICENSE-DATA). Third-party folders, such as `mujoco_menagerie/`,
`unitree_mujoco/` and `mpc-and-rl-benchmark/`, keep their own terms.
