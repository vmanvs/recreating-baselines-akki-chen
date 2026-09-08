# Independent Go1 MPC–RL Benchmark

This repository independently reimplements and extends the experiments in:

> S. Akki and T. Chen, “Benchmarking Model Predictive Control and Reinforcement
> Learning-Based Control for Legged Robot Locomotion in MuJoCo Simulation,”
> *IEEE Access*, 2025. DOI:
> [10.1109/ACCESS.2025.3582523](https://doi.org/10.1109/ACCESS.2025.3582523).

The published artifact contains trajectories, plots, and videos, but not the PPO
environment/checkpoint or the Go1 MJPC task. Consequently, this project does
**not** claim bitwise or numerical reproduction of the authors' controllers. It
reconstructs the stated experimental question using public implementations,
documents every new choice, and tests whether the qualitative conclusions hold
under controlled variation.

## Research question

Do the relative advantages of PPO and predictive-sampling MPC remain stable
across gait phase, disturbance direction, model mismatch, terrain, and online
compute budget?

The first milestone is a faithful independent baseline at a commanded forward
velocity of 0.5 m/s. The intended extension is a phase- and compute-conditioned
robustness envelope rather than a comparison based on a single perturbation.

## What exists now

- [`docs/reproduction-audit.md`](docs/reproduction-audit.md): what is and is not
  recoverable from the paper.
- [`docs/research-plan.md`](docs/research-plan.md): hypotheses, experimental
  phases, and decision gates.
- [`configs/paper_protocol.json`](configs/paper_protocol.json): machine-readable
  published values and explicitly unresolved fields.
- [`src/go1_benchmark/metrics.py`](src/go1_benchmark/metrics.py): controller-
  independent metric definitions.
- [`tests/test_metrics.py`](tests/test_metrics.py): executable checks for those
  definitions.

## Local setup

The metric layer intentionally uses only the Python standard library.

```bash
uv venv --python 3.11
uv sync
uv run python -m unittest discover -s tests -v
```

MuJoCo Playground and MJPC will be pinned as external controller backends after
their Go1 tasks have been validated on the Kaggle runtime. Generated rollouts,
videos, checkpoints, and caches stay outside Git; small manifests, aggregate
tables, figures, and environment locks belong in Git.

## Reproducibility language

Use **reported** for numbers copied from the paper, **independently
reimplemented** for our controllers, and **replicated** only for findings that
survive the preregistered evaluation with uncertainty estimates.
