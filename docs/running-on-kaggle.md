# Running on Kaggle

Use [Train and verify a Go1 PPO policy](train-from-scratch.md) for the current
installation, training, evaluation, and checkpoint-continuation commands.

GPU setup requires both `--extra rl --extra cuda`. Run subsequent commands with
`uv run --no-sync` to keep those optional dependencies installed. Checkpoint
evaluation reads the saved run configuration automatically.
