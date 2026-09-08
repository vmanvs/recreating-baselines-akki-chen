# Running the implemented PPO baseline on Kaggle

This is a real training run over MuJoCo Playground's Go1 environment with the
command fixed to `[0.5, 0.0, 0.0]`. It is not the unavailable authors' SB3 MDP.

## GPU notebook setup

Select **T4 x2**, turn Internet on, clone this repository, and enter it. Install
the pinned RL extra without allowing upstream `main` to drift:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra rl
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Do not start training unless that check prints `gpu`. The locked `rl` extra
installs Playground's CUDA 12 JAX dependencies inside the project environment.

Use one GPU for the first seed:

```bash
CUDA_VISIBLE_DEVICES=0 uv run go1-train-ppo \
  --profile paper_budget_proxy \
  --seed 0 \
  --output /kaggle/working/persist/checkpoints/ppo-paper-proxy-seed-000
```

The `paper_budget_proxy` profile applies the published one-million-step,
`5.9e-4` learning-rate, and 0.2 clipping values to the fully specified public
Go1 MDP. The manifest states this limitation explicitly.

If the proxy budget does not produce a viable gait, run the public tuned budget:

```bash
CUDA_VISIBLE_DEVICES=0 uv run go1-train-ppo \
  --profile playground_reference \
  --seed 0 \
  --output /kaggle/working/persist/checkpoints/ppo-playground-seed-000
```

Every run writes:

- `manifest.json`: code, upstream revision, environment, PPO, hardware, and seed.
- `progress.jsonl`: append-only training/evaluation measurements.
- `checkpoints/`: resumable Brax checkpoints.
- `final_metrics.json`: final training return values.

Do not run multiple seeds until seed 0 shows sustained locomotion. A failed
paper-budget proxy is retained as a result rather than silently receiving extra
training.

## Evaluate the paper's displayed push

Evaluation switches to the full-collision Go1 model, fixes the initial yaw to
zero, disables observation noise by default, and applies a constant force to the
torso through MuJoCo's `xfrc_applied` array:

```bash
uv run go1-evaluate-ppo \
  --profile paper_budget_proxy \
  --checkpoint /kaggle/working/persist/checkpoints/ppo-paper-proxy-seed-000/checkpoints \
  --force-n 150 \
  --direction-deg 0 \
  --push-start-s 5 \
  --push-duration-s 0.2 \
  --output /kaggle/working/persist/results/ppo-seed-000-push-pos-x
```

This writes a compressed state trajectory and a summary containing non-foot
ground-contact failure, recovery time, tracking RMSE, forward distance, and CoT.
