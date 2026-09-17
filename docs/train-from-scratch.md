# Train and verify a Go1 PPO policy on Windows

The canonical workspace is `D:\Code\recreating-baseline-master`.
This pipeline trains Brax PPO from random weights on the pinned MuJoCo
Playground Go1 task, with a privileged critic. It is an independent baseline,
not an exact recreation of the authors' unavailable SB3 environment.

## Why JAX reports only a CPU

Native Windows JAX does not support NVIDIA CUDA. A successful import followed
by `[CpuDevice(id=0)]` is expected there. The earlier GPU assertion was for
Linux, not native PowerShell. Installing a Windows CUDA toolkit will not fix
this platform limitation. For GPU use on this Windows machine, run the Linux
environment inside WSL2. JAX lists WSL2 NVIDIA support as experimental.
See the [JAX platform matrix](https://docs.jax.dev/en/latest/installation.html).

The local hardware check found a GeForce GTX 1650 with 4 GB VRAM and Windows
driver 592.00. Windows can see the GPU. At the setup check, WSL had no installed
Linux distributions. WSL GPU training has therefore not been verified here.

## 1. Install Ubuntu under WSL2 once

Run in **Administrator PowerShell**:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if requested, open Ubuntu, and create its Linux username and
password. Then in PowerShell:

```powershell
wsl --update
wsl --list --verbose
```

Ubuntu should show version 2. If it shows 1, run
`wsl --set-version Ubuntu 2`. Launch it with `wsl -d Ubuntu`.
See [Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).

## 2. Install the separate Linux environment once

All commands in this section run in **Ubuntu Bash**, not PowerShell:

```bash
sudo apt update
sudo apt install -y git curl
nvidia-smi
```

If that command is not on PATH, try `/usr/lib/wsl/lib/nvidia-smi`.
If neither can access the GPU, resolve WSL/Windows-driver access before
installing training dependencies. Do not install a Linux NVIDIA display driver
inside WSL. WSL uses the Windows driver.
See [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/go1-ppo"
export UV_CACHE_DIR="$HOME/.cache/uv"
cd /mnt/d/Code/recreating-baseline-master
uv sync --locked --python 3.11 --extra rl --extra cuda
uv run --no-sync python -c "import sys,jax,mujoco; print(sys.executable); print('JAX',jax.__version__,'MuJoCo',mujoco.__version__); print(jax.devices()); assert jax.default_backend() == 'gpu'"
```

Keep the existing Windows `.venv` separate. Do not activate or reuse it in
Ubuntu. The variables above put Linux packages on the Linux filesystem while
reading the same canonical source files from drive D. The `cuda` extra installs
the locked Linux CUDA 12 JAX dependencies. Do not manually upgrade JAX outside
the lock. First environment construction downloads pinned Menagerie assets, so
keep Internet available.

If JAX still reports CPU inside Ubuntu, check that `sys.executable` points into
the Linux environment above, rerun the sync with both extras, and read any CUDA
plugin initialization warnings. Check that `JAX_PLATFORMS` is not forcing CPU
and `CUDA_VISIBLE_DEVICES` is not hiding the GPU.

## Dependency compatibility

Brax is pinned to upstream commit
`303d89dfac42919e5fde7c91f24c28a505373b24`, which replaces the removed
`jax.device_put_replicated` calls. JAX is pinned to 0.10.2.
Brax still reports version 0.14.2 at this commit, so the version string alone
does not identify the fixed build; preserve `uv.lock` with results.

If your earlier install failed with that AttributeError, rerun the exports and
`uv sync --locked --python 3.11 --extra rl --extra cuda` inside Ubuntu.
No manual edits to site-packages or CUDA reinstall are needed. Restart training
with a new output directory, since the failed attempt wrote run metadata.
The regression test `tests/test_brax_compatibility.py` executes a tiny upstream
test environment to check actual PPO initialization and updates. It does not
verify Go1 learning quality or the full workload's GPU memory requirements.

## 3. Start each later session

Open Ubuntu and run:

```bash
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/go1-ppo"
export UV_CACHE_DIR="$HOME/.cache/uv"
cd /mnt/d/Code/recreating-baseline-master
```

You do not need to reinstall uv or packages each session. Repeat sync only when
dependencies change or the environment needs repair. Use `uv run --no-sync`
to keep the installed optional dependencies.

## 4. Train from random weights

In Ubuntu, after the GPU check succeeds:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync go1-train-ppo \
  --profile playground_reference --seed 0 \
  --num-envs 128 --num-eval-envs 4 --num-evals 21 \
  --output outputs/ppo-seed-000
```

This requests 200 million environment steps. It is a real training run, not a
smoke test. Compilation happens before useful progress. Actual step counts can
exceed the budget because of PPO batch sizes. No pretrained policy is used.

128 training environments and 4 evaluation environments are conservative
starting overrides for this small GPU, **not a verified 4 GB fit**. PPO rollout
storage, updates, and compilation also consume memory. If it runs out of memory,
try 64 or 32 environments and 1 evaluation environment with a new output folder.
The environment count must divide 8192. Reducing it does not reduce the fixed
PPO batch size, so this alone may not solve memory exhaustion. Further batch
configuration changes or a larger GPU may be needed. No training-time estimate
or successful full run on this GPU is established.

For the separate one-million-step budget experiment:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync go1-train-ppo \
  --profile paper_budget_proxy --seed 0 \
  --num-envs 128 --num-eval-envs 4 \
  --output outputs/ppo-paper-budget-seed-000
```

This applies selected reported settings to our public environment, not the
authors' complete configuration. Keep these results separate from longer runs.

## 5. Verify the saved policy

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync go1-evaluate-ppo \
  --checkpoint outputs/ppo-seed-000/checkpoints \
  --duration-s 20 --force-n 0 --warmup-s 2 \
  --max-rmse-m-s 0.15 --require-pass \
  --output outputs/verify-ppo-seed-000
```

The latest numeric checkpoint is selected. The evaluator restores its network
and normalizer and uses the saved experiment settings. Verification uses a
deterministic policy, canonical home pose, no observation noise, and full body
collisions. Training uses feet-only collisions, so high reward need not imply
a verification pass.

`summary.json` reports `nominal_pass`: finite signals, no observed non-foot
ground contact, no termination, positive forward displacement, and planar
tracking RMSE at most 0.15 m/s after two seconds. Failure exits nonzero after
saving artifacts. This is our engineering criterion, not a paper value or proof
of repeatability. `trajectory.npz` contains states, actions, actuator forces,
velocities, and contact flags. `--noise-level 1` enables observation noise.

Contacts and power are sampled at the control rate, which can miss shorter
events. CoT is an estimate, not publication-ready energy evidence.

After nominal walking passes, test a push:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync go1-evaluate-ppo \
  --checkpoint outputs/ppo-seed-000/checkpoints \
  --duration-s 20 --force-n 150 --direction-deg 0 \
  --push-start-s 5 --push-duration-s 0.2 \
  --output outputs/push-ppo-seed-000-pos-x
```

World-frame directions: 0 for +x, 180 for -x, 90 for +y, 270 for -y.
This rectangular pulse has 30 N s impulse. Match the source waveform before
calling it a matched reproduction. Rollout continues after failure to record
recovery; observed body contact remains flagged.

## Native PowerShell CPU alternative

Use this only if choosing CPU execution instead of WSL GPU training:

```powershell
Set-Location D:\Code\recreating-baseline-master
uv sync --locked --python 3.11 --extra rl
uv run --no-sync python -c "import jax; print(jax.devices())"
uv run --no-sync go1-train-ppo --allow-cpu --profile playground_reference --seed 0 --num-envs 32 --num-eval-envs 1 --num-evals 21 --output outputs/ppo-cpu-seed-000
uv run --no-sync go1-evaluate-ppo --checkpoint outputs/ppo-cpu-seed-000/checkpoints --duration-s 20 --force-n 0 --warmup-s 2 --max-rmse-m-s 0.15 --require-pass --output outputs/verify-ppo-cpu-seed-000
```

A CPU device is expected here, so there is no GPU assertion. These are real
training commands, but 200 million steps on CPU can be impractical. CPU imports
and command interfaces were checked; a complete native Windows training and
checkpoint-restoration run has not been validated. Do not use Bash line
continuations or `CUDA_VISIBLE_DEVICES=0 command` syntax in PowerShell.

## Persistence, source tracking, and continuation

Outputs under `outputs/` live on
`D:\Code\recreating-baseline-master\outputs` and survive terminal closure and
reboot. The Linux environment and downloaded packages also persist. Training
itself is not kept running by saving files: leave the process running and keep
the machine awake. Uncheckpointed work can be lost on shutdown or interruption.

Preserve each complete run folder: `manifest.json`, `experiment.json`,
`model.xml`, `progress.jsonl`, `checkpoints/`, and final metrics when available.
Also preserve verification artifacts, source code, and `uv.lock`. The XML is
only the top-level scene and still requires pinned Playground/Menagerie assets.
Back up results separately; ignored output directories are not saved by Git.

Use Windows Git from PowerShell for this linked worktree. Its Windows `.git`
pointer may not resolve with Linux Git inside WSL; do not rewrite it just for
training. The trainer can then record a null Git revision. Before running,
record `git rev-parse HEAD` and preserve any uncommitted source changes with your
run notes, since the commit alone does not capture those changes.

To continue, replace NUMERIC_CHECKPOINT below with an existing checkpoint
directory, and run in Ubuntu:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync go1-train-ppo \
  --profile playground_reference --seed 0 \
  --num-envs 128 --num-eval-envs 4 --num-evals 21 \
  --resume outputs/ppo-seed-000/checkpoints/NUMERIC_CHECKPOINT \
  --output outputs/ppo-seed-000-continuation
```

Keep environment and network settings identical. This Brax version restores the
policy, critic, and observation normalizer, but starts a new optimizer and step
counter. The step budget is for this new invocation. This is warm-start
continuation, not exact interrupted-training resume. Nonempty output folders
are rejected.
