# Triangular pulse follow-up

This is a separate 36-trial push follow-up, not a replacement for the original
`outputs/paper-2026-09-30` dataset. The trained calf-clearance PPO checkpoint and
MJPC controller are unchanged. Nominal tests are not repeated because they have
no push pulse.

Two spawned CPU processes run independent native MuJoCo models. Each MJPC
controller retains four planning threads. Worker affinity is disjoint and math
library thread counts are limited before importing JAX. Each trial has its own
folder; only the parent writes the dataset index. The process-global MJPC sensor
callback is never shared across concurrent trials.

The pulse is symmetric triangular, peaks at 50/100/150 N, lasts 0.2 s, and starts
at 5 s. Its intended impulse is 5/10/15 N s, half the rectangular impulse at the
same peak and duration. Each trial retains the original 12 s horizon and checks
physical failures every 4 ms. These are not the paper's declared 50 s trials.

## Run in WSL or Linux

Use the already configured PPO environment and built native MJPC library.
From the canonical master checkout in WSL:

```bash
cd /mnt/d/Code/recreating-baseline-master
export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=src

/home/manv/.venvs/go1-ppo/bin/python -u \
  -m go1_benchmark.parallel_paper_dataset \
  --root "$PWD" \
  --protocol configs/paper_triangular_protocol.json \
  --output outputs/paper-triangular-2026-09-30 \
  --library /mnt/d/Code/recreating-baseline/.build/mjpc-go1/libgo1_mjpc.so \
  --workers 2
```

On another Linux device, substitute the actual repository, Python and library
paths. Do not change the planner thread setting to get more parallel workers.
Keep total workers conservative for RAM and CPU availability.

The same command resumes completed trials if identities still match. An
incomplete trial directory is retained and blocks automatic resume so it can
be inspected. Never remove or overwrite a partially collected scientific trial
without recording what happened. The collection status and trial index update
after each completed trial, not after each simulated second.

## Audit and compare

After all 36 trials finish:

```bash
/home/manv/.venvs/go1-ppo/bin/python scripts/analyze_triangular_dataset.py \
  --dataset outputs/paper-triangular-2026-09-30 \
  --rectangular outputs/paper-2026-09-30 \
  --output results/paper-triangular-2026-09-30
```

This verifies frozen source, library and policy hashes, common model and initial
poses, actual force profiles, intended and delivered impulse, work integration,
failure evidence and complete trial coverage. It exports a Markdown report,
JSON counts, Wilson intervals and PNG/PDF figures versus both peak and impulse.
The generic original nominal-analysis script is not used for this push-only set.

For the current launched batch, one-shot post-processing is already attached.
It waits for this collection process to finish, audits the data, generates the
report and figures, then creates and CRC-checks
`outputs/paper-triangular-2026-09-30.zip` with SHA-256 sidecars. It does not launch
new trials or schedule recurring runs. Its log is
`/mnt/d/Code/recreating-baseline/.build/triangular-postprocess-2026-09-30.log`.
Keep the laptop awake until collection and post-processing finish.

## Interpretation limits

The current run is on battery at the user's request. Its decision wall times
also include contention from concurrent CPU workers. Do not use those timings
as a serial hardware or real-time controller benchmark. Simulated time advances
by the same 4 ms steps regardless of how long planning takes on the host.

Improvement under triangular pulses would reflect a changed disturbance, not an
improved policy. No policy training, reward changes, planner tuning or GPU
backend changes are performed. Three initial-condition seeds per cell and one
trained PPO seed do not establish a maximum robust force. MJPC sampling remains
unseeded, so its comparison is descriptive rather than an exactly paired causal
pulse-shape experiment.

Before collection, 39 unit/native tests passed. A separate short verification
ran both controllers in two real worker processes, checked exact serial versus
parallel PPO physics trajectories, and verified completed-trial resume.
