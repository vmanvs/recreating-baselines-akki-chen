# MJPC development tuning, 29 September 2026

## Outcome

The selected gait-aware configuration passed the nominal and walking checks in
all three 12 s development repeats. There were no non-foot ground contacts at
any 4 ms physics step. This is a promising offline baseline, not a real-time
controller, an exact reproduction of the authors' task, or a completed paper
benchmark. No PPO training or policy changes were made.

The default `configs/mjpc_go1.json` now uses the selected settings. The exact
confirmation candidate is `configs/mjpc_go1_development.json`. Original starting
settings are archived as `configs/mjpc_go1_initial.json`.

## Confirmation results

All repeats used the same settings, a 0.5 m/s forward command, 12 s duration,
and 2 s warmup for steady metrics. Initial joint perturbations had standard
deviation 0.01 rad, using seeds 0, 1 and 2. These are initial-pose seeds, not
trained-policy seeds. Upstream sampling remains unseeded.

| Repeat | Speed (m/s) | Planar RMSE (m/s) | Phase match | One swing | Non-foot contact | Mean plan (ms) |
|---|---:|---:|---:|---:|---|---:|
| 0 | 0.3948 | 0.1363 | 87.82% | 70.06% | None | 230.4 |
| 1 | 0.3902 | 0.1445 | 88.52% | 72.06% | None | 235.3 |
| 2 | 0.4011 | 0.1319 | 87.67% | 69.06% | None | 231.2 |

All-feet-airborne fractions were 0.20%, 0%, and 0.20%. Minimum steady calf
clearance across legs was 6.84, 2.05, and 3.15 mm respectively. The configured
15 mm clearance is a soft objective, not a guaranteed constraint. Positive
50 Hz work CoT was 1.2420, 1.2830, and 1.2830; corresponding 250 Hz estimates
were 1.4523, 1.4893, and 1.4899. These are not yet directly comparable to PPO.

Every measured plan exceeded the 20 ms budget. P95 latency was 256.7, 260.6,
and 253.7 ms. Simulation waited for planning and did not delay actions by that
amount. These results do not demonstrate real-time control.

Three short successful repeats do not establish a robust success probability.
Speed remains biased below the command. Gait metrics use instantaneous
penetrating contacts, unlike PPO's filtered contacts. Heading/path drift needs
review in longer runs.

## Changes and search limits

We explored 26 adaptive short development configurations, then ran three
fixed-configuration confirmation repeats. This was not an exhaustive search,
Optuna study, or controlled ablation. Multiple settings and task versions
changed, so the trials do not establish which single term caused an improvement.

The initial objective often tipped the robot into hip or calf contacts.
Increasing clearance and foot-height weights alone did not solve this. The
selected task adds a capture-point balance proxy, angular-velocity penalty,
body-frame foot-placement reference, support-force schedule, and direct
non-foot collision penalty. These are explicit task priors, not joint-action
generators or recovered author settings. Actual upstream MJPC SamplingPlanner
still optimizes all 12 joint position targets through simulated candidates.

The task now contains 58 residual entries in 13 quadratic terms. Each term's
cost is one half its weight times the squared residual norm. The selected
planner uses a 0.24 s horizon, 128 candidates, three linear spline nodes, four
iterations, exploration 0.03, and four threads. Noise standard deviation is
exploration times HALF the actuator control-range width. Foot-height and stride
amplitudes ramp over 1 s; the velocity objective targets 0.5 m/s from the start.

The bridge caches fixed model identifiers and uses an immutable task snapshot
in the sensor callback, avoiding unnecessary serialization of rollout workers.
Upstream planner source was not changed. Rebuild the library after updating.
Native tests compare every residual to an independent Python reference at four
gait phases, perturbed poses, and an intentionally low collision pose.

One-iteration solver and warm-start sensitivity were separately investigated.
Disabling warm starts or increasing solver iterations did not yield a successful
candidate in those trials. The selected settings retain original physics.
This does not prove a solver bug. Some solver and earlier screening runs
overlapped, so their latencies are not controlled speed comparisons. The final
three confirmations ran sequentially without concurrent rendering.

## Artifacts

All outputs, including unsuccessful trials, are retained locally under:

`D:/Code/recreating-baseline/outputs/mjpc-tuning-2026-09-29/`

The confirmation report is `confirmation-01/development-report.json`. Each
repeat contains `model.mjb`, `experiment.json`, `manifest.json`, `summary.json`,
and `trajectory.npz`. Trajectories record joint states, actual torques, commands,
foot positions and forces, contacts, calf clearance, task residuals, per-term
costs, and planning latency. They are development data, not paper test results.

The replay uses the first confirmation, not a searched-for best-looking run.
It plays at simulation time, not the slower wall-clock speed.

MuJoCo: 3.12.0. MJPC revision:
`ff572a21e7c2bf9fda62e1862a758da7e9a8719b`.

Confirmation library SHA256:
`621a5b0727f3477c507aa03864f8df74d21544eb178a50434455fa293528fe9a`.

Exact confirmation configuration SHA256:
`81e5ee170e0011c63372531a5d6d3a088644e4371c091319de352d8727aded76`.

## Rebuild and repeat

Use the existing pinned environment. On this computer:

```bash
cd /mnt/d/Code/recreating-baseline-master
source /home/manv/.venvs/go1-ppo/bin/activate
export PYTHONPATH="$PWD/src"
python scripts/build_mjpc.py --jobs 2
GO1_MJPC_LIBRARY="$PWD/.build/mjpc/libgo1_mjpc.so" python -m pytest tests/test_mjpc.py -q
python scripts/tune_mjpc.py --configs configs/mjpc_go1.json \
  --duration-s 12 --warmup-s 2 --repeats 3 \
  --initial-joint-noise-rad 0.01 --require-walking \
  --output outputs/mjpc-development-confirm-new
```

The repeat runner snapshots configurations and rejects library changes during
validation. Use a new output directory. On another Linux machine, use its
checkout/environment paths and rebuild there.

## Before paper data and figures

1. Keep settings fixed and visually review the replay, including heading and
   foot placement. Parameter changes mean new development work.
2. Align native MuJoCo and PPO evaluation physics, observations, contact
   filtering, failure checks, work sampling, and commands. Prefer a common
   backend. Disclose MJPC's full-state information and gait priors.
3. Predeclare longer nominal runs, push shape/duration/directions/force grid,
   repeat counts, and failure/recovery rules. Reserve new initial states and do
   not tune using those test results.
4. Include failed runs and the offline timing limitation. Do not report a push
   threshold unless the nominal controller passes first.
5. Then create comparative speed/error, gait/contact, work/CoT, recovery,
   success-versus-force, and planner-latency figures. Do not relabel these pilot
   runs as the final paper benchmark.
