# Paper dataset results, 30 September 2026

All 56 frozen trials were collected and audited.

The plant, physical checks and measurements are shared native MuJoCo. Policies are frozen. Trials stop at the first physical failure. All failed trials remain in the dataset.

| Controller | Nominal pass | Completed | Survivor speed (m/s) | Survivor RMSE (m/s) | Survivor mechanical CoT |
|---|---:|---:|---:|---:|---:|
| PPO reference | 0/5 | 0/5 | N/A | N/A | N/A |
| PPO gait | 0/5 | 0/5 | N/A | N/A | N/A |
| PPO gait + calf | 5/5 | 5/5 | 0.4218 | 0.1013 | 1.3333 |
| MJPC | 2/5 | 2/5 | 0.3935 | 0.1404 | 1.4881 |

Speed, RMSE and CoT in this table describe completed nominal trials only. Early failures have no steady-state estimate; their time to failure and failure geometry are in trials.csv. CoT integrates positive actuator work at 250 Hz and divides by mass, gravity and net forward displacement. It is not electrical energy.

| Controller | Direction | Force (N) | Impulse (N s) | Strict success | Physically completed | Prepush qualified | Full pulse delivered |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO gait + calf | 90° | 50 | 10 | 0/3 | 0/3 | 3/3 | 3/3 |
| PPO gait + calf | 90° | 100 | 20 | 0/3 | 0/3 | 3/3 | 3/3 |
| PPO gait + calf | 90° | 150 | 30 | 0/3 | 0/3 | 3/3 | 3/3 |
| PPO gait + calf | 270° | 50 | 10 | 0/3 | 0/3 | 3/3 | 3/3 |
| PPO gait + calf | 270° | 100 | 20 | 0/3 | 0/3 | 3/3 | 3/3 |
| PPO gait + calf | 270° | 150 | 30 | 0/3 | 0/3 | 3/3 | 3/3 |
| MJPC | 90° | 50 | 10 | 1/3 | 3/3 | 2/3 | 3/3 |
| MJPC | 90° | 100 | 20 | 0/3 | 0/3 | 2/3 | 1/3 |
| MJPC | 90° | 150 | 30 | 1/3 | 1/3 | 3/3 | 1/3 |
| MJPC | 270° | 50 | 10 | 0/3 | 1/3 | 3/3 | 3/3 |
| MJPC | 270° | 100 | 20 | 0/3 | 1/3 | 2/3 | 3/3 |
| MJPC | 270° | 150 | 30 | 0/3 | 0/3 | 2/3 | 2/3 |

Success requires prepush qualification, completing the trial without failure, and returning below 0.15 m/s planar error for at least 0.5 s. This tolerance is prespecified and differs from earlier pilot evaluations. Prepush failures do not establish a disturbance threshold. Push directions are world-frame ±Y; headings may drift.
Physical completion is reported separately from strict success. A contact-free trial may fail prepush tracking qualification or the sustained recovery criterion. The physical-completion plot does not certify recovered command tracking.

## Interpretation limits

Only one trained seed per PPO variant is available. The nominal comparison is descriptive, not a causal reward ablation. Five initial poses and three push trials per cell give wide intervals. Wilson intervals describe these finite trial outcomes; they do not establish broad generalization. Upstream MJPC sampling is unseeded. The disturbance grid is coarse and does not identify a maximum robust force.

All pushes start at 5 s, so gait-aware controllers are tested at one scheduled onset phase. Results cannot establish robustness across other push phases or onset times. The reference PPO is the public Playground-based policy, not the original authors' SB3 controller. policy-summary.csv retains actual checkpoint steps, requested training budgets, architecture and training settings; these variants are selected frozen policies rather than a budget-controlled causal comparison.

All controllers execute synchronously offline. Latency includes observation/host handling and blocked actor inference or MJPC planning, excludes policy compilation, and is measured on this machine. No delayed-action real-time experiment was performed. MJPC receives exact full state; PPO receives its saved observation with noise disabled. The common native backend is an evaluation transfer from MJX training. No terrain or physical-robot tests were run.

The laptop temporarily ran on battery during MJPC 50 N -Y push trials. Reported clock dropped to 1.7 GHz and measured planning rose near 500–600 ms, then returned near 230 ms after AC power was connected. Raw timings retain this transition and hardware diagnostic records. The latency figure uses nominal trials; timing is not an isolated fixed-power hardware benchmark.

## Files

- trials.csv: all outcomes, including early failures
- joint-work.csv: per-joint positive work for every trial; durations and completion flags retained
- policy-summary.csv: evaluated checkpoint identities and training settings
- cot-sampling-sensitivity.csv: paired 50 Hz and 250 Hz endpoint estimates for completed nominal trials
- training-curves.csv: existing training evaluation logs through the selected checkpoints; these are not held-out benchmark results
- nominal-summary.csv: nominal counts, Wilson intervals, and survivor metrics with sample counts
- push-summary.csv: disturbance outcomes, qualification and delivery counts
- analysis.json: machine-readable aggregates
- artifact-sha256.json: raw artifact integrity inventory
- PNG and PDF figures: survival, speed, paths, pushes, latency, gait contacts, joint torques, tracking and energy

World-frame tracking error, final heading and lateral drift are retained in trials.csv and nominal-summary.csv. Nominal success uses the frozen body-frame criterion; it does not certify negligible straight-line drift. The illustrative joint-torque plot uses the shared 2–4 s interval of seed 100, before its first failure, and is not a controlled joint-effort experiment.
The CoT sampling figure compares two endpoint quadratures of the same trajectories. The 250 Hz estimate is the declared metric, not a demonstrated continuous-time ground truth. No physics time-step convergence test was performed.

Shared model SHA256: `82b91868325657efad18102e39201936c86ba1bdf93bc7f582fec0640377b1e4`
Protocol SHA256: `4a1e9ed0efba87df77162063a9945f5828ac0f66abfecda8e54c257cf96afb81`
