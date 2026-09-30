# Triangular pulse follow-up

All 36 triangular trials were retained and audited against saved native MuJoCo physics. The original rectangular dataset is unchanged.

Only the commanded pulse shape changed: 0.2 s symmetric triangles at 50/100/150 N deliver 5/10/15 N s rather than rectangular 10/20/30 N s. The policy, planner configuration, library, full-collision plant, initial poses, failure checks and tracking criteria match the original tests.

| Pulse | Controller | Direction | Peak N | Impulse N s | Strict success | Physical completion | Prepush qualified | Full pulse |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rectangular | PPO gait + calf | 90 | 50 | 10 | 0/3 | 0/3 | 3/3 | 3/3 |
| rectangular | PPO gait + calf | 90 | 100 | 20 | 0/3 | 0/3 | 3/3 | 3/3 |
| rectangular | PPO gait + calf | 90 | 150 | 30 | 0/3 | 0/3 | 3/3 | 3/3 |
| rectangular | PPO gait + calf | 270 | 50 | 10 | 0/3 | 0/3 | 3/3 | 3/3 |
| rectangular | PPO gait + calf | 270 | 100 | 20 | 0/3 | 0/3 | 3/3 | 3/3 |
| rectangular | PPO gait + calf | 270 | 150 | 30 | 0/3 | 0/3 | 3/3 | 3/3 |
| rectangular | MJPC | 90 | 50 | 10 | 1/3 | 3/3 | 2/3 | 3/3 |
| rectangular | MJPC | 90 | 100 | 20 | 0/3 | 0/3 | 2/3 | 1/3 |
| rectangular | MJPC | 90 | 150 | 30 | 1/3 | 1/3 | 3/3 | 1/3 |
| rectangular | MJPC | 270 | 50 | 10 | 0/3 | 1/3 | 3/3 | 3/3 |
| rectangular | MJPC | 270 | 100 | 20 | 0/3 | 1/3 | 2/3 | 3/3 |
| rectangular | MJPC | 270 | 150 | 30 | 0/3 | 0/3 | 2/3 | 2/3 |
| triangular | PPO gait + calf | 90 | 50 | 5 | 2/3 | 3/3 | 3/3 | 3/3 |
| triangular | PPO gait + calf | 90 | 100 | 10 | 0/3 | 0/3 | 3/3 | 3/3 |
| triangular | PPO gait + calf | 90 | 150 | 15 | 0/3 | 0/3 | 3/3 | 3/3 |
| triangular | PPO gait + calf | 270 | 50 | 5 | 3/3 | 3/3 | 3/3 | 3/3 |
| triangular | PPO gait + calf | 270 | 100 | 10 | 0/3 | 0/3 | 3/3 | 3/3 |
| triangular | PPO gait + calf | 270 | 150 | 15 | 0/3 | 0/3 | 3/3 | 3/3 |
| triangular | MJPC | 90 | 50 | 5 | 1/3 | 3/3 | 2/3 | 3/3 |
| triangular | MJPC | 90 | 100 | 10 | 1/3 | 2/3 | 2/3 | 2/3 |
| triangular | MJPC | 90 | 150 | 15 | 0/3 | 2/3 | 3/3 | 2/3 |
| triangular | MJPC | 270 | 50 | 5 | 1/3 | 1/3 | 3/3 | 3/3 |
| triangular | MJPC | 270 | 100 | 10 | 0/3 | 2/3 | 2/3 | 3/3 |
| triangular | MJPC | 270 | 150 | 15 | 1/3 | 2/3 | 1/3 | 3/3 |

## Interpretation

Triangles reduce total impulse at the same peak, so any improved completion is a response to a different disturbance, not an improved or retrained controller. Equal impulse also does not imply equal peak force or force history. Three initial-condition trials per cell and one trained PPO seed are insufficient to establish a maximum robust force.

The upstream MJPC sampler is unseeded. Matched poses do not guarantee paired stochastic planning trajectories. Any MJPC difference includes sampling variability, so this is a descriptive follow-up, not an isolated causal shape experiment.

Triangular trials ran in two isolated CPU processes with unchanged four-thread MJPC settings and 4 ms physics checks. Decision wall times are resource-contended and must not replace the original serial latency measurements. No GPU backend, training, reward changes or controller tuning were used.

Trials remain 12 s with pushes at 5 s, rather than the original paper's declared 50 s. Existing 30 s nominal tests remain in the original dataset. Gait onset phase, terrain and training-seed generalization were not expanded.

Files: push-comparison.json, audit.json, artifact-sha256.json, and peak/impulse PNG/PDF outcome figures. Raw trajectories, physics, summaries, worker placement, frozen policies, source and library remain in the dataset directory.
