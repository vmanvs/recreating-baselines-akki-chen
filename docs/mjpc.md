# Go1 MJPC baseline

This is actual upstream MJPC `SamplingPlanner`, compiled with a small native
bridge and our own Go1 task. It is not the older handwritten-gait sampling
controller and is not an exact reproduction of the paper authors' task.
No PPO retraining is needed to install or run it.

## Build once on Linux / Pop!_OS / WSL Ubuntu

Use the canonical master checkout. On this Windows computer, enter Ubuntu WSL,
not PowerShell, for the commands below:

```bash
cd /mnt/d/Code/recreating-baseline-master
source /home/manv/.venvs/go1-ppo/bin/activate
export PYTHONPATH="$PWD/src"
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential cmake ninja-build git
python scripts/build_mjpc.py --jobs 2
```

On a different Linux computer, use that computer's repository path and PPO
virtual environment. If starting from a fresh environment, follow
`docs/train-from-scratch.md` to install the pinned `rl` dependencies first;
installing them does not run training. MJPC evaluation does not import JAX.
The new command can also be run as a module without reinstalling the project.

The first build downloads pinned MJPC and Abseil sources. Later builds reuse
`.build/mjpc`. No GUI, gRPC, GPU, or unrelated robot model download is required.
The bridge links against the active Python wheel's MuJoCo headers and library.
The tested wheel version is MuJoCo 3.12.0. `mujoco_compat.h` adapts removed error
helpers and the added ray-query normal argument for the pinned upstream source;
it does not change the predictive-sampling algorithm.
Rebuild after changing the MuJoCo version or moving the virtual environment.
Do not copy a Windows environment or compiled binary to Linux, or commit builds.
On a memory-limited laptop keep `--jobs 2`, or reduce to 1.

## Evaluate

```bash
python -m go1_benchmark.evaluate_mjpc \
  --duration-s 50 --seed 0 \
  --output outputs/mjpc-nominal-000
```

Use a new output directory each time. The runner reports completed control
steps, simulation time, elapsed time, speed and planner latency every simulated
second. This is synchronous offline evaluation: simulation waits for planning.
A completed run does not imply the controller meets a real-time deadline.
Add `--require-pass` when you want a failed nominal test to exit nonzero.

```bash
python -m go1_benchmark.evaluate_mjpc \
  --duration-s 50 --force-n 150 --direction-deg 0 \
  --push-start-s 5 --push-duration-s 0.2 --push-shape triangular \
  --output outputs/mjpc-push-plus-x-000
```

Directions: 0 = +X, 180 = -X, 90 = +Y, 270 = -Y. Rectangular and triangular
pulses are both supported. Default is rectangular to match the current PPO
evaluator, NOT the paper's triangular maximum-disturbance protocol.
150 N over 0.2 s gives 30 N s rectangular or 15 N s triangular. Do not compare
force thresholds across different pulse shapes. The native planner does not
receive future disturbance forces: it reacts to the observed disturbed state.

`--initial-joint-noise-rad 0.02 --seed 1` enables seeded initial joint jitter.
The upstream planner uses unseeded `absl::BitGen`; `--seed` does NOT seed MJPC's
sampling. Runs have stochastic planner variation even at the canonical start.
Do not claim bitwise reproducibility or independent trained-policy seeds.

If Go1 mesh assets are not found, pass `--menagerie-root /path/to/mujoco_menagerie`.
By default the runner looks in Playground's downloaded assets, then the local
repository's Menagerie checkout. It does not silently download assets.

## Task and planner settings

`configs/mjpc_go1.json` contains all starting settings. The controller directly
optimizes position-target splines; there is no sinusoidal joint-action generator
or learned policy inside MJPC. A four-beat foot-height reference is a task prior.
Disabling `Gait`, `Posture` or `CalfClearance` weights is an explicit task ablation,
not an exact recovery of the paper's four residuals. The default includes these
terms to support the same walking objective as the gait-aware PPO policy.

| Term | Residual |
|---|---|
| Height | Torso height above floor minus 0.278 m |
| Velocity | Local forward/lateral velocity minus command |
| Upright | Torso local Z axis minus world Z |
| Heading | Torso forward-axis XY components minus world +X |
| Effort | Actual actuator forces divided by actuator force limits |
| Posture | Joint positions minus home pose |
| Gait | Foot sphere-center height minus radius and swing-height reference |
| CalfClearance | Positive deficit below 15 mm calf capsule clearance |

MJPC applies its quadratic norm to each residual vector and weights the terms.
Default horizon is 0.4 s, 64 candidates, 5 linearly interpolated spline nodes,
2 sampling iterations per 20 ms control interval, and 4 worker threads.
Exploration is 0.08 times each actuator's full control-range width, not 0.08 rad.
These are starting values requiring locomotion validation and tuning. Freeze
chosen settings before collecting reported test results; do not tune on tests.

## Outputs and comparison limits

Each run saves `summary.json`, `manifest.json`, effective `experiment.json`,
`trajectory.npz`, and `model.mjb`. The manifest records source revision, native
library and model hashes, MuJoCo version, arguments and RNG limitations.
Trajectories contain all 12 actual actuator forces, positions, velocities,
position commands, instantaneous contacts, calf clearance and planner latency.
Summary includes speed error, distance, contact failure, recovery time and CoT.

Two energy estimates are recorded:

- `positive_work_cot`: 50 Hz endpoint estimate, as in the current PPO evaluator.
- `positive_work_cot_physics`: 250 Hz endpoint estimate from every physics step.

They are positive mechanical work divided by mass, gravity and forward distance,
not electrical energy. Nonpositive displacement produces an undefined CoT, not
a fabricated finite value. Non-foot contact is additionally checked at every
physics step. This is stricter than PPO's current 50 Hz failure check. MJPC gait
metrics use instantaneous penetrating contacts, while PPO uses filtered contact
flags. Do not compare gait percentages or failure rates as identical metrics
until those measurement conventions are aligned.

The physical Go1 XML, home pose, 4 ms timestep, PD gains 35/0.5 and full collision
model match PPO evaluation settings. MJPC executes in native MuJoCo while current
PPO evaluation uses MJX. A common-backend validation and matching observation
conditions are still needed before claiming an apples-to-apples controller
benchmark. This implementation observes full simulator state without sensor
noise; its information advantage must be stated when compared with PPO.

Do not report legacy `benchmark_metrics.py`'s hardcoded paper CoT/force thresholds
as measurements of this controller. Do not use `run_go1_simulation.py --controller
rl` for a learned PPO comparison: that still runs the handwritten legacy gait.

## Verification without training

```bash
GO1_MJPC_LIBRARY="$PWD/.build/mjpc/libgo1_mjpc.so" \
  python -m pytest tests/test_mjpc.py -q
```

The native test compares all 40 C++ residual entries against an independent
Python implementation at four gait phases and exercises actual MJPC planning.
Without that environment variable the native test is explicitly skipped;
passing the configuration tests alone is not confirmation of a working build.

### Local integration check, 2026-09-29

The MuJoCo 3.12.0 native build succeeded, and all 10 MJPC tests passed, including
the native residual/planner test. A 2 s development rollout completed with
finite states and no MuJoCo warnings, but FAILED the locomotion criterion due
to non-foot contacts. Mean planning time was approximately 100 ms against a
20 ms control interval. These are integration diagnostics, not publishable
benchmark results. The starting controller needs task/planner tuning and longer
evaluation before inclusion as a performance baseline. No Go1 PPO training was
run as part of this integration.

Local diagnostic artifacts are in the other checkout's
`D:/Code/recreating-baseline/outputs/mjpc-integration-2026-09-29-01`.
The locally verified library is at
`D:/Code/recreating-baseline/.build/mjpc-go1/libgo1_mjpc.so`; inside WSL use the
corresponding `/mnt/d/Code/...` path with `--library`, or run the normal build
command above to build into this checkout's default `.build/mjpc` directory.

## Sources

- https://github.com/google-deepmind/mujoco_mpc
- https://github.com/google-deepmind/mujoco_mpc/blob/main/docs/OVERVIEW.md
- https://doi.org/10.1109/ACCESS.2025.3582523

MJPC revision: `ff572a21e7c2bf9fda62e1862a758da7e9a8719b`.
The upstream sources retain their Apache-2.0 license. The bridge does not patch
or relabel the upstream planner as newly invented software.
