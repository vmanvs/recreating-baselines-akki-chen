# Paper dataset protocol

The main test protocol was frozen on 30 September 2026 before collecting test
outcomes. It compares three saved PPO policies and the fixed tuned MJPC task.
The canonical configuration is `configs/paper_protocol.json`. The collection
copies that configuration, evaluator source, checkpoint identities and library
hashes into a self-contained dataset manifest.

## Main experiment

- Five nominal trials per controller, requested duration 30 s.
- Initial joint jitter: independent normal draws with 0.01 rad standard deviation,
  using seeds 100 through 104. The same seed gives exactly the same initial pose
  for every controller. These seeds were not used in MJPC development.
- Final calf-clearance PPO and MJPC additionally receive rectangular lateral
  pushes of 50, 100 and 150 N, in world directions 90 and 270 degrees. Each cell
  contains three trials, initial-pose seeds 100, 101 and 102.
- Push duration 0.2 s, onset 5 s, trial duration 12 s. Intended impulses are
  10, 20 and 30 N s. Save actual delivered impulse, including truncated pulses.
  The force is applied through MuJoCo's trunk-body external-force field, at
  that body's center of mass, with no separately specified torque.
- Fixed command: 0.5 m/s forward, zero lateral and yaw command.

This gives 56 trials: 20 nominal and 36 disturbance trials. One trained seed is
available per PPO variant. The nominal comparison therefore describes these
saved policies and cannot establish a causal reward effect or generalization
over training seeds. MJPC exploration uses upstream unseeded randomness.
The reference PPO is the public Playground-based policy, not the authors'
SB3 controller. Actual checkpoint steps and training settings are exported.
Push onset is fixed at one scheduled gait phase; this protocol does not test
robustness across onset phases.

## Shared evaluation

All four controllers use the same compiled full-collision native MuJoCo model,
with 4 ms physics, 20 ms control, position actuators, Kp 35 and Kd 0.5. Keep the
original Newton solver settings and collision geometry. Tests compare the
physical model fields to pinned Playground, and compare the native actor input
to the upstream observation at perturbed states and several gait phases.

PPO uses its latest saved checkpoint, its saved normalizer and network, noiseless
observations, deterministic inference, and the saved action scale of 0.5 rad.
The actor sees local velocity, gyro, gravity, relative joint angles, joint
velocities, previous-action history and command; gait variants also see sine
and cosine phase. Preserve the pinned upstream action-history update ordering.
The privileged critic input is not used by the actor. Evaluation does not reset
automatically after an episode-length counter.

MJPC receives exact position, velocity and time, plus the declared task priors.
This information difference is part of the controller comparison. This native
evaluation is a backend transfer from PPO's MJX training; it is not a repeat of
the old MJX-only evaluation.

## Failure and success rules

Check failure at every physics step. Stop at the first penetrating non-foot
floor contact, inverted torso, nonfinite state, MuJoCo warning or controller
error. Save its time and geometry. Stopping a trial is a failed outcome, not
missing data. A small calf penetration is still a failure under this conservative
contact definition.

Nominal success requires completing 30 s, no failure, positive forward progress,
and planar body-frame velocity RMSE at most 0.15 m/s after a 2 s warmup.
Report gait phase agreement, exactly-one-swing fraction and all-air fraction
separately, using instantaneous penetrating foot-floor contacts for both
controllers. Score the same four-beat schedule for the reference PPO even
though it was not trained to follow that schedule.

Disturbance success requires no failure through 12 s, prepush tracking RMSE
at most 0.15 m/s over 2 to 5 s, and postpush tracking error at most 0.15 m/s
continuously for at least 0.5 s. Report recovery only for qualifying successful
trials. Prepush failures do not demonstrate a disturbance force threshold.
This recovery tolerance was frozen for this experiment and differs from pilot
evaluations. World-frame pushes are not necessarily exactly body-lateral when
the controller's heading drifts.

Integrate positive actuator mechanical power at 250 Hz. Save a 50 Hz endpoint
estimate for sampling sensitivity. CoT uses positive work divided by robot mass,
gravity and net forward displacement. It is mechanical CoT, not electrical
energy. Report steady metrics and CoT of complete nominal trials separately
from early failure trajectories, without assigning failed runs a fabricated
steady speed or CoT.

Trials run sequentially. PPO actor compilation occurs before trials, and its
CPU inference blocks until output is available. Decision time includes Python
observation/host overhead. Both controllers execute synchronously in simulation;
an over-budget decision is not artificially delayed. These experiments measure
offline control performance and computational latency, not delayed-action
real-time robustness.

## Outputs and commands

On this machine, from WSL:

```bash
cd /mnt/d/Code/recreating-baseline-master
export PYTHONPATH="$PWD/src"
export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
python -m go1_benchmark.paper_dataset \
  --protocol configs/paper_protocol.json \
  --library /mnt/d/Code/recreating-baseline/.build/mjpc-go1/libgo1_mjpc.so \
  --output outputs/paper-2026-09-30
python scripts/analyze_paper_dataset.py \
  --dataset outputs/paper-2026-09-30 \
  --output outputs/paper-2026-09-30/analysis
```

Use `/home/manv/.venvs/go1-ppo/bin/python` if that environment is not activated.
On another Linux machine, rebuild the library and use that checkout's paths.
Collection resumes completed trials only if frozen source, protocol, policy and
library hashes match. An incomplete trial directory is retained for inspection;
the runner does not silently delete or rerun it. Use a new dataset directory for
any changed configuration or new experiment.

Each trial contains its model, inputs, summary, 50 Hz trajectory and 250 Hz
physics log. The dataset contains a manifest, frozen source, a trial index and
collection status. Analysis verifies all planned trials, common model hashes,
identical initial states, physics timing, integrated work and delivered impulse.
It exports all-trial and aggregate tables, Wilson 95% intervals and PNG/PDF
scientific figures. Representative traces use initial seed 100, rather than
selecting the best outcome.

The raw dataset remains in ignored `outputs/`. The completed collection archives
the three exact selected checkpoints, their training manifests and progress
logs under `frozen-policies/`. It also retains the native C++ task, Linux library,
reproduction tools and collection log. Analysis verifies the archived checkpoint
and library hashes against the original frozen manifest.
The manifest's protocol and configuration hashes refer to the byte-exact
`protocol-input.json` and `mjpc-config-input.json`. The adjacent `protocol.json`
and `mjpc-config.json` are normalized JSON copies of the same settings and can
have different byte hashes because keys and whitespace were rewritten.

Small curated results, tables and PNG/PDF figures are in
`results/paper-2026-09-30/`; binary trajectories and policy weights are not
committed. Preserve `outputs/paper-2026-09-30.zip` outside this machine as well.
The ZIP contains the raw dataset, checkpoints, analysis and representative
videos. The Linux library is archived for identity, not promised portable across
machines. Rebuild it with the pinned source when moving to another environment.

The selected checkpoints are at 217,907,200 steps for reference and gait PPO,
and 229,376,000 steps for calf-clearance PPO. These actual counts exceed their
requested 200-million-step budgets and are not identical. Existing training
curves use different reward definitions and must not be compared as if they
shared a common return scale.

The observation/model parity, MJPC and metric test suite passed 26 tests before
collection. Actual saved actor restoration and short native execution checks
also passed for all three policies. This does not establish numerical equivalence
between long native and MJX rollouts.

`hardware-context.json` records the Ryzen 5 4600HS CPU, 6 physical cores and
12 logical CPUs visible to WSL2, approximately 3.5 GiB WSL RAM, and four MJPC
planner threads. PPO inference uses CPU JAX. Environment configuration and
lock files are retained beside the runtime package versions in the manifest.
Rendering takes place after all controller timing measurements.
During the 50 N -Y MJPC trials, the laptop temporarily ran on battery and
reported a lower clock. Latency rose to approximately 500–600 ms, then returned
near 230 ms after the user connected AC power. The dataset retains both power
diagnostic records and the unmodified timings. The latency figure uses nominal
trials; the raw push timings also include this power transition.

```bash
python scripts/render_paper_dataset.py \
  --dataset outputs/paper-2026-09-30 \
  --output outputs/paper-2026-09-30/videos
```

The replay compares nominal seed 100 and the 50 N +Y push at seed 100.
Each panel is clearly marked when its trial has stopped; a retained final pose
does not imply that the failed controller continued walking.

## Limits for writing

Describe this as an independent reconstruction and empirical comparison.
Distinguish these held-out initial poses from the adaptive MJPC development
trials. No exact author reproduction, real-time execution, terrain robustness,
physical-robot validity, causal reward ablation or maximum force claim follows
from this protocol. Three trials per disturbance cell give wide intervals.
Changes motivated by these outcomes require a separately labeled follow-up
experiment and must not overwrite this dataset.
