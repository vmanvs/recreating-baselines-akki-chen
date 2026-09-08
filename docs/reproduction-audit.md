# Reproduction audit

## Bottom line

The paper's five experiment families can all be reconstructed, but the authors'
exact controllers cannot be reconstructed uniquely from the publication and
artifact. We therefore separate **protocol reproduction** from **controller
reproduction** and **numerical reproduction**.

| Component | Status | What we can do |
|---|---|---|
| Go1 model in MuJoCo | Reconstructable | Pin the public MuJoCo Menagerie model and record any modifications. |
| Straight walking at 0.5 m/s | Reconstructable | Define one shared command and evaluation wrapper. |
| PPO algorithm family | Partially specified | Use the reported PPO settings, but define and publish a new MDP and train multiple seeds. |
| Authors' PPO policy | Not recoverable | No observation/action/reward definition, full configuration, seed, normalization state, or checkpoint. |
| Predictive-sampling algorithm | Reconstructable | Use a pinned public MJPC implementation. |
| Authors' Go1 MJPC controller | Not recoverable | The Go1 task source, costs, horizon, samples, spline settings, rates, and actuator configuration are absent. |
| 150 N, 0.2 s directional pushes | Reconstructable | Recreate the displayed 30 N s impulse and publish exact timing and application point. |
| Maximum-disturbance test | Reconstructable with a new procedure | Use bracketed/binary search and repeated trials instead of unspecified force increments. |
| Failure definition | Mostly specified | Ground contact by a non-foot body; add an explicit observation window and recovery criterion. |
| Mechanical CoT | Reconstructable | Implement the paper's positive-work convention and report sensitivity to alternative conventions. |
| Slippery/uneven terrain | Concept only | Define friction, geometry, seeds, and severity levels because the paper does not provide them. |

## Scope statement

All controllers in this repository are independent reimplementations. Published
numbers are reference observations, not pass/fail thresholds. We will call a
qualitative conclusion replicated only if it survives matched scenarios,
multiple seeds, and uncertainty estimates.

## Reproduction coverage

- **Experiment design:** five of five reported experiment families are feasible.
- **Published PPO instantiation:** not exactly recoverable.
- **Published MJPC instantiation:** not exactly recoverable.
- **Exact numerical tables/plots:** not reproducible without those controller
  instantiations.
- **Qualitative claims:** testable, which is the scientifically important part.

## Public evidence

The paper specifies PPO with `MlpPolicy`, learning rate `5.9e-4`, 128 rollout
steps, one million training timesteps, and clip range 0.2. It specifies MJPC
predictive sampling but omits its planner/task parameters. The public artifact
describes itself as the data and plots used in the paper and exposes trajectory
columns; it does not provide the controller sources.
