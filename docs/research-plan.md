# Research plan and decision gates

## Primary question

Are comparisons between PPO and predictive-sampling MPC invariant to gait
phase, disturbance geometry, model mismatch, terrain, and online compute budget?

## Preregistered hypotheses

- **H1 — phase dependence:** recovery probability depends materially on the
  support phase at which an otherwise identical impulse is applied.
- **H2 — rank interaction:** the PPO–MPC performance difference changes with
  push direction, phase, or MPC compute budget; a single global ranking is
  inadequate.
- **H3 — seed uncertainty:** between-policy PPO seed variance is non-negligible
  relative to the reported controller difference.
- **H4 — compute frontier:** increasing MPC planning budget changes the
  robustness/latency trade-off, so comparisons require a declared deadline.

These are hypotheses, not assumed findings. Null or contradictory results must
be retained.

## Phase 0: shared definitions

1. Pin robot model, simulator, controller repositories, Python, compiler, and
   hardware description.
2. Define actuator semantics and physics/policy/planner frequencies.
3. Implement controller-independent logging and metrics.
4. Save a manifest with every rollout.

Exit criterion: metric tests pass and a passive/standing smoke rollout produces
a complete manifest.

## Phase 1: independent baseline

1. Train a minimum of five PPO seeds on the same Go1 task.
2. Implement a Go1 predictive-sampling MJPC task using the same actuator model.
3. Require both controllers to sustain 0.5 m/s nominal walking before tuning
   robustness.
4. Report nominal tracking, CoT, torque saturation, and controller latency.

Exit criterion: at least three viable PPO seeds and one stable MPC configuration
complete the nominal protocol. Failed seeds remain part of the training result.

## Phase 2: robustness-envelope pilot

- Eight gait-phase bins.
- Eight horizontal push directions.
- Adaptive search over impulse rather than a single force.
- At least three MPC compute budgets.
- Paired scenarios and explicit random seeds.

Primary endpoint: impulse corresponding to 50% failure probability (`I50`) with
a bootstrap confidence interval. Secondary endpoints: recovery time, integrated
tracking error, CoT, peak torque, saturation time, latency, and deadline misses.

Exit criterion: quantify interaction effect sizes and uncertainty before adding
terrain or proposing a new controller.

## Phase 3: extension selected by evidence

- If failures are strongly phase-/state-predictable, test a risk-triggered
  PPO-to-MPC intervention controller.
- If reward components produce measurable gradient conflict, compare scalar PPO
  with objective-wise normalization/GCR-PPO and only then a GDPO-inspired method.
- If compute dominates MPC ranking, develop a budget-adaptive planner policy.

The method is selected after the benchmark identifies a mechanism. “Try more
algorithms” is not itself a contribution.
