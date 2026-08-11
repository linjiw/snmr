# E68 — `walk3_subject1` Specialist Feasibility Extension

**Registered:** 2026-08-08 after E67 stopped at its teacher gate and before resuming the
specialist.  This is an exploratory nuisance-model calibration experiment, not a continuation
that can retroactively validate E67.  **Status: complete; frozen endpoint failed.**

The final `model_15998.pt` evaluation improved over the 8k source checkpoint but remained
below both behavioral gates: completion 0.6543, mean survival 7.9747 s, and joint RMSE
0.2024 rad (1,024 rollouts, seed 404).  The required values were 0.80 completion and 9.0 s
survival.  Its SHA-256 is
`ee8f69161d299516a6144964c6ed98194cf1f26c1aa9dc91356337774d07d32d`.

Per the frozen decision rule, the `walk1_subject5 + walk3_subject1` pair is closed.  No
intermediate checkpoint will be evaluated and no further budget will be added under E68.

## Motivation and disclosed prior result

The E67 `walk3_subject1` specialist improved substantially during its fixed 8,000-iteration
budget (training mean episode length approximately 100, 111, 128, and 211 steps at 2k, 4k, 6k,
and 8k).  Its frozen evaluation nevertheless missed both behavioral gates: completion 0.5615,
mean survival 7.2881 s, and joint RMSE 0.2233 rad.  The other specialist passed.

The question here is only whether more optimization of the difficult specialist makes the
planned two-clip task technically feasible.  E68 makes no SNMR-versus-time representation
claim and runs no student.

## Frozen intervention

- Resume exactly the E67 8k checkpoint with SHA-256
  `12f3e92b2d58a748dea768fcf9e442329470348d54c06af63041df5d2a6db32d`.
- Keep the motion, reward override, PPO configuration, 512 training environments, seed 0,
  simulator, and randomization settings unchanged.
- Add exactly 8,000 PPO learning iterations.  Holosoma resumes from stored iteration 7,999, so
  its final filename is expected to be `model_15998.pt`.
- Do not evaluate or select an intermediate checkpoint.  Evaluate only the final checkpoint.
- Use the original E67 gate unchanged: 1,024 phase-stratified 10-s rollouts, seed 404,
  completion >=0.80, survival >=9.0 s, and finite joint RMSE.

## Decision

- If the final specialist passes, the teacher interface is feasible.  Register a fresh E69
  representation experiment in a new output root and retrain every student from scratch.  Do
  not reuse the invalid E67 explicit-student artifact.
- If it fails, stop this clip pair.  The next paper-valid direction is either a newly registered
  reference-only clip-pair screen whose selected clips both pass fixed specialist gates, or a
  paper rewrite that reports teacher feasibility as the blocking result.  Do not increase the
  E68 budget or choose an intermediate checkpoint after seeing the endpoint.
