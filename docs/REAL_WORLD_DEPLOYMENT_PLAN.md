# E70 Real-World Deployment Readiness

**Audit date:** 2026-08-10 (Phase E update 2026-08-12 — see addendum at end)  
**Robot target:** Unitree G1, 29 DOF, 50 Hz WBT runtime  
**Verdict:** repeated production CPU loopback gate passed; **not cleared for hardware**

## What is ready

Holosoma's pinned checkout already includes a G1 whole-body inference runtime, Unitree SDK
interface, 50 Hz motion clock, per-joint gains/action scaling from ONNX metadata, stiff-start pose,
manual policy stop, and joystick emergency kill.  The E70 simulator uses the hardware-limit G1
model and the WBT configuration includes startup friction, base-center-of-mass, encoder-bias, and
push randomization.  These are useful foundations, not a sim-to-real validation.

`scripts/export_e70_policy_onnx.py` now closes the model-format gap for fixed, preplanned motions.
It embeds the motion reference and either its frozen SNMR trajectory or the explicit goal path,
reproduces the exact observation normalization, and emits Holosoma's standard WBT outputs:
`actions`, `joint_pos`, `joint_vel`, and `ref_quat_xyzw`.  It also embeds the robot URDF, ordered
joint names, KP/KD, per-joint action scales, artifact hashes, and input contract.

Two seed-0 `walk1_subject1` qualification exports currently live under the generated
`exports/e70_video/deployment/` directory:

| Candidate | Simulation role | Size | ONNX Runtime parity |
| --- | --- | ---: | ---: |
| explicit | higher-reliability capability control | 5.0 MB | max abs 3.31e-6 |
| SNMR | exclusive research interface | 12 MB | max abs 3.03e-6 |

Both original exports pass `onnx.checker` and a CPU ONNX Runtime parity gate at the first, middle,
and final motion frames with threshold 1e-5.  That check was necessary but not sufficient: the
original explicit export later produced hard-limit targets on 494/500 production-runtime
fault-injection steps and is ineligible for deployment.

The active deployment-only candidate is
`e70_seed0_explicit_walk1_subject1_safe95.onnx` (SHA-256
`ea2510fbdc0fda2648f46cd853ef165ecd4da94af9bf95ab7a6835c1dc351813`).  It embeds a per-joint
action envelope whose resulting position targets stay within 95% of each URDF joint range.  It
preserves the frozen training checkpoint and observation/command semantics.  This is a deployment
safety transformation, not a new E70 result.

Regenerate the bounded explicit candidate with:

```bash
source scripts/activate_snmr.sh
PYTHONPATH="$PWD" "$WBT_PYTHON" scripts/export_e70_policy_onnx.py \
  --student /data/robotixx/snmr-research/e70/students/seed0_explicit/c_prior_explicit_student.pt \
  --motion /data/robotixx/snmr-research/e69/motions/walk1_subject1_mj_z.npz \
  --out exports/e70_video/deployment/e70_seed0_explicit_walk1_subject1_safe95.onnx \
  --safety-limit-fraction 0.95
```

## Qualification results

The bounded candidate passes a 500-step no-command construction of the actual production
`WholeBodyTrackingPolicy` contract:

| Check | Result |
| --- | ---: |
| Input/output, observation, joint-order, gains/scale, URDF limits | pass |
| Non-finite/deadline/runtime-clip steps | 0 / 0 / 0 |
| Hard-limit/safety-envelope violations | 0 / 0 |
| Inference latency, p99 / maximum | 3.155 / 8.095 ms |
| Minimum commanded hard-limit margin | 0.01309 rad |

The first v2 CPU MuJoCo loopback used only interface `lo`, initialized the first pose under a
virtual gantry, started both the policy and clip while supported, and released after an upright
preflight.  It reached motion timestep 700, including 500 registered frames after release.  It
showed that production inference could control the simulator, but freezing a walking reference at
an arbitrary terminal frame and then jumping to the stiff startup pose was not a safe stop:

| Loopback observation | Result |
| --- | ---: |
| Minimum unassisted base height | 0.574 m |
| Minimum unassisted up-axis | 0.855 |
| Commanded joint-limit violations | 0 |
| Worst measured joint-limit margin | -0.00279 rad |
| Motion end at timestep 700 | reached |
| Stop-to-stiff-hold transition | failed; robot fell |

That original sequence was therefore a failed gate with a useful positive sub-result: production
inference controlled the CPU simulator through the intended interface for the registered motion
window, but its terminal-frame lifecycle was unsafe.  No physical robot command was sent.

A second, declared 90%-range envelope also passed the 500-step computation contract and increased
the minimum commanded margin to 0.02618 rad.  Under the identical loopback it removed the
unassisted measured-limit overshoot (minimum margin +0.02536 rad), but it was not behaviorally
equivalent: base height reached 0.352 m, up-axis reached 0.446, and three sampled velocity-limit
violations occurred before stop.  Safe90 is discarded.  This falsifies “clip harder at deployment”
as a sufficient fix; any future tighter-envelope candidate must be optimized with that projection
in the loop.

The retained v2 handoff protocol fixes the actual lifecycle rather than clipping harder.  It
completes the stiff-hold prompt before the first physics step, runs 500 registered unassisted WBT
frames, and then switches to Holosoma's default zero-velocity locomotion safety policy before the
walking reference freezes.  Three consecutive identical trials pass:

| Worst case across three trials | Result |
| --- | ---: |
| Unassisted base height / up-axis | 0.768 m / 0.974 |
| Safety-hold base height / up-axis | 0.746 m / 0.983 |
| Unassisted measured / commanded limit margin | +0.1205 / +0.0349 rad |
| Safety-hold measured / commanded limit margin | +0.1051 / +0.0498 rad |
| Non-finite, joint-limit, velocity-limit, torque-limit samples | 0 / 0 / 0 / 0 |
| Physical robot commands | 0 |

The machine-readable aggregate is
`e70_seed0_explicit_walk1_subject1_safe95.loopback_safety_handoff_v2.summary.json`.  This clears the
selected highest-value CPU runtime/handoff gate.  It remains one preplanned motion in one CPU
engine and is not evidence of hardware safety or sim-to-real transfer.

## Why hardware deployment is not yet authorized

The exclusive SNMR arm is a diagnostic interface, not the safest capability policy.  Its seed-0
general completion is 0.685, versus 0.925 for the explicit control; seed-1/2 confirmation is still
finishing.  If the frozen paper gate passes, its positive result is only the paired advantage over
matched time and shuffled controls; a scoped null or failed capability control will be reported
instead if registered.  None of those outcomes is a claim of hardware-grade reliability.  A real
deployment should first qualify the explicit policy and should not preserve the exclusive
bottleneck merely for experimental purity.

The export is also intentionally offline: the command at `t` and `t+0.1 s` is embedded from a
preplanned motion.  Live video/teleoperation would require an online, bounded-latency GMR+SNMR
producer, clock/buffer semantics, stale-command handling, and an independent safety monitor.
Installing Isaac Lab or SONIC does not close those interfaces by itself.

## Hard-gated path to the robot

1. **Frozen confirmation.** Finish E70 seeds 1--2, run the hierarchical analyzer, update the
   paper, and choose deployment candidates by a recorded rule.  Do not select a visually pleasing
   seed.
2. **Runtime sim-to-sim — passed for the selected candidate.** Safe95 passes exact contract loading,
   three production-interface motion windows, and three WBT-to-standing-safety-policy handoffs.
   Preserve the report hashes and repeat this gate for any changed model, runtime, or motion.
3. **Safety-state replay — passed for the registered terminal handoff.** The retained handoff has
   zero hard joint-limit, velocity-limit, or torque-limit samples and remains upright for the
   four-second observation window.  Mid-motion emergency switching and longer holds belong to the
   next robustness matrix and are not implied by this result.
4. **Robustness matrix.** Evaluate held-out friction, mass/CoM, actuator gain, encoder bias,
   observation noise, one-to-five-tick action/command delay, dropped latent updates, and pushes.
   Report worst decile as well as mean.  The existing E65 latent-hold result is informative but
   does not substitute for this policy- and engine-specific matrix.
5. **Independent engine.** Use Isaac Lab/Isaac Sim only here, as a cross-engine falsification
   test after the runtime contract is stable.  A new simulator install is therefore a Stage-5
   tool, not today's bottleneck.
6. **Hardware-in-the-loop.** Validate state parsing, joint permutation, signs, units, time stamps,
   and position targets with motors disabled or on a bench fixture.  Compare logged observations
   against the simulator contract.
7. **Gantry progression.** Explicit policy only: stiff hold, then low-amplitude segment at reduced
   action scale, then the full clip while tethered with two people and a physical E-stop.  Promote
   the SNMR policy only after it independently passes the same gates.  Untethered operation is a
   separate approval.

Every stage produces a machine-readable report with model/config hashes.  Failure at a stage
blocks the next one; it is not repaired by editing the threshold after seeing the run.

## Paper boundary

For ICRA, the defensible statement remains: all reported behavior is in simulation and sim-to-real
is unvalidated.  The ONNX export and production loopback can support a reproducibility or
future-deployment paragraph, but they are not real-world results.  A hardware video belongs in the
paper only after the staged gates above and must be labeled separately from simulation.

## Addendum — 2026-08-12 Phase E campaign (post-confirmation candidates, both walks)

Full results: `docs/SIM2SIM_VALIDATION_2026-08-12.md`; artifacts under
`exports/sim2sim_2026-08-12/` and `exports/sim2sim_review/`.

- **Stage 1 (frozen confirmation) is complete.** E70 closed positive at three seeds; the
  deployment candidates were selected by the recorded median-completion rule (explicit →
  seed 2, SNMR → seed 1), not visually.
- **Stages 2–3 re-pass for the deployment configuration.** All four safe95 exports pass the
  parity and 500-step production-contract gates; explicit × `walk1_subject1` passes the
  three-repeat loopback safety-handoff gate 3/3 with phase minima matching the seed-0
  record. The stage-2/3 result is seed-selection-robust.
- **New finding: the registered terminal handoff is motion-specific.** On
  `walk1_subject5` the fixed 27 s handoff lands at a marginal gait phase; most failures
  are upright walking followed by collapse after the switch to the zero-velocity safety
  policy. Before any hardware stage, the handoff frame must be chosen per clip at a
  double-support/low-velocity reference point and requalified — a new recorded protocol
  variant, not a threshold edit.
- **Tracking transfers.** CPU-loopback joint tracking RMSE (0.17–0.23 rad unassisted)
  matches the frozen Warp-eval values; the engine/runtime hop does not degrade tracking.
- The hardware boundary is unchanged: **not cleared for hardware**; stages 4–7 remain.
