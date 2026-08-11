# SNMR Research, Paper, Video, and Deployment Completion Goal

**Defined:** 2026-08-10 (America/New_York)  
**Active experiment:** E70 frozen two-walk confirmation  
**Decision horizon:** finish the submission evidence chain before opening a new training study  
**Hardware status:** not authorized; simulation and offline/runtime qualification only

## Goal statement

Complete one defensible evidence chain showing whether a frozen neural retargeting latent carries
trajectory information that a humanoid tracker can use beyond proprioception and absolute
within-clip time.  The chain is complete only when the unchanged, preregistered three-seed E70
assay has a final hierarchical analysis; every displayed manuscript number is generated from that
analysis; a provenance-bound, failure-honest simulation video visualizes policy-independently
selected rollouts; and the higher-reliability explicit controller passes the next real inference
runtime and CPU MuJoCo loopback gates.  No real robot command may be sent under this goal.

The intended paper claim is deliberately narrower than the deployment objective:

> An exclusive retarget-to-track command interface makes control-usable trajectory information
> measurable.  In a frozen two-walk Unitree G1 simulation assay, matched time, proprioception,
> phase-shuffled content, and command-destruction controls determine what crosses that interface.

The intended deployment conclusion is also narrow:

> The explicit-goal student is the first qualification candidate because it is the most reliable
> completed capability control.  The SNMR-only policy remains a research interface until it
> independently clears the same runtime, robustness, and hardware-safety gates.

## Why this is the most promising goal

The central uncertainty is no longer whether the simulator task is feasible.  The explicit
positive control reaches approximately 0.92 general completion for every completed training seed.
The completed hypothesis and time-control cells also show positive raw ambiguity-start A-minus-T
differences for all three seeds: +0.156, +0.275, and +0.145.  For the two seeds with a completed
shuffled-content cell, raw A-minus-S is +0.188 and +0.136.  These descriptive numbers are
encouraging but are not the final statistic; only the frozen analyzer may produce the aggregate
effect and confidence interval.

This path is higher value than adding a new robot, simulator, architecture, or benchmark today.
It converts already strong but incomplete evidence into a reviewable result, makes the qualitative
video subordinate to aggregate evidence, and advances deployment using the safer controller
without overstating the SNMR bottleneck's current reliability.

## Frozen scientific question and success rule

E70 asks whether the frozen SNMR latent supplies control-usable information beyond an absolute
within-clip time code when two walking trajectories contain similar current states but different
futures.  Its scope is the deterministic 64-d exclusive command student, two LAFAN1 walks, one
Unitree G1 model, and one simulator/controller family.

The primary endpoint is ambiguity-start completion over 69 reference-only state-matched frame
pairs.  The positive-content gate remains exactly:

1. The explicit positive control passes its frozen general-completion rule.
2. SNMR minus time is at least +0.10.
3. The paired 95% interval for SNMR minus time excludes zero.
4. The paired 95% interval for SNMR minus phase-matched shuffled content excludes zero.
5. SNMR minus time is positive on each of the two clips.

Training seed is retained as the outer uncertainty level and frame pair as the paired cluster.
No threshold, clip, seed, checkpoint rule, training budget, endpoint, or interpretation may change
after observing a result.  If the explicit capability gate passes but the content gate does not,
the result is a reportable scoped null.  If the explicit gate fails, the controller assay is
invalid and supports no representation comparison.  Neither outcome permits tuning E70.

## Latest audited state

| Work item | State on 2026-08-11 | Evidence boundary |
| --- | --- | --- |
| Seeds 0 and 1, all five arms | Complete | Eligible frozen reports |
| Seed 2 explicit, SNMR, and time | Complete | A-minus-T direction is positive descriptively |
| Seed 2 proprioception | Pending clean restart from round 0 | Interrupted round-1550 state is quarantined and ineligible |
| Seed 2 shuffled content | Pending after proprioception | Required for final A-minus-S |
| Three-seed analyzer | Pending | No final aggregate may be quoted yet |
| Paper | Positive, scoped-null, and invalid-assay branches each build as six-page letter PDFs with embedded fonts | Displays labeled seed-0 values until final macros exist |
| Video | Selection, capture, index, outcome-conditioned composition, and validation code frozen | Raw scientific captures intentionally pending final analysis |
| Deployment export | Safe95 explicit seed-0 ONNX passes the production WBT computation contract | Original export remains ineligible |
| Production loopback | Three consecutive WBT-to-safety-policy handoff trials pass | CPU simulation only; robustness and hardware gates remain |

At the resumed 2026-08-11 audit, one unrelated job had ended and 22.53 GiB was free, still below
the frozen 26,000-MiB E70 launch gate.  The recovery supervisor therefore correctly remained idle.
The gate must not be weakened and the remaining unrelated process must not be interrupted for this
experiment.  All 5 confirmation, 24 deployment, and 12 paper-video frozen hashes still match.
The same condition persisted for three consecutive resumed audits through 20:39 UTC; the detached
recovery and postprocessing supervisors remain responsible for automatic continuation.

## Completion contract

### A. Experiment complete

- Seed-2 proprioception restarts at round 0 under the unchanged launcher.
- Seed-2 shuffled content completes after it.
- The analyzer accepts exactly seeds 0, 1, and 2 and 69 paired clusters.
- The final JSON includes all five arms, per-seed effects, per-clip A-minus-T effects, hierarchical
  intervals, input hashes, and the frozen content-gate verdict.
- Frozen confirmation hashes still match after completion.

### B. Paper complete

- `paper/e70_results.tex` is generated from the final analyzer JSON, never transcribed manually.
- The abstract, teaser, result table, effect intervals, interpretation, and limitations all agree
  with the generated macros and the preregistered outcome language.
- Generated booleans make every result-bearing passage follow one of three fail-closed outcomes:
  positive-content pass, explicit-pass/content-null, or explicit-capability invalidation.
- The PDF is anonymous, US letter, no more than eight pages including references, and has all fonts
  embedded and subset.
- The final text says “evaluated teacher checkpoint,” scopes E70 to two walks and simulation, and
  does not present ONNX export as real-world evidence.

### C. Simulation video complete

- Render exactly the six frozen manifest captures; do not replace a failed or visually awkward
  example with a better-looking rollout.
- Bind exact starts, checkpoint, teacher, motion, evaluator, camera, report, and raw-video hashes.
- Preserve terminations and label every panel with completed/terminated status and survival time.
- Compose the registered 70-second storyboard at 1920x1080, 30 fps, H.264/yuv420p, progressive,
  and at most 19,000,000 bytes.
- Populate aggregate cards only from the final analyzer.
- Make the result card fail closed on the explicit capability gate, and state that production CPU
  loopback/handoff passes while robustness, HIL, tethered hardware, and sim-to-real remain open.
- Inspect the contact sheet and full video for framing, label accuracy, reset leakage, clipping,
  and misleading synchronization before calling the video final.
- Bind that review to the exact MP4 and contact-sheet hashes, then require the final bundle auditor
  to cross-check paper, analyzer, captures, video validation, code freeze, and review.  The
  supervisor's `POSTPROCESS_COMPLETE` marker alone means encoding finished, not visual acceptance.
- Generate the review record with `scripts/record_e70_visual_review.py` only after watching the
  full MP4; the tool refuses incomplete checklists, missing media, and accidental overwrite.

### D. Pre-hardware deployment gate complete

The frozen seed-0 explicit export is the first candidate; it was selected as the capability
control, not because of video appearance.  Before any hardware request:

1. Load it through the actual `WholeBodyTrackingPolicy` ONNX contract with the production
   observation order, joint order, gains, action scales, URDF, and four required outputs.
2. Run the production inference process against CPU MuJoCo over loopback at 50 Hz with no physical
   network interface.
3. Record inference latency, missed deadlines, non-finite values, clipped actions, commanded joint
   targets, joint-limit margin, and termination/survival.
4. Require zero non-finite commands, exact 29-joint identity/order, zero hard joint-limit breaches,
   and a functioning stop-to-safe-hold transition.  Report latency percentiles and any tracking
   regression; do not invent a tolerance after observing it.
5. Only after loopback passes, preregister latency/dropout, encoder-bias, gain, friction, mass/CoM,
   and push robustness tests.  Hardware-in-the-loop and tethered gantry work are later, separately
   approved stages.

The current candidate passes the computation portion of this gate.  The original
export produced hard-limit targets on 494 of 500 stationary fault-injection steps.  A replacement
export with an ONNX-embedded 95%-of-range command envelope passes all 500 production-runtime
steps: zero non-finite outputs, deadline misses, hard-limit violations, or safety-envelope
violations; p99 inference latency is 3.155 ms against the 20 ms deadline and the minimum hard-limit
margin is 0.01309 rad.

Early CPU loopbacks established two failure modes.  Releasing the gantry before policy activation
caused a fall, and freezing a continuous walking reference at timestep 700 produced an unstable
terminal pose.  Repeated traces showed the fall began after the registered 500-frame motion window
and before the later stop command.  A hold-last prototype was therefore discarded: freezing a
single-support target is not a standing controller.

A preregistered 90%-range envelope isolated the tradeoff rather than solving it.  It restored a
+0.0254-rad measured margin during unassisted motion, but base height fell to 0.352 m, up-axis to
0.446, and velocity-limit violations appeared before the stop.  Post-hoc clipping is therefore
discarded as the deployment fix.  If later robustness tests require a tighter projection, that
candidate must be optimized with the exact projection active rather than clipped only at export.

The retained solution uses Holosoma's existing production dual-mode architecture.  Physics does
not advance while the runtime is blocked on stiff-hold confirmation; the explicit WBT policy then
starts under gantry support, runs 500 registered frames after release, and hands off before
terminal freeze to the default zero-velocity locomotion safety policy.  Three consecutive
loopback trials pass every unchanged check.  Across those trials the worst unassisted height and
up-axis are 0.768 m and 0.974; the worst four-second safety-hold values are 0.746 m and 0.983.
There are zero non-finite, measured/commanded joint-limit, velocity-limit, or torque-limit samples
in every phase, and no physical command is sent.  This completes the highest-value CPU pre-hardware
runtime gate.  It does not authorize hardware or establish sim-to-real performance.

## Execution order and stopping rules

1. Preserve both detached supervisors and monitor the 26,000-MiB capacity gate.
2. While GPU-blocked, complete only work that cannot influence E70: exact-runtime contract checks,
   CPU loopback tooling, paper compliance, and artifact documentation.
3. When E70 completes, inspect the analyzer before any prose edit.  Accept the registered positive,
   scoped-null, or invalid-assay branch without changing the endpoints.
4. Generate the final paper, then raw captures, then composition, then visual review.
5. Do not start a new learning experiment until the final E70 paper/video bundle is internally
   consistent and archived.

Stop and preserve evidence if a hash changes, a partial stable capture exists, a final analysis
lacks a registered seed/cluster, the explicit runtime violates a hard safety contract, or a step
would send commands outside loopback simulation.

## Most promising next research direction

After the submission-critical bundle closes, extend the ambiguity assay from a two-walk proof to
held-out trajectory generalization.  The key question is not merely whether a latent beats time on
two memorized clips, but whether an exclusive content representation selects the correct future
for unseen motions and larger motion families while retaining compactness and robustness.

The next study should therefore preregister a multi-trajectory train/held-out split, reference-only
ambiguous-state selection, matched absolute-time and phase-shuffled controls, an explicit capability
control, and per-motion paired uncertainty.  Its primary metric should remain future-sensitive
completion at ambiguous starts.  Only after that representation generalizes should the project add
an online GMR+SNMR producer for live commands.  This direction directly attacks E70's largest
limitation—breadth—while preserving the causal measurement contract that currently differentiates
the work.

## Immediate next actions

1. Continue monitoring the unchanged E70 recovery and post-processing supervisors; the 26,000-MiB
   launch gate remains authoritative.
2. Preserve the passing three-repeat WBT-to-safety-policy summary as the CPU runtime gate; do not
   reinterpret it as real-world evidence.
3. After the E70 paper/video bundle closes, preregister the robustness matrix over latency/dropout,
   encoder bias, gain, friction, mass/CoM, pushes, and multiple motion starts.  Require the same
   safety handoff and report worst-decile results.
4. On final analyzer arrival, generate the outcome-conditioned paper macros, frozen captures,
   final video, contact sheet, and completion audit in that order.
