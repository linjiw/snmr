# MorphoRetarget execution plan

**Advisor plan date:** 2026-08-29
**Execution revision:** 2026-09-01
**Status:** foundation sprint in progress; no MorphoRetarget learning claim established

This file is the missing plan referenced by
`ADVISOR_GUIDANCE_2026-08-29_MORPHORETARGET.md`. It incorporates the owner's 2026-09-01 scope
corrections. The advisor document remains the source record; this is the executable, falsifiable
plan.

## 1. Claim and first-paper boundary

The target claim is:

> One variable-DoF model receives canonical human motion and a standardized, identity-free
> RobotSpec and retargets an entirely unseen humanoid without a robot-specific head, prompt,
> checkpoint, or target-robot adaptation.

The first paper is restricted to 19--32 body-DoF humanoids, flat ground, feet as the primary
contacts, and a small declared semantic manifest. It does not claim arbitrary topology, fingers,
objects, torque-policy synthesis, universal tracking, hardware readiness, or simulator gradients.

The program has three ordered scientific questions:

1. Can explicit kinematics generalize retargeting to an unseen embodiment?
2. Once that works, can explicit dynamics cause useful, controlled changes in the output?
3. Once a qualified verifier can localize failures, can local repair improve trackability without
   semantic collapse?

No later mechanism may be used to obscure a failure of an earlier question.

## 2. Execution graph: parallel infrastructure and science tracks

P2 verifier qualification and P3 kinematic model integration answer independent questions and are
not serial prerequisites for each other. After the foundation freeze they proceed in parallel:

```text
                         Foundation freeze
                                |
                 HumanMotionSpec + G1 asset contract
                         /                     \
                        /                       \
      Track A: verifier infrastructure     Track B: embodiment science
      frozen G1 tracker                    PM01 diagnosis
      PhysX + MJWarp qualification         graph-token contract
      20 stratified references             fixed-G1 amortization
                        \                       /
                         \                     /
                  held-out kinematic generalization ladder
                                |
                   only after kinematic G2 passes
                                |
              full RobotSpec + qualified physics preferences
                                |
                      localized repair + distillation
```

Hard dependencies:

- No cross-backend evidence before HumanMotionSpec and the exact G1 asset/controller mapping are
  frozen.
- No physics critic or preference labels before the frozen-tracker verifier qualifies.
- No dynamics-conditioned learning before the held-out kinematic experiment passes.
- No repair before the model responds causally to registered dynamics interventions.
- No flow matching unless deterministic mode averaging is measured and best-of-N has a real gain.
- No sequence-level PPO or hardware in the first method gate.

## 3. P0 foundation freeze

### Required contents

- `snmr/robot_spec.py`, `snmr/verification.py`, simulator runners, and their tests;
- HumanMotionSpec and graph-token contracts;
- exact `snmr_commit`, `newton_commit`, and `isaac_lab_commit`, with explicit dirty flags;
- the command, environment, config hash, controller hash, motion hash, and complete referenced-asset
  bundle hash in every worker report;
- independent worker-side capture and hashing of the bytes actually consumed;
- an explicit unavailable state, never a null value interpreted as clean;
- corrected E80 `mZf` status and the previously missing plan file.

The provenance system stops expanding after it can answer whether an anomalous result came from a
changed RobotSpec, asset mapping, motion preprocessing, tracker/controller, or model.

### Exit criterion

A clean checkout reproduces the same invariant hashes and gate booleans, and registered numeric
metrics within frozen tolerances excluding wall time. Recorded experiment directories are never
overwritten during an audit.

## 4. P1 / G0 canonical contracts

G0 is binary. Partial implementation is progress, not a partial gate pass.

### HumanMotionSpec v0.1

Every training tensor must bind:

- source dataset, sequence, subject, clip range, split, and source SHA-256;
- source FPS, explicit 50 Hz timestamps, and applied position/quaternion/contact/mask resampling;
- Z-up, forward +X, pelvis-rooted frame, absolute/relative root-height meaning, meters, radians,
  and internal wxyz quaternions;
- body ordering and validity mask;
- normalized torso, thigh, shin, upper-arm, and forearm descriptors;
- probabilistic left/right contacts in [0, 1], with declared foot-speed and height-clearance
  thresholds;
- ordered transformations and preprocessing version;
- SHA-256 over canonical, labelled flat buffers.

The first adapter only needs to make the existing LAFAN1/AMASS/GMR/SNMR paths explicit. It is not a
general motion-file framework.

### Exact G1 asset parity

Freeze one three-way mapping:

- the G1 MJCF actually used by the MuJoCo/MJWarp path;
- the G1 URDF/USD actually imported by Isaac Lab;
- the joint, body, actuator, timing, reset, and semantic-link map used by the frozen tracker.

Before rollouts, mechanically test:

- joint names, order, type, axis, limits, and nominal pose;
- pelvis, torso, head, hands, and feet under 10,000 deterministic in-limit configurations;
- wxyz/xyzw and world/body transforms;
- standing root height and foot-sole/collision offsets;
- position-target interpretation, action scale, PD gains, effort/velocity limits;
- simulation rate, control rate, decimation, latency, and reset state;
- independent asset, motion, controller, config, and sample-buffer hashes.

Frozen numerical criterion:

```text
max key-link position error < 1.0 mm
max key-link quaternion geodesic error < 1e-3 rad
```

The criterion is not relaxed after observing results. A unavailable or timed-out PhysX worker is a
G0 failure, not a skipped pass.

## 5. P2 / G6 strong-verifier qualification

This track is infrastructure and runs in parallel with kinematic model integration.

### Frozen contract

Use one G1 whole-body tracker checkpoint with identical:

- observation fields and normalization;
- action interpretation and scale;
- controller gains and limits;
- control/simulation timing;
- reset and termination rules.

Run it in the tracker-native PhysX/Isaac Lab backend and a paired Newton/MJWarp backend. Newton is a
secondary verifier and disagreement detector, not a separate research storyline.

### Qualification set

Use 20 registered references stratified across easy/fast walk, turns and sharp cuts, crouch/squat,
kick, single support, large arm motion, jump/airborne, and difficult transitions. For each motion,
evaluate effort scales 0.5, 1.0, and 1.5; the broader diagnostic sweep may retain 0.75 and 1.25.

Report full-clip survival, failure time, tracking error, torque/velocity saturation, contact
disagreement, overflow, reset/mapping disagreement, and failure taxonomy:

- contact chattering;
- joint saturation;
- numerical instability;
- fall/COM failure;
- mapping or reset error.

### Registered gate

- at least 90% full-clip survival on baseline 1.0x GMR references in each backend;
- absolute paired failure-time difference below 0.1 s when both fail;
- Spearman rank correlation above 0.85 across motion difficulty and torque scales;
- stable ordering under two independent registered reset seeds;
- no unresolved controller, state-map, or hash mismatch.

The earlier open-loop PD pilot remains a wiring diagnostic only. It is not a reward or ranker.

### Selective preference rule

A candidate pair becomes a training label only when the controller-independent L1 screen, PhysX
tracker, and MJWarp tracker agree on its ordering. Otherwise emit `uncertain/abstain` and preserve
the disagreement. The simulator is a selective oracle with an audited confidence boundary, not
absolute ground truth.

## 6. Two-day PM01 diagnosis before encoder decisions

The legacy PM01 holdout is one failed robot, not proof about every morphology. Before attributing
29.59 cm versus 5.72 cm to architecture, run two checkpoint-level probes:

1. On trained targets, keep the target topology/static rows/limits fixed and swap only the pooled
   embodiment code. Test whether outputs change and whether code-space distance predicts the
   direction of change.
2. On the exact LORO model, localize FK/keypoint and joint-angle error by body/joint to distinguish
   diffuse extrapolation from a small semantic-correspondence failure.

Decision mapping:

- insensitive to code -> data/reliance failure; a new encoder alone is not the remedy;
- sensitive but nonsmooth/OOD -> morphology coverage and coherent variants are primary;
- localized semantic error -> correspondence/anchor representation is primary.

If the original LORO checkpoint is absent, report the localization diagnosis as inconclusive and
rerun it under a registered replacement checkpoint. Do not infer localization from the all-robot
checkpoint.

## 7. P3 kinematic RobotSpec architecture

Reuse the validated human temporal encoder and replace legacy 8D target conditioning with:

```text
human motion tokens -> temporal encoder ------------------+
                                                         |
RobotSpec nodes -> graph/topology embedder ---------------+-> cross attention
                                                              + tree-distance bias
                                                              -> shared per-joint head
```

The first model receives kinematics only:

- parent-child topology and shortest-path distance;
- joint type, axis, limits, and local transform;
- link length/basic collision geometry;
- depth and child count;
- declared semantic/contact roles and bilateral symmetry.

It does not receive names, paths, hashes, robot/family IDs, learned robot prompts, mass, inertia,
torque, gains, latency, simulation timestep, or a robot-specific output head.

Attention uses `bias[i,j] = -gamma * tree_distance(i,j)`. A shared lightweight head emits one
bounded scalar for every revolute joint token. Padding and joint masks support heterogeneous DoF.
Parent indices remain structural metadata and are not numeric model features because their values
depend on serialization.

Required tests:

- joint-permutation equivariance with inverse-permuted output equality;
- renamed links and altered MJCF serialization with invariant numeric outputs;
- variable-DoF shape/mask behavior;
- no identity fields;
- joint-limit-by-construction output;
- continuous response to counterfactual limb proportions.

### Fixed-G1 entry gate

Before morphology training, the graph-conditioned decoder on fixed G1 must:

- be within 5% of the GMR teacher on human-side semantic error;
- introduce no joint/geometry violation increase;
- not worsen contact timing consistency;
- pass full-sequence stitching without boundary jumps;
- pass the adversarial serialization tests.

Failure here is an integration/data/decoder problem. Do not add robots or physics.

## 8. Variant generator as a first-class method component

Every coherent variant manifest declares seed, parent family, exact changed dimensions, source and
output hashes, and consistency checks. Perturbations include:

- link lengths and limb proportions;
- joint limits;
- waist/ankle DoF presence where the architecture supports that topology;
- mass distribution and inertias for later full-RobotSpec experiments;
- collision/contact geometry.

Geometry and dynamics change coherently: link-volume or declared density determines mass, and
inertia is recomputed rather than independently jittered. Family and procedural seeds are disjoint
across train/test.

For every held-out robot, report its dimensionless RobotSpec-feature distance to the nearest
training robot and plot error versus distance. A one-point T1 result is not by itself a general
zero-shot curve.

## 9. P4 preregistration: MR-E1-KIN-LORO

The first learning experiment isolates kinematic conditioning. It does not include full RobotSpec,
a physics critic, repair, or flow.

### Generalization ladder

1. hold out a coherent same-topology variant;
2. hold out one entire training family, such as H1-2;
3. hold out Booster T1-29DoF entirely.

This distinguishes interpolation, cross-family transfer, and cross-topology extrapolation before
the final all-or-nothing holdout.

### Registered T1 design

```yaml
experiment: MR-E1-KIN-LORO-T1
train_families: [G1, H1, H1-2]
coherent_variants_per_family: 8
held_out_robot: T1-29DoF
motion_count: 100
motion_categories:
  [walking, turning, running, squat, crouch, kick, dance, jump,
   asymmetric_upper_body, recovery_transition]
statistical_unit: clip
confidence_interval: paired_clip_bootstrap
```

Registered arms:

1. manually configured GMR-T1 quality reference;
2. T1-specific supervised upper bound;
3. shared robot-ID model, with T1 mapped to the registered nearest training identity;
4. strong nearest-transfer baseline: nearest training robot selected in RobotSpec descriptor space,
   with its GMR output transferred through a declared joint/semantic map;
5. kinematic RobotSpec graph model.

The strong transferred-output baseline replaces a deliberately weak "nearest ID only" reading.

### Independent semantic metrics

GMR may be a teacher and quality reference, but it is not the semantic ruler. Primary semantic
metrics are computed from the human motion and robot FK independently:

- normalized human keypoint to robot semantic-keypoint error;
- end-effector relative position/orientation error;
- contact timing agreement;
- root heading, displacement, and normalized height trajectory;
- motion-energy/amplitude retention by semantic group.

Also report joint-limit, geometry/self-collision, foot-clearance, and temporal-boundary violations.
Teacher imitation error is diagnostic only.

### Primary decision

The kinematic RobotSpec model must significantly beat both shared robot-ID and strong nearest
transfer on held-out T1 under paired clip bootstrap, without increased hard-constraint violations
or global motion-amplitude collapse. Report every action category, not only the pooled mean.

Internal effect-size summary:

```text
gap recovery = (E_nearest - E_RobotSpec) / (E_nearest - E_T1_upper)
```

Provisional engineering targets are <=5 cm FK keypoint MPJPE and <0.1% joint-limit violations.
They do not replace the comparative, human-side primary decision. Stop/refactor if MPJPE exceeds
7 cm or violations exceed 1%.

Required causal checks:

- counterfactual limb-length changes with fixed names/order/topology must cause smooth, sensible
  output changes;
- adversarial node order, renaming, and equivalent MJCF serialization must preserve output after
  inverse permutation.

### Failure policy

- Fixed G1 fails: repair contract/coordinates/decoder/loss/stitching only.
- Fixed G1 passes but graph does not beat nearest transfer: inspect morphology coverage, semantic
  anchors, topology use, and feature-distance extrapolation; do not add dynamics.
- Graph beats nearest but remains below the T1 upper bound: expand coherent family coverage or
  capacity; this is the useful intermediate outcome that licenses the next stage.

## 10. Later gates, deliberately frozen now

Only after MR-E1 passes:

1. add missing dynamics features such as latency, simulation timestep/decimation, actuator mode,
   torque-speed curve, and richer contact geometry, with availability masks;
2. generate controlled candidates over time scale, root-height/amplitude residuals, contact timing,
   pelvis trajectory, and local joint splines;
3. train only on consensus preferences from L1 + both qualified trackers;
4. test a causal torque-only intervention for monotonic local time/amplitude response;
5. introduce low-dimensional CEM/MPPI repair with a hard human-side semantic gate;
6. distill verified repairs back into the amortized model;
7. finally test downstream tracker sample efficiency/full-clip/OOD utility under the same recipe.

## 11. Closed or out-of-scope branches

- E80 latent-specific dropout fusion: stopped at seed 0 by its registered rule; outage training is
  retained as a recipe result.
- Open-loop PD proxy: diagnostic only, never a preference label or RL reward.
- Flow matching: frozen until measured mode averaging and best-of-N gain.
- Sequence-level PPO: frozen; any later RL begins with low-dimensional local repair operators.
- Hardware: out of scope until frozen-tracker and repeated safety gates pass.
- Broad simulator-ensemble paper: out of scope; Newton remains a paired verifier/disagreement
  detector and possible batched repair backend.

## 12. Immediate 72-hour checklist

- [ ] Commit the foundation to `feat/morpho-retarget-foundation` and tag the archival commit.
- [ ] Produce a clean provenance artifact with SNMR/Newton/Isaac revisions and byte-bound inputs.
- [x] Complete and test HumanMotionSpec.
- [x] Run the deterministic 10,000-pose G1 MuJoCo half of the FK gate and record the live PhysX
  worker timeout as `g0_evaluated=false`, `g0_pass=false`; the cross-backend gate remains open.
- [x] Save the topology-safe PM01 conditioning-swap diagnosis. Exact LORO localization remains
  registered follow-up because the historical checkpoint is unavailable.
- [x] Complete the kinematic graph tokenizer and adversarial permutation tests at the contract
  level; learned-model equivariance remains a later gate.
- [x] Do not launch large training, physics-label generation, dynamics learning, repair, or
  hardware in this iteration.
