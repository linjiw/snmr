# Learning-based retargeting research status and experiment review

**Date:** 2026-09-01
**Latest program:** MorphoRetarget
**Overall status:** fixed-G1 single-window wiring gate passed; controller/standing G0 fails closed;
full fixed-G1 and held-out-T1 claims remain unqualified
**Evidence basis:** frozen source, clean-checkout artifacts, and bounded iteration
`autoresearch/iterate-260901-0111/` plus `autoresearch/iterate-260901-0341/`
**Frozen source:**
`feat/morpho-retarget-foundation@078d3116aa233ca09c5c68134d9fd5dd5eb55944`
**Current implementation:**
`feat/morpho-retarget-kinematic@554022268b27e4b72953a60038a1394c45be358d`
**Archive:** `autoresearch/iterate-260901-0111/foundation_freeze_078d311/`; annotated tag
`morpho-retarget-foundation-2026-09-01`

## Executive verdict

The latest research direction is well defined, but its central unseen-embodiment learning claim
has not yet been tested. The target is a single variable-DoF retargeter that reads an unseen humanoid's physical
specification rather than its identity, produces semantically faithful motion without a
robot-specific head, prompt, or fine-tuning, and improves robust closed-loop trackability under
independent physics verification.

The current work has established the prerequisite RobotSpec, canonical `HumanMotionSpec`, safe
pair adapter, graph-token, learned kinematic decoder skeleton, teacher-independent semantic
metric, provenance, FK-parity, and verification-report contracts. It has also diagnosed one
important property of the legacy PM01 failure: trained SNMR uses its embodiment conditioning, but
the surviving artifacts cannot determine whether the exact LORO failure was coverage,
extrapolation, or semantic misalignment. The new RobotSpec-conditioned decoder has now passed one
registered 1,000-step, 128-frame fixed-G1 overfit gate from scratch, including its learned root and
bounded 29-joint output. It has **not** passed full-corpus fixed-G1 qualification, human-side
semantic non-inferiority, held-out-T1 generalization, strong-tracker PhysX/MJWarp ranking parity,
repair, or downstream data utility. The honest maturity label is **single-window wiring and
representational-capacity gate passed; overall G0 not passed**, not a G1/G2 paper result.

The fixed-G1 gate finished with a 98.4526% registered last-20-versus-first-20 loss reduction,
`2.0269 cm` teacher-FK MPJPE, `0.01963 rad` joint MAE, `1.8409 mm` local-root position MAE,
`0.09347 rad` root-orientation geodesic error, zero joint-limit violations, and
`4.7684e-7` serialization error. Every named model block had finite nonzero gradients on all 1,000
steps. This establishes that the integrated architecture can represent and optimize one frozen G1
teacher window; it does not evaluate validation clips or the teacher-independent semantic ruler.

The strongest independent infrastructure result remains the clean 10,000-pose live-PhysX FK
subgate: maximum
MJCF-to-USD key-link error is `3.932e-6 m` and maximum orientation geodesic error is
`3.759e-6 rad`, both far inside the frozen strict `1e-3` limits. The same clean bundle also exposes
the strongest boundary: controller/standing-state parity has 14 hard failures and 8 missing live
checks, including incompatible action scaling, effort/friction/reset contracts, and no identical
frozen checkpoint across PhysX and MJWarp. Therefore the FK subgate passes while overall G0 remains
false. Separately, open-loop PD replay still fails too quickly and responds non-monotonically to
torque scale, so it must not be used as a candidate ranker, physics preference label, or RL reward.

## The latest research goal

The program's governing claim is:

> A single retargeter reads an unseen humanoid's physical specification, rather than its identity,
> and produces motion that survives independent physics verification.

Operationally, the model should consume human motion, an identity-free `RobotSpec`, and an optional
task objective; decode a variable number of robot joints; require no robot-specific output head,
learned robot prompt, retargeting-weight tuning, or adaptation on the held-out robot; and expose
localized failures that can be repaired and distilled.

The intended primary result is semantic non-inferiority on held-out robots together with a
significant improvement in robust full-clip tracking success, using the same qualified tracker
recipe and no held-out-robot retargeter tuning or adaptation. "Semantic" must be measured on the
human-side task contract--scaled human/robot keypoint error, contact-timing agreement, and relative
end-effector pose--rather than distance to GMR output. GMR-T1 remains an independently scored
optimization baseline, not both teacher and evaluation ruler. The first-paper scope is humanoid
whole-body motion on flat ground with foot contacts. It does not require arbitrary robot topology,
fingers, object interaction, torque-policy generation, a morphology-universal tracker, or simulator
gradients.

Source: [advisor guidance](ADVISOR_GUIDANCE_2026-08-29_MORPHORETARGET.md), especially the project
definition, staged training plan, E0--E10 experiments, and G0--G6 gates.

## Baseline carried into MorphoRetarget

SNMR is already a substantial learning-based retargeting baseline. That should be preserved, but it
does not establish the new claim.

| Evidence | Current result | Meaning for MorphoRetarget |
| --- | ---: | --- |
| Paired teacher corpus | 77 LAFAN1 clips x 5 robots, 2.48M frames | Enough to validate the existing supervised pipeline; not enough morphology diversity for zero-shot embodiment claims |
| G1 specialist | Public BENCH-v2: 3.66 cm MPJPE, 95% CI [3.46, 3.86]; historical sparse gate: 2.18 cm | Amortized GMR imitation works on a fixed robot; 3.66 cm supersedes 2.18 cm for public comparison |
| Shared five-robot model | Public BENCH-v2: 2.92--6.04 cm; historical E04 16-window eval: 3.16--6.67 cm; zero joint-limit violations | A shared model can serve trained robots with modest sharing cost |
| PM01 leave-one-robot-out | Historical matched 16-window eval: 29.59 cm versus 5.72 cm in-training, 5.2x worse | Existing 8D static conditioning fails the only completed unseen-robot holdout; the new diagnosis rules out wholesale conditioning neglect but cannot identify the exact missing-checkpoint LORO mechanism |
| New kinematic graph path | Padded variable-node tokens plus a tree-biased graph encoder, human cross-attention, learned root, and one shared bounded per-joint head | Shape/gradient/limit/renaming/permutation/counterfactual tests pass; a registered 128-frame fixed-G1 overfit passes, but no full fixed-G1 or held-out result exists |

The LORO failure is not a side note. It is the main reason the project has moved from "shared latent
across trained robots" to explicit physical conditioning, coherent morphology augmentation, and a
true held-out-embodiment benchmark. The new pooled-code swap shows that the old network does use
conditioning, so simply making the conditioning path louder is not a justified fix. Coverage and a
nonsmooth, identity-like learned geometry are leading hypotheses, while semantic mapping remains
unresolved. See [the PM01 diagnosis](../autoresearch/iterate-260901-0111/pm01_conditioning_diagnosis.md),
[the experiment log](EXPERIMENT_LOG.md), and [program consolidation](PROGRAM_CONSOLIDATION.md).

## Work completed for the new goal

### 1. RobotSpec v0.1

`snmr/robot_spec.py` now represents frames, semantic anchors, the kinematic tree, link geometry and
collision proxies, mass/COM/inertia, joint axes and limits, actuator limits and gains, and control
and simulation timing. It provides:

- separate kinematic, dynamics, and complete specification hashes;
- identity-free, dimensionless model features with availability masks for missing dynamics fields;
- round-trip serialization and validation;
- coherent dynamics twins that preserve the kinematic hash;
- explicit exclusion of names, paths, asset hashes, and trainable robot IDs from numeric model
  features.

This is currently an MJCF-first RobotSpec adapter. URDF/USD bundles are bound by the G0 harness and
their live key-link FK parity now passes, but no general URDF/USD-to-RobotSpec adapter exists and
the feature bundle is deliberately not connected to old SNMR checkpoints.
The 37D node vector plus four availability fields also does not yet expose every proposed physical
quantity: latency and simulation timestep are absent as explicit model features, collision geometry
is compressed to a radius, and only torque scaling has been experimentally exercised.

### 2. Canonical HumanMotionSpec v0.1

`snmr/motion_spec.py` now provides the canonical human-motion language that was previously missing:

- immutable source identity and strict clip provenance, including motion ID, optional subject,
  half-open source-frame range, split, ordered transformations, and preprocessing version;
- canonical Z-up, forward +X, pelvis-origin, absolute-world-Z root height, meters/radians, and
  `wxyz` quaternion semantics;
- explicit source FPS/timestamps and a strictly uniform 50 Hz target grid containing only complete
  20 ms ticks within source duration;
- cubic position interpolation with a declared short-sequence linear fallback and SciPy quaternion
  SLERP, without appending a misleading short final target interval;
- root/body positions and orientations, probabilistic left/right contacts, a fail-closed validity
  mask, and extraction of normalized torso, thigh, shin, upper-arm, and forearm lengths from
  declared bilateral landmarks;
- a layout/dtype-normalized tensor-buffer SHA-256 plus a separate specification hash binding the
  frames, contact protocol, source identity, and transformation provenance;
- strict JSON round trip, finite/shape/unit-quaternion/contact checks, rejection of invalid source
  samples before interpolation, and rejection of finite float64 values that would overflow or
  underflow the canonical float32 hash representation.

`snmr/motion_adapter.py` now safely binds the current LAFAN1 pair-NPZ path to this contract. It
loads captured bytes with `allow_pickle=false`, keeps the raw human-source identity separate from
the robot-specific pair-container hash, and hashes the declared segment landmark mapping. This is
one concrete migration path, not proof that all AMASS, LAFAN1, GMR, and existing SNMR loaders have
been migrated.

### 3. Kinematic RobotSpec graph path

`snmr/robot_tokens.py` is a parameter-free tensorization layer for heterogeneous RobotSpecs. It
emits padded variable-node batches, node and joint masks, structural parent metadata, all-pairs
tree distances and attention bias, per-node joint bounds, topology features, and separate dynamics
availability. The registered first mode is `feature_set="kinematic"`; it zeros live dynamics
availability and selects only kinematic fields. The full 37D fields remain available for a later,
separately gated dynamics experiment.

Adversarial tests permute link and joint serialization, rename links and joints, batch different
DoF counts, compare kinematic twins, and remove authored free-root spawn pose from model features.
After inverse permutation, the contract tensors agree exactly. Each batch records separate hashes
for exact model-visible buffers and audit-only source identities.

`snmr/morpho_model.py` now implements the bounded kinematic architecture skeleton: shared node
projection, shortest-path-biased graph self-attention, human-temporal cross-attention, and one
shared scalar head mapped into each joint's limits. It rejects full/dynamics tokens, malformed tree
metrics, non-finite limits, empty roots/DoFs, and tampered feature/mask contracts. Learned forward
tests cover gradients, variable-DoF padding, opaque renaming, serialization permutation, padded
co-batches, dynamics-twin isolation, and a counterfactual link-transform response.

`snmr/morpho_integration.py` now reuses the existing human temporal encoder, supplies the graph
decoder, and predicts a learned heading-local root pose; the overfit runner reconstructs that root
in world coordinates for FK. Contacts remain deliberately absent from the network and must be
derived from predicted sole FK. The first from-scratch single-window optimization passes all
registered gates, including gradients through the human encoder, robot graph encoder, shared joint
head, and root head. The architecture still uses dense attention and supports one scalar revolute
joint per child link. There is no full-corpus fixed-G1 qualification, semantic qualification,
coherent-asset training, generalization staircase, or held-out T1 result.

### 4. Teacher-independent semantic evaluation

`snmr/semantic_metrics.py` evaluates human motion and robot FK directly, with no GMR/teacher target
argument. A fixed seven-role correspondence binds local semantic point and orientation offsets;
robot normalization comes from the hash-matched RobotSpec; root motion comes from the declared FK
root link; and foot contacts are derived from calibrated sole FK rather than candidate-supplied
labels. Reports include robot-scale meter and normalized keypoint errors, relative end-effector
pose, root/contact timing, per-anchor two-sided amplitude/energy/jitter gates, and hashes for the
motion, RobotSpec/asset, exact FK/contact buffers, correspondence, thresholds, and metric schema.

This prevents GMR from being both teacher and ruler and blocks obvious scale, root, contact, and
motion-collapse attacks. The first protocol is deliberately narrow: seven complete semantic roles,
flat-ground kinematic sole contact, and preregistered robot-specific frame/point calibration. It is
not a physics-feasibility metric.

### 5. Morphology variant boundary and G1 MJCF materializer

`snmr/robot_variants.py` currently emits deterministic same-topology **virtual** RobotSpec probes
for tokenizer/model counterfactual tests. Their exact canonical JSON bytes are hash-bound and the
manifest is deeply immutable, but it explicitly records `backend_compatible=false` and refuses
simulator use. Free-root spawn pose is never perturbed, and nonzero nominal joint poses fail closed
until nominal-pose FK is qualified.

`snmr/mjcf_variants.py` now also materializes exact same-topology variants for the frozen G1 MJCF
dialect from captured bundle bytes. A link owns its outgoing child anchors: local lengths and
geometry scale by `s`, mass by `s^3`, and inertia by `s^5`; bilateral semantic paths share draws,
the free-root spawn and `qpos0` remain fixed, and zero perturbation is byte-identical. The manifest
hash-binds the semantic roles and realized body-scale groups, and a dataset consumer must re-hash
the emitted backend bundle before the archival-ready guard accepts it. The real G1 test exercises
all registered `+/-15%` seeds 0--7; every bundle compiles with 50 bodies/29 hinges and passes the
implemented transformation, nominal-FK-finite, and limit checks.

This is still not the P4 training set. It supports only the frozen, explicit-inertial G1 MJCF
dialect and preserves the same topology. Its perturbations are independent bilateral
semantic-path-depth draws rather than coherent human segment-family variables; nonzero joint-limit
shifts are disabled until mirrored physical signs are qualified. Collision/contact behavior,
standing state, and randomized-pose cross-asset FK have not been qualified, no corresponding
URDF/USD variants exist, and no actual teacher/training corpus has been generated from these
assets. The virtual RobotSpec probes remain useful only for model counterfactual tests. The earlier
lossy marginal descriptor was not retained as a purported strong baseline.

### 6. Fail-closed worker provenance

`snmr/provenance.py` snapshots the exact bytes a worker consumes, hashes referenced MJCF bundles
rather than only entrypoint paths, materializes immutable private copies for path-only simulator
APIs, detects source mutation, and records explicit SNMR/Newton/Isaac Lab Git revisions with dirty
or unavailable state. Required and optional digests in RobotSpec and verification records now
reject non-hexadecimal 64-character strings. The MuJoCo and Newton runners recompute motion,
referenced-asset, controller, and shared candidate-contract hashes inside their own environments.

The archival source commit is `078d3116...`. Detached clean SNMR and Newton worktrees plus the clean
Isaac Lab checkout regenerated the pilot, and every worker report records all three exact revisions
with `dirty=false`. A second clean run reproduced the invariant hashes, gate booleans, and
registered metrics (excluding wall time). This satisfies the bounded foundation-freeze exit
criterion. It does not satisfy program G0 or qualify either rollout backend.

### 7. Backend-neutral verification reports

`snmr/verification.py` defines a versioned JSON contract for backend identity, motion/spec/controller
hashes, metrics, overflow state, pass/fail, and typed frame intervals. Validation fails closed on
non-finite metrics and rollout states, malformed required or optional hashes, intervals outside a
clip, incompatible interval coordinate systems, and overlapping intervals of the same type. A
shared rollout-contract hash now lets paired runners establish that motion, asset bundle,
controller, and candidate configuration are the same before comparing outcomes.

### 8. Dynamics-twin, cross-solver, and G0 FK runners

The new scripts hold motion, robot kinematics, PD gains, timing, and initialization fixed while
scaling only effort limits. MuJoCo CPU is the first implementation and Newton/MJWarp is the second
solver. Quaternion conversion and controller hashes are explicit in their reports.

The G0 harness generates a deterministic 10,000 x 29 in-limit sample buffer and MuJoCo key-link FK
reference, binds the G1 MJCF/URDF/USD bundles and semantic key frames, and freezes strict 1.0 mm /
1e-3 rad maximum-error thresholds. A parent supervisor promotes a result only after observing
clean Isaac framework shutdown and deleting the private materialized USD bundle. The latest live
run passes the FK subgate. A separate controller/standing audit fails closed on source, asset,
checkpoint, or live-evidence mismatch.

These components are described in the
[MorphoRetarget foundation report](MORPHORETARGET_FOUNDATION_2026-08-30.md), with machine artifacts
under `autoresearch/iterate-260830-0026/`. The motion, tokenizer, provenance, PM01 diagnosis, and G0
hardening are recorded under `autoresearch/iterate-260901-0111/`.

## Latest experiment review

### A. Fixed-G1 single-window amortization: keep as a wiring/capacity gate

The preregistered seed-0 run trained the 1,583,754-parameter kinematic integration from scratch for
1,000 AdamW steps on canonical frames `[0,128)` of `walk1_subject1` at 50 Hz. The model predicted
the heading-local root pose and all 29 bounded G1 joints; no teacher root was copied. Every frozen
single-window gate passed:

| Measure | Before | After / registered statistic | Gate |
| --- | ---: | ---: | ---: |
| Total loss | 1.9795268 | 0.0130085 | last-20/first-20 drop >=75% |
| Last-20 versus first-20 loss reduction | -- | 98.452599% | pass |
| Teacher-FK MPJPE | 1.6904538 m | 0.02026875 m | <0.05 m |
| Joint MAE | 0.4857043 rad | 0.01962682 rad | <0.10 rad |
| Local-root position MAE | 1.6605420 m | 0.001840892 m | <0.05 m |
| Root-orientation geodesic | 2.0068941 rad | 0.09347169 rad | <0.15 rad |
| Joint-limit violations | 0 | 0 | exactly 0 |
| Inverse-permutation maximum error | -- | 4.7683716e-7 | <=1e-5 |

The human encoder, robot graph encoder, root head, and shared joint head each had finite nonzero
gradients on all 1,000 steps. The final state-dict SHA-256 is
`943ac36b79f43c2fba1367de6270e95bf79ac9039ccf142b3ba5a3e9943912db`; the serialized checkpoint
SHA-256 is `fafcf42b67d2fb828e7b5e1e158efc9865af98e4d95bbf0c591edb09e73143e0`; the exact aligned
window-buffer SHA-256 is `ab89dad9d4f5e1ee764daf610db450a2d5913198cbab0936b8657bdf59f8804e`.
The provenance-rerun report bytes hash to
`c1b079bb7d6cce77f83f893ca97ae8edf0d448b6bfb3afeea97637f1730a42ef`, and its artifact
manifest records canonical manifest hash
`cb31a20bcb626469f9d318419d10088ede6291a85410b3e0ad4a5fc2c534d530`.

The initial run failed to discover external Newton and Isaac Lab repository paths in its report.
That omission does not affect these CPU/MuJoCo-only model metrics because neither external
simulator was executed or consumed by the experiment. A write-once provenance rerun explicitly
records clean SNMR `5679fc784cee09c76b597ba4e7b6f615289217db`, Newton
`7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a` with `dirty=true`, and clean Isaac Lab
`3c6e67bb5c7ada942a6d1884ab69338f57596f77`. Its complete canonical contract, config, gates,
gradient audit, before/after metrics, state-dict hashes, full loss history, and serialization result
are exactly identical to the initial run, and `checkpoint.pt` is byte-identical. The provenance
rerun is therefore the preferred provenance reference; its external revisions are provenance, not
simulator qualification. Newton is honestly recorded dirty, so this is not a fully clean external
environment, but Newton was not imported or executed by the model experiment. See
[`g1_overfit_seed0_provenance_rerun`](../autoresearch/iterate-260901-0341/g1_overfit_seed0_provenance_rerun/report.json).

Assessment: keep. This proves end-to-end wiring, optimization, and representational capacity for
one fixed-G1 teacher window. It is not full fixed-G1 validation, teacher-independent semantic
non-inferiority, morphology generalization, T1 evidence, contact quality, or physics evidence.

### B. RobotSpec torque-twin contract: keep

One G1 reference, 325 frames at 50 Hz, was evaluated at four effort-limit scales.

| Effort scale | Fixed-reference max requested torque / limit | MuJoCo open-loop survival |
| ---: | ---: | ---: |
| 0.50 | 0.641 | 0.18 s |
| 0.75 | 0.427 | 0.18 s |
| 1.00 | 0.321 | 0.18 s |
| 1.25 | 0.256 | 0.18 s |

All six **pilot-local** checks pass:

1. the kinematic hash is unchanged;
2. dynamics hashes change;
3. identity-free model features change with torque;
4. normalized fixed-reference demand is monotonic;
5. the inverse scaling relation is exact;
6. the 0.50x condition emits a localized low-margin interval at frames 37--41.

Assessment: this is good causal instrumentation. It proves that a future model can be tested for
using motor-strength information. It does **not** prove that the current retargeter uses that
information or that any generated motion adapts correctly.

### C. This open-loop-PD proxy as a physics reward: kill

All four MuJoCo conditions diverge after 0.18 s even though the script's one-step fixed-reference
PD-demand proxy reports zero saturation. Rollout saturation is non-monotonic with motor strength.
The two Newton/MJWarp endpoint runs also fail after 0.22 s.

Assessment: the proxy is confounded by an unqualified open-loop controller and has no useful
candidate ordering. Keep it only as a wiring and gross-failure diagnostic. The earlier E18 language
about measured trackability equivalence should not be carried forward as strong physics evidence.

### D. MuJoCo CPU versus Newton/MJWarp pilot: keep as a diagnostic only

| Effort scale | Pass agreement | First-failure delta | Failure-type Jaccard | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 0.50 | yes, both fail | 0.02 s | 0.667 | Partial failure-category agreement |
| 1.25 | yes, both fail | 0.00 s | 1.000 | Same observed failure categories |

Neither run overflowed, and the recorded candidate/controller/spec identifiers match. The runners
now re-hash consumed motion bytes, referenced asset bundles, controller state, and a shared
candidate configuration independently. This is still one motion, two torque endpoints, and a
rejected controller, so it is not evidence of ranking parity, strong trackability, robust success,
or solver-independent learned improvement.

### E. PM01 failure diagnosis: informative, but exact mechanism remains inconclusive

The historical E06 PM01-LORO checkpoint and per-frame predictions are missing, so the 29.59 cm
failure cannot be localized exactly by root, body, or joint. A topology-safe intervention was
therefore run on the frozen E67 all-five reproduction: keep each target robot's raw 8D node
features, topology, limits, FK graph, and human latent, and replace only the pooled 32D embodiment
code with another trained robot's code.

The correct code ranked first for all five trained robots. The least harmful wrong code increased
MPJPE by 1.95x to 12.75x depending on target; the worst wrong code increased it by 5.84x to 15.91x.
Across 20 off-diagonal swaps, learned-code distance versus swap MPJPE had Spearman rho 0.580
(`p=0.0074`). This rules out the hypothesis that the legacy network simply ignored conditioning.
It does not establish that its code is a smooth morphology coordinate: G1 and Toddy are mutually
nearest in code distance yet mutually worst swaps.

Assessment: coverage/extrapolation with an identity-like or nonsmooth conditioning geometry is the
leading explanation, but remains unproven. A localized semantic-mapping failure is unresolved.
The exact LORO checkpoint must be restored or the registered LORO run repeated before choosing
between those mechanisms. This diagnosis supports building and testing the explicit graph contract;
it is not evidence that the new tokenizer or a future RobotSpec model solves PM01.

### F. G0 qualification: FK passes; controller/standing contract fails

The shutdown stall was localized to Isaac application teardown rather than pose acquisition. The
worker now releases SimulationContext callbacks, closes the USD stage, journals a fail-closed
pending result, and lets a parent process observe framework shutdown and delete the private asset
bundle before promotion. From detached clean SNMR and Newton worktrees plus clean Isaac Lab, the
live worker evaluated all 10,000 poses and seven registered links:

| FK measure | Observed maximum | Frozen threshold | Result |
| --- | ---: | ---: | --- |
| Link position L2 | 3.9321826e-6 m | < 1.0e-3 m | pass |
| Quaternion geodesic | 3.7592296e-6 rad | < 1.0e-3 rad | pass |

The pose source is live `Articulation.data.body_link_{pos,quat}_w`; no offline URDF substitute was
used. The final report records simulation-context release, stage closure, clean framework exit,
and private-bundle deletion. The FK artifact therefore legitimately carries
`g0_evaluated=true/g0_pass=true` for this subgate.

The separate controller/standing audit compares the exact saved PhysX and E67 MJWarp recipes,
checkpoints, G1 URDF/MJCF/USD bundles, and current adapter sources. It records 14 hard failures and
8 missing checks. The principal failures are:

- PhysX uses action scale 1.0 while E67 uses `0.25 * effort_limit / Kp`;
- reset/randomization contracts differ;
- MJCF hip-roll motors clip at 88 Nm versus 139 Nm in the saved config;
- MJCF joint friction is 0.1 Nm versus zero in the config/Isaac path;
- the MuJoCo path writes nominal joints before `mj_resetData` with no proven post-reset write;
- the two backends do not use one identical frozen checkpoint.

Live nominal/controller snapshots, same-motion/same-seed reset evidence, training-time Holosoma/USD
identity, and explicit MJCF velocity-limit enforcement are also missing. Assessment: **the FK
subgate passes, but overall G0 remains false**. The exact bundle is
[`g0_qualification_clean_f6e5ff0_v2`](../autoresearch/iterate-260901-0111/g0_qualification_clean_f6e5ff0_v2/README.md).

### G. Adjacent latent-command program: close the latent-specific seed-0 branch

The original E80-A snapshot said the `mZf` treatment was pending, but its artifacts subsequently
completed. The document now includes a superseding stop section. Re-running the frozen paired
analyzer gives:

| Contrast/cell | Paired completion difference | 95% cluster CI | Registered reading |
| --- | ---: | ---: | --- |
| `mZf - mE`, clean | -0.0088 | [-0.0234, 0.0059] | Point estimate narrowly meets the -0.01 clean-cost bound, but uncertainty crosses it |
| `mZf - mE`, f=0.3 / 5--25 ticks | +0.0156 | [-0.0068, 0.0381] | Far below the +0.10 primary target |
| `mZf - mE`, f=0.3 / 25--50 ticks | -0.0107 | [-0.0361, 0.0146] | No treatment benefit |
| `mZf - mTl`, f=0.3 / 25--50 ticks | -0.0664 | [-0.0908, -0.0420] | The content-free live clock is better |
| `mZf - mTl`, ambiguity f=0.3 / 25--50 ticks | -0.0557 | [-0.0789, -0.0328] | The registered content-sensitive cell also loses |

The largest `mZf - mE` gain over the entire severity sweep is +0.0156, below the preregistered
seed-0 stop threshold of +0.05. Therefore the latent-specific fusion branch should stop after seed
0 without tuning or seeds 1--2. The robust result that survives is about **training with command
outages**, not about an SNMR-latent advantage. This closes a stale open item and reinforces the
decision to put the next scientific effort into cross-embodiment RobotSpec conditioning.

The analyzer revision, input-report manifest hashes, compact cell outputs, and decision are saved in
[the E80 mZf seed-0 status artifact](../reproducibility/reports/e80_mzf_seed0_status_2026-09-01.json).
Any later mutation of the external reports is detectable against that artifact.

This adjacent result is not evidence for or against MorphoRetarget's unseen-robot hypothesis; it is
included so that the current research record does not carry an obsolete pending claim.

### H. Sim-to-sim export and hardware boundary: not deployment-ready

The four recorded E70 ONNX candidates each have three repeated pre-hardware loopback handoffs. Only
the seed-2 explicit `walk1_subject1` candidate passes all three repeats. Seed-1 SNMR
`walk1_subject1` passes 2/3, seed-1 SNMR `walk1_subject5` passes 1/3, and seed-2 explicit
`walk1_subject5` passes 1/3; all three therefore fail their repeated summary gates, with the failing
runs showing unsafe base state and/or safe-hold joint-limit violations. Every report records zero
physical robot commands sent.

Assessment: the export pipeline and certified video bundle are useful engineering and presentation
evidence, but they do not establish hardware readiness or strong physics verification. The new
program should keep hardware out of scope until the paired frozen-tracker and safe-reference gates
are passed. Evidence is in `exports/sim2sim_2026-08-12/*.loopback_safety_handoff.summary.json`.

## Program gate scorecard

| Program gate | Status on 2026-09-01 | Evidence and missing work |
| --- | --- | --- |
| G0 Contract | **Not passed; FK subgate passed** | Live PhysX evaluates 10,000 poses with 3.93e-6 m / 3.76e-6 rad maxima and clean shutdown. The controller/standing audit has 14 hard failures and 8 missing live checks, so overall G0 remains false. |
| G1 Amortization | **Single-window subgate passed; qualification not met** | The new RobotSpec pipeline passes its preregistered 1,000-step, 128-frame G1 wiring/capacity gate with 2.03 cm teacher-FK MPJPE and all gates true. No full train/validation run, stitching, virtual-head calibration, or human-side semantic non-inferiority result exists. |
| G2 Embodiment | **Not met; fixed-G1 capacity evidence only** | PM01 LORO is 5.2x worse. The kinematic graph path now has one learned fixed-G1 window result and a bounded same-topology G1 MJCF materializer. No materialized three-family teacher corpus, strong nearest-transfer comparison, generalization staircase, or held-out T1 result exists. |
| G3 Dynamics | **Not met; torque instrumentation ready** | Torque twins and localized reports work. Broader dynamics feature/intervention coverage is untested; no model is conditioned on the new features and no simulator-derived dynamics labels or learned time-warp response exist. |
| G4 Repair | **Not started for this pipeline** | Prior contact-projection machinery is useful infrastructure, but failure-type-specific local repair, strong-rollout validation, and repair distillation are not demonstrated. |
| G5 Utility | **Not started** | No matched tracker-training comparison of GMR, neural, and repaired MorphoRetarget datasets exists. |
| G6 Cross-sim | **Diagnostic pilot only** | FK asset mapping now passes, but the saved controller/checkpoint contracts do not match. Missing one frozen paired tracker, 20 stratified references, dynamics-twin ranking, robustness metrics, and the disagreement taxonomy. |

No core MorphoRetarget paper claim is currently established. The foundation is useful because it
makes the next failures interpretable, not because it should be presented as learned retargeting.

## Research assessment

### What is now known

- Learned amortized retargeting is viable on fixed and trained robot embodiments.
- The current representation failed zero-shot decoding on the only completed holdout, PM01 (29.59
  versus 5.72 cm); multi-holdout sufficiency remains untested. The legacy trained network is not
  conditioning-insensitive, but the exact LORO mechanism remains inconclusive because its
  checkpoint is missing.
- Human motion now has a strict source/provenance, frame, timebase, contact, scale, validity, and
  normalized-buffer hash contract. Invalid samples are rejected before interpolation rather than
  silently entering training tensors.
- RobotSpec can now be tensorized into padded kinematic or full variable-node batches. The
  tree-biased/cross-attention decoder passes a learned, from-scratch fixed-G1 single-window gate,
  including its root and all 29 bounded joints, while preserving inverse-permutation agreement to
  `4.7684e-7`. This is wiring and capacity evidence, not full fixed-G1 or learned generalization
  evidence.
- The frozen G1 MJCF can now be materialized into same-topology bilateral path-depth variants with
  parent-owned anchors and constant-density `s/s^3/s^5` scaling. Seeds 0--7 at `+/-15%` pass the
  implemented G1 checks and carry semantic/group hashes plus a consumer re-hash guard. They are not
  yet a teacher-labelled training corpus or a cross-asset/collision-qualified variant family.
- Human-side semantic evaluation no longer uses GMR as its ruler. Its scale, root, contact,
  correspondence, thresholds, and exact buffers are hash-bound, with per-anchor anti-collapse and
  anti-jitter gates.
- A standardized physical contract can expose causal dynamics interventions without identity
  fields such as names, paths, hashes, or robot IDs in the numeric model features. Resistance of a
  future learned model to family memorization remains untested.
- Failure reports can retain frame-level evidence across simulator boundaries.
- G1 MJCF-to-live-PhysX/Isaac-USD key-link FK passes the registered 10,000-pose maximum-error gate.
  This does not qualify the tracker because controller, reset, and checkpoint parity fail.
- GMR is appropriate as a broad kinematic teacher but cannot supervise dynamics adaptation: its
  output does not change when only torque, mass, or latency changes.
- This open-loop-PD setup is unqualified and non-ranking; the current plan therefore defers
  sequence-level PPO.

### Principal unresolved scientific risks

1. **Embodiment generalization:** three training robot families plus coherent variants may still be
   too narrow for a genuine T1 holdout.
2. **Permutation and topology:** the trained single-window result passes inverse-permutation
   inference, but one G1 window cannot expose family memorization and no cross-topology
   generalization has been measured.
3. **Dynamics supervision:** identity-free features can change without a model learning to use
   them. Labels must come from controlled candidates and qualified rollout preferences, not GMR.
4. **Verifier bias:** tracker weakness can be mistaken for retargeter weakness unless each robot's
   tracker first clears a teacher-reference qualification gate under the same recipe.
5. **Solver dependence:** agreement that a catastrophically bad rollout fails is easier than
   agreement on close candidate rankings. The latter is the evidence the learning loop needs.
6. **Semantic collapse during repair:** optimizing survival can produce conservative standing or
   low-amplitude motion unless semantic non-inferiority and refusal behavior are hard gates.
7. **Data leakage and variant validity:** asset names and serialization are excluded from the model
   path, and G1 simulator assets can now be materialized, but their draws are same-topology
   path-depth perturbations without collision/standing/random-FK qualification or teacher data.
   P4 still needs physically coherent segment-family variants across G1/H1/H1-2,
   split-by-seed/family, a strong semantic/path-aware nearest baseline, and trained-model
   anti-memorization tests.

## Recommended next execution order

P2 and P3 are **parallel tracks**, not a serial dependency. G0 asset/controller parity is a
prerequisite for cross-backend P2 conclusions, while P3's kinematic unit tests, fixed-G1
integration, and human-side held-out evaluation can proceed without physics preference labels. P2
asks whether rollout rankings are trustworthy; P3 asks whether explicit kinematics improves
embodiment generalization. The tracks converge only after the kinematic G2 decision, when a
qualified verifier would be used for dynamics preferences and repair.

### P0. Foundation freeze: completed

Source commit `078d3116...` was replayed from detached clean SNMR and Newton worktrees against the
clean Isaac Lab revision. The regenerated reports independently re-hash consumed motion bytes,
referenced assets, controller state, and shared rollout configuration. A second clean run matched
the invariant RobotSpec hashes, gate booleans, registered non-wall-time metrics, and 10,000-pose
NPZ hash. No recorded experiment output was overwritten. The complete suite passed 553 tests with
5 skips and 27 pre-existing warnings.

The freeze manifest is
[`foundation_freeze_manifest.json`](../autoresearch/iterate-260901-0111/foundation_freeze_078d311/foundation_freeze_manifest.json).
Provenance work now stops unless a later gate exposes a concrete missing identity.

### P1. Complete G0 with paired asset and controller parity

The canonical motion contract, byte-level asset binding, joint/key-frame mapping, and random-pose
FK parity are complete. The remaining work is no longer an unspecified PhysX pose-path problem;
it is a concrete controller/checkpoint contract repair:

- choose one frozen G1 tracker checkpoint and reproduce its observation normalization in both
  backends;
- make action scaling, action/torque clipping, PD gains, effort/velocity limits, friction, timing,
  latency, and reset/randomization semantics identical;
- move or prove the MuJoCo nominal-state write after `mj_resetData`;
- capture live nominal/controller snapshots and same-motion/same-seed reset state from both
  backends;
- bind the exact training-time Holosoma revision and USD, or retrain under a newly frozen recipe.

Exit criterion: the controller report has zero hard failures and zero missing checks. The already
passing FK tolerance must remain frozen.

### P2-A. Parallel infrastructure track: qualify the strong verifier

Export one frozen G1 tracker checkpoint with identical observations, actions, timing, and resets to
PhysX and MJWarp. Evaluate 20 stratified GMR references plus registered dynamics twins. Report
teacher-reference qualification, pass agreement, failure-time delta, candidate-rank correlation,
saturation, robustness, overflow, and the full disagreement taxonomy.

Accept a preference label only when the controller-independent screen and both strong backends
agree. Preserve all other cases as disagreement data.

Exit criterion: the tracker clears the teacher-reference floor in both backends and the registered
ranking/agreement thresholds are met. If not, diagnose mapping/controller parity before training.

Do not generate physics preference labels or train a physics critic until this track passes. A
single primary backend may support the first kinematic paper result; paired MJWarp remains the
robustness gate for later physics-label claims.

### P3-B. Parallel science track: integrate the kinematic RobotSpec model

- Promote the passed single-window fixed-G1 gate to the registered 20k all-train-clips screen, then
  a fresh 50k qualification only after full-sequence stitching and the virtual-head semantic
  calibration are frozen.
- Turn the bounded G1 MJCF materializer into a physically interpretable segment-family generator,
  qualify collision/standing/random-pose FK, then extend it to the required G1/H1/H1-2 assets and
  generate teacher-labelled training data. Each consumer must independently re-hash its bundle.
- Retain explicit robot-ID and a strong nearest-transfer baseline--the nearest training robot's GMR
  output transferred through a declared joint/semantic mapping--so graph conditioning must earn its
  place.
- Run adversarial renamed/reordered serialization and limb-length counterfactual inference after
  training, not only on random initialization.

Exit criterion: full fixed-G1 amortization is within about 5% of GMR human-side semantic error with
no increased violations, then the kinematic graph beats ID/strong-nearest-transfer on a held-out
robot. The passed single-window teacher-relative gate is only the entry condition for this work.

### P4. Run the first falsifiable learned embodiment experiment

Use a kinematic-only registered design before adding any dynamics conditioning:

- train: G1, H1, H1-2, each with 8 coherent variants;
- hold out: T1 29-DoF in its entirety;
- motions: 100 stratified clips;
- arms: GMR-T1 optimization baseline, T1-specific supervised upper bound, shared robot-ID, strong
  nearest transfer, and kinematic RobotSpec;
- unit of analysis: clip, with per-robot results and paired bootstrap confidence intervals.

Precede the full T1 holdout with a generalization staircase: hold out a coherent same-topology
variant, then one training family, then T1. Report each held-out robot's distance to the nearest
training morphology and error versus that distance. The first decision is whether kinematic
RobotSpec beats robot-ID and the strong nearest-transfer baseline without increasing limits or
collapsing motion amplitude. If it does not, stop before dynamics training and fix representation,
semantic alignment, or morphology coverage.

### P5. Add dynamics learning, then repair

Only after the kinematic G2 decision **and** verifier qualification, build controlled candidate
sets over time scale, root-height/amplitude residuals, contact timing, and local splines. Use L1
plus the qualified PhysX/MJWarp trackers to create selective Pareto preferences, abstaining on
solver disagreement. Train a feasibility critic and generator through ranking and supervised
self-training. Use low-dimensional CEM/MPPI for localized repair before considering an RL repair
operator policy.

Do not introduce flow matching until a deterministic model exhibits measured multi-solution
averaging and best-of-N provides a real gain.

## Reproduction and audit trail

The latest foundation records the following workflow. This audit form writes fresh outputs to a
temporary directory rather than overwriting the recorded iteration:

```bash
SNMR_AUDIT_DIR="$(mktemp -d)"

.venv/bin/python scripts/experiment_dynamics_twins.py \
  --newton-root /path/to/clean/newton \
  --isaac-lab-root /path/to/clean/IsaacLab \
  --out "$SNMR_AUDIT_DIR/dynamics_twins_mujoco.json"

/home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json "$SNMR_AUDIT_DIR/dynamics_twins_mujoco.json" \
  --scale 0.5 --isaac-lab-root /path/to/clean/IsaacLab \
  --out "$SNMR_AUDIT_DIR/newton_torque_0.5.json"

/home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json "$SNMR_AUDIT_DIR/dynamics_twins_mujoco.json" \
  --scale 1.25 --isaac-lab-root /path/to/clean/IsaacLab \
  --out "$SNMR_AUDIT_DIR/newton_torque_1.25.json"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q
```

The current contract and fail-closed G0 artifacts can be checked without overwriting the iteration:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_motion_spec.py tests/test_robot_tokens.py tests/test_provenance.py \
  tests/test_fk_parity_contract.py tests/test_robot_spec.py tests/test_verification.py

SNMR_G0_DIR="$(mktemp -d)"
.venv/bin/python scripts/g0_fk_parity_mujoco.py \
  --num-samples 10000 --seed 0 \
  --out-npz "$SNMR_G0_DIR/reference.npz" \
  --out-json "$SNMR_G0_DIR/reference.json"

/home/robotixx/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  scripts/g0_fk_parity_physx.py \
  --reference-npz "$SNMR_G0_DIR/reference.npz" \
  --reference-json "$SNMR_G0_DIR/reference.json" \
  --batch-size 256 --headless \
  --out-json "$SNMR_G0_DIR/physx.json"

# Expected to exit 1 until the registered controller mismatches are repaired.
/home/robotixx/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  scripts/g0_controller_parity.py \
  --physx-fk-report "$SNMR_G0_DIR/physx.json" \
  --out-json "$SNMR_G0_DIR/controller.json"
```

The PhysX worker must be run in the Isaac/Holosoma simulator environment. The supervisor promotes
the worker journal only after clean framework exit and private-bundle deletion. Its absence or
failure remains a false gate, never a reason to run an offline substitute.

The stale E80 treatment verdict was rechecked without writing artifacts:

```bash
.venv/bin/python scripts/analyze_e78_dropout.py \
  --treatment /data/robotixx/snmr-research/e78_masked_fusion/seed0_mZf:d_prior_explicit_snmr \
  --reference /data/robotixx/snmr-research/e78_masked_fusion/seed0_mE:c_prior_explicit

.venv/bin/python scripts/analyze_e78_dropout.py \
  --treatment /data/robotixx/snmr-research/e78_masked_fusion/seed0_mZf:d_prior_explicit_snmr \
  --reference /data/robotixx/snmr-research/e78_masked_fusion/seed0_mTl:d_prior_explicit_snmr

.venv/bin/python scripts/analyze_e78_dropout.py --grid ambiguity \
  --treatment /data/robotixx/snmr-research/e78_masked_fusion/seed0_mZf:d_prior_explicit_snmr \
  --reference /data/robotixx/snmr-research/e78_masked_fusion/seed0_mTl:d_prior_explicit_snmr
```

Archival state:

- The source contracts are frozen at `078d3116...` on `feat/morpho-retarget-foundation`; the
  annotated tag names the evidence-bearing branch commit.
- The clean bundle records SNMR `078d3116...`, Newton `7bb6d02...`, and Isaac Lab `3c6e67b...`, all
  with `dirty=false`. The user's separate Newton working checkout remains untouched.
- The pre-freeze torque and FK artifacts remain historical evidence. The
  `foundation_freeze_078d311/` directory supersedes their provenance and bundle-hash contracts.
- The current clean G0 qualification bundle binds SNMR `f6e5ff0...`, Newton `7bb6d02...`, and
  Isaac Lab `3c6e67b...` with `dirty=false`; the exact Holosoma inputs are byte-hashed and its
  checkout is honestly recorded `dirty=true` because the USD is generated/untracked.
- The current branch adds the safe pair adapter, learned kinematic integration/root path,
  independent semantic benchmark, virtual counterfactual probes, and bounded G1 MJCF materializer.
  The preferred fixed-G1 provenance rerun uses clean SNMR `5679fc7...`; commit `5540222...` adds
  the materializer. Its unused Newton checkout is disclosed as dirty rather than treated as clean.
- Final single-thread CPU verification after the learned gate and materializer changes passes 638
  tests with 5 skips and 27 known warnings in 177.40 s.
- Both learned runs wrote fresh directories; their checkpoints are byte-identical. No generated
  experiment directory was overwritten.

## Bottom line

The program has moved from an underspecified idea about "RL retargeting" to a sharper, falsifiable
research program. Motion, robot, provenance, semantic-evaluation, variable-DoF decoding, learned
root prediction, a bounded G1 variant generator, and live-FK contracts now exist; the 10,000-pose
cross-asset FK subgate and the fixed-G1 single-window wiring/capacity gate pass. The zero-shot
retargeter still has not been trained across morphologies or evaluated on T1, and the saved tracking
stacks demonstrably do not share one controller/checkpoint contract. The next engineering
milestone is repairing that specific G0 controller contract and qualifying the frozen tracker. In
parallel, the next scientific milestone is the registered full-corpus fixed-G1 screen and semantic
qualification, followed by a physically qualified multi-family variant corpus, generalization
staircase, and held-out T1 experiment. Dynamics conditioning, physics preferences, and repair
remain downstream of both relevant gates.
