# Learning-based retargeting research status and experiment review

**Date:** 2026-09-01
**Latest program:** MorphoRetarget
**Overall status:** pre-training foundation; core contracts implemented; program Gate G0 fails closed
**Evidence basis:** repository working tree, recorded artifacts, and bounded iteration
`autoresearch/iterate-260901-0111/`
**Repository state audited:** `feat/morpho-retarget-foundation@8f5ed72c6b45868913e1fd2e53b88ea89a17a031`
plus uncommitted work

## Executive verdict

The latest research direction is well defined, but its central learning claim has not yet been
tested. The target is a single variable-DoF retargeter that reads an unseen humanoid's physical
specification rather than its identity, produces semantically faithful motion without a
robot-specific head, prompt, or fine-tuning, and improves robust closed-loop trackability under
independent physics verification.

The current work has now established the prerequisite RobotSpec, canonical `HumanMotionSpec`,
graph-token, provenance, FK-parity, and verification-report contracts. It has also diagnosed one
important property of the legacy PM01 failure: trained SNMR uses its embodiment conditioning, but
the surviving artifacts cannot determine whether the exact LORO failure was coverage,
extrapolation, or semantic misalignment. It has **not** trained a RobotSpec-conditioned model,
demonstrated a learned response to kinematics or dynamics, run the held-out T1 experiment,
established strong-tracker PhysX/MJWarp ranking parity, repaired failures, or shown downstream data
utility. The honest maturity label remains **pre-training foundation; G0 not passed**, not "six
gates passed." The earlier six passing checks are local checks for one torque-twin pilot, not the
program's G0--G6 paper gates.

The strongest positive result is now a reproducible set of fail-closed input contracts: exact
motion/provenance hashes, a serialization-equivariant variable-node tokenizer, and a deterministic
10,000-pose MuJoCo FK reference. The strongest boundary is equally important: the PhysX worker did
not return comparison poses, so the FK gate records `g0_evaluated=false` and `g0_pass=false` rather
than substituting offline URDF FK or relaxing the gate. Separately, open-loop PD replay still fails
too quickly and responds non-monotonically to torque scale, so it must not be used as a candidate
ranker, physics preference label, or RL reward.

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
| New graph-token contract | Padded variable-node tensors, exact inverse-permutation agreement, tree distances, per-node limits, and separated dynamics availability | The tensor interface now exists and is adversarially tested; no learned graph encoder, per-joint model, fixed-G1 integration, or held-out result exists |

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

This is currently an MJCF-first RobotSpec adapter. URDF/USD bundles are bound by the G0 harness,
but no URDF/USD RobotSpec adapter or measured cross-asset parity has been demonstrated, and the
feature bundle is deliberately not connected to old SNMR checkpoints.
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

This completes the in-repository motion schema and resampling contract. It does not prove that all
AMASS, LAFAN1, GMR, and existing SNMR loaders have been migrated to it, nor does it pass cross-asset
G0 by itself.

### 3. Kinematic RobotSpec graph-token contract

`snmr/robot_tokens.py` is a parameter-free tensorization layer for heterogeneous RobotSpecs. It
emits padded variable-node batches, node and joint masks, structural parent metadata, all-pairs
tree distances and attention bias, per-node joint bounds, topology features, and separate dynamics
availability. The registered first mode is `feature_set="kinematic"`; it zeros live dynamics
availability and selects only kinematic fields. The full 37D fields remain available for a later,
separately gated dynamics experiment.

Adversarial tests permute link and joint serialization, rename links and joints, batch different
DoF counts, and compare kinematic twins. After inverse permutation, the contract tensors agree
exactly. A shared limit-by-construction helper maps per-node logits into each active joint's range.

This is **not** a learned RobotSpec model. No graph/topology encoder, cross-attention module,
temporal integration, trained per-joint head, fixed-G1 amortization result, or held-out T1 result has
been produced. The current token contract also rejects multiple scalar joints attached to one link;
true multi-DoF joint types would require separate joint tokens.

### 4. Fail-closed worker provenance

`snmr/provenance.py` snapshots the exact bytes a worker consumes, hashes referenced MJCF bundles
rather than only entrypoint paths, materializes immutable private copies for path-only simulator
APIs, detects source mutation, and records explicit SNMR/Newton/Isaac Lab Git revisions with dirty
or unavailable state. Required and optional digests in RobotSpec and verification records now
reject non-hexadecimal 64-character strings. The MuJoCo and Newton runners recompute motion,
referenced-asset, controller, and shared candidate-contract hashes inside their own environments.

This is meaningful provenance hardening, not an archival freeze. The SNMR working tree remains
uncommitted, the current Newton checkout is still dirty, and a clean-checkout reproduction has not
yet satisfied the foundation exit criterion.

### 5. Backend-neutral verification reports

`snmr/verification.py` defines a versioned JSON contract for backend identity, motion/spec/controller
hashes, metrics, overflow state, pass/fail, and typed frame intervals. Validation fails closed on
non-finite metrics and rollout states, malformed required or optional hashes, intervals outside a
clip, incompatible interval coordinate systems, and overlapping intervals of the same type. A
shared rollout-contract hash now lets paired runners establish that motion, asset bundle,
controller, and candidate configuration are the same before comparing outcomes.

### 6. Dynamics-twin, cross-solver, and G0 FK runners

The new scripts hold motion, robot kinematics, PD gains, timing, and initialization fixed while
scaling only effort limits. MuJoCo CPU is the first implementation and Newton/MJWarp is the second
solver. Quaternion conversion and controller hashes are explicit in their reports.

The G0 harness now generates a deterministic 10,000 x 29 in-limit sample buffer and MuJoCo key-link
FK reference, binds the G1 MJCF/URDF/USD bundles and semantic key frames, and freezes strict
1.0 mm / 1e-3 rad maximum-error thresholds. Its PhysX side fails closed when Isaac Lab is
unavailable or fails to return comparison poses; it does not substitute a different FK engine.

These components are described in the
[MorphoRetarget foundation report](MORPHORETARGET_FOUNDATION_2026-08-30.md), with machine artifacts
under `autoresearch/iterate-260830-0026/`. The motion, tokenizer, provenance, PM01 diagnosis, and G0
hardening are recorded under `autoresearch/iterate-260901-0111/`.

## Latest experiment review

### A. RobotSpec torque-twin contract: keep

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

### B. This open-loop-PD proxy as a physics reward: kill

All four MuJoCo conditions diverge after 0.18 s even though the script's one-step fixed-reference
PD-demand proxy reports zero saturation. Rollout saturation is non-monotonic with motor strength.
The two Newton/MJWarp endpoint runs also fail after 0.22 s.

Assessment: the proxy is confounded by an unqualified open-loop controller and has no useful
candidate ordering. Keep it only as a wiring and gross-failure diagnostic. The earlier E18 language
about measured trackability equivalence should not be carried forward as strong physics evidence.

### C. MuJoCo CPU versus Newton/MJWarp pilot: keep as a diagnostic only

| Effort scale | Pass agreement | First-failure delta | Failure-type Jaccard | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 0.50 | yes, both fail | 0.02 s | 0.667 | Partial failure-category agreement |
| 1.25 | yes, both fail | 0.00 s | 1.000 | Same observed failure categories |

Neither run overflowed, and the recorded candidate/controller/spec identifiers match. The runners
now re-hash consumed motion bytes, referenced asset bundles, controller state, and a shared
candidate configuration independently. This is still one motion, two torque endpoints, and a
rejected controller, so it is not evidence of ranking parity, strong trackability, robust success,
or solver-independent learned improvement.

### D. PM01 failure diagnosis: informative, but exact mechanism remains inconclusive

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

### E. G0 10,000-pose FK parity: harness complete, gate fails closed

The MuJoCo worker generated and reproduced a deterministic 10,000 x 29 in-limit pose buffer at seed
0, key-link poses for pelvis, torso, head, hands, and feet, and bound MJCF, URDF, USD, sample, and
output hashes. The registered pass thresholds remain strict maximum position error below 1.0 mm
and orientation geodesic error below 1e-3 rad.

The ordinary SNMR environment correctly reported Isaac Lab unavailable. A capped attempt in the
Holosoma `hssim` environment initialized Isaac Sim 5.1, loaded the generated G1 USD, and observed 29
joints, but returned no articulation body-pose tensor before the cap. There are consequently no
PhysX error metrics. Both artifacts record `g0_evaluated=false` and `g0_pass=false`.

Assessment: the executable protocol, asset binding, sample buffer, semantic link mapping, and
MuJoCo reference are useful completed infrastructure. G0 has **not** passed, and the timeout does
not support any FK-parity conclusion. The next attempt must diagnose the PhysX articulation path
and return the registered poses; offline URDF FK is not an acceptable substitute for the named
PhysX gate.

### F. Adjacent latent-command program: close the latent-specific seed-0 branch

The repository's latest E80-A document says the `mZf` treatment was pending, but its artifacts are
complete under `/data/robotixx/snmr-research/e78_masked_fusion/seed0_mZf/`. Re-running the frozen
paired analyzer gives:

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

### G. Sim-to-sim export and hardware boundary: not deployment-ready

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
| G0 Contract | **Not passed; fails closed** | `HumanMotionSpec`, graph/token contracts, worker byte snapshots, revision fields, G1 MJCF/URDF/USD bundle hashes, key-frame mappings, and the deterministic 10,000-pose MuJoCo reference now exist. The PhysX worker returned no poses, so no cross-asset maximum errors exist and both G0 booleans are false. Controller/standing-state parity and clean archival reproduction also remain. |
| G1 Amortization | **Historical evidence only** | Existing G1 SNMR imitates GMR well. The new RobotSpec-conditioned pipeline has not been integrated or tested on fixed-G1 amortization. |
| G2 Embodiment | **Not met; token contract only** | PM01 LORO is 5.2x worse. The diagnosis rules out conditioning-insensitivity but leaves exact coverage versus semantic failure unresolved. A serialization-equivariant tokenizer and bounded variable-DoF output utility exist, but no learned RobotSpec graph encoder/conditioned decoder, coherent-variant training, strong nearest-transfer comparison, generalization staircase, or held-out T1 result exists. |
| G3 Dynamics | **Not met; torque instrumentation ready** | Torque twins and localized reports work. Broader dynamics feature/intervention coverage is untested; no model is conditioned on the new features and no simulator-derived dynamics labels or learned time-warp response exist. |
| G4 Repair | **Not started for this pipeline** | Prior contact-projection machinery is useful infrastructure, but failure-type-specific local repair, strong-rollout validation, and repair distillation are not demonstrated. |
| G5 Utility | **Not started** | No matched tracker-training comparison of GMR, neural, and repaired MorphoRetarget datasets exists. |
| G6 Cross-sim | **Diagnostic pilot only** | Two endpoints agree that a bad open-loop rollout fails, and shared candidate hashes now establish input identity. Missing a frozen tracker, 20 stratified references, dynamics-twin ranking, robustness metrics, and a disagreement taxonomy. The separate PhysX FK attempt produced no comparison poses. |

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
- RobotSpec can now be tensorized into padded kinematic or full variable-node batches whose
  structural outputs are exactly inverse-permutation equivalent under tested serialization and
  renaming interventions. This is an input/output contract result, not a learned generalization
  result.
- A standardized physical contract can expose causal dynamics interventions without identity
  fields such as names, paths, hashes, or robot IDs in the numeric model features. Resistance of a
  future learned model to family memorization remains untested.
- Failure reports can retain frame-level evidence across simulator boundaries.
- A 10,000-pose MuJoCo FK reference can be reproduced and asset/sample identity can be audited, but
  PhysX parity is unknown because the comparison worker returned no poses.
- GMR is appropriate as a broad kinematic teacher but cannot supervise dynamics adaptation: its
  output does not change when only torque, mass, or latency changes.
- This open-loop-PD setup is unqualified and non-ranking; the current plan therefore defers
  sequence-level PPO.

### Principal unresolved scientific risks

1. **Embodiment generalization:** three training robot families plus coherent variants may still be
   too narrow for a genuine T1 holdout.
2. **Permutation and topology:** the parameter-free token/output contract passes adversarial
   serialization and renaming tests, but no learned encoder/decoder has demonstrated the same
   invariance or cross-topology generalization.
3. **Dynamics supervision:** identity-free features can change without a model learning to use
   them. Labels must come from controlled candidates and qualified rollout preferences, not GMR.
4. **Verifier bias:** tracker weakness can be mistaken for retargeter weakness unless each robot's
   tracker first clears a teacher-reference qualification gate under the same recipe.
5. **Solver dependence:** agreement that a catastrophically bad rollout fails is easier than
   agreement on close candidate rankings. The latter is the evidence the learning loop needs.
6. **Semantic collapse during repair:** optimizing survival can produce conservative standing or
   low-amplitude motion unless semantic non-inferiority and refusal behavior are hard gates.
7. **Data leakage:** asset names and serialization are excluded from the tested token contract, but
   coherent variants and family similarity still require split-by-seed/family, renamed-asset
   inference, and trained-model anti-memorization tests.

## Recommended next execution order

P2 and P3 are **parallel tracks**, not a serial dependency. G0 asset/controller parity is a
prerequisite for cross-backend P2 conclusions, while P3's kinematic unit tests, fixed-G1
integration, and human-side held-out evaluation can proceed without physics preference labels. P2
asks whether rollout rankings are trustworthy; P3 asks whether explicit kinematics improves
embodiment generalization. The tracks converge only after the kinematic G2 decision, when a
qualified verifier would be used for dynamics preferences and repair.

### P0. Freeze and make the foundation reproducible

- Commit the bounded foundation on `feat/morpho-retarget-foundation` and bind that exact SNMR
  revision in regenerated artifacts.
- Re-run the pilot from a clean SNMR checkout and a clean Newton checkout; the current Newton build
  is recorded as `7bb6d02...+dirty`.
- Use the implemented worker-side motion and referenced-asset snapshots, source-revision fields,
  and shared candidate hash to regenerate the old pilot rather than retroactively treating its
  weaker manifest as complete.
- Stop extending provenance after a clean checkout reproduces the same hashes, gate booleans, and
  tolerance-bound metrics.

Exit criterion: a clean checkout reproduces the same invariant RobotSpec hashes and gate booleans,
registered metrics within frozen tolerances excluding wall time, and focused/full test results
without modifying recorded run directories.

### P1. Complete G0 with paired asset and controller parity

The canonical `HumanMotionSpec`, byte-level asset binding, key-frame mapping, registered random pose
buffer, and MuJoCo reference are complete. Repair the live PhysX articulation path and then
mechanically verify the exact G1 MJCF against the generated Isaac Lab URDF/USD asset:

- joint names, ordering, types, axes, and limits;
- root and key-link FK across random in-limit poses;
- quaternion and world/body frame conventions;
- standing state and collision geometry;
- position-action interpretation, PD gains, effort limits, control rate, simulation rate, and
  latency;
- identical motion, controller, and mapping hashes in exchanged reports.

Exit criterion: all schema/order/frame/timing checks pass and key-link FK is within a frozen
tolerance before any cross-backend rollout comparison.

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

- Reuse the existing SNMR human temporal encoder and connect only the implemented kinematic token
  subset first.
- Add the learned topology encoder, cross-attention, and shared per-joint head on top of the tested
  serialization-equivariant variable-node contract.
- Generate coherent same-topology variants with family/seed-disjoint splits.
- Retain explicit robot-ID and a strong nearest-transfer baseline--the nearest training robot's GMR
  output transferred through a declared joint/semantic mapping--so graph conditioning must earn its
  place.
- Run adversarial renamed/reordered serialization inference on the trained model, not only the
  tokenizer.

Exit criterion: fixed-G1 amortization is within about 5% of GMR semantic error with no increased
violations, then the kinematic graph beats ID/strong-nearest-transfer on a held-out robot.

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
  --out "$SNMR_AUDIT_DIR/dynamics_twins_mujoco.json"

/home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json "$SNMR_AUDIT_DIR/dynamics_twins_mujoco.json" \
  --scale 0.5 --out "$SNMR_AUDIT_DIR/newton_torque_0.5.json"

/home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json "$SNMR_AUDIT_DIR/dynamics_twins_mujoco.json" \
  --scale 1.25 --out "$SNMR_AUDIT_DIR/newton_torque_1.25.json"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
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
```

The PhysX worker must be run in the Isaac/Holosoma simulator environment. Its absence or failure to
return body poses is a recorded false gate, not a reason to run an offline substitute.

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

Current workspace cautions:

- The dedicated branch exists as `feat/morpho-retarget-foundation`, but `HEAD` is still
  `8f5ed72c...`; the complete foundation remains uncommitted working-tree material.
- The new modules, scripts, tests, docs, and both bounded iteration directories are untracked;
  README, experiment-log, E80, and trackability files are modified.
- New reports can bind SNMR, Newton, and Isaac Lab revisions explicitly, but the recorded SNMR tree
  is dirty and Newton remains `7bb6d02...+dirty`. The older torque pilot retains its weaker
  provenance and should be regenerated after the freeze.
- The hardening ledger records 78 focused passes and 3 skips. This is evidence for the bounded
  contracts, not a clean archival full-suite result.
- No generated experiment directory was overwritten during this review.

## Bottom line

The program has moved from an underspecified idea about "RL retargeting" to a sharper, falsifiable
research program. The foundation can now represent and audit controlled physical changes and
exchange localized evidence between solvers, canonicalize human motion, and tensorize variable
robot graphs without serialization identity. That is real progress. The learned zero-shot
retargeter itself, however, has not been built or evaluated under the new contract. The next
engineering milestone is the fail-closed G0/qualified-tracker path; in parallel, the next scientific
milestone is fixed-G1 integration followed by the small kinematic-only held-out-T1 experiment.
Dynamics conditioning, physics preferences, and repair remain downstream of both relevant gates.
