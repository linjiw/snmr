# E71 Same-State Source-Valid Command Swap — Frozen Preflight Protocol

**Drafted:** 2026-08-13  
**Revised:** 2026-08-14  
**Status:** revised design frozen for preflight; **no simulator run; not preregistered**  
**Authority:** follows `docs/0813-designdoc.md`; additive to frozen E70  
**Execution rule:** no simulator cell may run until the B4/GPU gate permits it.  A smoke cell
may use a `DRAFT` manifest; every confirmatory cell requires the post-smoke manifest status
`PREREGISTERED` and exact hash agreement.

## 1. Decision, estimand, and licensed claims

E71 tests whether replacing one source-valid SNMR command stream with the other, while holding
one frozen controller and the complete initial simulator and policy state fixed, changes the
rollout toward the replacement stream's associated reference branch.

“Source-valid” is deliberate.  Each command value is copied without modification from a real
E70 trajectory, but a crossed `(state A, command B)` or `(state B, command A)` combination is a
counterfactual recombination outside the observed joint state-command support.  The explicit
controller gate restricts inference to crossed starts on which a capable explicit interface can
realize both branches; it does not make the crossed combination observationally in-support.

The primary estimand is conditional on exactly three frozen SNMR controllers and the
explicit-feasible subset of the 69 registered starts:

```text
mean over fixed controllers and eligible pairs of
  mean over physical state sides of [C(command B) - C(command A)].
```

A positive directional result licenses:

> Among explicit-feasible matched starts from two known walks, replacing the source-valid SNMR
> command stream while holding each frozen controller and complete initial state fixed shifted
> the one-second rollout toward the replacement stream's associated reference branch.

“Shifted toward” and “selected” are separate conclusions.  Selection additionally requires the
absolute branch signs in §9.  Neither result separates dynamic trajectory content from static
clip identity, establishes semantic understanding, supports held-out-motion or embodiment
generalization, or estimates a population over newly trained controllers.

E71 is evaluation-only.  It does not retrain or select an E70 student, alter the frozen E70
pairs or outcomes, or tune a representation after viewing a rollout.

## 2. Methodological basis and evidence ladder

The construction is **interchange-style**, not a formal interchange-intervention or causal-
abstraction result.  It holds the base state and model fixed while inserting a value produced by
another source input, following the closest formal precedent of
[Geiger et al. (2022)](https://proceedings.mlr.press/v162/geiger22a.html).  E71 intervenes on an
external command boundary, not an aligned hidden variable.

The experiment advances beyond destruction.  Destruction and amnesic interventions establish
behavioral reliance, but not which valid target a channel selects
([Elazar et al., 2021](https://aclanthology.org/2021.tacl-1.10/)); recent VLA tracing likewise
combines knockouts with rollout behavior
([VLA-Trace, 2026](https://arxiv.org/html/2605.30117v1)).

The explicit feasibility restriction is motivated by work showing that goals must be feasible
from the current state and that arbitrary cross-trajectory pairs can be unconnected
([Nair et al., 2020](https://proceedings.mlr.press/v100/nair20a.html);
[Ke et al., 2025](https://proceedings.mlr.press/v267/ke25a.html)).  Those papers motivate the
problem; they do not validate this assay's particular gate.

| Evidence rung | E70/E71 status before execution |
| --- | --- |
| Structural exclusivity | E70 pass |
| Behavioral necessity | E70 pass under three destruction modes and three explicit seeds |
| Source-valid target-directed shift | Open; E71 primary question |
| Absolute branch selection | Open; stronger E71 interpretation |
| Beyond absolute time | E70 pass on two known walks |
| Beyond static identity/phase | Open after E71 |
| Held-out generality | Open |

## 3. Frozen scientific inputs

The eventual machine-readable freeze manifest must bind these inputs:

| Input | Path | SHA-256 |
| --- | --- | --- |
| Ambiguity selector | `autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json` | `3c03b89e2bda939e9a8c5a6dd58caeb771944bb9e5755b14eb37ab75ed9c502e` |
| Teacher manifest | `/data/robotixx/snmr-research/e70/teacher_manifest.json` | `2d0005a7a9504056ce5944a71388e3992d0e4ee30bae441da26a460a5b163504` |
| Walk A motion | `/data/robotixx/snmr-research/e70/motions/walk1_subject1_mj_z.npz` | `b78f294395a5c74f37edc1c09dd6de0909d966a9b7fd948291c8dd803f7106aa` |
| Walk B motion | `/data/robotixx/snmr-research/e70/motions/walk1_subject5_mj_z.npz` | `d8de93425c14e90dce2930450d722d3eb2b6fcbb09e9c4ff3d59725025424f51` |

The pair is `walk1_subject1,walk1_subject5`, in that order, with all 69 frozen windows.

| Seed | Explicit SHA-256 | SNMR SHA-256 |
| ---: | --- | --- |
| 0 | `8fb6f2e1645d8070e978c34f642f82b5be8b43a67c3f11e0adf87ea1fa272916` | `f88984971c3435e3c377f038ed2ef5abef788aa1a0f68a80ad7011b23bb9b93a` |
| 1 | `a6a53f09e6bfc1f886aa331330c7456fa7bdd2cf59a2450eecda3d4788cb80d5` | `185ac3991cb6bcdd451d719c72d5d72273d4351a45bd93e7f532ebeaec730d38` |
| 2 | `c8f262f1284eb883fdd2a07c1713241e6dd00c6c7449c7d811fe892b82f69779` | `6d23363133df4ba30f6ddb12887aa22b932f6255a9e27031bdf57daf683e42c1` |

The manifest must also bind the evaluator, reset and body-fix layers, latent and distillation
runtimes, analyzer, launcher, bundle auditor, this protocol, Holosoma and SNMR revisions,
CUDA/MuJoCo environment, six checkpoints, output locations, evaluation conditions, and the
full-state tensor contract.  No such final manifest exists yet.

## 4. Four-cell intervention

For pair `i`, let `s_i^A,s_i^B` be the two reference-derived physical starts and
`u_i^A,u_i^B` the two unmodified command streams:

| Physical start | Command A | Command B |
| --- | --- | --- |
| `s_i^A` | `AA` | `AB` |
| `s_i^B` | `BA` | `BB` |

Every policy-seed report contains `69 × 4 = 276` cells.  Within a state-side comparison, the
weights, normalization, simulator model, complete Markov state, controller/history state,
evaluation seed, and physical start are fixed; only the command route changes.

The 69 pairs are not independent repetitions.  They are grouped into the same 12 connected
10-second temporal components frozen for E70, with sizes
`[1,2,2,3,3,4,4,7,7,9,13,14]`.

## 5. Runtime, reset, and routing contract

### Independent cursors

The existing Holosoma `MotionCommand.time_steps` couples the physical reset, observation,
termination, teacher route, and latent lookup.  E71 introduces:

- `state_start_steps`, used to initialize the robot state;
- `command_start_steps`, authoritative for command `motion_ids/time_steps`, explicit motion
  observation, teacher route, and SNMR latent lookup.

`snmr/integration/counterfactual_eval.py` wraps the repaired `wbt_bodyfix` reset and rewrites the
physical start from `state_start_steps - 1`; the normal shared zero-action warm-up advances to
the requested observation frame.  State and command cursors must have an in-clip preceding
frame and the full 10-second rollout horizon.

### Nominal deterministic evaluation

The confirmatory environment uses evaluation seed 404, deterministic prior inference, and the
frozen normalizers.  It must realize all of the following, fail closed, and record them in each
report and the manifest:

- no actor or other observation noise;
- no startup physics/domain randomization and no reset-state randomization;
- no pushes, action delay, randomized PD gains, torque RFI, terrain-spawn randomization, or
  initial-pose noise;
- no adaptive timestep sampler and deterministic Torch execution;
- the pinned 29-DoF, 50-Hz-policy/200-Hz-physics, plane-terrain MJWarp runtime contract.

Seeding alone is not accepted as proof of nominal conditions.

### Full-state and semantic audits

Immediately after reset and before a learned action, every same-state command pair must agree
to `1e-6` in both raw and normalized 90-D proprioception and in every available
command-independent Markov/controller tensor.  The frozen tensor-name contract includes local
qpos/root state, qvel, actuator/integration history, warm-start and applied-force state,
rigid-body/contact state, current and previous actions, processed action/torque buffers, default
pose, and any backend state exposed by the pinned runtime.  Missing or newly appearing fields
invalidate the run.  This follows MuJoCo's requirement to preserve complete integration state
for reproducibility
([MuJoCo state API](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html);
[reproducibility guidance](https://mujoco.readthedocs.io/en/3.1.2/computation/)).

The evaluator must additionally prove semantic routing rather than trusting tensor width:

1. the 154-D actor observation is exactly the configured concatenation of actions, base angular
   velocity, DoF position, DoF velocity, 64-D motion command, and motion-reference orientation;
2. command cursors and motion IDs equal the registered command side;
3. latent windows equal direct lookup at command offsets `[0,5]` and remain equal across
   physical state sides for a fixed command;
4. the one-step warm-up alone uses task-before-termination callback order, while the learned
   rollout restores the pinned MJWarp task-after-termination order;
5. warm-up produces no reset and exactly one environment step;
6. MJWarp's accumulated overflow bitmask remains exactly zero after warm-up and after every
   rollout transition.

Any audit failure aborts without an interpretable report.

### Termination isolation

Reference-dependent bad-tracking termination is suppressed for the first 50 policy steps.
Nonfinite state remains a fail-closed guard.  No primary-horizon reset is permitted.  After
future sample 50 is recorded, the normal reference termination is restored through the
10-second rollout.  Consequently, post-one-second survival and completion are descriptive
interface outcomes, not part of the primary branch estimand.

## 6. Frozen branch coordinate

The branch goal is mapped G1 joint position plus joint velocity, 58 dimensions.  Per-dimension
scale `sigma` is pooled once over the two complete motions; values below `1e-6` are replaced by
`1.0`, matching the frozen selector.

Only post-action future samples enter the branch statistic.  For rollout `r`, branch
`k ∈ {A,B}`, and `N ∈ {25,50}` future samples:

```text
Q_k(r,N)  = mean_{t=1..N,dim} ((g_r(t)-g_k(t))/sigma_dim)^2
Q_AB(N)   = mean_{t=1..N,dim} ((g_A(t)-g_B(t))/sigma_dim)^2
C(r,N)    = (Q_A(r,N)-Q_B(r,N)) / (Q_AB(N)+1e-8)
```

`C=-1` for an exact A-branch rollout and `C=+1` for an exact B-branch rollout.  For fixed pair
and physical state side:

```text
delta_swap(i,state,N) = C(s_i,u_i^B,N) - C(s_i,u_i^A,N)
delta_pair(i,N)       = mean_state delta_swap(i,state,N)
```

Positive `delta_swap` is a directional shift toward B under command replacement.  Absolute
selection additionally requires `C(command A)<0` and `C(command B)>0`.  The report retains raw
`Q_A,Q_B,Q_AB`, `sqrt(Q_A)`, `sqrt(Q_B)`, and `C` for every cell.  Average-trajectory and
terminal/miss-style reporting are complementary precedents in official trajectory benchmarks
([nuScenes prediction metrics](https://www.nuscenes.org/prediction)).

## 7. Endpoints and uncertainty

### Confirmatory endpoint

- Primary horizon: `N=50` future-only samples, or 1.0 seconds.
- Unit: one state-averaged `delta_pair` per eligible pair for each of three fixed controllers.
- Point: equal mean over eligible pairs and the three fixed controllers.
- Interval: 10,000-replicate percentile bootstrap, RNG seed 7104, resampling the eligible E70
  temporal components with one common component-weight vector across all three controllers.

The primary interval conditions on the three controllers.  This is intentional: bootstrap
coverage can be unreliable with only three training runs
([Agarwal et al., 2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)).
It must not be described as inference over a population of newly trained policies.

A crossed seed × temporal-component bootstrap, using common component weights and independent
seed weights, is reported only as a sensitivity analysis.  Treating the same pair IDs as
independent observations nested within each seed is prohibited; seeds and temporal regions are
crossed factors
([Owen and Eckles, 2012](https://arxiv.org/abs/1106.2125)).

### Secondary endpoints

- the same coordinate and swap effect over `N=25` future-only samples;
- coordinate and effect by physical state side and training seed;
- command-consistent branch-choice rates;
- first `z_cmd`, first student action, and routed explicit-teacher action;
- a first-action teacher-alignment margin,
  `||a_student-a_teacher(opposite command)||_2 -
  ||a_student-a_teacher(supplied command)||_2`, analyzed with the same fixed-controller
  temporal-component interval.  This is a registered secondary only: it measures specialist-
  teacher target alignment and does not separate teacher/clip identity from finer trajectory
  content;
- survival and completion after reference termination is restored;
- raw branch errors and audit diagnostics.

No secondary endpoint rescues a failed primary.  Numerical reproduction of diagonal E70
rollouts is not required or interpretable because E71 deliberately removes E70 randomization
and suppresses reference termination during the primary horizon.  Structural cursor, semantic
observation, latent-routing, and state-equality audits replace that stale requirement.

## 8. Explicit feasibility gate

Run the three frozen explicit students first.  A pair is eligible only when, for **each**
physical state side, the two conditions `C(command A)<0` and `C(command B)>0` hold jointly in at
least two of three explicit seeds at 1.0 seconds.

The assay is valid only if at least 20 pairs spanning at least 6 of the frozen 12 temporal
components remain.  Otherwise the explicit gate is a terminal invalid-assay result.  Pair IDs,
component IDs, rule, thresholds, input paths, and hashes are written once to an immutable gate
artifact before any SNMR report is admitted.

This is a feasibility restriction, not evidence for SNMR.  All SNMR inference is explicitly
conditional on the selected support; the all-69 descriptive distributions remain available for
transparency.  No pair may be added or removed after an SNMR rollout is inspected.

## 9. Confirmatory decision branches

After a valid explicit gate, run the unchanged four-cell grid for SNMR seeds 0, 1, and 2.

The **directional shift gate** passes only when:

1. the fixed-controller temporal-component interval has lower 95% bound above zero; and
2. mean `delta_swap` is positive for every training-seed × physical-state-side cell.

The stronger **selection gate** passes only when the shift gate passes and aggregate
`C(command A)<0<C(command B)` holds separately from both physical state sides.

| Outcome | Frozen interpretation |
| --- | --- |
| Explicit pair/component gate invalid | crossed-state assay invalid; no SNMR inference |
| Explicit valid; shift fails | E70 retains necessity/beyond-clock results, but fixed-policy target-directed shift is not established |
| Shift passes; selection fails | changing only the source-valid command caused a directional shift toward its associated branch in these fixed-policy rollouts on the selected support; do not say “selects” |
| Shift and selection pass | source-valid replacement shifts behavior and meets the aggregate branch-selection sign criterion on the selected support |

Report effect sizes, raw coordinates, all controller/state directions, and both intervals.
Do not describe a barely positive result as large.

## 10. Artifact state machine and isolation

Use a new root, provisionally `/data/robotixx/snmr-research/e71/`.  Never write beneath an E70
student directory.  Reports, gate, analysis, manifest transitions, and final certificate are
write-once and atomic.  The launcher must enforce this state machine without skipped states:

```text
DESIGN_DRAFT
  -> DRAFT_MANIFEST
  -> FOUR_CELL_SMOKE_PASSED
  -> PREREGISTERED_MANIFEST
  -> EXPLICIT_0_1_2_COMPLETE
  -> EXPLICIT_PREFLIGHT_CERTIFIED
  -> EXPLICIT_GATE_INVALID [terminal]
     or EXPLICIT_GATE_VALID
        -> SNMR_0_1_2_COMPLETE
        -> FINAL_ANALYSIS_WRITTEN
        -> BUNDLE_CERTIFIED
```

Smoke output uses a distinct protocol and can never enter the confirmatory analyzer.  The
`DRAFT` is immutable; promotion writes a distinct `PREREGISTERED` child rather than editing it.
The child must bind the parent DRAFT, smoke report, independently recomputed smoke certificate,
postprocess marker, 26,000-MiB capacity proof, owner, and timezone-aware date, while every
scientific/runtime field remains byte-for-byte unchanged.  The evaluator replays that lineage
before environment construction.  The bundle auditor independently replays every file hash,
validates the exact grid, state and routing audits, recomputes the explicit gate and final
analysis, and refuses a final bundle unless this transition is valid.

## 11. Current implementation status

| File | Purpose | Status on 2026-08-14 |
| --- | --- | --- |
| `snmr/integration/counterfactual_eval.py` | four-cell grid, independent cursors, normalized coordinate, same-state audits | implemented; CPU-test coverage present |
| `scripts/eval_e71_command_swap.py` | nominal Holosoma evaluator, semantic/full-state/routing audits, termination isolation, report writer | implemented; **not simulator-smoke-tested** |
| `scripts/analyze_e71_command_swap.py` | explicit sign gate, 12-component partition, fixed-controller primary interval, crossed sensitivity | implemented; CPU-test coverage present |
| `scripts/audit_e71_bundle.py` | manifest replay, explicit preflight, gate/analysis recomputation, final certificate | implemented; CPU-test coverage present |
| `scripts/prepare_e71_freeze.py` | write-once DRAFT and audited child-PREREGISTERED generator | implemented; temporary CPU dry run passed, no production manifest instantiated |
| `tests/test_counterfactual_eval.py` | grid, metric, state-audit regression tests | present |
| `tests/test_e71_command_swap_analysis.py` | report, gate, component and decision regression tests | present |
| `tests/test_audit_e71_bundle.py` | manifest and bundle fail-closed tests | present |
| `scripts/run_e71_command_swap.sh` | B4/logical-GPU/manifest/smoke gates and explicit-before-SNMR orchestration | transition enforcement implemented and CPU-dry-checked; simulator smoke pending |
| E71 freeze manifest | binds code, runtime, conditions, full-state contract, checkpoints and paths | pending; not instantiated |

No E71 simulator report, explicit gate, SNMR result, analysis, or certificate exists.  This is
a frozen preflight protocol; it does not record a preregistration.

## 12. Freeze and launch checklist

Do not change the status to preregistered until every pre-run item is complete:

- [x] reset layer, evaluator, analyzer, bundle auditor, and CPU regression tests implemented;
- [x] launcher/manifest transition state machine enforced and CPU-dry-checked without a GPU launch;
- [ ] `DRAFT` manifest created with all file hashes and exact runtime/state contracts;
- [ ] four-environment smoke proves callback order, cursor routing, semantic observation,
      latent lookup, nominal conditions, complete state equality, and 50 uncensored future steps;
- [ ] smoke artifact audited and excluded from confirmatory inputs;
- [ ] immutable `PREREGISTERED` child written with parent/smoke/certificate hashes, date, and
      owner before any 276-cell result;
- [ ] output paths and write-once explicit gate path frozen;
- [ ] B4/cross-project GPU gate released;
- [ ] `POSTPROCESS_COMPLETE` present and at least 26,000 MiB free GPU memory observable.

Current blocker on 2026-08-14: `/data/robotixx/snmr-research/e70/POSTPROCESS_COMPLETE` is absent,
and `nvidia-smi` cannot communicate with the NVIDIA driver, so the 26,000-MiB condition cannot
**[RETRACTED 2026-08-14: stale. The driver responds and 30,827 MiB are free; the memory condition IS
demonstrable. `POSTPROCESS_COMPLETE` remains absent, so the launch block stands on that gate alone.
A blocking defect in `counterfactual_eval.py:561-564` (pelvis vs `ref_body_index` root init) must
also be fixed before any DRAFT — see `docs/PLAN_2026-08-14.md` §4 Track D.]**
be demonstrated.  B4 retains priority.  No E71 simulator launch is authorized.

Once those gates recover, the only allowed transition is:

```bash
.venv/bin/python scripts/prepare_e71_freeze.py --status DRAFT \
  --out autoresearch/iterate-260813-2350/e71_freeze_draft.json
scripts/run_e71_command_swap.sh smoke
.venv/bin/python scripts/prepare_e71_freeze.py --status PREREGISTERED \
  --parent-draft autoresearch/iterate-260813-2350/e71_freeze_draft.json \
  --smoke-report /data/robotixx/snmr-research/e71/reports/smoke_explicit_seed0_pair0.json \
  --smoke-audit /data/robotixx/snmr-research/e71/smoke_audit.json \
  --owner '<named owner>' \
  --out autoresearch/iterate-260813-2350/e71_freeze_manifest.json
scripts/run_e71_command_swap.sh confirmatory
```

Every destination is create-if-absent.  A failed stage is inspected and resolved; no artifact is
overwritten or silently resumed.

## 13. What follows E71

If E71 establishes a shift, the next frozen-policy battery separates identity, phase, and local
motion using correct-clip phase shifts (`±0.25`, `±0.5 s`), first-frame or clip-mean constants,
and current/duplicated/two-sample latent windows.  These are perturbation diagnostics, not
source-valid crossed commands.

Only after target-directed behavior and identity/phase decomposition should the project spend
on fixed-decoder calibration or a second interface.  A frozen decoder is established motor-
prior machinery rather than the novelty itself
([AnyBody, 2026](https://arxiv.org/html/2606.29209v1)); the contribution would be its use as a
calibrated measurement instrument.  Likewise, explicit reference paths can suppress latent
utility, supporting the relational interpretation of representation value
([UniTracker, 2025](https://arxiv.org/html/2507.07356v2)).
