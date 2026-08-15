# Paper Strengthening Design — Target Specificity Before Breadth

**Date:** 2026-08-13  
**Revised:** 2026-08-14  
**Scope:** review of the current seven-page ICRA manuscript, frozen E70 artifacts, current
execution record, and the supplied external review guide.  This document proposes work; it
does not alter E70, the paper/video freeze, or any generated experiment directory.

## 1. Decision

Keep the paper as an **interface-measurement paper**, not an SNMR capability paper.  The frozen
E70 result is statistically credible and already answers the clock confound.  Its remaining
acceptance risk is interpretation: the current assay does not separate a static two-clip
identifier from time-varying command content, and its only fixed-policy intervention destroys
the command rather than replacing it with another source-valid command.

The highest-value next result is therefore a **same-state, source-valid command swap on the
frozen E70 policies**.  Each substituted command is an unmodified value from a real trajectory,
but the crossed state-command combination is a deliberate counterfactual recombination outside
observed joint support.  It should precede E72 noise titration, new capability baselines,
another robot, or a broader motion set.  Destruction shows that a channel is necessary; a
source-valid swap can test target-directed shift and, under a stronger sign criterion, branch
selection.

Recommended thesis:

> A learned humanoid command interface is credible only when it passes an evidence ladder:
> structural exclusivity, behavioral necessity, target-specific counterfactual swaps,
> shortcut-matched nulls, and held-out generalization.

Current-data-safe thesis, until the swap passes:

> Under a fixed training protocol and a two-walk matched-state assay, the frozen SNMR source
> produces more successful control than absolute within-clip time or a matched-phase
> wrong-trajectory source; this establishes clip-disambiguating, control-usable trajectory
> information, not future semantics or generalization.

## 2. Audited current status

### Submission state

- The audit began from clean commit `cbe1f9c`; `origin/main` matched the local branch before
  this design document was added.
- `paper/main.pdf` is a seven-page, US-letter, anonymous three-seed positive build.
- Frozen E70 is complete for seeds 0, 1, and 2.  The authoritative analyzer is
  `/data/robotixx/snmr-research/e70/analysis_seed0-1-2.json`, SHA-256
  `05ca3176c0a78eebc6ca49665ce092ddfe0ea423be51e7d61a0192d428ea9b5f`.
- The primary result is A--T `+0.191 [0.124, 0.274]` and A--S
  `+0.199 [0.127, 0.279]`, with positive A--T direction on both clips.
- Ambiguity completion is explicit `0.973`, SNMR `0.754`, time `0.562`, shuffled
  `0.552`, and proprioception `0.437`.
- The descriptive normalized content recovery is
  `(0.754-0.562)/(0.973-0.562) = 0.467`; the survival-time analogue is `0.541`.
  These are algorithm-relative effect sizes, not additive information decompositions.
- The preregistered secondary temporal-block analysis is complete over 12 blocks and remains
  positive: A--T `+0.191 [0.106, 0.280]`, A--S `+0.199 [0.117, 0.282]`.
- Zero, batch-shuffle, and marginal-random destruction collapse each of the three explicit
  students to `0.000` completion.  The manuscript currently reports this only for seed 0.
- B4 video re-freeze has owner approval, but the postprocess supervisor is still waiting.  Under
  the current execution contract, do not edit `paper/main.tex` until B4 completes or a new dated
  owner amendment explicitly reopens it.
- As of 2026-08-14, `/data/robotixx/snmr-research/e70/POSTPROCESS_COMPLETE` is absent and
  `nvidia-smi` cannot communicate with the NVIDIA driver.  **[RETRACTED 2026-08-14: stale. Driver
  590.48.01 responds and reports 30,827 MiB free of 32,607, above the 26,000-MiB gate. B4's real
  blocker is a MuJoCo offscreen-framebuffer crash in capture 1 of 6; see
  `docs/EXECUTION_PLAN_REPORTS.md` and `docs/PLAN_2026-08-14.md`.]**  The frozen 26,000-MiB launch
  condition therefore cannot be demonstrated; no E71 simulator run is authorized.
- B5 submission has not been recorded as complete.

### Evidence ladder

| Rung | Question | Current status | Paper implication |
| --- | --- | --- | --- |
| Structural exclusivity | Can the goal reach the decoder outside `z_cmd`? | **Pass** after the name-derived layout repair and regression tests | Keep as a contract, not an architectural assertion |
| Behavioral necessity | Does a trained controller depend on `z_cmd`? | **Pass**: three destruction modes collapse all three explicit seeds | Update the PDF from seed 0 to all seeds |
| Target-directed shift/selection | Does one source-valid command versus another shift, and possibly select, the corresponding future from the same complete initial state? | **Missing** | Highest-priority experiment |
| Beyond time | Does SNMR beat an equally trained absolute-time route? | **Pass** on two known walks | Frozen E70 headline |
| Beyond identity/phase | Is the useful signal more than clip identity plus progress? | **Open** | Add phase/static/ID controls |
| Held-out generality | Does the conclusion survive unseen motions, another interface, or another robot? | **Open** | Post-submission unless target specificity closes early |

### Immediate manuscript inconsistencies

1. `paper/main.tex` still says a temporal-block or non-overlapping analysis "is needed" even
   though the final preregistered 12-block analysis is complete and positive.
2. The abstract, contribution list, and instrument section still describe command destruction
   on the seed-0 explicit student, although all three explicit seeds now collapse under all three
   destruction modes.
3. The introduction motivates the learned interface with contact reasoning, uncertainty, and
   cross-embodiment structure, while the paper later reports weak unsupervised contact content,
   failed unseen-robot decoding, and no interaction motions in the tracking assay.  The opening
   should instead motivate **attribution and identifiability**.
4. "A causal interface instrument" is broader than the present evidence.  Before a source-valid
   swap, prefer "exclusive interventional interface assay" and reserve causal language for
   within-policy destruction.
5. Figure 1 shows the architecture and aggregate bars but not the critical construction:
   similar present states, divergent futures, and different rollouts under source-valid
   commands.
6. The upstream two-teacher table and detailed probes consume space while providing weaker
   headline evidence than the missing target-specific intervention.

## 3. Non-goals

- Do not try to beat the explicit ceiling.  E52 v4 already found the explicit-plus-SNMR arm null
  over three seeds, and the E70 explicit arm is at teacher parity.
- Do not change E70 thresholds, pairs, seeds, analyzer, reports, or interpretation sentence.
- Do not treat E72 Gaussian noise as a substitute for a source-valid command swap.  Noise measures
  sensitivity; it does not establish which future the command requests.
- Do not claim semantics, future intent beyond identity, embodiment generalization, hardware
  validation, or sim-to-real transfer from the present result.
- Do not put the Phase E CPU-MuJoCo loopback campaign ahead of the causal controls.  It is useful
  engineering evidence and can remain one sentence, a video, or supplementary material.

## 4. Workstream P0 — zero-new-training paper corrections

**When:** prepare the patch now, apply only after B4's freeze permits it.  
**Cost:** less than one day.  
**Scientific risk:** none; all values already exist in frozen artifacts.

### Changes

1. Replace the limitations sentence promising a temporal analysis with the completed result:
   the 12-block secondary bootstrap is directionally consistent and both intervals exclude zero.
2. Replace every "seed-0 destruction" statement with the all-seed result.  Generate the numbers
   from a small hash-checking renderer rather than hand-transcribing them.
3. Introduce two explicitly descriptive scalars:

   ```text
   CNM(U) = Y_U - max_{N in nulls} Y_N
   NCR(U) = (Y_U - Y_strongest_null) / (Y_explicit - Y_strongest_null)
   ```

   Report CNM `+0.191` and NCR `0.47` for ambiguity completion.  State that separately trained
   arms make NCR an algorithm-relative recovery score, not an information fraction.
4. Rewrite the first two introduction paragraphs around deterministic-clip non-identifiability:
   time indexes the full target on a deterministic trajectory, so tracking success cannot
   identify motion-specific command content.
5. Change the contribution wording from a generic causal-instrument claim to the evidence
   ladder and its currently passed rungs.
6. Keep the single-clip failure prominent.  It is the conceptual result that turns the paper
   from a positive SNMR case study into a reusable evaluation standard.

### Acceptance criteria

- All new numeric prose is generated from hash-checked JSON.
- The primary E70 table, intervals, and preregistered verdict are unchanged.
- Positive, scoped-null, and invalid-assay branches still compile.
- PDF remains at most eight pages, with embedded fonts, no overfull boxes, and no anonymity
  findings.

## 5. Workstream P1 — same-state counterfactual command swap

**Priority:** highest.  
**Type:** evaluation-only on frozen E70 checkpoints.  
**Estimated execution:** six policy evaluations plus analysis; implementation and validation
are more substantial than GPU time.  
**Isolation:** use a new artifact root and never edit frozen E70 files or write under
`/data/robotixx/snmr-research/e70/`.

### 5.1 Scientific question

For each frozen ambiguity pair, evaluate the complete four-cell grid:

```text
(state a, command a)    (state a, command b)
(state b, command a)    (state b, command b)
```

Holding the student weights and complete initial simulator and policy state fixed, does changing
only the upstream command move the action and rollout toward its associated branch?  Each
command is copied unmodified from a real E70 trajectory, but a crossed state-command combination
is a deliberate counterfactual recombination outside observed joint support.  The construction
is interchange-style, not a formal causal-abstraction result
([Geiger et al., 2022](https://proceedings.mlr.press/v162/geiger22a.html)).

This differs from E70's S arm, which trains a separate controller while the environment couples
physical reset, reference, teacher route, and latent cursor.  P1 decouples physical-start source
from command/reference source inside one frozen controller.

### 5.2 Reset, nominal-runtime, and routing contract

The additive reset layer provides `state_start_steps` for physical initialization and
`command_start_steps` for motion IDs, time cursor, explicit motion observation, teacher route,
and SNMR latent lookup.  The shared zero-action warm-up advances both routes to the requested
observation frame.

The run fails closed unless all of the following hold:

- raw and normalized 90-D proprioception agree to `1e-6` across commands at a fixed physical
  start;
- the full command-independent Markov and policy state agrees to `1e-6`, including backend
  qpos/qvel/act/history/warm-start/control/applied-force state, rigid-body/contact state, and
  current/previous/processed action buffers.  MuJoCo requires complete integration state for
  reproducibility
  ([state API](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html);
  [reproducibility guidance](https://mujoco.readthedocs.io/en/3.1.2/computation/));
- the 154-D actor observation is reconstructed semantically from the named action, angular-
  velocity, DoF, 64-D motion-command, and orientation terms rather than accepted by width;
- command cursor and motion ID equal the registered command side, and the
  `[z_t,z_{t+0.1}]` lookup matches that cursor while remaining invariant to physical state side;
- evaluation is nominal and deterministic: no actor/other observation noise, domain or reset
  randomization, pushes, action delay, PD-gain randomization, torque RFI, initial-pose noise,
  terrain randomization, or adaptive timestep sampling;
- state and command frames remain inside their respective clip for the full horizon, and the
  warm-up causes no reset.

Reference-dependent bad-tracking termination is suppressed through the first 50 policy steps;
nonfinite state remains fail closed.  Reference termination is restored only after the
one-second primary samples are recorded, making later survival and completion descriptive.

Current status on 2026-08-14:

- the reset layer, evaluator, analyzer, bundle auditor, and their CPU regression tests exist;
- the evaluator has **not** been simulator-smoke-tested;
- the launcher, independent smoke certificate, and one-way DRAFT-to-child-PREREGISTERED
  transition are enforced and CPU-dry-checked; a production DRAFT is still pending;
- no E71 simulator report, gate, analysis, bundle certificate, or preregistration exists.

### 5.3 Evaluation order and fail-closed gate

1. Create a hash-complete `DRAFT` manifest and run one four-environment smoke report under its
   separate, non-confirmatory protocol.
2. Only after smoke proves callback order, full-state equality, zero MJWarp overflow, semantic
   observation/latent routing, nominal conditions, and 50 uncensored future steps, write a
   distinct immutable `PREREGISTERED` child that hash-binds the DRAFT, report, independent smoke
   certificate, B4 marker, capacity proof, owner, and date before any 276-cell result is opened.
3. Run all three frozen explicit students and independently certify their report set.
4. A pair is eligible only when, from **each** physical state side,
   `C(command A)<0<C(command B)` holds jointly in at least two of three explicit seeds.
5. Require at least 20 pairs spanning at least 6 of the frozen 12 temporal components.
   Otherwise stop with an invalid-assay result before SNMR.
6. Write pair/component IDs, thresholds, and explicit report hashes once to an immutable gate.
7. After a valid gate, run the three SNMR students, write the frozen analysis, and require the
   independent auditor to recompute the gate and analysis before bundle certification.

This gate is a goal-feasibility restriction, motivated by work on reachable goal relabeling and
state-conditioned subgoal feasibility
([Nair et al., 2020](https://proceedings.mlr.press/v100/nair20a.html);
[Ke et al., 2025](https://proceedings.mlr.press/v267/ke25a.html)).  Those precedents do not
validate this threshold or place crossed state-command combinations back on observed joint
support.

The 1-second horizon is deliberate.  It contains exactly 50 post-action samples; the secondary
0.5-second horizon contains 25.  Numerical diagonal reproduction against E70 is not a gate:
E71 removes E70 randomization and changes primary-horizon termination.  Cursor, semantic-
routing, nominal-condition, and complete-state audits are the replacement validity checks.

### 5.4 Metrics and uncertainty

With 58-D joint position/velocity goal `g`, pooled scale `sigma`, and `N ∈ {25,50}` future-only
samples:

```text
Q_k(r,N) = mean_{t=1..N,dim} ((g_r(t)-g_k(t))/sigma_dim)^2
Q_AB(N)  = mean_{t=1..N,dim} ((g_A(t)-g_B(t))/sigma_dim)^2
C(r,N)   = (Q_A(r,N)-Q_B(r,N)) / (Q_AB(N)+1e-8)
Δswap(x,N) = C(x,u_b,N) - C(x,u_a,N)
```

An exact A rollout has `C=-1`; an exact B rollout has `C=+1`.  Positive `Δswap` establishes a
directional shift but not absolute branch selection; selection requires
`C(command A)<0<C(command B)`.

The primary coordinate averages the future rather than reducing it to a terminal point;
official trajectory benchmarks provide precedent for reporting average-trajectory and
terminal/miss distances as complementary views
([nuScenes prediction metrics](https://www.nuscenes.org/prediction)).

The primary point is the equal mean over eligible pairs and three **fixed** controllers of the
two state-side effects at 50 samples.  The 10,000-replicate, seed-7104 primary interval
resamples eligible E70 temporal components with one common component-weight vector across all
controllers.  A crossed seed × temporal-component bootstrap is sensitivity-only.  Pair-wise
resampling nested within seed is prohibited because controller and temporal region are crossed
factors ([Owen and Eckles, 2012](https://arxiv.org/abs/1106.2125)).  With only three training
runs, the claim remains conditional on these controllers
([Agarwal et al., 2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)).

Secondary outcomes are `C` and `Δswap` over 25 samples, raw `Q_A,Q_B,Q_AB`, branch-choice rates,
first `z_cmd`/student/teacher actions, per-seed and per-state direction, and post-primary
survival/completion.  The stored actions also define a registered secondary margin:
`||a_student-a_teacher(opposite)||_2-||a_student-a_teacher(commanded)||_2`, with the same
fixed-controller temporal-component interval.  It measures specialist-teacher target alignment,
not content beyond teacher/clip identity.  No secondary endpoint rescues a failed primary result.

### 5.5 Registered interpretation branches

- **Explicit gate fails:** crossed-state assay invalid; make no SNMR inference.
- **Directional shift fails:** E70 remains a valid training-algorithm contrast, but
  target-directed use on a fixed controller is not established.
- **Directional shift passes:** the fixed-controller temporal-component interval has lower bound
  above zero and every seed × state-side mean effect is positive.  Say only that source-valid
  replacement shifts the rollout toward its associated branch.
- **Selection also passes:** aggregate `C(command A)<0<C(command B)` additionally holds
  separately from both physical state sides.  Say that the result meets the aggregate branch-
  selection sign criterion on the explicit-feasible two-walk support.

Do not require SNMR to match the explicit effect size.  Neither positive branch distinguishes
static identity from time-varying content.

## 6. Workstream P2 — identity, phase, and local-motion decomposition

**Priority:** second.  
**Type:** mostly evaluation-only on the same frozen SNMR policies.  
**Launch rule:** preregister after P1's evaluator passes synthetic and reset-equality tests, but
before inspecting any P2 rollout.

### Evaluation-only source interventions

Evaluate the unchanged target trajectory while changing the source latent window:

| Intervention | Construction | Question |
| --- | --- | --- |
| Intact | `[z_t, z_{t+0.1}]` | Frozen baseline |
| Current repeated | `[z_t, z_t]` | Is the 0.1 s preview necessary? |
| Correct clip, shifted phase | `[z_{t+δ}, z_{t+δ+0.1}]`, `δ ∈ {-0.5,-0.25,+0.25,+0.5}s` | Is time-varying trajectory state used? |
| Clip mean repeated | `[mean(z_clip), mean(z_clip)]` | Can a static identity proxy retain performance? |
| First-frame repeated | `[z_0, z_0]` | Can a valid clip-specific constant retain performance? |

Use the first second as the confirmatory horizon so all shifted commands remain within the
original ten-second support.  Keep the target reference and physical start unchanged.  Treat
the clip mean as a synthetic, outside-source-support diagnostic; do not give it the
interventional status of a source-valid command swap.

### Interpretation

- Smooth degradation with `|δ|` plus failure of static codes supports time-varying trajectory
  state beyond identity.
- Static codes retaining most of intact performance means the E70 advantage is largely clip
  identity.  This is publishable if stated directly.
- `[z_t,z_t]` matching intact means the claimed command content is current-state/identity, not
  evidence that the `+0.1 s` sample provides future intent.

### Optional retrained controls

Only if the evaluation-only decomposition remains ambiguous, train three seeds each under the
unchanged E70 recipe:

- **ID only:** a fixed random code per clip;
- **time + ID:** the existing shared time code plus a fixed clip code.

Time+ID is a memorization oracle for two known deterministic clips.  If it matches SNMR, the
latent behaves operationally like a learned lookup key.  If SNMR exceeds it under the bounded
trainer, claim only greater **algorithm-relative accessibility**, not greater raw information.

## 7. Workstream P3 — calibrated instrument and generality

These are valuable only after target specificity is resolved.

1. **Known-content calibration (existing E73 direction).** Prefer a common frozen explicit
   decoder with source-specific adapters.  Calibrate with nested PCA dimensions and an
   interpretable physical-information ladder.  This turns the assay into an instrument rather
   than a collection of separately trained policies.
2. **Noise titration (current E72 draft).** Run after P1/P2.  It can validate dose sensitivity,
   but its result cannot replace target specificity.  Amend its rationale so it is presented as
   a sensitivity curve, not the next missing causal rung.
3. **Held-out motion.** Mine multiple matched-present/divergent-future pairs across train and
   held-out clips, freeze the selector without student outcomes, and require a passing explicit
   capability gate.  This is the cleanest answer to the memorized-two-walk criticism.
4. **Second interface or robot (existing E74/T1 directions).** Use the unchanged assay only
   after its calibration and target-specific behavior are established on G1.

## 8. Manuscript redesign after results

### Recommended structure

1. **Introduction: why tracking success cannot identify command content**
2. **An evidence ladder for learned command interfaces**
3. **Exclusive counterfactual interface assay**
4. **Instrument validation and matched nulls**
5. **Case study: what the SNMR interface carries**
6. **Validity audits, scope, and reproducibility**
7. **Conclusion**

### Space budget

Move to the supplement:

- most of the two-teacher absorption result and Table II;
- the online-distillation loop figure;
- detailed probe numbers;
- extended defect chronology.

Keep in the main paper:

- the single-clip time result;
- evidence-ladder table;
- E70 five-arm result with CNM/NCR;
- all-seed destruction and temporal robustness in compact form;
- P1 command-swap result and one decomposition result;
- a concise validity-audit box.

### Main figure

Replace the current architecture-plus-bars figure with one visual question:

1. two overlaid humanoid poses at the matched present;
2. colored futures at `+0.5 s` and `+1.0 s`;
3. the exclusive `z_cmd` path;
4. the same physical start rolled out under source-valid command `a` versus `b`;
5. a compact CNM/command-swap forest plot.

The reader should be able to infer: **same state, different source-valid command, which future is
followed?**

### Claim language

Before P1 passes:

- use "exclusive interventional assay";
- say "clip-disambiguating trajectory information";
- state explicitly that static identity is not separated.

After the P1 directional-shift gate passes:

- say that changing only the source-valid command, with the controller and complete initial
  state fixed, caused a directional shift toward the associated branch on the explicit-feasible
  two-walk support;
- state that the crossed state-command recombination was outside observed joint support and
  that inference is conditional on the three evaluated controllers;
- do not say "selected" unless the stronger aggregate sign criterion passes separately from
  both physical state sides.

After both P1 gates pass, add only that the result also meets the registered aggregate branch-
selection sign criterion.  Do not upgrade either branch to semantics, held-out generalization,
or embodiment transfer.

Recommended title:

> **What Crosses the Boundary? A Counterfactual Assay for Humanoid Command Interfaces**

Alternative:

> **Beyond the Clock: Measuring Control-Usable Information in Humanoid Motion Interfaces**

## 9. Execution order and stopping rules

| Order | Deliverable | Cost | Continue when |
| --- | --- | ---: | --- |
| 0 | Finish or explicitly amend B4 and restore observable GPU availability | current queued work | `POSTPROCESS_COMPLETE` exists and at least 26,000 MiB free memory is observable |
| 1 | P0 patch prepared, not applied during freeze | <1 CPU day | generated values and all branches verify |
| 2 | Instantiate the already CPU-verified state machine's hash-complete production `DRAFT` | <1 CPU day after gates recover | manifest/auditor replay passes with no frozen hash drift |
| 3 | Run and audit the separate four-environment smoke, then make the one-way `PREREGISTERED` manifest transition | small simulator smoke | callback order, complete-state equality, semantic routing, nominal conditions, and 50 uncensored samples pass |
| 4 | Run all three explicit controllers and independent preflight certification | about 3 evals | at least 20 pairs across at least 6/12 temporal components pass the registered signs on both states |
| 5 | Freeze the explicit gate, run all three SNMR controllers, analyze, and audit the final bundle | about 3 evals | report the registered shift/selection branch without tuning |
| 6 | P2 phase/static battery | roughly 1--2 GPU hours | only if it can affect claim precision before the paper lock |
| 7 | Rewrite and main-figure replacement | 1--2 writing days | all numbers are machine-generated and PDF is compliant |
| 8 | ID/time+ID training | moderate | only if P2 does not resolve identity |
| 9 | E73 calibration, held-out motions, E74 second subject | high | post-submission program |

Global stop rules:

- stop if any frozen E70 hash changes;
- do not launch while the B4 marker is absent, the NVIDIA driver is unavailable, or the
  26,000-MiB condition cannot be demonstrated;
- stop if the full command-independent Markov/policy state differs across commands after reset,
  or if semantic observation, cursor, latent routing, nominal-condition, warm-up, or termination-
  isolation audits fail;
- do not require numerical diagonal reproduction of E70 under E71's changed nominal and
  primary-termination conditions;
- stop before SNMR evaluation if the explicit cross-start gate fails;
- never change pair inclusion, horizons, or thresholds after viewing SNMR results;
- admit reports only through the frozen manifest/state machine and require independent auditor
  recomputation before a final certificate;
- preserve every null outcome and select manuscript language from prewritten branches.

## 10. Minimum publishable versus strongest submission

### Minimum publishable revision

- apply P0;
- rewrite the introduction around non-identifiability;
- add the evidence ladder and CNM/NCR;
- accurately state that identity versus dynamic content is open;
- finish B4/B5 with no new scientific claim.

This is already a serious, defensible measurement paper.

### Recommended stronger submission

- minimum revision plus P1;
- replace the main figure with the same-state source-valid command swap;
- add at least the phase-shift/current-repeat subset of P2;
- move weaker upstream breadth material to the supplement.

This directly answers the strongest reviewer attack without broadening the paper into a
capability benchmark.

### Full benchmark program

- recommended submission plus fixed-decoder calibration, held-out motion pairs, and a second
  external interface;
- package the exclusivity contract, selector, matched nulls, analyzer, assay card, and
  preregistration template as the reusable artifact.

That program is a follow-on paper.  It should not delay the current submission unless the P1
reset contract cannot be made valid.
