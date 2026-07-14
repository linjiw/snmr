# SNMR Design Review, Structured Brainstorm, and Proposed Plan Revision

- **Review date:** 2026-07-14 (evening; written while E35 fight1 continuation was at ~4.7k/7k
  iterations)
- **Scope:** `NEURAL_RETARGETING_DESIGN.md`, `docs/EXPERIMENT_LOG.md` (through E35),
  `docs/NEURAL_RETARGETING_RESEARCH_sol.md` (the 2026-07-13 audit), `docs/PAPER_DRAFT.md`, and
  direct reads of run artifacts (`runs/gate1_g1/`, `runs/skate_structure/`,
  `runs/wbt_horizon_calibration/`, `runs/ablations/SUMMARY.md`).
- **Method:** the review applies the structured research-ideation frameworks
  (problem/solution-first, abstraction ladder, tension hunting, cross-pollination, what-changed,
  failure analysis, simplicity test, stakeholder rotation, composition/decomposition,
  two-sentence test), then converges to a ranked, preregisterable proposal set.
- **Status precedence:** `NEURAL_RETARGETING_RESEARCH_sol.md` remains the frozen decision record.
  This document is the **proposed next revision** — nothing here overrides a frozen protocol until
  explicitly adopted.

---

## 1. Where the project actually is (independent re-read, 2026-07-14)

### 1.1 The one-paragraph state

SNMR's core amortization result is solid and honest: one 1.6M-parameter network distills a GMR IK
teacher across five trained humanoids (3.2–6.7 cm held-out MPJPE, zero limit violations), with a
quantitatively characterized shared-but-not-invariant latent (CKA 0.91, retrieval R@1 0.75, MLP
attacker 0.91). Both preregistered contact routes have now **failed as registered**: the
factorized soft objectives (Gate 1: best arm C4 reaches 0.264 m/s vs the ≤0.08 endpoint, 0/3
seeds) and the deployable windowed C6 projection (0.099 m/s + MPJPE guard fail). Crucially, the
**same frozen C6 solver passes every guard under a teacher-height oracle mask (0.0056 m/s,
MPJPE 3.82 cm, jerk within guard)** — so the contact problem has been localized from "objective
design" to **contact-mask precision at inference**. On the tracking side, the 18-run WBT training
pilot passed its artifact contract with modest mixed effects, but the 5,400-rollout independent
evaluation returned an **undertrained control** verdict (0% ten-second completion both arms), and
the in-flight E35 horizon calibration shows the failure was budget, not source: GMR walk1
completion climbs 3% → 48% → 88% across 2k/4k/8k iterations.

### 1.2 Evidence ledger, restated per gate

| Gate | Question | Verdict as of tonight | Load-bearing artifact |
|---|---|---|---|
| 0 | Provenance/measurement | **Complete** (throughput still blocked on idle-GPU timing) | manifests, dual-run evaluator repro |
| 1 | Soft contact objectives | **Failed endpoint, closed.** C4 = robust regularizer (−62% speed, MPJPE 3.09 cm), not a fix | E30–E32, `runs/gate1_g1/` |
| 1-C6 | Constrained projection | Source mask **fails** (0.099 m/s, MPJPE 4.28); teacher oracle **passes all guards** (0.0056 m/s) | E29, `runs/skate_structure/windowed_c6_*` |
| 2A/B | WBT plumbing + pilot | Passed; descriptive only (joint-err favorable 8/9, ref-err unfavorable 1/9) | E27/E33 |
| 2 eval | Independent rollouts | **Undertrained control** — 0/2700 completions both arms at 1k iterations | E34 |
| 2 cal | Horizon calibration (E35) | **In flight.** walk1: 88% @8k (passes); dance2: 10% @8k (**fails the ≥25% floor**); fight1 training | `runs/wbt_horizon_calibration/reports/` |
| 3 | Sharing cost | Diagnosed: no pervasive gradient conflict; S2 width / S3 adapters are next | E28 |
| 4 | Generalization | LORO fails 5.2×; `zr_decode_prob` wired, untested | E06/E19/E21 |
| 5 | Representation | "Aligned, not invariant"; controls specified, not run | E04/E11/E23 |
| 6 | Temporal modeling | Current ablation invalid (no positional encoding); framewise is default | E08, audit P1 |

### 1.3 Two new facts found in this review (not yet in any doc)

**(a) E35 will almost certainly fail its promotion rule, and the failure mode is per-clip
heterogeneity, not global budget.** The frozen rule requires pooled completion ≥50% **and every
clip ≥25%** at the promoted horizon. Live reports: walk1 = 0.03/0.48/0.88 and dance2 =
0.00/0.02/0.10 at 2k/4k/8k. Even a perfect fight1 leaves dance2 at 10% ≪ 25%, so the earliest
compliant horizon does not exist within 8k, and the protocol's own branch fires: *stop before
extending the 18-policy matrix and reassess the default 30k schedule/config.* Section 4.2
proposes that reassessment now, **before** any SNMR checkpoint is evaluated at long horizons —
this is the last moment endpoint amendments can be made blind to source identity.

**(b) A learned contact mask already exists and is decent, for free.** The Gate-1 C1 (BCE-only)
checkpoint's contact head, evaluated on held-out windows, scores **F1 0.87 / precision 0.89 /
recall 0.87 against the source-contact stance mask** (`runs/gate1_g1/screen/c1_bce_seed0/
benchmark.json`). C1 was retained as a *negative control for kinematics* — correctly, since
classification alone moved no motion metric — but its head is exactly the "trained contact head"
lever that E26c-3/E29 identified as the bottleneck for the projection path. Two caveats found in
the same artifact: the head's agreement with the *teacher-height* mask is poor (F1 0.18) largely
because that mask's evaluation prevalence is only 0.089 with ~34 samples/window — see §1.4.

### 1.4 A measurement-power concern to fix before the next contact decision

The primary Gate-1 endpoint (teacher-height stance speed) is computed under a mask that covers
only ~9% of evaluation frames (~34 samples per 192-frame window; `fight1_subject3` has zero
support in every arm; the Gate-0 bootstrap CI on this metric spans `[0.11, 0.56]` m/s across
clips). The source-contact mask has ~10× the support (~257 samples/window, prevalence 0.67) and
is equally non-circular with respect to decoded velocity (it is built from *human* height+speed).
The teacher-height mask is the stricter, more contact-faithful definition; but a primary endpoint
this sparse is fragile. **Proposal: make teacher-height and source-contact coprimary in the next
preregistration**, with the pass rule requiring both. Also audit one definitional wrinkle: both
trainer and benchmark normalize foot height by the *window-local* minimum (`ground_z=None`),
so 64-frame training labels and 192-frame evaluation labels are not the same distribution —
cheap to quantify, and it partially explains the C1 head's low teacher-height agreement.

---

## 2. Framework-guided review of the research position

### 2.1 Problem-first vs. solution-first (F1)

SNMR is honestly **solution-first** (a shared-latent neural retargeter seeking its problems), and
the audit already forced the right discipline: each claimed benefit was converted into a testable
problem statement. Re-verifying the two genuine problems the solution must address:

1. *Throughput/config cost:* real — GMR is 35–170 FPS CPU-sequential with 30 hand-tuned configs;
   any group retargeting hours of motion × robots pays this. SNMR at ~2k FPS batched is a real
   answer **once controlled timing exists** (still Gate-0-blocked on an idle GPU).
2. *Downstream tracking data quality:* real (the GMR paper's own central result) — but SNMR's
   evidence remains "not detectably worse," and the decisive experiment is stalled on control
   calibration, not on SNMR.

The third originally-claimed problem — cross-embodiment unification — has fractured under
testing into one supported piece (aligned representation) and two failed pieces (positive
transfer, zero-shot decoding). The plan below stops treating those as one goal.

### 2.2 Tension hunting (F3): the core tension is now precisely characterized

The project's defining trade-off, with three experiments' worth of measurement behind it:

> **Fidelity-to-teacher ↔ physical consistency.** Distillation dominates every soft contact
> penalty at weights that don't hurt MPJPE (E10a, E32); the student's ~3 cm errors differentiate
> into 0.3–0.7 m/s stance velocity (E24: smooth, chain-correlated xy oscillation, not jitter).

The Gate-1 + E29 results resolve the *artifact vs. fundamental* question cleanly: the tension is
**an artifact of trying to buy physicality with training-time penalties**, and it dissolves under
a different mechanism — exact constraint satisfaction at inference (the oracle projection passes
everything at negligible fidelity cost). This is the literature-consensus hybrid (Villegas'21
contact-aware refinement, UnderPressure cleanup, X-Morph's correction stage): **amortize the
solve, project the constraints.** The remaining gap is *which frames to constrain* — a
classification problem, on which the free C1 head already scores 0.87 F1 against one valid mask.

A second tension worth naming: **preregistration rigor ↔ iteration speed.** The frozen-protocol
discipline has caught real errors (evaluator zero-bias, analyzer all-finite bug, C2's failed
recalibration) and is the reason the negative results are publishable. Its cost is that both
contact routes ran to completion before the mask-precision diagnosis emerged. The plan below
keeps the discipline but adds explicitly *diagnostic* (non-confirmatory) pilots before each
frozen study, which the Gate-1 calibration stage already models well.

### 2.3 What changed (F5): four facts that reshape the plan

1. **WBT runs locally** (E20→E27). Physics-in-the-loop supervision (NMR/ReActor-style) moved
   from "needs external IsaacSim" to "one more local GPU job." This makes the audit's
   physics-repaired branch *executable*, not aspirational.
2. **The bottleneck moved.** Before E29, contact looked like an objective-design problem; the
   oracle result relocates it to mask precision at inference. Every plan written before E29
   overweights training-time levers.
3. **A trained mask exists as a byproduct** (§1.3b) — the highest-leverage next experiment costs
   roughly one evaluation run, not a retrain.
4. **The completion endpoint has a measured budget curve** (E35 partial): 1k iterations →
   ~1 s survival; 8k → 88% on walk but 10% on dance. Any confirmatory WBT design that ignores
   per-clip difficulty will fail its own floors.

### 2.4 Failure analysis (F6): root causes, one line each

| Failure | Root cause (measured) | Wrong fix (falsified) | Right fix (evidence) |
|---|---|---|---|
| Foot skate | Smooth chain-correlated stance xy oscillation from ~3 cm amortization error (E24) | Penalties (E10a/E32), low-pass (E22), position lock (E24), correction smoothing (E26 σ>0) | Windowed projection with a **precise mask** (E29 oracle) |
| C6 deployable | Dilated source mask over-covers swing → root-correction saturation → MPJPE fail (E29) | Loosening bounds (MPJPE already failing) | Better mask (learned / decoded-height) |
| WBT verdict | Control undertrained ~30× vs default schedule (E34/E35) | Blaming either retargeting source | Budget calibration + per-clip horizons (§4.2) |
| LORO zero-shot | Conditioning = lossy bag of local features; 5 robots ≪ diversity needed (audit P1) | More training on same 5 | Few-shot adapters first; augmentation later (Gate 4) |
| Sharing cost | *Not* pervasive gradient conflict (E28: mean cosines +0.12–0.30) | PCGrad/GradNorm first | S2 width, S3 adapters at matched exposure |

### 2.5 Simplicity test (F7): three places to simplify

1. **The mask may not need learning at all.** E24 showed decoded foot *heights* are correct
   (mean z matches teacher to 3 decimals); only the *velocity*-gated detector under-fires. A
   height-hysteresis mask **on the decoded feet** has never been tested as the projection driver
   (E26's "decoded-mask" arm used the speed-gated detector). This is a zero-training arm that
   should sit beside the learned head in the mask study.
2. **Framewise stays the default architecture** (E08 + audit P1). No order-aware temporal work
   until it blocks a claim — Gate 6 stays deprioritized.
3. **The paper story simplifies to "amortize + project."** One network for speed and sharing; a
   windowed projection for exactness; the mask is the interface between them. This is both the
   honest current state and the literature-standard architecture.

### 2.6 Cross-pollination (F4) and composition (F9) — kept short

- *Animation contact cleanup* (UnderPressure, Villegas'21): labels from a clean signal + an
  optimization pass — exactly the C6-with-learned-mask design; cite as mechanism precedent.
- *Amortized inference + test-time refinement* (amortized IK literature): frames the paper's
  positioning — SNMR is an amortizer whose residual constraint error is projected out.
- *Composition worth registering:* **C4-regularized checkpoint + learned mask + C6 projection.**
  C4 halves the raw stance speed (0.264 vs 0.701 m/s) and *improves* MPJPE (3.09 cm), so the
  projection starts 2.7× closer to the target with more MPJPE headroom. Register as a separate
  arm, not a bundle.
- *Composition to defer:* predicted-contact flags exported into WBT reward shaping, and the
  latent-as-command interface (N10/C2). Both are real openings; neither is on the critical path.

### 2.7 Stakeholder rotation (F8), abbreviated to the two that bind

- **The RL practitioner** needs references that train policies as well as GMR's, delivered fast.
  Verdict: blocked on Gate 2 calibration, not on more kinematic polish. Don't let contact work
  starve the WBT queue.
- **The reviewer** will ask (i) is the projection just IK smuggled back in? — answer with
  runtime numbers and the throughput story (projection is windowed, batched, and optional per
  deployment); (ii) is anything better than the teacher? — answer with throughput, sharing, the
  representation analysis, and (if E34's descriptive joint-RMSE signal survives a calibrated
  rerun) tracking non-inferiority with a *trained* control.

---

## 3. Diverge → converge: candidate directions and ranking

Candidates generated across frameworks (12 raw), then filtered by the explain-it test,
problem-first check, simplicity, stakeholder benefit, and feasibility on one A10G:

| # | Candidate | Source frame | Verdict |
|---|---|---|---|
| 1 | **Mask-precision study driving the frozen C6 solver** (learned head, decoded-height hysteresis, C4-init) | F6/F7/F9 | **Rank 1 — do now** |
| 2 | **Gate-2 horizon reassessment + blind endpoint amendment** | F5/F8 | **Rank 2 — decide this week** |
| 3 | Physics-repaired supervision: repair student outputs in sim (PD replay / short-horizon tracker), retrain on repaired targets (NMR direction) | F4/F5 | Rank 3 — conditional on #1 failing |
| 4 | Gate-3 S2/S3 width-vs-adapters at matched exposure | audit | Rank 4 — next GPU idle block |
| 5 | Paper reframe to "amortize + project" with claims-ledger update | F7/F10 | Rank 5 — cheap, after #1 reads out |
| 6 | Retargeting/tracking co-optimization (ReActor-style bilevel) | F4 | Defer: expensive; only if #1 and #3 both fail |
| 7 | `zr_decode_prob` G-source study (Gate 4) | audit | Defer: after contact/tracking |
| 8 | Embodiment augmentation + LORO revisit (Gate 4) | audit | Defer |
| 9 | Representation controls + contrastive/part-wise latents (Gate 5) | audit | Defer: paper-depth, CPU-friendly filler |
| 10 | Data-scale ablation within LAFAN1 (no prior work ablates data scale) | F5/lit gap | Defer: cheap paper depth, one GPU day |
| 11 | Contact flags → WBT reward shaping | F9 | Parked: needs Gate 2 healthy first |
| 12 | Latent-as-command tracking interface (N10) | design §N10 | Parked: Phase-4 scope |

Kill notes: #6 fails feasibility *now* (bilevel RL on one shared A10G while Gate 2 is unresolved);
#11/#12 fail the problem-first check until a calibrated WBT baseline exists to compare against.

---

## 4. The proposal (refined winners)

### 4.1 Rank 1 — Gate 1b: mask-precision study (preregister, then run)

**Two-sentence pitch.** SNMR's contact failure is now localized to inference-time stance
classification: the frozen windowed projection passes every preregistered guard under an oracle
mask and fails only under the heuristic source mask. We test whether a deployable mask — a
trained contact head (already at 0.87 F1) or height-hysteresis on the decoded feet (heights
already correct) — closes the oracle gap, turning the existing solver into a passing pipeline
with zero new training-time machinery.

**Why this does not violate the Gate-1 stop rule.** The stop rule forbids further *soft-penalty
sweeps* and *projection-bound tuning*. This study changes neither weights nor bounds: it varies
only the mask input to the already-frozen solver, which is exactly the "learned/physics-repaired
contact labels" branch the audit itself prioritizes (E29 decision; E26c-3 conclusion).

**Design (frozen items).** Same 42-window/7-clip evaluator, same committed C6 solver and
hyperparameters as E29, G1 only. Arms, each = one projection run + one benchmark:

| Arm | Mask source | New training? |
|---|---|---|
| M0 | Source-contact heuristic (E29 baseline, fails) | no |
| M1 | Height-hysteresis on **decoded** feet (enter 0.03 / exit 0.05, window-local ground) | no |
| M2 | C1 checkpoint's contact head, thresholded at 0.5 (+ hysteresis on probabilities) | no |
| M3 | Contact head retrained on **teacher-height** labels atop the **C4 seed-0 checkpoint** (BCE 0.25, all other weights frozen at C4's) | one 50k G1 run (~1 h) |
| M4 | Teacher-height oracle (upper bound, E29 result) | no |

Pre-study audit (one CPU script, do first): quantify the window-local vs clip-local ground
normalization gap in teacher-height labels (§1.4), and report each mask's precision/recall
against the oracle mask *before* any projection runs, so mask quality and projection outcome can
be attributed separately.

**Endpoints (coprimary, per §1.4):** teacher-height stance speed ≤0.08 m/s **and** source-contact
stance speed ≤0.10 m/s; guards unchanged from E29 (MPJPE ≤ +0.5 cm relative and ≤4.0 cm absolute
reported separately; jerk ≤1.2×; zero limit violations; penetration bounds). Promotion: the
simplest passing arm (M1 < M2 < M3) goes to a three-seed replication *of its mask-generating
model* only if the mask depends on training (M3); M1/M2 need no replication beyond the existing
seeds since the solver is deterministic.

**Decision rules.**
- Any deployable arm passes → contact is closed as "predict + project"; validate on the other
  four robots; update claims ledger C5.
- All deployable arms fail but mask-vs-oracle precision explains the residual (high mask
  precision still fails) → the problem is *not* classification; escalate to Rank 3
  (physics-repaired supervision) and do not iterate further on masks.
- Mask precision is the measured blocker (low precision → failure) → one registered iteration on
  the mask *model* (better labels, e.g. clip-local ground), not on the solver.

**Two-week pilot content:** the audit script + M0/M1/M2/M4 (all zero-training, ~a day of
compute), M3 behind them. Strongest objection: *"a projection stage concedes that the network
didn't learn contact."* Response: correct, and measured — that concession is the paper's finding
(three falsified training-time mechanisms with matched controls), and the hybrid is the
literature-standard deployment (Villegas'21; X-Morph; UnderPressure). The network's job is
speed, sharing, and 95% of the pose; the projection's job is the last 5 cm/s.

### 4.2 Rank 2 — Gate 2 reassessment: act on E35 before SNMR sees long horizons

E35 will formally fail its promotion rule on dance2 (§1.3a). The frozen branch says: stop, do not
extend the 18-policy matrix, reassess the 30k default. The reassessment should be written **now,
while only GMR has been evaluated at long horizons** — after SNMR long-horizon numbers exist, any
endpoint amendment is post hoc. Proposed, in order:

1. **Finish E35 as frozen** (fight1 completes tonight); record the formal FAIL and the three
   calibration curves as the negative-control budget study.
2. **Preregister Gate-2 v2 with per-clip horizons and amended endpoints, blind to SNMR:**
   - Per-clip training budget = the E35-measured horizon reaching ≥50% GMR completion, capped at
     30k (walk1 → 8k; dance2/fight1 → one continuation each of the *existing* E35 run to 16k/30k
     to locate their curves before committing the matrix — two more GPU jobs, not eighteen).
   - Endpoints: keep the two coprimary margins (completion −5 pp; joint RMSE +10%) but add a
     preregistered **survival-time coprimary** (relative margin, e.g. −10%) so the study remains
     interpretable if a hard clip plateaus below 50% completion even at 30k — dance2's curve
     (0→2→10% over 2k/4k/8k) makes this a live risk.
   - Assay floor per clip, not pooled: a clip enters the confirmatory set only if its GMR control
     passes ≥50% completion at its assigned budget; clips that never pass are reported as
     excluded-with-curve, not silently dropped.
3. **Only then extend the matched matrix.** Cost control: at ~2 h per 8k-iteration run on the
   A10G, 18 runs × walk-class budgets are feasible; 18 × 30k are not (~10 days GPU-solid). The
   per-clip budget rule is what keeps the confirmatory study runnable on this box; if dance2
   needs 30k, run it as a 2-source × 3-seed single-clip add-on rather than blocking the rest.
4. **Keep the E34 descriptive signals in view but unweighted:** SNMR's joint-RMSE CI
   `[−5.43%, +0.26%]` at 1k iterations *hints* the retargeting sources are close; only the
   calibrated rerun can say so.

### 4.3 Rank 3 — physics-repaired supervision (conditional; specify now, spend later)

If Gate 1b fails with mask precision exonerated, the audit's remaining branch is
physics-in-the-loop. The locally-executable instantiation (new since E27): roll the current best
policy (or PD replay for a v0) on SNMR references in MuJoCo/Warp, harvest the *simulated* foot
trajectories during detected stance, and retrain the retargeter with those as the supervision
target for the feet (NMR's repair-then-retrain, scoped to the contact subsystem). Preconditions
before any spend: Gate 2 v2 has produced at least one ≥50%-completion policy (the repairer must
be competent), and a one-clip pilot shows repaired targets actually differ from teacher targets
where the student skates. Do not start this while Rank 1 is unresolved — it is strictly more
expensive and its motivation partially evaporates if projection passes.

### 4.4 Ranks 4–5 — scheduled, not urgent

- **Gate 3 S2/S3** (width vs adapters, matched exposure, 3 seeds): next idle GPU block after
  Gate 1b and the Gate-2 continuations; target ≤0.5 cm sharing gap without >2× parameters.
  E28 already rules out PCGrad/GradNorm as first-line.
- **Paper reframe:** adopt the "amortize + project" story; move C5 from "endpoint met with
  trade-off open" (current draft overstates — that was the framewise blend-2 arm that fails the
  jerk guard) to whatever Gate 1b returns; add the Gate-1 negative-result table and the E35
  budget curves as contributions (matched-control negative results are this project's
  differentiator — AdaMorph publishes no baselines or ablations at all).
- **Cheap paper depth when GPU is busy (CPU-friendly):** Gate-5 measurement controls
  (permutation/time-shift CKA and retrieval controls), the LAFAN1 data-scale ablation
  (no published retargeting work ablates data scale), and controlled throughput timing the next
  time the A10G is idle (the last open Gate-0 item).

### 4.5 Explicitly deferred (unchanged from the audit, re-affirmed)

Gate 4 (`zr_decode_prob`, embodiment augmentation, LORO breadth), Gate 5 interventions
(contrastive/part-wise/adversarial), Gate 6 temporal study, AMASS scale-up, T1 WBT port,
latent-as-command (N10), ReActor-style bilevel. None blocks the two decisive claims.

---

## 5. Proposed goal restatement and claim deltas

**Revised program goal (two sentences, F10):** Humanoid motion retargeting today forces a choice
between slow, hand-configured per-robot optimization and fast neural prediction that violates
contact; this matters because retargeting quality and throughput jointly gate whole-body-tracking
training at scale. SNMR resolves the tension by amortizing the optimization teacher into one
shared multi-embodiment network and projecting the small residual constraint error at inference,
which works because the network's remaining error is precisely localized (stance-frame
classification), measured, and correctable by a windowed constrained solve.

**Claims-ledger deltas vs. the audit version:**

| Claim | Audit verdict | Proposed update |
|---|---|---|
| Contact-consistent decoding | Not supported | Pending Gate 1b; if a deployable mask passes → "contact-consistent via predict+project, with matched negative results for training-time objectives" |
| Benefits motion tracking | Not tested | "Not testable at pilot budget (measured); calibrated study preregistered" — cite E34/E35 curves |
| Positive transfer | Not supported | unchanged; add E28 (conflict exonerated) to the mechanism story |
| Temporal transformer | Unsupported | unchanged; framewise default stated in method |
| Throughput | Measurement incomplete | unchanged until idle-GPU timing (schedule with §4.4) |

**Additions to the no-go list:**

14. Do not evaluate any SNMR checkpoint at horizons >1k iterations before the Gate-2 v2
    endpoints are frozen (preserves the blind-amendment window).
15. Do not tune the C6 solver (bounds, weights, iterations) inside Gate 1b; only the mask varies.
16. Do not promote a mask arm on the teacher-height endpoint alone; both coprimary masks must
    pass (measurement-power rule, §1.4).

---

## 6. Execution order and budget (one A10G, shared)

| Order | Work | Compute | Blocking? |
|---:|---|---|---|
| 0 | E35 fight1 finishes + formal E35 verdict | in flight | — |
| 1 | Gate 1b audit script + M0/M1/M2/M4 (zero-training arms) | ~1 GPU-day mixed | contact claim |
| 2 | Gate-2 v2 preregistration (writing, blind window) | none | tracking claim |
| 3 | dance2/fight1 continuations to 16k/30k (curve location) | ~2–6 GPU-days background | tracking claim |
| 4 | Gate 1b M3 (C4+BCE retrain) if M1/M2 fail | ~1 h + eval | contact claim |
| 5 | Gate-2 v2 matched matrix at per-clip budgets | dominant cost — bounded by budget rule | tracking claim |
| 6 | Gate 3 S2/S3; throughput timing; CPU paper-depth items | fill idle | paper depth |

The two decisive claims (contact, tracking) have independent critical paths and share no
artifacts; interleave them by GPU availability exactly as the audit's parallelism rule intends.

---

## 7. Review verdicts on the three source documents

- **`NEURAL_RETARGETING_DESIGN.md`** — historically valuable and now correctly demoted to a
  design record. Its status banner is accurate. One stale item: §0.4b/N6's "primary fix"
  narrative (training-time contact head) is superseded by Gate 1's negative result; the banner
  covers this, but a one-line pointer at §N6 to Gate 1b (this doc) would prevent misreads.
- **`docs/EXPERIMENT_LOG.md`** — exemplary. Append-only with correction history (E04, E10a, E26b
  evaluator bug) is the project's strongest methodological asset. Gap: E35's partial results and
  formal verdict need an entry once fight1 lands; suggest also logging the C1 contact-head F1
  numbers (§1.3b) since they motivate Gate 1b.
- **`docs/NEURAL_RETARGETING_RESEARCH_sol.md`** — the audit's structure (gates, frozen protocols,
  stop rules) demonstrably worked: it caught the undertrained control before a false verdict and
  killed two dead-end mechanism families with attribution. Two amendments proposed: the
  coprimary-mask rule (§1.4) and the Gate-2 v2 per-clip-budget design (§4.2). Both are additive;
  no frozen decision is reopened.
- **`docs/PAPER_DRAFT.md`** — claim C5's "[ENDPOINT MET]" tag is ahead of the evidence (the
  passing framewise arm fails the jerk guard; the passing-jerk arm fails speed; the full C6
  passes only under the oracle). Align C5 with the Gate 1b outcome before any external sharing.

---

## Appendix: raw candidate list retained per brainstorming protocol

All twelve diverge-phase candidates are kept in §3's table, including killed ones — the parked
items (#11 contact-flags-as-reward, #12 latent-command) are the most likely to recombine with a
healthy Gate-2 pipeline in a follow-up paper.
