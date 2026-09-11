# MINIMAL METHOD: Bounded Correction → Intent Filter → Distillation (CVD)

**Hypothesis under test:** RobotSpec-conditioned, intent-preserving retargeting corrections learned from closed-loop tracking experience transfer to an unseen humanoid.

**One-line method:** freeze the existing proposal, search a bounded C² spline correction on training robots against a frozen per-robot tracker, reject any candidate that fails a human-side intent filter *before* it is ever scored for execution, then distill only accepted corrections into a token-conditioned residual head that adds zero parameters for a new robot.

---

## 1. Correction parameterization η

**η is a clamped cubic B-spline coefficient field over one bounded failure interval, additive in the robot's configuration space.**

**Channels (DoF-agnostic, addressed by semantic role, not index):**

| block | channels | bound (L∞ on the realized trajectory) |
|---|---|---|
| root translation Δxyz | 3 | **0.04 m** |
| root yaw Δψ | 1 | **0.12 rad** |
| leg-chain joint residual Δq | one per joint on the two root→foot chains (**12 on G1 29-DoF**) | **0.35 rad** |

The leg set is resolved at runtime by walking `snmr/mjcf_variants.py:545 _path_to_root(spec, endpoint)` from each foot link named in the `SemanticManifest` (`/home/robotixx/snmr/snmr/robot_spec.py:179`). Roll and pitch of the root are **excluded** (they are a dynamics claim, not a kinematic correction). Every joint not on a leg chain — arms, waist, neck — is **excluded by construction**.

**Knots:** uniform spacing Δ = 8 frames = **0.16 s** at 50 Hz, clamped cubic, first two and last two control points pinned to zero. Interval length capped at **192 frames** (the repo's existing window, `scripts/eval_footlock.py --window 192`); longer failures split at that boundary.

**Dimensionality:** free coefficients per channel = ⌈L/8⌉ − 1. For a typical 2 s interval (L=100): 12 per channel × 16 channels = **192**. Worst case (L=192): 23 × 16 = **368**. Hard cap dim(η) ≤ 384 — tractable for CEM/MPPI with 512 samples × 8 iterations.

**Bound justification, from the repo's own numbers:**
- `0.04 / 0.12 / 0.35` are the *exact* defaults at `/home/robotixx/snmr/scripts/eval_footlock.py:191-193`, and they are the configuration under which `windowed_contact_projection` produced the only **all-guards-pass** result in the repository: stance speed 0.502 → 0.0056 m/s, MPJPE 3.66 → 3.82 cm, jerk 693 → 697 (teacher-height oracle mask). So the box is *known sufficient* (a ~90× skate reduction fits inside it) and *known non-destructive* (+0.16 cm MPJPE, +0.6% jerk). Do not invent new bounds; reuse these and cite that row.
- Knot spacing 0.16 s bandlimits η to ≈3 Hz. Justified two-sidedly: E22 shows the residual foot error is **smooth systematic error, not jitter** (σ=1 Gaussian low-pass moves 0.093 → 0.081 m/s), so no high-frequency content is needed; but the DLS correction has **within-stance structure** (smoothing the correction at σ=1 re-admits skate, 0.05 → 0.19 m/s), so the band must still reach stance-cadence harmonics. 0.16 s sits between those two facts.
- Jerk guard ≤ **1.2×** baseline (existing repo guard). It is not vacuous: DLS at `blend=2` fails it at 1.35×.

**What η structurally cannot do (this is the point):**
- **No time channel** → cannot slow down, cannot time-warp. Not a penalty; not representable.
- **Zero-clamped at interval endpoints, no global multiplicative parameter** → cannot globally shrink amplitude; every correction is locally supported and C²-continuous with the uncorrected trajectory.
- **Upper body excluded** → the four upper-body semantic anchors (torso, head, L/R hand) move only through the root block, which is itself bounded at 0.04 m / 0.12 rad.

Two of the three named failure modes (shrink, slow down) are removed by the parameterization. Only "delete a hard transition" survives to be caught by the constraint set.

---

## 2. Intent-preservation constraint set D_intent ≤ ε

Evaluated on the **human-side** ruler `/home/robotixx/snmr/snmr/semantic_metrics.py:732 evaluate_semantic_retargeting` — which scores the candidate against the source human motion and **never** against a teacher — over the seven frozen anchors (pelvis, torso, head, L/R hand, L/R foot).

**Block A — absolute envelope (already frozen in code, hash-bound).**
`SemanticThresholds` at `snmr/semantic_metrics.py:266-279`: amplitude ratio ∈ [0.50, 1.50], energy ∈ [0.25, 4.00], jitter ∈ [0.25, 4.00], with a `__post_init__` assertion that every band brackets 1.0 and a `sha256` property. All 7 anchors × 3 statistics must pass.

**Block B — non-regression against η = 0 (the operative constraint).** Block A alone is far too loose: it permits halving the motion. So bind the correction to the *proposal*, per anchor *a*, statistic *s* ∈ {amplitude, energy, jitter}, ratio *r*:

1. `log r_s^η(a) − log r_s^0(a) ≥ −log(1.05)` — never lose more than 5% of the uncorrected statistic (one-sided; blocks shrinkage and slow-down).
2. `|log r_s^η(a)| ≤ |log r_s^0(a)| + log(1.05)` — never end up more than 5% further from the human than the proposal already was.

Growth *toward* the human is unbounded (that is an improvement). ε = log(1.05) = **0.04879** for all three statistics. Amplitude (`_per_anchor_amplitude`, RMS of mean-centered position) is the shrink detector; energy (`_per_anchor_energy`, mean squared velocity) is the slow-down detector.

**Block C — transition preservation (new code, ~60 lines).** The Block A/B statistics are window means; a single deleted foot-strike or direction reversal is invisible to them. Define events **on the human source**:
- event set = `snmr/verification.py:341 localize_threshold_failures` applied to each human anchor's acceleration magnitude, `threshold` = the 95th percentile of *that clip's own human* acceleration, `comparison="greater"`, `min_frames=2`, `merge_gap_frames=2`.
- per event: peak robot anchor acceleration in a ±8-frame neighborhood must satisfy `P^η / P^0 ≥ 0.95`, and the argmax time shift must be ≤ 8 frames (one knot — η cannot represent a finer shift, so a larger one is a bug).

**Block D — zero-time-warp audit (assertion).** Cross-correlation argmax lag between η and 0 anchor traces must be exactly 0 for all seven anchors. Structurally guaranteed; the assertion catches implementation error.

**How every threshold is set without touching validation data:**
1. Block A: already frozen in the repository with a content hash, predating this experiment (it is the A1 T5 gate).
2. Block B/C 5%: imported verbatim from the advisor's preregistered G4 endpoint, `docs/ADVISOR_GUIDANCE_2026-08-29_MORPHORETARGET.md:1289` ("semantic loss < ~5%"), written 2026-08-29.
3. Block C event definition: a percentile of the **human source clip**, computed with no reference to any model output or score.
4. **Discipline rule (adopt verbatim from GenTrack, arXiv:2608.01410):** D_intent never enters the search objective and never enters the distillation loss. It is a hard filter. Report every intent metric as held out.

**Hard prerequisite.** D_intent requires a `SemanticManifest` + `SemanticCorrespondence` for each robot. It exists **only for G1** (`snmr/g1_semantics.py`; `snmr/robot_spec.py:179` calls it "the bounded manual annotation allowed for a new robot"). Materialized G1 limb-length variants inherit it: link names are preserved so `g1_robot_fk_trajectory`'s manifest-equality check at `snmr/g1_semantics.py:166` passes, and `g1_lafan1_semantic_correspondence` already exposes the anchor points as arguments with a docstring stating the A2 policy ("a materialized variant must transform these points by the same realized body/geometry scales"). A **real** unseen robot (T1/N1/PM01) has no manifest, and D_intent cannot be evaluated there until seven anchor points are hand-measured.

---

## 3. Acceptance rule (decision procedure)

For each (robot *r*, clip *c*, failure interval *I*, candidate η):

**Stage 0 — admissibility (closed form, free).** η inside the box; C² by construction; zero at both endpoints; resulting q inside the robot's joint limits (`limit_violation_fraction == 0`, already measured by `scripts/benchmark.py`); no increase in penetration fraction.

**Stage 1 — intent feasibility (CPU, no simulator).** Blocks A ∧ B ∧ C ∧ D. **Fail ⇒ reject immediately, never scored for execution.** Ordering is the whole design: a candidate that violates intent never produces a reward number, so intent cannot be traded.

**Stage 2 — execution screen (CEM/MPPI inner objective, thousands of evals).** Stage-1-feasible candidates only; infeasible get −∞. Objective = open-loop PD replay survival (`/home/robotixx/snmr/scripts/trackability_proxy.py`, MuJoCo, no training, works on any MJCF) plus the existing kinematic guards: stance foot speed, penetration fraction, jerk ratio ≤ 1.2×, MPJPE-to-proposal. This is a *screen*, not a decision — E18 showed its SNMR-vs-GMR deltas sit inside between-window spread (0.87 s vs 0.82 s) while it does discriminate at +0.3 rad noise (0.98 → 0.67 s).

**Stage 3 — acceptance (frozen tracker, GPU, CEM winner only).** Roll out robot *r*'s frozen qualified tracker on the corrected vs uncorrected reference, paired, same start grid, same seed — the E67/E70 protocol (`torch.linspace` start grid, `HORIZON_STEPS=500`, seed 404, `scripts/eval_e67_teacher.py:75-86`). Accept η iff **all four** hold:
1. `completion(η) − completion(0) ≥ +0.10` (the advisor's G4 +10 pp);
2. the 95% **paired cluster bootstrap** lower bound (10,000 replicates, clip as cluster — design already in `scripts/analyze_wbt_rollouts.py`) is **> 0**;
3. `mean_survival(η) ≥ mean_survival(0) − 0.05 s`;
4. Stage 1 **re-checked on the full stitched clip** after overlap-add composition (`snmr/morpho_stitching.py`), because interval-local acceptance can compose badly.

**Eligibility filter (E77's saturation lesson).** Exclude any (r, c) whose uncorrected completion ≥ 0.95 (no headroom) or ≤ 0.05 (floored). A curve measured outside the functional window is uninterpretable regardless of execution quality.

**Never accept per-rollout.** E76 measured ~18% of individual rollout outcomes flipping on re-run under identical inputs while the arm-level contrast agreed to 0.003. Acceptance is a clip-aggregate decision with the bootstrap, always.

**Disclosed conservative confound:** each robot's frozen tracker was trained on that robot's *uncorrected* reference, so it is biased toward η = 0. The bias runs against the hypothesis. State it in one sentence rather than let it be asked.

---

## 4. Distillation target and loss

**Model.** `f_θ(robot_tokens, proposal_window, interval_descriptor) → (η̂, gate_logit)`.
- Conditioning: `RobotGraphTokenizer("kinematic")` (`snmr/robot_tokens.py`) — 24 kinematic + 5 topology features, `dynamics_available` zeroed at `:249-252`, with the fail-closed guard at `snmr/morpho_integration.py:142-148` that *refuses* to bind nonzero dynamics availability.
- Trunk: reuse the tree-biased attention stack from `snmr/morpho_model.py:80/159/214`, **frozen**. The 1,583,754-parameter retargeter stays the proposal generator; the correction head is new and small. Freezing means accepted-correction data cannot damage the proposal.
- **Architectural commitment that carries the paper:** the head emits *one K-vector of spline coefficients per robot joint token*, masked to the leg chain, plus 4 root channels. An unseen robot therefore requires **zero new parameters** — no per-robot prompt bank, no per-robot output head. This is precisely what AdaMorph (arXiv:2601.07284) and X-Morph (arXiv:2606.30290) structurally cannot say, and it must be verified as literally true of the implementation before anything is written up.
- **Do not build the head inside `FixedTargetMorphoRetargeter`** (`snmr/morpho_integration.py:279`): it binds one `RobotTokenBatch` at construction time. The head must take tokens as a forward argument.

**Target.** The accepted set A = {(r, c, I, η\*)} from Stage 3 — only accepted corrections, plus every eligible interval where the search found **no** accepted η, labelled η\* = 0.

**Loss.**
```
L = Huber( B(η̂) , B(η*) )  +  λ · BCE( gate_logit , 1[interval is correctable] )
```
- `B(·)` evaluates the spline on the frame grid, so the loss lives in **trajectory space** (coefficient-space error is not the quantity that matters); the evaluation is linear, so this is a single differentiable term.
- Each channel is normalized by its own box bound (0.04 / 0.12 / 0.35) before the Huber, so root translation and joint residual contribute comparably. This is scale matching, not a hyperparameter.
- Output projection `tanh × bound` → η̂ is inside the box by construction; no bound penalty term exists.
- λ = 1. The gate head is **not optional**: a model trained only on accepted corrections will hallucinate corrections on intervals where none exists.
- **No intent term in the loss.** Intent is enforced at inference by the same Stage-1 filter, which is CPU-only and available on the unseen robot. Deployed pipeline on a new robot: tokenize URDF → frozen proposal → kinematic interval detector (no sim) → η̂ + gate → Stage 1 → accept, else fall back to η = 0. **Zero rollouts, zero search, zero new parameters, and a hard intent guarantee.**

**Mandatory negative control.** Train an identical head on **shuffled robot tokens** (the spec of a different robot). If held-out performance is unchanged, the conditioning is inert and the result is "a better average retargeter," not transfer. Run this before committing to the framing.

---

## 5. Falsification conditions — what would tell them to stop

Run in this order; each is a stop point.

**F0 — no ruler on the held-out robot.** If no `SemanticManifest` + seven measured anchors can be produced for the held-out robot, D_intent cannot be evaluated there and the experiment as specified cannot run. **Stop before any GPU.**

**F1 — instrument not qualified (must be settled first).** Write the tracker qualification floor **before any rollout is observed**. The only occurrence of the phrase in the repository is `docs/MORPHORETARGET_B1_DECISION_2026-09-01.md:24` and it defines **no number**; a completed, unscored T1 checkpoint already exists (`model_39999.pt`, 2026-09-02 11:20). Preregister the E67 precedent: ≥0.80 completion, ≥9.0 s survival on the **uncorrected teacher** reference, per-clip not macro. If robot *r*'s tracker fails its own floor, no correction result on *r* is interpretable — that is instrument failure, not hypothesis failure, and must not be reported as either.

**F2 — the target does not exist (most likely failure, and it is cheap).** Run the oracle search on **one** training robot first. If the accepted set A is empty or covers <50% of eligible intervals at +10 pp, there is nothing to distill. **Stop before building the head.** The repo's own prior points this way: E50-A ran an unbounded version of exactly this correction and passed physics (stance speed ratio 0.33/0.16 vs a ≤0.50 gate, zero penetration) while **failing fidelity at 9.7 cm heading-local MPJPE against a 5 cm gate, flat in time** — literally "improved execution by turning it into a different, easier movement." Under Stage-1-before-Stage-2 ordering that entire solution is rejected. The bounded operator is a different operator (E50-A replaced the reference wholesale), so its failure is not predetermined — but this is the headline risk and it costs one CPU-day to find out. (Caveat: E50-A's 9.7 cm was measured under DEFECT-1, body-position reward at 0.0; it needs a post-fix rerun before being quoted as final.)

**F3 — intent is what is being traded.** If the accepted rate is materially positive only when Block B's 5% bound is loosened, **and** the achieved amplitude/energy ratios of accepted η cluster below 1.0 — specifically, median `log r_energy^η − log r_energy^0 < −0.05` with the bound removed — the gains are the degenerate solution. **Stop.** This is the advisor's stated kill criterion, and note the sign: the repo's existing T5 failures are all **high-side** amplification (1.51 vs a 1.50 bound), so a low-side cluster is a new and disqualifying signal.

**F4 — transfer is not demonstrated (the primary test).** On the held-out robot, predicted-η completion gain over η = 0, measured with that robot's frozen tracker on the E67 protocol with the paired cluster bootstrap: if the 95% CI **contains zero**, transfer is not demonstrated. Hard kill if the CI **upper bound < +0.02** — at most a fifth of what the search itself achieves on training robots.

**F5 — conditioning is inert.** If the shuffled-RobotSpec head matches the real head on the held-out robot within the CI, the model learned a robot-independent average correction. The transfer claim dies; a smaller "better default retargeter" claim survives.

**F6 — amortization fails (the decomposition that matters).** Run the oracle search *on the held-out robot too*. If oracle search gains ≥ +10 pp but predicted η̂ gains < +3 pp, then corrections **exist** for the unseen robot but **cannot be predicted** without searching it. That is a clean, publishable negative and it is the single most informative run in the plan — it separates "corrections exist" from "corrections transfer." Do not skip it to save GPU.

**F7 — flat scaling.** Held-out gain as a function of N training robots with slope indistinguishable from zero. Per the novelty audit, a positive slope is the one result that cannot be reassembled from AdaMorph + ReActor; a flat slope means there is no transfer phenomenon, only an average. Be honest that at N ≤ 4 this figure is under-powered — if the budget permits only one headline, **F4/F6 outrank F7**.

---

## 6. Module-by-module

### Write (new)

| path | what | size / risk |
|---|---|---|
| `/home/robotixx/snmr/snmr/correction_basis.py` | `BoundedSplineCorrection`: interval, 8-frame knots, zero-clamped ends, channel bounds mirroring `eval_footlock.py:191-193`, spline eval to Δ-trajectory, box projection, dim accounting, leg-chain resolution via `_path_to_root`. Pure tensor code, no simulator. | ~250 lines, low risk |
| `/home/robotixx/snmr/snmr/repair_intent.py` | D_intent Blocks A/B/C/D. Blocks A/B wrap `semantic_metrics.py:732`; Block C is new (human-side event localization + peak preservation); Block D is an assertion. | ~400 lines, **MEDIUM–HIGH risk** |
| `/home/robotixx/snmr/snmr/correction_search.py` | CEM/MPPI over η with the Stage 0→1→2 gating; interval enumeration via `verification.py:341/404`. | ~350 lines, medium |
| `/home/robotixx/snmr/snmr/correction_head.py` | Per-joint-token coefficient head + gate head; `tanh × bound` output; takes `RobotTokenBatch` as a forward argument. | ~200 lines, low |
| `/home/robotixx/snmr/scripts/search_corrections.py` | Per-(robot, clip) driver; writes the accepted-correction ledger with input SHAs. | ~200 lines |
| `/home/robotixx/snmr/scripts/eval_wbt_multimotion.py` | **Fork of `scripts/eval_e67_teacher.py`; delete the `num_motions != 1` guard at lines 75-76** and build a per-motion stratified start grid. The multi-motion mapping is **already implemented** in `snmr/integration/wbt_bodyfix.py:85-133` (`searchsorted` on `motion_end_idx`, assigns `motion_ids`). | **~60 lines — highest value per hour in the list** |
| `/home/robotixx/snmr/scripts/train_correction_head.py` | Distillation trainer + shuffled-spec control arm. | ~250 lines |
| `/home/robotixx/snmr/scripts/eval_correction_transfer.py` | Held-out-robot evaluation: η̂ vs 0 vs oracle-search vs shuffled-spec, paired bootstrap extracted from `scripts/analyze_wbt_rollouts.py`. | ~200 lines |
| tests | `tests/test_correction_basis.py` (bounds, C², zero endpoints, no-time-warp), `tests/test_repair_intent.py` (a synthetic shrunk motion must fail Block B; a synthetic slowed motion must fail Block B-energy; a synthetic deleted-transition motion must fail Block C and **pass** Blocks A/B — that last one is the test that proves Block C is load-bearing) | required |

**The MEDIUM–HIGH risk is the `repair_intent.py` adapter, and it is the single item most likely to blow the schedule.** `evaluate_semantic_retargeting` hard-requires a `HumanMotionSpec` with `contact_origin == "derived"`, `robot_fk.frames == human_motion.frames`, and `robot_fk.timestamps_s` matching `human_motion.timebase.target_timestamps` to within `_TIME_ATOL`. The corrector harness works in 192-frame windows in the scaled-human-heading frame. **Check first** whether `/home/robotixx/snmr/snmr/motion_spec.py` and `/home/robotixx/snmr/snmr/human.py` already build a `HumanMotionSpec` for the seven benchmark `VAL_CLIPS`; if that path is not already wired, budget a full day rather than half.

### Reuse unchanged

- `/home/robotixx/snmr/snmr/semantic_metrics.py` — `evaluate_semantic_retargeting` (:732), `SemanticThresholds` (:266), `_per_anchor_amplitude/_energy/_jitter` (:645-665). The source-intent ruler. **Do not** use `snmr/morpho_a1_posttests.py` as an intent metric — it measures spec-conditioning adversaries, a different question.
- `/home/robotixx/snmr/snmr/g1_semantics.py` — `g1_semantic_manifest()` (:53), `g1_lafan1_semantic_correspondence()` (:84, already parameterized for variant anchor transforms), `g1_robot_fk_trajectory()` (:154, works unchanged for limb-length variants since link names are preserved).
- `/home/robotixx/snmr/snmr/verification.py` — `localize_threshold_failures` (:341), `offset_failure_intervals` (:404), `VerificationReport.validate` (:156-232). The bounded-interval primitive; no new interval machinery needed.
- `/home/robotixx/snmr/snmr/morpho_stitching.py` — overlap-add + seam diagnostics for composing per-interval corrections into a full clip, with the seam-vs-elsewhere magnitude statistics that make Stage-3 step 4 auditable.
- `/home/robotixx/snmr/snmr/robot_tokens.py` + `/home/robotixx/snmr/snmr/morpho_integration.py` + `/home/robotixx/snmr/snmr/morpho_model.py` — conditioning trunk, frozen. Keep `RobotGraphTokenizer("kinematic")` and the fail-closed dynamics guard exactly as they are.
- `/home/robotixx/snmr/scripts/trackability_proxy.py` — Stage 2. `/home/robotixx/snmr/scripts/eval_e67_teacher.py` — Stage 3 template. `/home/robotixx/snmr/scripts/eval_footlock.py` + `/home/robotixx/snmr/snmr/metrics.py` — guard suite, mask catalogue, 2000-sample clip bootstrap.
- `/home/robotixx/snmr/snmr/mjcf_variants.py:562/1168` + `/home/robotixx/snmr/snmr/robot_variants.py` (`CoherentVariantConfig`) — the training-robot set.
- `/home/robotixx/snmr/scripts/analyze_wbt_rollouts.py` — **extract** the 10,000-replicate paired cluster bootstrap into a generic arm-vs-arm analyzer. Treat `runs/wbt_independent_eval_protocol.sh` as a design template only: it pins holosoma `eebdcf4` and `--wbt-metrics` flags that no longer exist in the local clone.

### Robot set, motion set, and cost

- **Motion set: the walk family only** (walk1_subject1, walk1_subject5, walk3). The deployable source-contact mask is P 0.711 / R 0.941 / F1 0.799 on walk1_subject5 versus P 0.004 on dance2 and P 0.000 on fight1 — the "all deployable masks fail" aggregate is near-degenerate on 3 of 7 clips. Trackers qualify here (E70 two-walk student 0.9248) and nowhere else. **Do not re-attempt the 8-clip heterogeneous set** — 0.378 / 0.465 / 0.478 across three budgets is a settled negative.
- **Robots: G1 + 3 coherent limb-length variants (train), 2 further variants at the extremes (held out).** Variants inherit the semantic manifest, which is the only reason D_intent is evaluable on the held-out robot at all. Booster T1 is the **tier-2** held-out robot and is gated on F0 (seven hand-measured anchors) — it is the real-robot claim, and it should be attempted only after the variant result exists.
- **GPU: one night per robot for the Stage-3 verifier.** Measured: 8,192 envs × 40,000 iters = 15 h 16 m on the T1 PhysX recipe (~7.86 B samples, ~20× E53's failing mjwarp budget). Six robots ⇒ six nights, one-time. Stage-3 *rollouts* are cheap once the tracker exists; the CEM inner loop never touches a learned tracker. Head training is ~1 GPU-hour. Needs exclusive GPU (only ~9 GB of 32,607 MiB currently free; use `scripts/resume_when_gpu_exclusive.sh`).
- **Run order:** F0/F1 (write the qualification floor — do this today, a completed unscored checkpoint already exists) → build `correction_basis` + `repair_intent` + the multimotion evaluator (~2–3 CPU-days) → **F2 on one training robot** → only then spend the six GPU nights → distill → F4/F5/F6 → F7 if budget remains.

### Two housekeeping items that block provenance, not science

Commit the 49-path working tree: all 51 frozen artifacts under `autoresearch/iterate-260901-1222/` record `dirty=true` and none records `false`, so no hash in the MorphoRetarget partition is verifiable from a clean checkout. And split the inverted gate label at `/home/robotixx/snmr/snmr/morpho_a1_evaluation.py:157-158` into `model_amplitude_collapse` (< min) and `model_amplitude_amplification` (> max) **before** this experiment starts reporting amplitude ratios — the current property mislabels high-side amplification as collapse, and this design's entire kill criterion turns on which side of 1.0 the violation falls.