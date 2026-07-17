# Flow Retargeting v2: Investigation Goals and Preregistered Protocol

Drafted 2026-07-17 from the E43–E45 post-mortem (`docs/FLOW_MATCHING_RETROSPECTIVE.md`) and the
verified literature pass (`docs/FLOW_RETARGETING_LITERATURE.md`). Status: REGISTERED, NOT
STARTED. Side-project rules from v1 carry over: no GPU while a main-line job runs, stop rules
fire at gates, no post-hoc threshold tuning. Main-line priorities (B1 confirmatory WBT matrix,
sharing-cost screen, B-multi calibration) remain ahead of everything here.

## 0. The central question, sharpened

v1 asked "does physics-guided flow sampling fix skate?" and failed for reasons that were
diagnosable *before* training: a Dirac conditional (guidance provably annihilated by posterior
covariance — Feng et al. 2025) on a noise-brittle regression latent (guidance pushes exactly
where the decoder is undefined). The v2 question is therefore NOT "tune the sampler" but:

> **Which of {substrate regularization, non-degenerate conditional targets, corrector
> formulation} is necessary and sufficient for inference-time physical correction of SNMR
> outputs to pass the frozen deployable-mask endpoints — and does any of it transfer to
> better WBT tracking?**

Three sub-questions, in forced dependency order:

- **V-A (substrate):** does a noise-regularized latent + noise-finetuned decoder make latent
  correction (even plain descent) deployably effective? This is testable WITHOUT any flow.
- **V-B (distribution):** can we manufacture a genuinely non-degenerate conditional target —
  (i) decodable robot encodings (`zr_decode_prob`-style joint finetune) and/or (ii) a
  multi-solution dataset (physics-repaired GMR variants) — so a flow has something to model?
- **V-C (corrector):** given V-A (and optionally V-B), which corrector wins at matched compute:
  Adam z-descent (v1 control), D-Flow-style z0 optimization through the flow, or
  interval-restricted norm-scheduled guidance? The literature predicts control-style
  optimization; v1's 5× result agrees.

## 1. What we may claim already (inputs, not open questions)

| Fact | Source |
|---|---|
| Guidance on a Dirac transport is structurally inert; scale cannot fix direction | E45 + arXiv:2502.02150 [3-0] |
| The decoder+FK cost gradient is ~orthogonal to the productive descent direction in the v1 latent | E45 (cos 0.05) |
| Adaptive (Adam) descent extracts 5× more than clamped guidance, still misses endpoints ~2.5× | E44 |
| Oracle contact mask makes every corrector family pass | Gate 1b + E44 |
| Deterministic-AE latents are off-manifold-brittle; regularization helps interpolation/decoding | arXiv:2412.04755 [2-1] + ACAI/RAE |
| Guidance, if ever reused: mid-t interval only, norm-scheduled (Polyak/TFG), manifold-projected | arXiv:2404.07724, 2403.12404, 2409.15761, 2311.16424 [all 3-0] |
| ODE-sampler stochasticity corrects toward the model's own marginal only — never a diversity source | arXiv:2206.00364 [3-0] |

## 2. Preregistered gates

Shared eval protocol: identical to v1 (VAL_CLIPS × 6 windows of 64, frozen Gate-1b guards,
paired raw baseline, NFE/iteration-matched controls). New arms must beat the *v1 z-descent
row* (0.221 m/s teacher-height), not just raw.

### V0 — noise-regularized substrate (cheapest decisive experiment; CPU eval, ~1 GPU-day train)

Finetune ONLY the decoder (+ embodiment encoder) of `phase1_g1_large` for 20k steps with
Gaussian latent augmentation: `z' = z_h + σ·ε`, σ ∈ {0.1, 0.3} (relative to per-dim latent
std), distill loss unchanged against the same teacher. Two checks:

- **V0a fidelity guard:** clean-z MPJPE within +0.2 cm of the un-finetuned decoder. (If noise
  training costs clean accuracy, record the Pareto and pick σ=0.1.)
- **V0b descent transfer (the gate):** re-run the v1 z-descent control (identical cost,
  iters, trust region) on the noise-finetuned decoder. PASS if teacher-height stance speed
  improves ≥ 25% over the v1 value (0.221 → ≤ 0.166 m/s) with all non-speed guards passing.
  PASS confirms "substrate was binding"; FAIL kills the substrate hypothesis for this decoder
  scale and V-C proceeds only under V1.

### V1 — decodable robot encodings (unblocks a real conditional; ~1 GPU-day)

Resume `phase1_g1_large` for 20k steps with `zr_decode_prob = 0.5`: with prob 0.5 decode from
the robot-teacher encoding z_r instead of z_h (the Phase-2-deferred experiment), plus the V0
noise augmentation on whichever latent is decoded. Gates:

- **V1a:** Dec(z_r) MPJPE ≤ 3 cm (v1's F0 bar — it failed at 50 cm; this is the direct fix).
- **V1b:** Dec(z_h) MPJPE regression ≤ +0.3 cm vs before.
- PASS unlocks a *non-Dirac* flow target: p(z_r | z_h) where z_r ≠ z_h by construction
  (measured human↔robot latent gap: CKA 0.91 but MLP-separable 0.91). Whether that gap gives
  usable conditional variance is measured, not assumed: report per-window ‖z_r − z_h‖ and the
  flow's endpoint diversity before any F2-style screen.

### V2 — multi-solution supervision screen (exploratory; bounded at 2 GPU-days)

Manufacture K=4 physically-distinct solutions per training window on ONE robot (G1): teacher
GMR + oracle-mask windowed projection variants under jittered projection bounds, each vetted
by the trackability proxy (`scripts/trackability_proxy.py`, PD-replay survival ≥ GMR's on the
same window). Train the V-B flow on the resulting one-to-many dataset. Gate: flow endpoint
diversity ≥ 5% of ‖z‖ (v1: 0.4%) AND unguided fidelity within F1 bounds. This tests whether
*data*, not architecture, was the diversity bottleneck. NOTE: the SSL/multi-solution
literature legs came back unverified — this arm is exploratory by construction and cannot
close any line on its own.

### V3 — corrector shoot-out (the decision gate; runs only after V0 or V1 passes)

At matched compute (25 decoder+FK gradient evaluations), on the winning substrate:

| Arm | What it tests |
|---|---|
| Adam z-descent (v1 control, re-run) | baseline carried forward |
| D-Flow-style: optimize z0 by backprop through the (V1/V2) flow | control-formulation hypothesis (OC-Flow family) |
| Interval-restricted, Polyak-normalized, endpoint-projected guidance (mid-t only) | "guidance done right" per findings 5/6/8 |

PASS = any deployable-signal arm meets BOTH frozen speed endpoints with all guards
(teacher-height ≤ 0.08, source-contact ≤ 0.10). Secondary readout: which arm wins and by how
much — this is the publishable comparison regardless of PASS/FAIL. FAIL of all arms closes
inference-time latent correction permanently; the remaining lever is physics-repaired
supervision *of SNMR itself* (main-line Track A5), which V2's dataset machinery would already
have prototyped.

### V4 — WBT hand-off (unchanged from v1 F3; only after V3 PASS and B1 resolution)

Third arm in the calibrated reference-quality comparison. Non-inferiority framing identical
to B1 (completion −5 pp, RMSE +10%).

## 3. Explicit non-goals

- No reflow/NFE-1 distillation before V3 passes.
- No SDE/stochastic-sampler work as a diversity mechanism (verified: corrects toward the
  model's own marginal only).
- No text conditioning, no safety-gate replication from SafeFlow.
- No SSL pretraining arm (masked motion modeling etc.) until the verified-literature gap is
  filled — the deep-research pass returned zero surviving claims on SSL motion latents; a
  targeted follow-up review is a prerequisite, not a formality.

## 4. Budget and ordering

V0 (1 GPU-day) → V1 (1 GPU-day) can run whenever the GPU is idle between main-line jobs;
V2 (2 GPU-days) only if V1 passes but its conditional variance is too small to model;
V3 is CPU-heavy eval (fine anytime); V4 competes with main-line WBT budget and must queue
behind B1. Total worst case ≈ 4 GPU-days + CPU eval — about half of v1's end-to-end cost,
because v1's machinery (trainer, evaluator, diagnostics, tests) is reused unchanged.
