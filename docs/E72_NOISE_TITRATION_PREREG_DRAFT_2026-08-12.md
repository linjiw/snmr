# E72 — Command-Channel Noise Titration (Instrument Sensitivity Curve)

**Status: DRAFT preregistration — not frozen, not launched.** Freeze rule: record this file's
SHA-256 in `docs/EXECUTION_PLAN_REPORTS.md` before the first rollout; no edits after that
except a dated addendum. GPU rule: E72 may not start until the B4 paper video has composed
(`POSTPROCESS_COMPLETE` exists); it never contends with the 26,000-MiB video gate.

## Question

Is the quantity our assay measures *graded*? E70 established a binary contrast (latent beats
matched nulls; destroyed channel collapses). A measurement instrument is more convincing if it
shows a dose–response: corrupt the command channel by calibrated amounts and the measured
content should decay smoothly from the intact value, through the time-null level, toward the
destruction floor. This validates instrument sensitivity without touching any E70 claim.

## Why this is cheap and safe

The frozen trainer's evaluation path already implements eval-time command corruption
(`E52_EVAL_NOISE_CMD`, Gaussian sigma in normalized units, `scripts/train_e52_dagger.py`
line 568/654 — a frozen file we invoke, never edit). E72 is therefore evaluation-only: no
training, no new channels, no frozen-file modifications, no effect on E70 artifacts.

## Registered design

- **Arms:** `snmr` (seeds 0–2, primary) and `explicit` (seeds 0–2, contrast — its curve shows
  how a full-information channel degrades, separating channel-content loss from generic
  controller fragility).
- **Noise levels:** sigma ∈ {0.1, 0.25, 0.5, 1.0, 2.0} normalized units. Sigma=0 is the
  existing frozen eval (reused as-is, not rerun).
- **Evaluations:** the unchanged general (1024 rollouts, eval seed 404) and ambiguity harnesses,
  per arm × seed × sigma → 60 new eval runs, all GPU-eval-only.
- **Analysis:** per-seed and mean curves of completion vs sigma per arm; the sigma at which the
  SNMR curve crosses the frozen time-null level (0.562 ambiguity) reported with a linear
  interpolation between adjacent levels. New non-frozen analyzer script; all displayed numbers
  machine-generated from its JSON.

## Registered outcomes (all reportable; none alters E70)

1. **Graded instrument (expected):** SNMR ambiguity completion decreases monotonically in sigma
   (tolerance: no increase >0.02 between adjacent levels), passes below the time-null level at
   some sigma*, and approaches the destruction floor at sigma=2.0. Claim licensed: "the assay
   measures a graded, causally manipulable quantity" — one sentence + one small figure.
2. **Threshold/cliff:** decay is monotone but abrupt (one adjacent-level drop >0.5 of the total
   range). Claim licensed: channel content is used near-binarily; report the cliff location. No
   gradedness claim.
3. **Non-monotone / noise-benefit anomaly:** any adjacent-level increase >0.02. Report as an
   instrument-sensitivity anomaly, investigate post-submission; no positive spin.

## Boundary

E72 numbers never merge into E70 tables or the frozen analyzer; they appear (if at all) as a
separate labeled figure/sentence. The paper's E70 claims are complete without E72; this is
supporting evidence for the *instrument*, aimed at the benchmark/assay positioning.

## Cost estimate

60 evals × ~2–4 min ≈ 2–4 GPU-hours, schedulable any time after the paper video composes.
