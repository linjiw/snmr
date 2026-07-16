# Phase 3: Multi-Clip Latent WBT Protocol - DRAFT (E39 resolved; baseline calibration pending)

**Drafted 2026-07-15 before the Phase-2 verdict.** Everything except the arm list is intended to
freeze as written; the arm list has two pre-registered variants conditioned on E39 (§4). Plan
context: `docs/WBT_LATENT_PLAN_v2.md` §5.4. This is the first setting where the literature
predicts the latent can win: across clips, z is no longer a function of a single clip's phase —
it must disambiguate motion identity/content, which the explicit per-frame command does not
carry across clip boundaries.

**E39 resolution, 2026-07-16:** L1 clears its feasibility floor in all six cells; S3 does not
replicate the frozen benefit rule; seed-0 C3 and S3 effects are indistinguishable. The registered
conditional arm set is therefore B/S3/L1/C3. This document is still **not launch-frozen**:
the 16k multi-clip budget is an unsupported extrapolation from one 8k single-clip baseline.
Section 3A adds a baseline-only calibration gate before any latent arm spends GPU time.

## 1. Question

Does the frozen SNMR latent improve a *multi-clip* GMR-reference WBT policy, measured on
**held-out clips** (generalization) and on trained clips (capacity), relative to a matched
baseline policy?

## 2. Data (exported and schema-validated 2026-07-15)

- Train `motion_dir`: `runs/wbt_latent_gmr_multi/train/` — walk1_subject2, walk2_subject1,
  walk3_subject1, walk3_subject3, run1_subject2, sprint1_subject2 (75,184 frames @ 50 Hz,
  ~25.1 min). Walk-family only, per the E35 budget finding (dance/fight excluded until per-clip
  budgets exist).
- Held-out eval clips: `runs/wbt_latent_gmr_multi/heldout/` — walk1_subject5 (the Phase-0/1/2
  screen clip, never trained on in Phase 3) and run2_subject1.
- Every arm trains on the SAME augmented NPZs (GMR reference + `latent_z`); standard fields are
  byte-identical to plain GMR exports (validated), and the vanilla observation term ignores
  `latent_z`, so arms differ ONLY in observation functions.

## 3. Fixed settings

| Item | Value |
|---|---|
| Robot / env | `g1-29dof-wbt`, MuJoCo/Warp, 1,024 envs |
| PPO iterations | 16,000 (≈2× Phase-0 budget for ≈6× content; GMT/VMS precedent that multi-motion needs more) |
| Save interval | 4,000 |
| Training seed | 0 (single seed; Phase-3 is a screen — replication follows only for promoted arms) |
| Training data | `--command...motion-config.motion-dir runs/wbt_latent_gmr_multi/train/` (motion_dir takes precedence over motion_file) |
| Adaptive timestep sampler | holosoma default (on), identical across arms |
| Evaluation | Official terminal-aware evaluator, per clip: override `motion-dir ""` + `motion-file <clip>_mj_z.npz`; 100 phase-stratified 10-s windows; eval seed 404 |
| Eval clips | walk1_subject5, run2_subject1 (held-out primary); walk1_subject2, run1_subject2 (trained secondary) |

### 3A. Baseline-only budget gate (required before protocol freeze)

Train B-multi first and evaluate checkpoints at 4k, 8k, and 16k on the two held-out and two
trained clips. Select the earliest horizon with held-out mean completion >=50% and every
held-out clip >=25%; selection uses B only and remains blind to every latent arm. If 16k fails,
extend B-multi once to 32k and repeat. If 32k fails, stop this experiment and reduce motion
difficulty or fix multi-motion training rather than interpreting latent-arm failures under an
undertrained control.

The final driver must use the selected horizon for every arm and record the baseline-calibration
artifact hash. The current `scripts/run_wbt_latent_phase3.sh` is infrastructure, not launch-ready,
until this gate resolves its `ITERATIONS` value.

## 4. Arms (E39-resolved)

Always run:

1. **B-multi** — baseline observations (`motion_command`), actor+critic.
2. **S3-multi** — actor baseline; critic `motion_command_with_latent_preview`. Runs regardless of
   the E39 verdict on S3: cheap (no deployment change), and the multi-clip regime is where the
   latent hypothesis actually lives; the E36 near-miss justifies one multi-clip test even if
   single-clip replication is mixed. If E39 *replicated* S3, S3-multi is the headline arm.
3. **L1-multi** — actor `latent_preview_command` (latent-only); critic
   `motion_command_with_latent_preview`. E39 passed the gate in all six cells (72-77%
   completion), so this arm runs.

4. **C3-multi** — actor baseline; critic `motion_command_with_explicit_preview`. E39 triggered
   this condition: C3 RMSE effects (-3.7%/-5.1%) match S3 (-4.4%/-4.9%) at seed 0. S3 cannot be
   interpreted without this attribution control.

## 5. Endpoints and promotion (frozen now)

Primary: held-out-clip mean across the two held-out clips of (a) 10-s completion, (b)
joint-position RMSE. Secondary: trained-clip metrics, per-clip breakdowns, full official set.

- **S3-multi promotes** if, on held-out clips, it meets the Phase-0 rule vs B-multi (completion
  floor −5 pp AND ≥5% relative RMSE/survival improvement or +5 pp completion).
- **L1-multi is descriptive** (no promotion rule): its readout is the held-out completion gap vs
  B-multi, benchmarked against the single-clip gap (−16 pp at 8k iters). L1-multi *narrowing*
  the gap on held-out clips relative to single-clip would be the first positive-generalization
  evidence for the latent command; widening or collapse (<40% completion) closes the L1 line at
  this scale.
- Analyzer: `analyze_wbt_latent_phase2.py`-style paired per-clip cells (B-multi is the pairing
  baseline; same eval windows per clip across arms).

## 6. Validation gates before launch

1. Complete the baseline-only budget gate in §3A; freeze the selected horizon.
2. Multi-motion smoke run (~50 iters, 256 envs) on the train `motion_dir` for each observation
   function actually used: assert obs dims (identical to Phase-0/1 values), finite rewards, and
   correct latent concatenation (loader logs 6 motions / 75,184 frames).
3. Held-out eval smoke on one checkpoint: assert the report's `motion_file` resolves to the
   held-out NPZ and start-step grid covers the clip.
4. Driver + analyzer + arms sha256'd into `runs/wbt_latent_phase3/` at launch; this document's
   §3–5 frozen at that moment (arm list resolved per §4 with the E39 entry cited).

## 7. Budget

Baseline calibration costs one 16k run plus an optional 16k continuation. The resolved four-arm
matrix then costs roughly 4 × the selected baseline horizon plus evaluations. Do not quote the
earlier 18-22 GPU-hour estimate until §3A determines the horizon.
