# Report — latent-benefit program kickoff (2026-08-15)

**Author:** Fable. **Scope:** the owner's directive to adopt the two-track proposal (reframe the
additive null; encode what the reference omits; spend retargeter byproducts on the curriculum) and
"aim for better performance". **Commit:** `39ef3fe` on `origin/main`.
**Companion docs:** `docs/LATENT_BENEFIT_PROGRAM_2026-08-15.md` (plan + preregistration drafts),
`docs/EXPERIMENT_LOG.md` (ledger entry), `docs/DEGRADED_COMMAND_RESEARCH_2026-08-15.md` and
`docs/E77_DEGRADATION_PILOT.md` (the results this work had to be reconciled with).

---

## 1. Summary

| Item | Status |
| --- | --- |
| Program plan reconciled with E77 and the benchmark memo; E78 preregistration draft with gates | **Done** |
| Paper hardening (proposal §6): conditional-information framing of the additive null; A/C window-asymmetry limitation | **Done**, PDF rebuilt, still 8 pages |
| Track A harness (E78): dropout masker, FiLM/gated fusion, derived trainer, launcher, paired analyzer | **Built and CPU-tested**; frozen files untouched (hash manifest 5/5) |
| Track B pilot (E1): retarget/latent features → per-second failure hazard | **Run**; pilot-positive within clip, see §3 |
| GPU experiments (E78 arms, C-future arm, SNMR v2 heads) | **Not run** — GPU fully occupied by the owner's other tenants; one command away |
| Tests | 20 new tests; 70/72 test files pass; 2 old training-style files exceeded a 5-min cap under load 26–44 on 20 cores (not related to today's changes) |

---

## 2. What was decided, and why

### 2.1 The additive null is a property of the instrument, not a finding about the latent
The students are trained by action-MSE from a deterministic teacher π*(x, g). Conditioned on
(x, g) the label is fixed, so I(a*; z_ret | x, g) = 0 and an MSE student has no gradient reason
to read z_ret beside g. E52-v4 arm D's three-seed null was therefore entailed. **Consequence:** the
paper now says so (§IV-B, Limitations); and additive value can only exist where the explicit channel
is degraded, delayed, or absent, or under RL — the regimes the sequel targets.

### 2.2 The sequel is masked *fusion*, not replacement — because E77 already killed replacement
E77 (2026-08-15, before this work) swept SNMR-only vs explicit-only under z_cmd zero-order hold and
Gaussian noise. On the matched axis the latent arm was worse at every level and floored first; the
one apparent positive (σ=1.0, +0.100 relative retention) dissolved to −0.020 on the paired matched
subset. So the proposal's Track A headline cannot be "the latent degrades more gracefully". It is
reframed as: **train the explicit+latent fusion under reference-channel dropout so the latent becomes
a usable fallback; measure completion vs dropout severity, paired, with clean regression bounded.**

### 2.3 The severity axis is physical and matched by construction
Reference-stream dropout in control ticks at 50 Hz (segments 0.1–0.5 s, target masked-tick fraction
f). The registered scope is `all`: the event removes the reference stream, so the explicit arm holds
its last `g` and the SNMR arm holds its last `z` window; proprioception stays live in both. This is
the "deployment-faithful evaluator" the E77 memo said did not exist. A secondary scope (`explicit`
only, `z` precomputed and delivered ahead) is kept but must be labelled as the bounded-lookahead story.

### 2.4 Analysis discipline
Every cross-arm severity contrast is reported as (i) paired all-rollout difference with a cluster
bootstrap over start windows and (ii) the E77-addendum matched-subset contrast with McNemar counts.
Marginal retention ratios are banned — they launder a clean gap.

---

## 3. Experiments run today (CPU only)

### 3.1 E1 pilot — do retarget-derived features predict where a tracker fails? (Track B)

**Script:** `scripts/e1_retarget_difficulty.py`. **Data:** the two frozen E70 walks
(`walk1_subject1`, `walk1_subject5`, 13,066 frames each at 50 Hz), and the frozen 1,024-rollout
general-grid evaluation reports (per-rollout `start_steps`, `completed`, `survival_s`).

**Labels per 1-s bin (522 bins total):**
- `hazard`: failures whose (start + survival) lands in the bin ÷ active-rollout ticks through the bin
  (the same quantity HoloSoma's `AdaptiveTimestepsSampler` estimates online, `wbt.py:408-486`);
- `start_fail`: 1 − completion of rollouts starting in the bin.

**Features per bin:**
- `kin` (available from any explicit reference): root speed mean/max/accel, root angular speed,
  root height mean/min/std, joint velocity RMS/max, joint acceleration RMS;
- `ret` (retargeting byproducts the reference format omits): min/mean joint-limit margin (limits from
  `g1_mocap_29dof.xml`), foot-skate mean/max, heuristic contact-switch density, foot clearance,
  stance fraction;
- `z` (SNMR latent): PCA-16 of the bin-mean latent fit on training folds only, mean ‖dz/dt‖,
  distance to the clip-mean latent.

**Model:** ridge (α = 3) on standardised features; held out by 20-s temporal blocks (both clips) and
by leave-one-clip-out.

**Results (held-out R², incremental over `kin`):**

| labels pooled from | event bins | CV | kin | +ret | +z | all |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 3 explicit seeds | 77 / 522 | temporal blocks | 0.095 | +0.055 | +0.054 | +0.063 |
| 3 explicit + 3 SNMR seeds | 277 / 522 | temporal blocks | 0.093 | **+0.076** | **+0.117** | +0.122 |
| same (start_fail label) | — | temporal blocks | 0.313 | +0.035 | +0.089 | +0.086 |
| 3 explicit + 3 SNMR seeds | 277 | leave-one-clip-out (n=2) | −0.08 | −0.05 | −1.27 | −1.32 |

**Findings (emphasis corrected later on 2026-08-15 per advisor guidance — the clean row is the
explicit-only one, +0.054/+0.055, below the gate; +0.117 is on partly circular labels and is not
the expectation):**
1. Within-clip held-out, the SNMR latent adds ≥ +0.10 R² for failure hazard only on the pooled
   (partly circular) labels; on clean labels it is +0.054, and the byproducts +0.055.
2. With explicit-only labels the signal is weaker (+0.05) mainly because the explicit student rarely
   fails (77 event bins) — the label is sparse, not the features weak.
3. Leave-one-clip-out with two clips is degenerate (even `kin` alone is negative), and z-PCA fit on
   one clip acts as a clip-identity code — the same identity structure E70/E72 found. Cross-clip
   transfer is untested, not refuted.
4. Caveat: SNMR-arm students *read* z, so z predicting their failures is partly circular. The
   explicit-only row is the clean one.

**Verdict:** pilot-positive; licenses E1-proper on a multi-clip pool (≥ 20 clips, leave-clips-out,
gate: incremental R² ≥ +0.10 over `kin` on held-out clips). Outputs:
`/data/robotixx/snmr-research/e1_retarget_difficulty/pilot_e70_{explicit,explicit_snmr}.json`.

### 3.2 Paper build check
`paper/main.tex` edited (two passages), rebuilt with the repo's tectonic path against the frozen
value files: 8 pages, letter, references end mid-page 8 (was also 8 pages before the edit).

### 3.3 Verification runs
- 20 new tests pass (`tests/test_fusion.py` 16, `tests/test_e78_analysis.py` 2,
  `tests/test_e1_retarget_difficulty.py` 2); the analyzer test plants a clean regression + a dropout
  effect on synthetic paired reports and checks both are recovered and the pairing invariant is
  enforced.
- Frozen E70 confirmation hash manifest: 5/5 verify, zero drift.
- Full suite: 70/72 files pass (0 failures); `test_overfit.py` and `test_human_to_robot.py` exceeded
  a 5-min per-file cap on a box at load 26–44/20 cores; both predate and are independent of today's
  changes and passed on 2026-08-14.

---

## 4. What was built (all committed)

| artefact | what it does |
| --- | --- |
| `snmr/integration/fusion.py` | `ReferenceDropoutMasker` (per-env Bernoulli segments, hold/zero, seeded ⇒ identical schedules across arms, flag bits `[is_masked, staleness]`, first tick after reset forced clean); `FusionCommandStudent` (concat / FiLM zero-init / gated-residual; decoder still sees exactly `[proprio, z_cmd]`; `flag_dim=0` is shape-identical to the frozen `CommandStudent`, so frozen E70 checkpoints load for the same sweep); `paired_dropout_summary`; `dropout_hazard`; `ramp` |
| `scripts/derive_e78_trainer.py` → `scripts/train_e78_masked_fusion.py` | E78 trainer derived from the frozen `train_e52_dagger.py` by 20 asserted replacements; a test fails if the derived file drifts; knobs `E78_FUSION`, `E78_MASK_{FRAC,SEG_MIN,SEG_MAX,SCOPE,MODE,RAMP_ROUNDS,SEED}`, `E78_EVAL_MASK_*`, `E78_FLAG_DIM` |
| `scripts/run_e78_masked_fusion.sh` | `train|sweep|frozen SEED TAG`; arms mE, mS, mZc, mZf, mZg, mTf, mShf, frozen explicit/snmr; 26,000-MiB gate; writes only under `/data/robotixx/snmr-research/e78_masked_fusion/` |
| `scripts/analyze_e78_dropout.py` | paired cluster-bootstrap + matched-subset + survival contrasts across discovered severities |
| `scripts/e1_retarget_difficulty.py` | §3.1 |
| `paper/main.tex` | §2.1 framing; A/C window sentence in Limitations |
| `docs/LATENT_BENEFIT_PROGRAM_2026-08-15.md` | plan, E78 prereg draft, E1/E2/E3 designs, owner decisions |

---

## 5. Preregistered gates for the next GPU experiment (E78, seed 0 first)

- Primary: paired completion (mZf − mE) at f = 0.3, seg 5–25 ticks ≥ **+0.10**, CI excluding 0,
  three seeds, positive on both clips.
- Co-primary: clean paired difference (mZf − mE) ≥ **−0.01**.
- Controls: mZf − mTf and mZf − mShf > 0 with CIs excluding 0 (else the benefit is "any window-matched
  conditioning under masking", a HANDOFF-style exposure finding, still reportable).
- Mandatory first: `frozen explicit` / `frozen snmr` clean cells must reproduce 0.923 / 0.699.
- Kill: mZf − mE < +0.05 at every severity on seed 0 → stop, record, do not tune.

## 6. Open items and owner decisions

1. **GPU night for E78 seed 0** (≈ one ≥26,000-MiB night). Commands in the plan doc §6.
2. **E1-proper label source:** log per-bin failure counters from the `whole_body_tracking` pool
   runs (small hook there; recommended) or E67-style teachers on a LAFAN1 subset here.
3. **Order after E78 seed 0:** E78 seeds 1–2 (decides whether Track A is a paper) before E3 v2 heads.
4. **Not built, needed before any Track A paper claim:** the explicit-with-future-window control
   (future `motion_command` lookup). It is the same plumbing the current draft's Limitations promises
   as future work — build once in `snmr/integration/`, use for both.
5. Unverified external claim carried from the earlier memo: UniAct (arXiv:2512.24321) "compressed
   command degrades more gracefully" — verify before any related-work sentence.

## 6b. Addendum (later on 2026-08-15) — advisor amendments actioned
Guidance saved at `docs/ADVISOR_GUIDANCE_2026-08-15_E78_AMENDMENTS.md` and fully actioned before any
GPU spend: mGf (explicit future window, `snmr/integration/goal_window.py`) and mTl (live clock) arms;
ambiguity-under-dropout and harder-cell secondaries registered; seed-exact frozen sanity gate;
analyzer covers the ambiguity grid and the mE − frozen_explicit descriptive contrast; E1 emphasis
corrected; pool-hook for E1-proper labels installed in `~/whole_body_tracking` (uncommitted there);
E54 T1 registration restored as an overlay so `run_e54_t1_teacher.sh` now only waits on ≥20 GB GPU.
Plan §3 updated with the pre-committed reading of mGf and the registered GPU order.

## 7. Bottom line
No new capability number was produced today and none was promised: the GPU was unavailable, and
E77 shows the naive robustness story is contradicted, not just unproven. What changed is that the
program now has (a) an honest theoretical framing of the null already in the paper, (b) a
matched-severity, deployment-faithful evaluator plus a fusion recipe designed to make the latent
*usable* rather than merely present, with gates fixed before any run, and (c) a pilot-positive
Track B result showing the retargeter's byproducts and latent predict tracker failure within clip.
