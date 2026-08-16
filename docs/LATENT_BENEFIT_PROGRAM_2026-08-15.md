# Latent-benefit program — from a scoped null to a designed interface

**Date:** 2026-08-15. **Author:** Fable, on the owner's direction to adopt the two-track proposal
(reframe the additive null; encode the delta set; spend retargeter byproducts on the curriculum) and
"aim for better performance". **Status:** program plan + preregistration drafts. Nothing here touches
a frozen E70 artifact, and nothing here is on the 2026-09-15 critical path except §7.

This document (a) reconciles the proposal with what E77 established yesterday, (b) fixes the program
thesis and the two tracks, (c) preregisters the first experiment of each track in the repo's
fail-closed style, and (d) records what was built today so the GPU work is one command away.

---

## 0. Bottom line

1. **The additive null is entailed by the instrument.** Action-MSE distillation from a
   deterministic teacher π*(x, g) makes I(a*; z_ret | x, g) = 0. Arm D of E52-v4 could not have
   come out any other way. This is now stated in `paper/main.tex` (§IV-B and Limitations), which
   inoculates the submission and licenses the sequel.
2. **The sequel is not "latent replaces explicit".** E77 killed that on the only matched axis, and
   the addendum's paired reanalysis showed the one apparent positive was a laundered clean gap.
   The sequel is **masked-training fusion**: the explicit reference stays; the latent is made
   *usable* by training under reference-channel dropout, and the claim is graceful degradation at
   zero clean cost, measured with the paired matched-subset protocol E77 taught us.
3. **The severity axis is fixed in advance and is physical.** Reference-stream dropout in ticks
   at 50 Hz (segment 0.1–0.5 s, masked-tick fraction f). Both arms lose the *same* event; each holds
   its *own* last-valid input while proprioception stays live. That is the deployment-faithful
   design the E77 memo said needed a new evaluator; the evaluator now exists (§6).
4. **Track B starts positive.** The E1 pilot on the two E70 walks: SNMR-latent features add
   +0.117 held-out R² over kinematics for per-second failure hazard (temporal-block CV, pooled
   students), retarget byproducts +0.076. Cross-clip transfer cannot be tested with two clips —
   that is exactly why E1-proper needs the multi-clip pool. See §5.
5. **GPU:** every GPU item is queued behind the owner's other tenants (28.4/32.6 GB in use as of
   this writing). Nothing below runs until a ≥26,000-MiB window opens; the SNMR rule that E70-class
   jobs take the first such window still holds. CPU-side deliverables are complete (§6).

---

## 1. Reconciliation with E77 and the benchmark memo

| Proposal claim | Verified status | Consequence |
| --- | --- | --- |
| "Additive null is I(a*; z\|x,g)=0 by construction" | Correct; consistent with `BENCHMARK_QUESTION_2026-08-12.md` (teacher-parity ceiling) | Written into the paper (§7). |
| "Latent-backed tracker degrades gracefully when the reference channel degrades" | **Untested in the proposed form.** E77 tested *replacement* (SNMR-only vs explicit-only) under z_cmd hold and Gaussian noise and found the latent arm worse on the matched axis; the E77 addendum showed the σ=1.0 positive was a baseline-composition artefact | The sequel's arms are **fusion** arms trained under masking, and every contrast is reported on the paired matched subset with cluster-bootstrap CIs. Marginal retention is banned. |
| "Zero clean-condition cost" | Must be a registered gate, not an assumption: E36-era screens found latent-beside-explicit *degraded* tracking | Clean regression ≤ 1 pp is a hard co-primary gate. |
| "E70 already proved the two-frame window is control-usable" | Yes, but the E70 controls were window-matched; the A/C gap is not (now stated in Limitations) | E78 includes an explicit-with-future-window arm as the honesty control. |
| "Bandwidth story" | Dead (`DEGRADED_COMMAND_RESEARCH` §3) | Not pursued. Severity in milliseconds and masked-tick fraction only. |
| Contact F1 in z_ret ≈ 0.088 (must be put in deliberately) | Consistent with `EXPERIMENT_LOG.md` (z-linear AUROC 0.51–0.64) | E3 (SNMR v2 heads) is a real prerequisite for the contact half of the delta set. |
| UniAct claim of "compressed command degrades more gracefully" | **Not verified** by anyone in this repo | Verify before writing any related-work sentence; irrelevant to E78's mechanism (masked *fusion*, not compression). |

The one-line reconciliation: **E77 measured whether the latent is a better *replacement*; E78
measures whether it is a usable *fallback* once training makes it one.** Different mechanism, different
arm structure, same discipline.

---

## 2. Program thesis and tracks

**Thesis.** The retarget→track interface is a designed artefact, not a fixed format; retargeting
byproducts are free supervision for tracking. Information flows *forward* from retargeting (SNMR
already exists before any tracker is trained), which is the structural difference from
HOVER/ULTRA/AnyBody-style unified command spaces distilled *down* from a trained tracker.

- **Track A — the interface sequel (E78 → E4/E5/E6).** Masked-training fusion; headline is a
  completion-vs-dropout-severity curve, paired, with clean regression bounded. Venue target RSS 2027
  / ICRA 2028; CoRL 2027 if the cross-embodiment extension matures.
- **Track B — retarget-aware curriculum (E1 → E2 → E7).** Retarget-derived features predict where a
  tracker fails, before rollouts; warm-start HoloSoma's `AdaptiveTimestepsSampler` failure EMA and
  gate DR per segment. Slots into the GACL→RTW→CLIMB arc; connects to LUCID.
- **Enabler — SNMR v2 (E3).** Contact / difficulty / future-consistency heads on the frozen recipe.

Both tracks share the E70 machinery, the E77 paired analyzer discipline, and the new masker.

---

## 3. E78 — masked-fusion prototype (Track A, distillation harness). PREREGISTRATION DRAFT

Two-tier as proposed: prototype in the distillation harness (hours per arm), confirm survivors in
HoloSoma PPO (E4-RL, later). This section registers tier 1.

### 3.1 Design

- **Substrate:** the frozen E70 two-walk instrument (teachers, motions, 1,024-rollout general grid,
  evaluation seed 404). Trainer `scripts/train_e78_masked_fusion.py`, *derived* from the frozen
  `train_e52_dagger.py` by `scripts/derive_e78_trainer.py` (20 asserted replacements; a test fails
  if the derived file drifts). Frozen files untouched; hash manifests unaffected.
- **Reference dropout (train and eval):** `ReferenceDropoutMasker` — Bernoulli-segment process per
  env; segment length U[5, 25] ticks (0.1–0.5 s), hazard set from a target masked-tick fraction;
  `hold` mode replays the last valid value; two flag bits `[is_masked, staleness/max_seg]` go to the
  code encoder only. Training ramps hazard 0→target over 300 rounds; masker resets on episode end
  and forces the first post-reset tick clean.
- **Scope (registered primary):** `all` — the dropout event removes the reference stream, so the
  explicit arm holds `g` and the SNMR arm holds its `z` window; proprioception is live in both.
  Matched by construction in physical units. **Secondary:** `explicit` — only `g` drops while `z`
  (precomputed and embedded in the NPZ, i.e. deliverable ahead of time) stays fresh; this is the
  bounded-lookahead delivery story and must be labelled as such.
- **Fusion:** `concat` (E52 arm D + flags), `film` (explicit trunk, latent emits per-layer
  scale/shift, zero-initialised so the latent must *earn* influence), `gated`
  (`base(x,g̃) + σ(w)·res(x,z')`). Decoder input remains exactly `[proprio, z_cmd]`.

### 3.2 Arms (all trained with masking, scope `all`, target fraction 0.3, seed-matched)

| tag | encoder input | fusion | role |
| --- | --- | --- | --- |
| mE | explicit | — | reference arm |
| mZf | explicit + z window | film | **treatment** |
| mZg | explicit + z window | gated | treatment (alt. fusion) |
| mZc | explicit + z window | concat | fusion baseline |
| mS | z window only | — | replacement (E77 context) |
| mTf | explicit + E70 time code | film | control: window-matched, content-free |
| mShf | explicit + other clip's z at matched phase | film | control: identity vs content |
| frozen_explicit / frozen_snmr | E70 seed students, unmasked-trained | — | "what dropout does to today's students" |

**Not yet buildable, required before any paper claim:** *explicit-with-future-window* (g at t and
t+0.1 s through the same projection path). It needs a future `motion_command` lookup that the current
observation path does not expose; it is the honesty control the paper cannot ship without, and it is
the same plumbing the current draft's Limitations promises as future work.

### 3.3 Endpoints and gates

Severity grid: masked-tick fraction f ∈ {0.1, 0.3, 0.5} × segment {5–25, 25–50} ticks, plus clean.
Same `E78_EVAL_MASK_SEED=404` for every arm (identical dropout schedule ⇒ paired).

- **Primary:** paired all-rollout completion difference (mZf − mE) at f = 0.3, seg 5–25,
  cluster-bootstrap over start windows (`scripts/analyze_e78_dropout.py`).
  Gate: **≥ +0.10 with CI excluding zero, three seeds, positive on both clips.**
- **Co-primary:** clean paired difference (mZf − mE) **≥ −0.01** (no clean regression). If this fails,
  the robustness result is reported but the "zero clean cost" sentence is not written.
- **Matched-subset check (E77 discipline):** contrast on rollouts both arms complete cleanly must
  agree in sign with the primary; McNemar counts reported.
- **Controls:** mZf − mTf and mZf − mShf ≥ 0 with CIs excluding zero at the primary severity.
  If mTf matches mZf, the benefit is "any window-matched conditioning under masking", not the latent.
- **Survival time** as a secondary endpoint (lower variance; still not a safety metric —
  Limitations sentence carries over verbatim).
- **Kill:** if mZf − mE < +0.05 at every severity on seed 0, stop after one seed and record the
  negative; do not tune. If mZc ≥ mZf, drop FiLM and continue with concat (fusion form is not the
  claim).

### 3.4 Cost

Per arm: one training (~E70 cell cost, ~hours) + 7 evaluations (~minutes each). Seed 0 for all
seven arms first (kill check), then seeds 1–2 for {mE, mZf, mTf, mShf} plus the frozen sweep. Fits
in two or three ≥26,000-MiB nights.

### 3.5 What survives each outcome

- Positive on primary and controls → Track A paper spine; proceed to E4-RL (HoloSoma PPO
  observation term `retarget_latent`, masking event term, FiLM in `ppo_modules`) and E5/E6.
- Positive vs mE, null vs mTf → "manufactured ambiguity + any window-matched signal" finding; the
  paper becomes a HANDOFF-style exposure study; still publishable, more interpretable.
- Null → recorded in the negative-results ledger; Track B unaffected.

---

## 4. E1 — retarget features predict tracking difficulty (Track B). PILOT DONE, PREREG FOR PROPER

### 4.1 Pilot (2026-08-15, CPU, `scripts/e1_retarget_difficulty.py`)

Labels: per-1-s-bin failure **hazard** (failures landing in the bin / active-rollout ticks through
the bin — the quantity HoloSoma's `AdaptiveTimestepsSampler` estimates online) and start-failure
rate, from the frozen E70 general-grid reports. Features: `kin` (root speed/height, joint vel/acc
RMS), `ret` (min joint-limit margin from `g1_mocap_29dof.xml`, foot skate, heuristic contact-switch
density, clearance), `z` (PCA-16 of bin-mean latent fit on training folds, mean ‖dz‖, distance to
clip mean). Ridge, α = 3.

| labels | CV | kin R² | +ret | +z | all |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 explicit seeds (77 event bins / 522) | temporal blocks 20 s | 0.095 | +0.055 | +0.054 | +0.063 |
| + 3 SNMR seeds (277 event bins) | temporal blocks 20 s | 0.093 | **+0.076** | **+0.117** | +0.122 |
| same | leave-one-clip-out (n = 2) | −0.08 | — | collapse | collapse |

Reading: within-clip held-out, the latent clears the +0.10 pilot gate and the byproducts come close;
across clips nothing generalises with two clips (PCA on z fit on one clip is a clip-identity code —
the E70/E72 finding again). Caveat: SNMR-arm students *read* z, so z predicting their failures is
partly circular; explicit-only gives +0.054. Outputs:
`/data/robotixx/snmr-research/e1_retarget_difficulty/pilot_e70_{explicit,explicit_snmr}.json`.

### 4.2 E1-proper (needs the multi-clip pool)

- **Labels:** `bin_failed_count` EMA per motion from a vanilla HoloSoma multi-clip run
  (`AdaptiveTimestepsSampler`, `wbt.py:408-486`) exported at checkpoints. HoloSoma is pinned at
  `20699ff` and bound in the reproducibility index, so the export is an snmr-side wrapper in the
  style of `wbt_latent.py` (`snmr/integration/wbt_curriculum.py`, to write), not a fork edit.
- **Design:** ≥ 20 clips; leave-clips-out CV; feature groups as above; gate **incremental R² ≥ +0.10
  over `kin` on held-out clips** for hazard. z-features must be evaluated with PCA fit on training
  clips only (already the case) and additionally with per-clip mean-centering to strip identity.
- **Data source decision (owner):** the pool currently training under `~/whole_body_tracking`
  (`config/motion_pool_train_converted.yaml`) is the natural label source if its per-bin failure
  counters can be logged; otherwise E67-style teachers on the LAFAN1 subset.

### 4.3 E2 — proactive curriculum

Arms: uniform / failure-EMA (default) / EMA warm-started from E1 prediction / warm-start from
kinematic-only prediction. Gate: ≥ 20 % fewer samples to 0.8 macro completion, 3 seeds, matched
final performance. Implementation: one-line initialisation of `bin_failed_count` from a predicted
vector, via the same snmr-side wrapper. Runs only after E1-proper passes.

---

## 5. E3 — SNMR v2 heads (enabler)

`snmr/model.py` already has the N6 per-node contact head (`predict_contact`, off by default; +1.5 cm
MPJPE at weight 0.5 in earlier runs). Plan: sweep weight ∈ {0.05, 0.1, 0.2} and a stop-gradient
variant; add a difficulty head regressing free byproducts (teacher–student FK residual, min limit
margin, foot skate, contact-switch density — the same quantities as E1's `ret` group, so E1 and E3
share one feature module); add an InfoNCE future-consistency objective predicting z_{t+Δ}, Δ ≤ 0.5 s.
**Freeze rule (register before training):** maximise held-out contact F1 subject to MPJPE ≤ v1 + 1.0 cm;
rerun the full §V probe battery on the frozen v2. GPU: SNMR retrains, moderate; not before E78 seed 0.

---

## 6. Built today (all CPU-verified; 20 new tests pass; full suite unaffected)

| artefact | purpose |
| --- | --- |
| `snmr/integration/fusion.py` | `ReferenceDropoutMasker`, `FusionCommandStudent` (concat/film/gated, flag bits, `flag_dim=0` loads frozen E70 students), `paired_dropout_summary`, `dropout_hazard`, `ramp` |
| `scripts/derive_e78_trainer.py` → `scripts/train_e78_masked_fusion.py` | E78 trainer derived from the frozen harness by asserted replacements; staleness test |
| `scripts/run_e78_masked_fusion.sh` | `train|sweep|frozen SEED TAG`; 26,000-MiB gate; writes only under `/data/robotixx/snmr-research/e78_masked_fusion/` |
| `scripts/analyze_e78_dropout.py` | paired, cluster-bootstrap, matched-subset analysis (E77 discipline) |
| `scripts/e1_retarget_difficulty.py` | E1 features/labels/CV; pilot outputs above |
| `tests/test_fusion.py`, `tests/test_e78_analysis.py`, `tests/test_e1_retarget_difficulty.py` | 20 tests |
| `paper/main.tex` | §7 edits; still 8 pages (references end mid-page 8) |

**When a ≥26,000-MiB window opens** (order matters; each is one command):

```
scripts/run_e78_masked_fusion.sh frozen 0 explicit      # sanity: clean must reproduce 0.923
scripts/run_e78_masked_fusion.sh frozen 0 snmr          #          and 0.699
for t in mE mZf mTf mShf mZc mZg mS; do scripts/run_e78_masked_fusion.sh train 0 $t; done
for t in mE mZf mTf mShf mZc mZg mS; do scripts/run_e78_masked_fusion.sh sweep 0 $t; done
python scripts/analyze_e78_dropout.py \
  --treatment /data/robotixx/snmr-research/e78_masked_fusion/seed0_mZf:d_prior_explicit_snmr \
  --reference /data/robotixx/snmr-research/e78_masked_fusion/seed0_mE:c_prior_explicit
```

The frozen sanity cells are mandatory before anything is interpreted (E72's rule).

---

## 7. Submission hardening done today (2026-09-15 track)

- §IV-B: the additive null is stated as entailed (I(a*; z_ret | x, g) = 0 under deterministic-teacher
  action-MSE) and the regimes where additive value *can* exist are named as untested.
- Limitations: registered contrasts A–T/A–S are window-matched; the A/C gap confounds representation
  with a 100 ms horizon and is not a registered contrast; window-matched explicit arm is future work.
- Build verified with the repo's tectonic path; 8 pages, letter, references end mid-page 8.
- **Not done, GPU-gated:** the C-future arm itself. If a window opens *before* the text lock and the
  owner wants it, it needs the future-`motion_command` lookup from §3.2 first — the same plumbing E78
  needs — so build it once, in `snmr/integration/`, and use it for both.

---

## 8. Owner decisions requested

1. **GPU tenancy for E78 seed 0** (≈ one night). Nothing runs until you free ≥ 26,000 MiB.
2. **E1-proper label source:** log per-bin failure counters from the `whole_body_tracking` pool runs
   (needs a small hook there), or train E67-style teachers on a LAFAN1 subset here.
3. **Order of GPU spend after E78 seed 0:** E78 seeds 1–2 (Track A) before or after E3 v2 heads.

Recommendation: 1 → E78 seed 0 in the next free night; 2 → the pool runs (labels are free);
3 → E78 seeds first (they decide whether Track A is a paper).
