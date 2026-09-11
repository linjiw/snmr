> **Amended 2026-09-10 — read [`EXPERIMENT_PROGRAM_AMENDMENT_2026-09-10.md`](EXPERIMENT_PROGRAM_AMENDMENT_2026-09-10.md) first.**
> The row-minus-column reading of the crossed matrix is withdrawn; Panel-F tracker-training
> clips are tracker-seen, not held out; the 500-step horizon is a 10-s window; the two-arm
> development comparison replaces the 6 × 6 matrix as the next step; the three-seed E70
> variance is a pilot estimate, not a universal constant; scaling in N is secondary. The text
> below is preserved as written for the record.

Everything is verified against the machine. Here is the program.

---

# SNMR Cross-Embodiment Experiment Program
**Frozen 2026-09-10** · single RTX 5090 (shared, ~9 GB free right now) · /data 118 GB free

Three artifacts were built and committed as `bc43ba5` while designing this; they are the program's frozen inputs, each with its generator so a clean checkout reproduces the hashes:

- `/home/robotixx/snmr/reproducibility/reports/xembodiment_split_manifest_2026-09-10.json` (sha `063b6da8…`) — gen: `/home/robotixx/snmr/scripts/build_xembodiment_split.py`
- `/home/robotixx/snmr/reproducibility/reports/xembodiment_morphology_distance_2026-09-10.json` — gen: `/home/robotixx/snmr/scripts/build_xembodiment_morphology_distance.py`
- `/home/robotixx/snmr/reproducibility/reports/xembodiment_robot_annotation_2026-09-10.json` (sha `9cb0c73b…`) — gen: `/home/robotixx/snmr/scripts/build_xembodiment_annotation.py`

## 0. Five facts measured today that constrain every choice below

1. **Zero-new-parameters is real and already verified.** `scripts/verify_zero_new_parameters.py` runs green: one **847,361-parameter** `KinematicMorphoRetargeter` serves 7 robots spanning **19–29 DoF**; only output width tracks the robot. The decoder is a shared per-node scalar head (`nn.Linear(hidden,1)` at `snmr/morpho_model.py:239`) under `node_mask`. This is the one thing AdaMorph (per-robot prompt bank + per-robot output head) and X-Morph (per source-target pair) cannot say. **Verify-before-writing is discharged.**
2. **All five robots parse and batch today.** With hand-declared `SemanticManifest`s they tokenize into a single `(5, 38, 24)` padded batch, `joint_mask` sums `[29, 27, 24, 23, 22]`. The multi-robot *data* path exists; only the multi-robot *training loop* is missing (`FixedTargetMorphoRetargeter` hard-rejects batch>1 at `snmr/morpho_integration.py:135`).
3. **The repo's current motion split leaks, and I quantified it.** All **13/13** subject5 validation clips share a LAFAN1 sequence label with a training clip (`walk1_subject5` held out while `walk1_subject1` and `walk1_subject2` train). An "unseen motion" claim on that split is not defensible.
4. **holosoma WBT exists only for G1.** `config_values/wbt/` contains `g1` and nothing else. T1 tracking runs only through the custom overlay `snmr/integration/morphoret_t1_tracker.py` at a measured **15.3 GPU-h/run** (PhysX, 8192 envs, 40k iters). PM01/N1/Toddy have no holosoma robot package at all. **This forces the downstream experiment onto G1 as the held-out robot** — and independently confirms the EVIDENCE dossier's warning against making T1 the held-out robot while a T1 tracker is the verifier.
5. **The cheap tracker recipe is measured.** E67: `exp:g1-29dof-wbt simulator:mjwarp`, 512 envs, 8000 PPO iterations = **50 min = 0.83 GPU-h** (`train.log` 04:49:40→05:39:40 and 05:39:56→06:30:26), and it passes a ≥0.80 completion / ≥9.0 s gate. This single number makes the downstream experiment affordable.

Other measured units: MorphoRetarget training **10.1–22.9 steps/s** (`a1_screen_*/training_summary.json`) → **1.2 GPU-h per 50k-step run** budgeted; GMR teacher generation **77 clips × 1 robot ≈ 2 min wall on 20 cores = 0.7 CPU-h, 0 GPU-h** (from `/data/robotixx/pairs` mtime spans).

---

## 1. Split design

### 1.1 Motion partition (frozen; sha `063b6da8…`)

Nine coarse **action families** merge near-neighbour LAFAN1 sequence labels so that holding out a family cannot leave a near-duplicate in training (`fight`+`fightAndSports` → combat; `push`+`pushAndFall`+`pushAndStumble` → perturb; `walk`+`run`+`sprint` → locomotion; `fallAndGetUp`+`ground` → fall_recover).

| Partition | n | Contents |
|---|---|---|
| **Train** | 45 | locomotion, obstacle, dance, fall_recover — subjects 1–4 only |
| **Panel F** — familiar family, unseen clip & subject | 9 | `dance2_s5, fallAndGetUp1_s5, ground1_s5, obstacles1_s5, obstacles2_s5, obstacles6_s5, run1_s5, walk1_s5, walk3_s5` |
| **Panel U** — unseen family | 19 | aiming (5), jump (3), combat (5), perturb (6); of which 4 are also subject5 |
| **Quarantine** | 4 | `multipleActions*` — mixes categories, assignable to no family, excluded from train *and* eval |

Asserted in the generator and re-checked at load: train ∩ panel_U families = ∅; subject5 ∉ train; the three sets are pairwise clip-disjoint.

### 1.2 Robot partition

Leave-one-robot-out over **all five folds** (H-Zero discipline — with N=5, one favourable held-out robot reads as cherry-picking). Measured scale-free morphology distance and the analytic scale ratio:

| Held out | leg (m) | nearest training robot | distance | leg ratio | a-priori stratum |
|---|---|---|---|---|---|
| engineai_pm01 | 0.785 | fourier_n1 | **0.941** | 1.179 | near |
| fourier_n1 | 0.666 | engineai_pm01 | **0.941** | 0.848 | near |
| unitree_g1 | 0.766 | booster_t1_29dof | 1.133 | 1.172 | mid |
| booster_t1_29dof | 0.654 | unitree_g1 | 1.133 | 0.854 | mid |
| stanford_toddy | 0.300 | booster_t1_29dof | **1.667** | **0.459** | **extrapolation** |

Toddy is a 3.4 kg, 2.2×-smaller outlier. It is declared an extrapolation stratum **before any run**, so a Toddy failure is a scale-extrapolation finding, not a silent LORO failure — and so it cannot be quietly dropped after the fact.

### 1.3 The 3-cell table

| | **Panel F** (familiar family, unseen clip+subject) | **Panel U** (unseen family) |
|---|---|---|
| **Seen robot** (in training) | Cell D — reference floor | **Cell A: seen robot / unseen motion** |
| **Unseen robot** (held out) | **Cell B: unseen robot / familiar family** | **Cell C: unseen robot / unseen motion** |

A−D isolates motion generalization at fixed robot; B−D isolates robot generalization at fixed motion; C−D is the conjunction; and **C − (A−D) − (B−D) − D is the interaction** — the quantity that says whether the two generalizations compound or are independent. Report the interaction explicitly; it is the number that distinguishes "a phenomenon" from "two additive effects."

### 1.4 Leakage prevention, by channel

| Channel | Mechanism | Enforcement |
|---|---|---|
| **Clip** | train/F/U pairwise disjoint | assertion in `build_xembodiment_split.py`; manifest sha re-checked by every consumer |
| **Subject** | subject5 never trains | assertion; Panel U additionally stratified into subject-seen (15) vs subject-unseen (4) and **both strata reported** — if they differ, subject leakage is live and is disclosed rather than averaged away |
| **Augmentation-descendant** | no augmented clip may cross the partition | every derived reference (repaired, corrected, tracker-stitched) inherits its **parent clip id** in the npz; the loader refuses any derived clip whose parent is on the other side. `multipleActions` quarantined because it is a category-level descendant of several families |
| **Embodiment** | held-out robot's MJCF, RobotSpec, GMR IK config, teacher labels, and semantic anchors are all absent from every training input | fold manifest lists allowed `robot_spec_sha256`; the trainer raises on any token batch whose source hash is not on the list. `RobotGraphTokenizer` already carries `source_spec_hashes` for exactly this |
| **Ruler** | the human-side semantic ruler must not be fit on held-out material | `morphology_distance` normalization fit on **training robots only** (already the A2 rule); per-anchor thresholds frozen at the `snmr/semantic_metrics.py:267` values |

---

## 2. Baselines

Primary metric for all: **`keypoint_error_normalized_mean`** from `snmr/semantic_metrics.py` — scored human-source → target FK, *never* against the teacher (`independence_rule`, already in the A2 prereg). GMR is a baseline, not the ruler.

| # | Arm | What it needs | In repo today? |
|---|---|---|---|
| **B1** | **Configured GMR per-robot** | `bvh_lafan1_to_<robot>.json` (hand-tuned `human_scale_table` + `ik_match_table` weights/offsets) + robot MJCF | ✅ **Fully.** All five IK configs and MJCFs present at `/data/robotixx/snmr-externals/GMR/`; teacher pairs already generated for all 5×77. 0 GPU-h. |
| **B2** | **Strong nearest-embodiment transfer** | see §2.1 | ❌ **Not implemented.** Preregistered in `a2_prereg.json` but only for *same-topology* variants ("exact identity joint map"), which does not exist cross-robot. ~1 day. |
| **B3** | **Robot-ID-conditioned** | one learned embedding per training robot, no RobotSpec numerics; at test assign the registered nearest training identity | ❌ Needs the multi-robot loop + an ID embedding table. ~0.5 day on top of B5. **This is the AdaMorph analogue and the arm that proves conditioning is spec-derived, not identity-derived.** |
| **B4** | **Geometry-only MorphoRetarget** | the current `KinematicMorphoRetargeter` trained multi-robot; no correction, no physics | ⚠️ **Model yes, loop no.** Model verified across 7 robots; `FixedTargetMorphoRetargeter` binds exactly one robot. ~2 days to unbind. |
| **B5** | **Simple repaired references** | GMR output + windowed C6 projection / DLS foot-lock / Gaussian low-pass, under the deployable contact mask | ✅ **~90%.** All five corrector families implemented (`snmr/projection.py`, `snmr/footlock.py`, `snmr/polish.py`, `snmr/flow.py`, `snmr/losses.py`); harness `scripts/eval_footlock.py` with guards + 2000-sample clip bootstrap. Missing: a *motion-level* Gaussian low-pass arm (~15 lines; E22 says it will be near-inert — keep it as the **accessible-but-useless negative control**). |
| **B6** | **Full method** | B4 + a transferable correction learned from physics feedback | ❌ **Not buildable today.** Blocked by design: `MORPHORETARGET_PLAN_2026-08-29.md:63` — "No physics critic or preference labels before the frozen-tracker verifier qualifies," and no tracker is qualified. **Phase 2.** |
| **B7** | **Per-robot supervised upper bound** | fresh fixed-target model trained on the held-out robot's *own* GMR labels, train clips only | ✅ **Runs today.** `scripts/train_morpho_g1.py` + `FixedTargetMorphoRetargeter` is exactly this. Not a candidate — a ceiling. |

Plus two compute-matched per-robot rivals the NOVELTY dossier says the paper fails without: **B8 = G-DReaM-style fine-tune** (add the held-out skeleton, ~5 GPU-h) and **B9 = ReActor-style bilevel** (~6 GPU-h). The claim that must survive is *equal or better at strictly lower compute on the new robot*. Amortization, not accuracy, is the result.

### 2.1 B2 specified precisely enough to implement

Frozen in `xembodiment_robot_annotation_2026-09-10.json`:

1. **Source selection.** Nearest training robot by the scale-free descriptor (5 shape ratios `arm/leg, torso/leg, thigh/leg, shank/leg, shoulder_width/leg` ∥ 19 joint ROM widths in rad), each channel divided by its training-set std, distance = RMS of differences. Tie-break: lexicographic. *Scale-free is the point* — B2 removes pure scale analytically, so the residual distance must not re-encode it.
2. **Joint angles.** Copy through the **19-role common correspondence** (L/R × {hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll}, waist_yaw, L/R × {shoulder_pitch, shoulder_roll, elbow}), then clamp to the target's own `jnt_range`. Report clamp-activation fraction per joint.
3. **Uncorresponded DoF.** Held at nominal pose, and the **uncovered-DoF fraction is reported**: G1 10/29, T1 8/27, PM01 5/24, N1 4/23, Toddy 3/22. B2 is strongest exactly where this fraction is lowest — say so rather than let a reviewer find it.
4. **Root.** Copy heading-relative orientation unchanged; multiply all three root-translation coordinates by `leg_target / leg_source`, where leg length = mean over L/R of the summed parent→child anchor lengths on the RobotSpec path pelvis→ankle_roll. Measured ratios: G1←T1 1.172, T1←G1 0.854, PM01←N1 1.179, N1←PM01 0.848, Toddy←T1 0.459.
5. **Limb scaling.** Segment-wise, not global: thigh and shank each scaled by their own target/source ratio before IK-free angle copy, so a robot with an unusual thigh:shank split is not silently mis-served.
6. **Disclosure.** The 19-role map is **manual annotation** and is declared as such. B2 and the candidate pay this cost symmetrically — that symmetry is what makes the comparison fair.

---

## 3. The zero-shot protocol

Applies to the held-out robot in cells B and C.

### FORBIDDEN at test time — any one of these voids the fold

1. **Teacher labels on the held-out robot.** No GMR output for that robot in any form: not as supervision, not as initialization, not for model selection, not for normalizer statistics, not for scale calibration.
2. **Correction rollouts on the held-out robot.** No simulator trajectory, no privileged policy output, no stitched-back sim state (the E50-A operator).
3. **Adapters, prompts, ID embeddings, or output heads** instantiated for the held-out robot. *Enforced structurally, not by policy*: `verify_zero_new_parameters.py` must report an identical 847,361 parameter count for the held-out robot, and the fold manifest records it.
4. **Fine-tuning or any gradient step** after the fold's frozen checkpoint hash.
5. **Outcome-driven tuning.** No threshold, hyperparameter, checkpoint, clip, or held-out-robot choice may change after any held-out number is observed. Checkpoint selection is the **fixed 50,000-step endpoint**, never validation-based (inherited from `a2_prereg.json`).
6. **The held-out robot's IK config.** `bvh_lafan1_to_<robot>.json` encodes hand-tuned per-body scales (PM01: 0.85 legs / 0.80 arms). Reading it is reading a human's tuning for that robot.

### MUST BE SEPARATELY DISCLOSED — legitimate, but not free

| Item | Disclosure |
|---|---|
| **Target-robot tracker training** | The downstream verifier is a G1 policy trained on G1. **The retargeter is zero-shot; the controller is not.** This sentence appears in the same paragraph as every downstream number, or the result reads as zero-shot control. Report the tracker's own budget separately from the retargeter's. |
| **Manual semantic correspondences** | The 7-role `SemanticManifest` and the 19-role joint map (sha `9cb0c73b…`). Measured cost: ~20 min for all five once the body lists are in hand — but Toddy (`ank_roll_link` / `ank_roll_link_2`, `hand` / `hand_2`, gear bodies `2xl430_gears_*`) and T1 (`AL1`…`AL6`, `H1`, `H2`) required reading the kinematic tree, not the names. **The site's unsourced "1–2 days per robot" should be replaced by this measurement.** |
| **Hand-measured metric anchors** | The `SemanticAnchors` with metric `robot_point_local_m` exist for **G1 only** (`snmr/g1_semantics.py`). Held-out-robot scoring needs them for the held-out robot. Deriving them from realized inertial centres and contact-support centroids is the A2 `geometry_derived_points` rule — mechanical, but it is annotation and is declared. |
| **Ruler thresholds** | Frozen two-sided bands amplitude [0.50, 1.50], energy [0.25, 4.00], jitter [0.25, 4.00], registered before any held-out run. |
| **Fold-independent recipe** | One config hash shared by all folds and arms. |

### Negative control, run before the framing is committed (NOVELTY, non-negotiable)

**Shuffled-spec test.** At test time, feed the held-out robot's motion with a *mismatched* RobotSpec (each fold's spec permuted to another robot's). If held-out performance does not degrade, the conditioning is inert and the gains are just a better average retargeter. **Eval-only on frozen checkpoints, ~0.5 GPU-h.** This is the single cheapest experiment that can kill the paper, so it runs first.

---

## 4. The downstream-learning experiment

**Held-out robot = G1.** Forced by fact 4 (only G1 has a native holosoma WBT config) and independently correct by fact 5 (0.83 GPU-h vs 15.3). G1 is held out of the retargeter's training set; the retargeter trains on T1/PM01/N1/Toddy. G1's nearest neighbour is T1 (d=1.133), which is in training — so B2 is a genuinely strong rival here.

**Reference motion set = walk-family, 3 clips** (`walk1_subject5`, `walk3_subject5`, `run1_subject5` from Panel F). Justified by REPAIRABILITY: the deployable contact mask is P 0.711 / F1 0.799 on walk1 vs P 0.000 on fight1, and E53's 8-clip heterogeneous tracker failed its 0.6 gate three times (0.378 / 0.465 / 0.478) — importing that heterogeneity buys no science. Panel-U clips enter only through the **evaluation** panel, never as a training target for the tracker.

### 4.1 Matched recipe — one hash, no variation

`exp:g1-29dof-wbt simulator:mjwarp`, 512 envs, **8000 PPO iterations**, `save-interval 1000` (8 curve points), `E51_JOINT_POS_WEIGHT=1.0`, `E51_JOINT_POS_SIGMA=0.5`, `--randomization.ignore-unsupported True`. Evaluation seed 404, exact-state resets via `snmr/integration/wbt_bodyfix.py`, deterministic linspace start grid with the realized grid asserted.

**Masked/outage training is pinned at ONE level across every arm.** The repo's own single-seed E80-A result is **+0.447** at the severest dropout cell (independently recomputed from raw rollouts: 0.1064 → 0.5537). That is **2.3× the largest representation effect ever measured here** (+0.191) and **4.7× the effect this experiment is powered to detect at K=5** (§5). If it varies by arm it is not a confound, it *is* the result. One config hash, recorded in every policy's manifest, asserted equal across arms before analysis.

### 4.2 The crossed matrix

Six reference sources, each generating (a) a G1 tracker training reference set and (b) a G1 evaluation reference set:

`S1` GMR-G1 configured · `S2` strong nearest transfer T1→G1 · `S3` robot-ID model, nearest ID · `S4` simple repaired (footlock on GMR-G1) · `S5` full method zero-shot · `S6` per-robot supervised upper bound

**Training source × evaluation source = 6 × 6 = 36 cells**, on the 3-clip walk panel. Reading the matrix:

- **Diagonal alone is the trap** the advisor names: a source that generates easier motions wins its own diagonal cell for free.
- **Column means** (fix the evaluation reference, vary the training source) are the honest comparison — *this is the primary endpoint*.
- **Row means minus column means** measures how much easier each source's own references are. A source with a high row mean and a low column mean has bought executability by simplifying the motion. Cross-check against the low side of the `amplitude_ratio` band (< 0.50) — the advisor's stated kill criterion, *the correction turned the motion into conservative standing*.

**Plus the common held-out evaluation panel**, which no arm generated: **Panel F (9 clips) + Panel U (19 clips) = 28 clips**, retargeted to G1 by a *single* fixed reference source held constant across every policy. Preregister the panel's source as **S1 (configured GMR-G1)** — the field-standard, arm-neutral choice — and state that every policy is evaluated on references it did not produce.

### 4.3 Endpoints

1. **Full-clip success** — completion rate over a 500-step horizon, E67 gate ≥0.80 completion / ≥9.0 s survival, on the common panel. Primary.
2. **AUC of the learning curve** — trapezoid over the 8 checkpoints (1k…8k iters) of completion vs environment steps, normalized by budget. Robust to where a curve happens to end.
3. **Steps-to-threshold** — first checkpoint reaching 0.50 completion (`GMR_COMPLETION_FLOOR` from `scripts/analyze_wbt_rollouts.py`), right-censored at 8k with censoring reported, not imputed.

The E77 rule governs threshold choice: **run the 2-arm pilot first** and confirm no arm is saturated or floored at 8k iterations. A curve measured outside the functional window cannot be interpreted no matter how well the rest is executed.

### 4.4 Seeds

**K = 5** (3 as the gating screen). §5 shows K=3 detects ≥0.123 completion units and K=5 detects ≥0.095. The effects in play are +0.447 (recipe — pinned), ~+0.19 (representation, historical), and an unknown downstream effect plausibly ~0.10. K=3 cannot resolve 0.10; K=5 barely can. **Do not spend the difference on more rollouts** — §5.

---

## 5. Statistics

### 5.1 What varies where

```
y[s,k,c,r] = μ + α[s] + b[k(s)] + g[c] + (αg)[s,c] + ε[s,k,c,r]
             ^      ^       ^        ^        ^          ^
          source  TRAINING  clip  source×clip      rollout
          (fixed)  SEED   (random) interaction    (nuisance)
                  (random)
```

- **Between policies:** `α` (the effect) and `b[k(s)]` (training-seed noise — different init, different env stream, different PPO trajectory). Shrinks **only** with more seeds.
- **Between rollouts:** `ε` — different start phase, different reset. Shrinks with more rollouts, and is **already negligible**.

E76 settles the rollout term empirically: **~18% of individual rollout outcomes flip on re-run under identical inputs, while the arm-level contrast agrees to 0.003.** Per-arm rollout noise is real and per-contrast rollout noise is not. At R = 512–1024 the `σ²_rollout/(K·C·R)` term is already far below the others; **more rollout repeats cannot shrink between-policy uncertainty.**

### 5.2 The variance components, computed from this repo's own three-seed data

From `reproducibility/reports/e70_seed0-1-2_analysis.json`, `snmr_minus_time` per seed = **0.15437, 0.27769, 0.14027**:

| Quantity | Value |
|---|---|
| mean difference | 0.19078 |
| **sd of the per-seed paired difference** | **0.0756** |
| SE at K=3 (seed-level) | 0.0436 |
| seed-level t (df=2) | 4.37, **p ≈ 0.048** |
| published clip-cluster-bootstrap SE | 0.0383 |

Cross-checked on `snmr_minus_shuffled` (0.18711, 0.13056, 0.28054): sd = **0.0757**. Two independent contrasts agree to three decimals on the seed-level sd, so σ_seed ≈ 0.076 completion units is a stable planning constant for this apparatus.

**Two consequences, both load-bearing.** First, the **seed-level SE (0.0436) exceeds the clip-bootstrap SE (0.0383)** — the repo's headline interval is a clip bootstrap and therefore *understates* policy-level uncertainty. Second, the headline result is **p ≈ 0.05 at the policy level** even though its clip interval looks comfortable. Both must be reported, and between-policy claims must use the seed-level SE.

### 5.3 Allocating n

Minimum detectable paired difference at 80% power, two-sided α=0.05, with σ_seed = 0.076:

| K (seeds) | MDE | verdict |
|---|---|---|
| 3 | 0.123 | screens only; cannot resolve a 0.10 downstream effect |
| **5** | **0.095** | **program default** |
| 8 | 0.075 | needed if the pilot shows σ_seed > 0.076 |
| 10 | 0.067 | not affordable at 0.83 GPU-h/policy × 6 arms |

Allocation rule, in priority order:

1. **Seeds K first** — the only lever on `σ²_seed/K`.
2. **Clips C second** — `σ²_(αg)/C`; paired cluster bootstrap over clips, 10,000 replicates, reusing the frozen design in `scripts/analyze_wbt_rollouts.py` (extract the bootstrap; the script's hardcoded `CLIPS`/`SOURCES`/`NAME_PATTERN` and ec2-user paths make it a template, not runnable code).
3. **Rollouts R last** — fix at 512 per (policy, clip) and never increase it. Extra rollouts buy precision on a term that is already invisible.

**Report both intervals side by side** for every headline contrast: a clip-cluster bootstrap *and* a seed-level paired interval. Where they disagree, the seed-level one governs. Pair on seed index across arms (same seed → same init stream), which is what makes σ = 0.076 a *difference* sd rather than an arm sd.

**Pre-registered power check:** if the X3 pilot's observed σ_seed exceeds 0.076, K rises to 8 before the confirmatory run — decided from the pilot, never from the result.

---

## 6. Cost accounting

Inference latency is the least of it. Every arm reports all six lines; an arm that hides one is not compute-matched.

| Line | Unit | Measured / estimated | Source |
|---|---|---|---|
| **C1 Teacher generation** | per robot, 77 clips | **0.7 CPU-h**, 0 GPU-h | `/data/robotixx/pairs` mtime spans, 1.7–2.4 min wall × 20 cores |
| **C1b Teacher *configuration*** | per **new** robot | **the real cost** — hand-tuned `human_scale_table` + `ik_match_table` offsets/weights. Sunk for these five; unbounded for a sixth | `bvh_lafan1_to_pm01.json` |
| **C2 Correction search** | per clip | Phase 2. CEM/MPPI over B-spline knot deltas is **not** among the five implemented families (~1 day new code); a full-trajectory C6 projection is CPU, minutes/window | `ADVISOR_GUIDANCE:1000-1025`; `snmr/projection.py` |
| **C3 Verifier (tracker) training** | per policy | **0.83 GPU-h** (G1 mjwarp 8k@512) · **15.3 GPU-h** (T1 PhysX 40k@8192) | E67 `train.log`; `morphoret_t1_tracker_20260901/train.log` |
| **C4 Retargeter training** | per 50k-step run | **1.2 GPU-h** (10.1–22.9 steps/s measured) | `a1_screen_*/training_summary.json` |
| **C5 Human onboarding** | per robot | `SemanticManifest` (7 roles) ≈ **20 min/robot measured** for all five today; 19-role joint map ≈ 30 min once; **metric `SemanticAnchors` exist for G1 only** and are the expensive remainder | authored and committed today, sha `9cb0c73b…` |
| **C6 Inference** | per clip | 671 fps CPU — **legacy 8-D model only**, and its "~4× faster than GMR" comparator is **unmeasured** (`teacher_timing: "not measured by this evaluator"`). MorphoRetarget has **no timing field in any artifact**. Do not import | `runs/bench_g1_v2.json`; `scripts/benchmark.py:460` |
| **C7 Storage** | program total | ~30 GB of 118 GB free. Reserve 40 GB; prune non-curve tracker checkpoints after each policy's evals are frozen | measured: e70 5.0 GB, e67 800 MB, T1 tracker 281 MB |

**The amortization ledger** — the table the paper lives or dies on, per *new* robot:

| Method | New-robot cost | Amortizes? |
|---|---|---|
| ReActor | ~6 GPU-h bilevel RL | no |
| G-DReaM | ~5 GPU-h retraining | no |
| OmniTrack | that robot's own privileged policy | structurally no |
| AdaMorph | new prompt bank + new output head + that robot's labels | structurally no |
| GMR | hand-tuned IK config (C1b) | no |
| **This work** | **C5 only: ~20 min annotation, 0 GPU-h, 0 new parameters** | **yes** |

---

## 7. Schedule

Approval threshold: **~20 GPU-h**. Everything runs `nice -n 10` behind `scripts/resume_when_gpu_exclusive.sh` — the other tenant holds ~23 GB of 32.6 GB right now.

| # | Experiment | GPU-h | Approval | Gate to proceed |
|---|---|---|---|---|
| **X0** | **Build.** Unbind `FixedTargetMorphoRetargeter` for multi-robot batches; B2 nearest-transfer; B3 ID-embedding table; crossed-eval harness; delete the `num_motions != 1` guard at `scripts/eval_e67_teacher.py:75-76` and build a per-motion stratified start grid (~60 lines — the multi-motion mapping already exists in `wbt_bodyfix.py:85-133`) | ~1 (smokes) | — | unit tests green; zero-new-parameters count unchanged |
| **X1** | **Repairability pilot, CPU half.** B5 arms 1–3 on the frozen 42-window protocol + the source-intent ruler. **0 GPU, ~2 days.** Runs in parallel with X0 | 0 | — | mask prevalence reported beside precision, per-clip not aggregate |
| **X2** | **LORO screen, seed 0.** B3 + B4 + B7 × 5 folds | **24** | ⚠️ **YES** (splits into 18 + 6 if preferred) | any fold beats B1 and B2 on Cell B |
| **X3** | **Shuffled-spec negative control.** Eval-only on X2 checkpoints | **0.5** | — | **HARD GATE.** Flat = conditioning inert = stop and reframe |
| **X4** | **Scaling curve in N.** N ∈ {1,2,3,4}, 2 nested orderings, seed 0, held-out G1 | **8.4** | — | positive slope; flat slope = transfer claim is not real, learn it here |
| **X5** | **Tracker qualification floor + eval mode.** Write the floor (value, reference set, horizon, rollout count) **before any rollout is observed** — a completed T1 checkpoint has sat unqualified since 2026-09-02; add `--mode eval` to `run_morphoret_t1_tracker.py` | ~1 | — | floor written, reviewed, hashed |
| **X6** | **X7 pilot.** 2 sources (S1 floor, S5 candidate) × 1 seed, 8k iters | **2** | — | **E77 window check**: neither saturated nor floored at 8k |
| **X7** | **Downstream screen.** 6 sources × 3 seeds + crossed matrix + curve evals | **24** | ⚠️ **YES** | column-mean separation exceeds the K=3 MDE of 0.123 |
| **X8** | **LORO confirmatory.** X2 + 2 more seeds | **48** | ⚠️ **YES** | — |
| **X9** | **Downstream confirmatory.** X7 + 2 more seeds (K=5) | **16** | — | — |
| **X10** | **Compute-matched rivals.** B8 G-DReaM-style fine-tune (~5) + B9 ReActor-style bilevel (~6), on the held-out robot | **11** | — | must run, or "just spend 6 h on the new robot" is unanswered |
| **X11** | **Phase 2: correction.** B6 — blocked until X5's verifier qualifies | TBD | ⚠️ **YES** | verifier qualified against a pre-written floor |

**Two nights' work gets the decisive answer.** X0+X1 are CPU/engineering. Then **X2 → X3 → X4 = 33 GPU-h** across two GPU nights, and X3 (0.5 GPU-h) can kill the framing before X8's 48 h are spent. Run X3 the moment X2's first fold lands, not after all five.

### Three things this program deliberately refuses

1. **It does not re-litigate the 8-clip heterogeneous tracker.** 0.378 / 0.465 / 0.478 across three budgets is a settled negative. The walk-family scope is a decision, not an inheritance.
2. **It does not let recipe vary.** +0.447 from masked training dwarfs every representation effect in this program's history. One config hash, asserted equal before analysis.
3. **It does not call B6 "the full method" until it exists.** Phase 1's claim is *geometry-only spec conditioning transfers to an unseen robot* — which, if X3 holds, is already the thing no prior work can evaluate, and is exactly the open question NMR's own limitations section names.