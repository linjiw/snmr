# Contact-mask audit, version 2: count-first metrics and a label-source audit

**Date:** 2026-09-10
**Status:** corrected measurement of a frozen artifact plus a new label-source audit. The
frozen 2026-07-14 audit (`runs/gate1b/mask_audit.json`) is unchanged and remains the record of
what Gate 1b was decided on. Every new number here has its own versioned artifact.
**Generators:** `scripts/audit_contact_masks.py` (v2), `scripts/audit_contact_oracle_labels.py`,
`scripts/build_locomotion_dev_set.py`; metric library `snmr/contact_audit.py`
(tests: `tests/test_contact_audit.py`).
**Provenance of this run:** commit `f4411ae3247e49b454c8b978c2c5aaf210f8d7db`, working tree
dirty; the tracked diff at the start of the sprint is
`reproducibility/reports/sprint_2026-09-10/tracked_dirty_diff_at_f4411ae.patch`
(`provenance_snapshot.json` alongside it). The auditor that produced the frozen report is
byte-identical to the pre-patch script in that commit (sha `7ff2bad4…`, recorded in the
frozen JSON).

## 1. The implementation defect

`mask_agreement` in the frozen auditor computed F1 through precision and recall and returned
`NaN` when `precision + recall == 0` (zero overlap with real errors, TP = 0, FP > 0, FN > 0)
and when either ratio was undefined (oracle-empty windows with candidate positives). `pooled`
then averaged the surviving per-window *ratios*, dropping `NaN` independently per metric,
while its docstring claimed confusion recomposition.

Consequences, all reproduced by `tests/test_contact_audit.py`:

- a failed window (F1 = 0 by the count definition) vanished from the F1 average;
- P, R and F1 were averaged over different window populations, so the reported triple could
  not describe any confusion matrix (frozen `jumps1_subject2`: P 0.057 from 6 windows, R 0.890
  from 1 window, F1 0.493 from 1 window);
- the two-window counterexample (one correctly detected contact; one window of 100 false
  positives) gives pooled F1 = 1.0 against a count F1 of 2/102.

The corrected definition is F1 = 2TP / (2TP + FP + FN) whenever the denominator is positive.
The only undefined case is the all-empty window (TP = FP = FN = 0), whose convention is an
explicit parameter (`--all_empty_policy`, default `undefined`, counted and reported, never
silently dropped).

## 2. What could and could not be recomputed here

The two checkpoints the frozen audit used are not on this host
(`runs/phase1_g1_large/ckpt_100k_final.pt`, sha `df9b56ac…`, and
`runs/gate1_g1/screen/c1_bce_seed0/ckpt.pt`, sha `ac0bd64a…`). The frozen JSON stores only
pooled ratios, not counts, so the model-dependent masks (`decoded_height`, `predicted_t0.5`,
`predicted_hyst`) **cannot be recomputed from the archived artifact**. They are listed as
skipped, with the reason, in the v2 output.

The checkpoint-free masks need only the pair NPZs and teacher forward kinematics, and all seven
pair files are present (sha256 recorded per clip). They were recomputed on the identical
42-window protocol. **The v2 run reproduces the frozen legacy aggregates to four decimals**
(source_contact 0.1317 / 0.9446 / 0.3587; window192 0.4267; window64 0.3699; identical
ground-gap statistics), which confirms the same inputs and code path.

Artifact: `runs/gate1b/mask_audit_v2_2026-09-10.json` (raw TP/FP/FN/TN per clip, window and
foot; micro, frame-level, labelled macro and legacy aggregates; skipped-mask list).

## 3. Corrected numbers for the deployable `source_contact` mask

Micro = ratios of summed counts over all 42 windows (the frozen protocol's windows do not
overlap, so the window-weighted and frame-level micro estimands coincide). "Legacy" is the
frozen rule reproduced from the same counts.

| Clip | micro P | micro R | **micro F1** | legacy F1 (windows contributing) | oracle prevalence | windows with any oracle positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| walk1_subject5 | 0.710 | 0.933 | **0.806** | 0.799 (6/6) | 0.482 | 6 |
| dance2_subject4 | 0.004 | 1.000 | **0.009** | 0.044 (1/6) | 0.003 | 1 |
| fight1_subject3 | 0.000 | undefined | **0.000** | undefined (0/6) | 0.000 | 0 |
| run2_subject1 | 0.040 | 0.965 | **0.078** | 0.101 (5/6) | 0.025 | 5 |
| jumps1_subject2 | 0.051 | 0.890 | **0.097** | 0.493 (1/6) | 0.036 | 1 |
| sprint1_subject4 | 0.061 | 0.856 | **0.113** | 0.154 (4/6) | 0.051 | 4 |
| aiming2_subject3 | 0.032 | 1.000 | **0.062** | 0.180 (2/6) | 0.025 | 2 |
| **aggregate (42 windows)** | 0.123 | 0.929 | **0.218** | 0.359 (19/42) | 0.089 | 19 |

Reading: the aggregate F1 the closure was argued from (0.359) came from 19 of 42 windows; the
count-first value over all 42 is 0.218. On `walk1_subject5` the correction is small
(0.799 → 0.806) because every window there had oracle positives. `jumps1_subject2`'s 0.493
was one window. `fight1_subject3` now has a defined F1 of 0: the candidate predicts stance on
63 % of samples and the oracle labels none.

The two teacher-height ground variants move the same way (window192: 0.427 → 0.253;
window64: 0.370 → 0.228 aggregate F1).

## 4. Label-source audit: why prevalence is near zero on six clips

`scripts/audit_contact_oracle_labels.py` inspects the oracle's ingredients per clip and foot
without any checkpoint (`runs/gate1b/oracle_label_audit_2026-09-10.json`;
walk-family run in `oracle_label_audit_walkfamily_2026-09-10.json`). The oracle is the ankle
body origin height from teacher FK, normalized by **that foot's clip-wide minimum**, with
enter ≤ 0.03 m / exit < 0.05 m.

Findings (per-foot values; "plateau" is the histogram mode of the lowest 5 cm of the height
distribution, i.e. where the foot actually rests):

| Clip | plateau − clip-min (L / R, m) | tilt at the minimum frame (L / R, deg; clip median ≈ 9–17) | oracle prevalence (L / R) | prevalence with the plateau as ground (L / R) |
| --- | --- | --- | --- | --- |
| walk1_subject5 | 0.022 / 0.027 | 5.1 / 13.7 | 0.526 / 0.374 | 0.707 / 0.687 |
| dance2_subject4 | 0.047 / 0.048 | 11.9 / 1.6 | 0.001 / 0.004 | 0.009 / 0.593 |
| fight1_subject3 | 0.053 / 0.003 | 17.2 / **49.9** | 0.002 / 0.001 | 0.154 / 0.001 |
| run2_subject1 | 0.048 / 0.048 | 4.9 / 8.9 | 0.011 / 0.039 | 0.531 / 0.555 |
| jumps1_subject2 | 0.048 / 0.038 | 26.9 / 3.4 | 0.028 / 0.007 | 0.404 / 0.101 |
| sprint1_subject4 | 0.048 / 0.043 | **37.9** / **31.3** | 0.013 / 0.052 | 0.753 / 0.782 |
| aiming2_subject3 | 0.048 / 0.048 | 8.6 / 15.6 | 0.023 / 0.036 | 0.637 / 0.704 |

Mechanism, stated as what was measured: on the six low-prevalence clips the per-foot clip
minimum lies 4–5 cm **below** the height at which the foot rests, and at that minimum frame the
foot is often strongly tilted or the origin is below zero (fight1 −0.030 / −0.050 m, dance2
−0.041 m). Because the enter threshold is 3 cm above the clip minimum, ordinary stance at the
plateau never enters the mask. On walk1_subject5 the gap is 2–3 cm, so stance is (mostly)
caught — and even there the right foot is under-labelled (0.374 vs 0.687 with the plateau).

What this does and does not establish:

- It **does** establish that the oracle's ground reference is a single-frame statistic that
  is not robust to one tilted or penetrating frame, and that this is what drives the
  prevalence split, not a property of fighting or dancing.
- It does **not** establish that a plateau ground fixes the label. The plateau detector itself
  fails where the lower tail is dominated by a dip (dance2 left 0.009, fight1 right 0.001,
  jumps1 right 0.101). Nor is any height-only heuristic force-verified contact.
- The teacher and human streams agree on convention (30 fps, equal frame counts, z up, robot
  pelvis ≈ 0.7 m, human toe height plausible); no frame or timestamp defect was found.
- Independently reviewed contact intervals do not exist in the repository. Until they do,
  every contact number is proxy-vs-proxy.

## 5. Locomotion development set (all clips marked development)

`reproducibility/reports/locomotion_dev_set_2026-09-10.json`, rule stated in the file: a
walk clip is *oracle-valid* iff for both feet the clip minimum is within 0.030 m of the
stance plateau, oracle prevalence is in [0.30, 0.80], and there are ≥ 50 stance runs.

| Result | Clips |
| --- | --- |
| **valid walk (4)** | walk1_subject1 (tracker-seen), walk1_subject2, walk1_subject5 (tracker-seen), walk2_subject4 |
| rejected walk (8) | walk2_subject1, walk2_subject3, walk3_subject1 (E67 specialist failed its gate), walk3_subject2, walk3_subject3, walk3_subject4, walk3_subject5, walk4_subject1 — all for an unsound ground reference on at least one foot |
| run/sprint (6) | outside the walking prevalence band by construction; a run-family band is not declared |

The important consequence: **"the oracle behaves on walking" is not a property of the walk
family.** It holds on 4 of 12 walk clips. Any locomotion pilot scored with this oracle must
draw from the valid list and say so.

## 6. Affected-claim list

| Claim / location | Status after v2 |
| --- | --- |
| `runs/gate1b/PROTOCOL.md` §"Pre-study audit summary" table (aggregate P/R/F1 for all six masks) | numbers are the legacy estimand; labelled as such here; the frozen file is left unchanged |
| PROTOCOL.md M2 selection rule "higher aggregate F1 between predicted_t0.5 and predicted_hyst" (0.3902 vs 0.3871) | decided under the legacy estimand; **not recomputable** here (checkpoints absent); the v2 auditor records both the legacy-rule and micro-rule choices when the checkpoints are present |
| `docs/CONTACT_MASK_AGGREGATE_ARTIFACT_2026-09-10.md` per-clip table | P/R are legacy ratio means; F1 column is inflated on dance2/jumps1/sprint1/aiming2/run2 (see §3); the qualitative split (walk1 vs the rest) survives |
| `docs/program/README.md:112`, `METHOD_SPEC…:172`, `EXPERIMENT_PROGRAM…:138`: "walk1_subject5 … F1 0.799" | interpretation correction: micro F1 is 0.806; the *scope* claim "the mask is good on walking" is narrowed to the 4 oracle-valid walk clips |
| CONTACT_MASK_AGGREGATE §"the oracle is broken on six of seven clips" | supported mechanistically (§4) but as "ground reference unsound", not "oracle wrong on those motions"; six clips are not shown to be mislabelled by force truth |
| Gate 1 / Gate 1b closure ("every deployable mask fails") | unchanged in outcome; the mask-quality aggregate it cited is now known to be a ratio mean over 19/42 windows |
| E70 completion / ambiguity results and every renderer-generated paper number | **not affected**; nothing here touches those artifacts and this audit must not be propagated into them |

## 7. What the milestone acceptance requires, and what is still open

Passing count/edge-case tests: done (13 tests). Corrected, provenance-linked contact report:
this document plus the v2 JSON. Small, valid locomotion development set: four clips, with the
rule and the rejections on record.

Open: independently reviewed contact intervals on at least the four valid clips; and
recomputation of the three model-dependent masks on a host that holds the two checkpoints
(the v2 auditor runs unchanged there without `--allow_missing_checkpoints`).
