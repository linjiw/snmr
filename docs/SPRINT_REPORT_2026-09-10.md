# Sprint report, 2026-09-10: from specification to one trustworthy (negative-by-exclusion) pilot result

**Revision under which everything here was produced:** commit
`f4411ae3247e49b454c8b978c2c5aaf210f8d7db` on `feat/morpho-retarget-kinematic`, working
tree **dirty** before the sprint began (8 tracked files modified, 49 untracked paths; exact
diff `reproducibility/reports/sprint_2026-09-10/tracked_dirty_diff_at_f4411ae.patch`,
snapshot `provenance_snapshot.json`). Every artifact written in this sprint records its own
`git` block; none of them was produced from a clean checkout, and a later commit must not be
represented as their training revision. **No GPU job was launched. Nothing was pushed to
`main`. No frozen report was overwritten.**

## The one decision

**Repair two named dependencies before any distillation; do not narrow the research
claim yet, and do not launch the multi-robot program.**

1. **The intent filter is not yet a usable gate.** With events defined on the proposal's own
   anchors and provisional tolerances, it rejects every non-identity candidate on all 24
   development windows (`events` 24/24 for smoothing, foot-lock and C6 projection). The
   spec's rule — events defined on the *human source* — needs the `HumanMotionSpec` adapter
   that the spec itself flagged as the schedule risk, plus tolerance calibration on the
   four-clip development set with the historical A1 T5 failures as regression cases.
2. **The closed-loop half of the pilot needs GPU approval.** Verifier, evaluator, start-grid
   split, command template, sample counts and cost basis are frozen in the manifest; two
   ~20-line evaluator changes (even/odd start subsets, second seed) and a reference re-export
   path are the only code dependencies. Budget: 12 evaluations, under 30 GPU-minutes.

What the CPU pilot *did* establish is an exclusion result, not a gain result: on the GMR-G1
references that the qualified tracker actually tracks, the two existing contact repairs
reduce stance foot speed 3–5× but violate the frozen Gate-1b guards (C6 projection: MPJPE
> 0.5 cm on 23/24 windows, jerk > 1.2× on 23/24; foot-lock DLS: jerk > 1.2× on 24/24), and the
only candidate that survives stage 0 is a near-no-op. Whether useful corrections exist is
**undetermined**: the outcome category is *filter/guard exclusion*, not *no observed gain*,
*search inadequacy*, *proxy failure* or *tracker failure*.

## Deliverable 1 — contact-audit patch, tests, corrected report

- `snmr/contact_audit.py` (count-first metrics; F1 = 2TP/(2TP+FP+FN); explicit all-empty
  policy; micro from summed counts; labelled macro; frame-level dedup; legacy aggregate for
  reconciliation) and `tests/test_contact_audit.py` (13 tests, including the two-window
  counterexample: legacy F1 1.0 vs count F1 2/102).
- `scripts/audit_contact_masks.py` v2: persists TP/FP/FN/TN per clip/window/foot, never
  writes over the frozen file, runs without checkpoints.
- Recompute: `runs/gate1b/mask_audit_v2_2026-09-10.json`. The two checkpoints are absent on
  this host, so the three model-dependent masks are skipped and listed; the checkpoint-free
  masks reproduce the frozen legacy aggregates to four decimals and then give the count-first
  numbers. Deployable `source_contact` aggregate F1: legacy 0.359 (from 19 of 42 windows) →
  micro **0.218** (all 42). walk1_subject5: 0.799 → 0.806. jumps1_subject2: 0.493 (one
  window) → 0.097. fight1_subject3: undefined → 0.
- Label-source audit: `scripts/audit_contact_oracle_labels.py`,
  `runs/gate1b/oracle_label_audit_2026-09-10.json` (+ `_walkfamily_`). Mechanism measured:
  on the six low-prevalence clips the per-foot clip minimum lies 4–5 cm below the stance
  plateau, at a tilted or penetrating frame, so ordinary stance never enters the 3 cm
  threshold. Not a property of fighting or dancing; not a proof of mislabelling either.
- Development set: `reproducibility/reports/locomotion_dev_set_2026-09-10.json`
  (generator `scripts/build_locomotion_dev_set.py`): **4 of 12 walk clips** are oracle-valid
  (walk1_subject1, walk1_subject2, walk1_subject5, walk2_subject4). "The oracle behaves on
  walking" is not a family property.
- Report and affected-claim list: `docs/CONTACT_AUDIT_V2_2026-09-10.md`. E70 metrics and
  paper numbers untouched.

## Deliverable 2 — guarantee/constraint tests and the method-spec amendment

- `snmr/correction_basis.py` + `tests/test_correction_basis.py` (14 tests): the halving
  counterexample, the ramp time-warp counterexample (rate 0.81–1.19 with fixed endpoints),
  pinned-two-coefficients is C¹ not C² (C² needs three per end, implemented as a constraint
  nullspace), knot spacing is not a cutoff (> 80 % energy above 3 Hz for alternating control
  points), and the contracts that hold: fixed grid, bounds, continuity through projection
  and stitching, joint limits by scaling with an explicit `baseline_invalid` abstention.
- `snmr/repair_intent.py` + `tests/test_repair_intent.py` (15 tests): separated desirable
  vs undesirable statistics, signed travel, windowed direction/speed, per-event timing and
  direction with nearest-peak matching, contact-sequence, explicit near-zero floors,
  `accept / fallback / abstain` decisions and coverage. Adversaries: amplitude halving,
  interior slow-down, direction reversal, removed foot-strike, shifted transition with zero
  global lag, jitter removal (must be accepted; the old ≥ 95 % rule would reject it).
- `docs/program/METHOD_SPEC_AMENDMENT_2026-09-10.md` governs over the spec; a pointer was
  added at the top of the spec. Dimensionality per channel is ⌈L/8⌉+3−6, not ⌈L/8⌉−1.

## Deliverable 3 — pilot manifest and verifier qualification

- `reproducibility/reports/pilot_manifest_g1_locomotion_2026-09-10.json`
  (generator `scripts/build_pilot_manifest.py`): G1 development pilot; separate fields for
  proposal-training robots, correction-label robots, tracker-training clips,
  verifier-qualification clips, final evaluation clips (none declared); tracker-seen vs
  retargeter-seen per clip; search (even linspace, seed 404) vs confirmation (odd, seed 405);
  headroom per clip before scoring (walk1_subject1 0.018, walk1_subject5 0.133); a
  headroom-relative threshold instead of +0.10; every candidate kept in the denominator;
  exact command template, sample counts, cost basis, approval requirement.
- `docs/VERIFIER_QUALIFICATION_G1_LOCOMOTION_2026-09-10.md`: the E70 seed-0 explicit two-walk
  student (sha `8fb6f2e1…`) passes the E67 floor on both clips on seeds 0–2 (0.98 / 0.86–0.87
  completion; 9.9 / 9.2 s survival). The multi-motion evaluator already exists in
  `scripts/train_e52_dagger.py`; deleting the `num_motions != 1` guard elsewhere was never the
  path. Qualified only on its own two clips; biased toward the no-op; 10-s horizon.

## Deliverable 4 — candidate ledger (CPU half)

`reproducibility/reports/pilot_candidate_ledger_cpu_2026-09-10.json`
(generator `scripts/run_pilot_candidate_bank_cpu.py`; 4 clips × 6 windows × 6 candidates;
provisional tolerances sha `091bf24f…`; closed-loop columns `null`, marked NOT RUN).

| Candidate | Stance speed, source mask (m/s) | MPJPE to proposal (cm) | Jerk ratio | Gate-1b guards (MPJPE ≤ 0.5 cm / jerk ≤ 1.2×) | Intent (provisional) | Stage-0 pass |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| no-op (GMR-G1 reference) | 0.108 | 0 | 1.00 | — | — | baseline |
| smooth σ=1 | 0.164 | 0.25 | 0.13 | 24/24 · 24/24 | events 24, jitter 11, contact 7 | 0/24 |
| smooth σ=2 | 0.239 | 0.77 | 0.03 | 8/24 · 24/24 | events 24, contact 17, jitter 14 | 0/24 |
| foot-lock DLS | 0.042 | 0.45 | 8.82 | 17/24 · **0/24** | events 24, jitter 24, contact 11 | 0/24 |
| C6 projection | 0.021 | 1.26 | 1.89 | **1/24 · 1/24** | events 24, jitter 24, contact 20 | 0/24 |
| spline-bounded (LS projection of C6's leg delta) | 0.122 | 0.31 | 1.00 | 23/24 · 24/24 | contact 17, events 2 | 7/24 |

Two observations the ledger adds, both as measurements, not conclusions: (i) the GMR
reference's own stance speed under the deployable mask is 0.06–0.135 m/s per clip, near the
0.10 endpoint, so the repair headroom on *these* references is small; (ii) Gaussian smoothing
of joint angles **increases** task-space hand jitter (window 0 of walk1_subject5, right hand:
15.6 → 77.9 m²/s⁴) although joint-space jerk falls 7×. This is consistent with redundant-joint
jitter that cancels at the end effector in the IK output; it is not diagnosed here, and it
means joint-space jerk is not a proxy for task-space smoothness on this data.

## Paper 1: interpretation tightened, no number changed

- `paper/main.tex` ~264: "lower bound on what a window-matched explicit arm would reach" →
  "a strong positive control under the implemented recipe, not a bound". ~274–279:
  destruction scoped to reliance on the channel under these interventions; exclusivity scoped
  to explicit exogenous reference inputs; proprioception and history still reach the decoder.
  The S wording in the manuscript was already correct and is unchanged. Rebuilt in a scratch
  copy with tectonic 0.17.0: **8 pages, 0 overfull boxes**; every renderer-generated `.tex`
  value file is untouched. `paper/main.pdf` in the repo was not regenerated.
- `docs/program/README.md`: "Identity was maximally available and bought nothing" replaced by
  the narrower statement; the withdrawn parameterization claims replaced; scaling demoted to
  secondary. `PAPER_STRATEGY` carries an amendment note under R1.
- Identity-separation generator: present on this branch (commit `1af217d`), **absent from
  `origin/main`** although the README that cites it is there — that is the reviewer's 404.
  It now emits a hash-bound artifact: `reproducibility/reports/e70_arm_identity_separation_2026-09-10.json`
  (A 8.0415 / S 8.0415 / T 0.0000 SD; input sha256 `b78f2943…`, `d8de9342…`; statistic and
  interpretation limits recorded). Fixing the public path requires pushing this branch or
  cherry-picking the script and artifact to `main`; not done here.
- Parameter counts, reconciled by instantiating the models on the real G1 MJCF:

  | Module / configuration | Parameters | Frozen | Per-robot buffers | Manual annotation |
  | --- | ---: | ---: | --- | --- |
  | `KinematicMorphoRetargeter` (joint decoder, human_token_dim 128) — the 7-robot zero-new-parameters check | 847,361 | 0 | none | `SemanticManifest` (7 roles) per robot |
  | `FixedTargetMorphoRetargeter` = human encoder 702,080 + joint decoder 847,361 + root head 34,313 — the A1/A2 proposal | 1,583,754 | 0 | 10 non-parameter tensors, 2,926 elements for G1's 38-node binding (grow with node count) | manifest + 19-role joint map; metric `SemanticAnchors` exist for G1 only |

  A constant count is an architectural property; it says nothing about transfer performance.
- Submission facts as verified 2026-09-10 (restated, not re-verified here): ICRA 2027 paper
  deadline September 15, 11:59 PM Pacific per RAS; internal date September 14; video pause
  September 10–16 and reopening 17–22 are intentional; eight pages all-inclusive; double
  anonymous; check the authenticated portal and the applicable AI-disclosure policy. **Not
  submitted on the user's behalf.**

## Program corrections (E/F)

`docs/program/EXPERIMENT_PROGRAM_AMENDMENT_2026-09-10.md` (pointer at the top of the
program): fixed-j comparison for the matrix, Panel-F tracker leakage, 10-s horizon,
geometry-only vs full as the primary two-arm comparison with GMR and simple repair as
baselines, GMR as a fixed non-neutral benchmark, pilot-based variance and separate seed/motion
heterogeneity, conditioning ablations with the binding kept correct, scaling secondary,
labelled outcomes for the learning phase, and the four separations to report.

## Tests

New tests: 13 + 14 + 15 = 42, all passing. Full suite (`pytest tests -q -x`, ROS `PYTHONPATH`
cleared, `tests/test_eval_morpho_g1_contract.py` excluded): **603 passed, 2 skipped, 1 error**
in 28 min; the error is the pre-existing untracked `tests/test_morphoret_t1_tracker.py`
importing `yaml`, which is not installed in `.venv` (environment, not this sprint's code), and
`-x` stopped the run there. The 33 files after it in sort order were rerun separately without
`-x`: 33 tests passed and none failed before a 58-minute limit killed the run without a
summary (pre-existing heavy tests, not this sprint's code). The suite was therefore **not
confirmed green end to end**; no completed test failed. Record:
`reproducibility/reports/sprint_2026-09-10/pytest_record.txt`.

## Not done, and why

- Closed-loop comparisons: not run (approval; GPU shared).
- Model-dependent mask recompute: checkpoints absent on this host.
- Independently reviewed contact intervals: none exist in the repository.
- Intent-tolerance calibration on development data: requires the human-side adapter first.
- Public-path fix for the identity script: requires a push, which is the user's call.
