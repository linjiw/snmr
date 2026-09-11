# Verifier qualification: the E70 two-walk explicit student as the pilot's closed-loop tracker

**Date:** 2026-09-10. **Scope:** qualification of an *existing* multi-motion policy and its
evaluator for the G1 locomotion development pilot
(`reproducibility/reports/pilot_manifest_g1_locomotion_2026-09-10.json`). No rollout was run
in this sprint; every number below is read from frozen E70 artifacts.

## 1. Which policy, exactly

| Item | Value |
| --- | --- |
| Policy | E70 seed-0 explicit-command DAgger student, arm `c_prior_explicit` (`E52_DET=1`, 2,000 rounds) |
| Checkpoint | `/data/robotixx/snmr-research/e70/students/seed0_explicit/c_prior_explicit_student.pt`, sha256 `8fb6f2e1645d8070…` |
| Motions it was trained on | `walk1_subject1` (E69 GMR-G1 reference, 13,066 frames at 50 Hz) and `walk1_subject5` (E67 reference); both hashed in the manifest |
| Teachers it was distilled from | the two E67/E69 PPO specialists in `e70/teacher_manifest.json` (per-clip checkpoints with sha256) |
| Observation normalization | the student consumes the specialists' frozen `actor_obs_normalizer_state_dict` through the E52 recipe; normalizers are frozen objects shared across arms (control audit, 2026-09-10) |
| Action ordering / bodies | holosoma `g1-29dof-wbt` (29 actions in MuJoCo dof order; 51 tracked bodies), the same environment the references were exported for (`scripts/export_wbt_gmr_batch.py`) |

## 2. Which evaluator, and why the `num_motions != 1` guard is irrelevant

The method spec proposed deleting the single-motion guard in `scripts/eval_e67_teacher.py`.
That file evaluates PPO *specialists*. The E70 students are evaluated by
`scripts/train_e52_dagger.py` in `E52_EVAL_ONLY=1` mode, which already:

- loads a multi-motion directory and checks it against the teacher manifest in lexical order;
- builds an **equal-count per-clip linspace start grid** (512 + 512 starts for 1,024 envs),
  one frame inside each clip boundary, and asserts the realized grid;
- routes teacher actions per motion id (`motion_ids` from `wbt_bodyfix.py`'s `searchsorted`);
- reports completion, survival and joint RMSE **per clip** as well as macro.

Removing a guard in the specialist evaluator would not have produced a qualified multi-motion
verifier; the qualified evaluator already exists and was used for every E70 number.

## 3. Qualification against the E67 floor (per clip, uncorrected reference)

Floor (E67 precedent, written before this sprint's use): completion ≥ 0.80 and survival
≥ 9.0 s on the uncorrected reference, per clip, on the 1,024-rollout / 500-step / seed-404
grid.

| Seed | Macro completion | `walk1_subject1` completion / survival | `walk1_subject5` completion / survival |
| --- | ---: | ---: | ---: |
| 0 | 0.9248 | 0.9824 / 9.92 s | 0.8672 / 9.25 s |
| 1 | 0.9199 | 0.9805 / 9.89 s | 0.8594 / 9.17 s |
| 2 | 0.9238 | 0.9785 / 9.85 s | 0.8691 / 9.24 s |

Both clips pass on all three seeds. Seed 0 is the pilot's verifier; seeds 1–2 are available
for a policy-seed sensitivity check of any accepted correction.

## 4. What the verifier is and is not qualified for

- **Qualified:** scoring corrected references of `walk1_subject1` and `walk1_subject5`
  against their own uncorrected references, under matched starts and horizon. This is the
  pilot's question.
- **Not qualified:** any motion it was not evaluated on. It has never been run on
  `walk1_subject2` or `walk2_subject4` (the two tracker-unseen development clips). Using it
  there requires a qualification run first, and a failure there is instrument failure, not
  a correction result.
- **Headroom:** `walk1_subject1` sits at 0.98 (headroom ≈ 0.018) — a completion gain cannot
  be measured there; it is kept in the denominator as no-headroom. `walk1_subject5` at 0.86–0.87
  is the clip with measurable headroom (≈ 0.13).
- **Disclosed bias:** trained on the uncorrected references, so biased toward the no-op
  candidate; this runs against the hypothesis.
- **Horizon:** 500 steps at 50 Hz is a 10-second window from each start, not full-clip
  success for a 261-second clip.

## 5. What blocks the closed-loop half of the pilot

1. Corrected references must be re-exported to the holosoma NPZ schema at 50 Hz through the
   same replay/resample path as the originals and carry a `latent_z` field for the loader
   (unused by the explicit arm).
2. The evaluator needs two small, committed changes: an even/odd start-grid subset flag
   (search vs confirmation) and a second evaluation seed. Reusing the existing seed-404 grid
   for both selection and confirmation would reuse feedback.
3. GPU access: the RTX 5090 is shared (≈ 22.6 GB held by another tenant at the time of
   writing). E70 ran 1,024 mjwarp envs on this GPU; each evaluation's log spans about 10 s.
   Twelve evaluations (6 candidates × search + confirmation) are budgeted under 30
   GPU-minutes. **Approval is required before launch**; nothing has been launched.

The exact command template, sample counts and cost basis are in the manifest.
