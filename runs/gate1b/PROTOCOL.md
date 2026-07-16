# Gate 1b: mask-precision study — FROZEN PROTOCOL

Frozen 2026-07-14, after the pre-study mask audit (`runs/gate1b/mask_audit.json`) and two
single-window plumbing smokes (`smoke_m1.json`, `smoke_m2.json`, walk1 window 0 only), and
BEFORE reading any full 42-window arm result. Registered design:
`docs/NEURAL_RETARGETING_RESEARCH_fable.md` section 4.1.

## What is frozen

- **Solver:** `snmr/projection.py` at commit `f6ed96a` — UNCHANGED (verified clean in git).
  All projection hyperparameters are the `eval_footlock.py` defaults used by E29:
  30 L-BFGS iterations, stance/stance-velocity weight 1000, deviation 0.1, velocity 0.5,
  acceleration 1.0, root translation bound 0.04 m, yaw bound 0.12 rad, joint delta bound
  0.35 rad, merge_gap 0, extend 0, min_stance_frames 1. Per no-go rule 15, none of these may
  change inside Gate 1b.
- **Evaluator:** `scripts/eval_footlock.py`, modified ONLY to add two mask sources
  (`decoded_height`; `--mask_ckpt` for a cross-model `predicted_contact`) and their provenance.
  Scoring, guards, decision logic, window protocol, and bootstrap are byte-identical in
  behavior to the E29 runs.
- **Base checkpoint:** `runs/phase1_g1_large/ckpt_100k_final.pt` (E03/E29 checkpoint).
- **Protocol:** 42 windows = 7 held-out clips x up to 6 windows x 192 frames, bootstrap seed 0,
  2000 samples, G1 only, CPU.

## Arms

| Arm | Mask | Command difference only |
|---|---|---|
| M0 | source_contact heuristic | E29 artifact reused (`runs/skate_structure/windowed_c6_source_velocity_full.json`) |
| M1 | decoded_height: hysteresis enter 0.03 / exit 0.05 on the base checkpoint's decoded feet | `--lock_mask decoded_height` |
| M2 | C1 contact head (`runs/gate1_g1/screen/c1_bce_seed0/ckpt.pt`), sigmoid >= 0.5 (variant selected by the audit's frozen higher-aggregate-F1 rule: `predicted_t0.5`) | `--lock_mask predicted_contact --mask_ckpt ...` |
| M4 | teacher_height oracle (upper bound) | E29 artifact reused (`..._teacher_oracle_velocity_full.json`) |

M3 (contact head retrained on teacher-height labels atop the C4 seed-0 checkpoint, BCE 0.25)
runs only if M1 and M2 both fail their endpoints.

## Endpoints and guards (frozen)

Coprimary endpoints, both required for a deployable pass (fable.md addition 16):

1. Teacher-height stance speed `<= 0.08 m/s` (the Gate 1 primary, clip-local hysteresis mask).
2. Source-contact stance speed `<= 0.10 m/s` (dense non-circular mask; teacher reference value
   on this evaluator is 0.1226 m/s, so this demands slightly better than teacher parity).

Guards, identical to E29: MPJPE `<= +0.5 cm` relative to the matched raw decode (3.66 cm) with
the absolute `<= 4.0 cm` target reported separately; DOF jerk `<= 1.2x` raw; zero limit
violations; penetration mean `<= raw + 0.002 m`; penetration fraction `<= raw + 0.02`.

## Decision rules (frozen, from fable.md 4.1)

- Any deployable arm (M1/M2/M3) passes both coprimaries and all guards -> contact closed as
  "predict + project"; validate on the other four robots; update claims ledger C5.
- All deployable arms fail AND the audit shows high mask-oracle agreement -> classification is
  exonerated; escalate to physics-repaired supervision (fable.md 4.3); stop mask iteration.
- Mask quality is the measured blocker -> exactly ONE registered iteration on the mask *labels*
  (pre-specified candidate: clip-local ground normalization for the deployable mask, motivated
  by the audit's ground-gap finding), never on the solver.

## Pre-study audit summary (context for interpretation, measured before this freeze)

Against the clip-local teacher-height oracle (prevalence 0.0887, ~34 samples/window):

| Candidate | Precision | Recall | F1 | Prevalence |
|---|---:|---:|---:|---:|
| M0 source_contact | 0.132 | 0.945 | 0.359 | 0.668 |
| M1 decoded_height | 0.167 | 0.743 | 0.392 | 0.439 |
| M2 predicted_t0.5 | 0.140 | 0.994 | 0.390 | 0.657 |
| (predicted_hyst) | 0.139 | 0.989 | 0.387 | 0.657 |
| oracle def., 192-window ground | 0.157 | 1.000 | 0.427 | 0.611 |
| oracle def., 64-window ground | 0.130 | 1.000 | 0.370 | 0.688 |

Ground-normalization gap: the 192-frame-window minimum foot height sits mean 4.9 cm (median
4.0 cm, p90 9.3 cm) above the clip minimum; 68% of windows exceed the 3 cm enter threshold.
**Interpretation registered before arm results:** every deployable mask's low precision vs the
oracle is dominated by ground normalization and mask density, not label noise — even the
oracle's own definition with window-local ground scores precision 0.157. The oracle is a
precision-biased, sparse ("deep stance only") mask. The open question the arms answer is
whether a dense deployable mask can pass the sparse scored endpoint without breaking MPJPE
(E29-M0 failed via root-correction saturation), or whether density itself is the failure mode.

## Registered label iteration M1b (amendment frozen 2026-07-14 ~22:15, before its run)

M1 and M2 have completed and both FAIL as frozen (`windowed_c6_m1_decoded_height_full.json`,
`windowed_c6_m2_c1head_full.json`):

| Arm | Teacher-height speed | Source-contact speed | MPJPE | Verdict |
|---|---:|---:|---:|---|
| M1 decoded_height (window-local ground) | 0.199 m/s | 0.236 m/s | 4.22 cm | fails both coprimaries + MPJPE |
| M2 C1 head t0.5 | **0.024 m/s (endpoint passes)** | 0.127 m/s | 4.61 cm | fails MPJPE (+0.95 cm), penetration fraction, source coprimary |

The failure pattern matches the pre-registered "density is the failure mode" branch: M2's dense
mask (prevalence 0.657, precision 0.140 vs oracle) passes the sparse speed endpoint but pins
swing frames, saturating root corrections (20/42 windows) and breaking MPJPE/penetration. Mask
precision is therefore the measured blocker, and the pre-specified single label iteration is
exercised exactly as registered:

- **M1b = decoded_height with CLIP-LOCAL decoded ground**: per-foot ground_z = minimum decoded
  foot height over the full clip (decoded by tiling the clip in consecutive 192-frame windows —
  deployable, uses no teacher signal), hysteresis enter 0.03 / exit 0.05 unchanged.
- Rationale: decoded foot heights are accurate (E24), so clip-local decoded ground should
  approximate the oracle's clip-local teacher ground; predicted effect is prevalence dropping
  from 0.44 toward the oracle's 0.09 with much higher precision.
- Everything else (solver, endpoints, guards, protocol) is unchanged. No further mask iteration
  is permitted after M1b: if M1b fails, M3 (already registered) runs; if M3 also fails,
  escalate to physics-repaired supervision per the frozen decision rules.

## M1b pre-run smoke status (observed after the amendment)

`smoke_m1b.json` selected zero stance samples in the first `walk1_subject5` window and returned
the solver's `empty_contact_support` result. The full-clip decoded foot minima are 5.1/6.9 cm
below that window's minima, so the clip-ground mask never enters contact there. This is a mask
support diagnostic, not a successful solver smoke. Audit all 42 windows for support, then use a
support-bearing window for projection plumbing before the full run; zero-support windows remain
mask failures under the frozen study.

## Final result and decision (2026-07-16)

The support audit found only 58 selected samples in 16,128 (`0.36%` prevalence): 8/42 windows
had any support and 34/42 were forced identity projections. M1b precision was `0.259`, recall
`0.0105`, and F1 `0.0201`. A support-bearing smoke exercised and accepted the real L-BFGS path,
but the unchanged full run failed both coprimaries (`0.479 m/s` teacher-height and `0.339 m/s`
source-contact speed) while preserving MPJPE at `3.66 cm`.

M3 then trained only the four newly introduced `decoder.contact_head.*` tensors for 50k steps
from the registered C4 seed-0 checkpoint. All 98 inherited tensors remained byte-identical. Its
held-out mask was denser and less precise than M2: `71.5%` prevalence, `0.0963` precision,
`0.6587` recall, and `0.3236` F1 against the `8.87%`-prevalence oracle. The frozen projection
failed both coprimaries (`0.430 m/s`, `0.254 m/s`) and the MPJPE, jerk, and penetration-fraction
guards (`6.51 cm` projected MPJPE).

Final verdict: `fail_close_mask_iteration`. No M3 seed replication is triggered and no further
mask or solver iteration is permitted in Gate 1b. The oracle result still establishes that the
projection machinery can close contact when constraints are correct; the deployable-mask study
establishes precision/density and clip-ground inconsistency as the blocker. The next contact
method is physics-repaired supervision, conditional on a competent calibrated WBT policy.
