# E69 — Reference-Only Pair Screen After Specialist Failure

**Registered:** 2026-08-08 after the frozen E68 endpoint failed and before screening the
remaining LAFAN1 clips.  E69 reads only GMR reference trajectories.  It does not read an SNMR
latent, specialist action, student checkpoint, or representation result.

## Purpose

E67 established a valid ambiguity construction but selected a second clip whose specialist did
not meet the quality gate even after the one registered 16k extension.  E69 searches for an
easier second trajectory that is still ambiguous with `walk1_subject5`, whose existing
specialist already passes.  `walk3_subject1` is excluded because its fixed feasibility budget is
closed; the anchor itself is also excluded.

## Frozen screen

- Anchor: `walk1_subject5`.
- Candidates: every other Unitree-G1 LAFAN1 pair file except `walk3_subject1`.
- Ambiguity thresholds: exactly E67's amended values (100 normalized-time bins, state RMS
  distance <=0.75 pooled SD, 1-s future RMS distance >=0.75 pooled SD, 11 future samples,
  >=0.5-s spacing, a full 10-s same-clip rollout after each start, and >=20 windows).
- Reference difficulty is the maximum of three candidate/anchor ratios: 95th-percentile joint
  speed RMS, 95th-percentile joint acceleration RMS, and 95th-percentile root angular speed.
  Denominators are floored at `1e-6`.
- A candidate must pass the ambiguity floor and have difficulty ratio <=1.25.
- Select lexicographically by: lowest difficulty ratio, most ambiguity windows, largest median
  eligible future distance, then clip name.  There is no manual clip choice after the report.

The report records every candidate, failure reason, input hash, raw difficulty statistic, and
selected ambiguity window.

**Screen result:** `walk1_subject1` was selected automatically with 69 windows and difficulty
ratio 0.852.  Its median eligible future distance is 1.113 pooled SD.  No policy result entered
this choice.  The GMR WBT reference has SHA-256
`599161a845870f830894fd538dcc3bd9ee36d55cb78425d3db2ec5b9656f50ba`; the aligned
GMR-reference + 128-d SNMR-latent motion has SHA-256
`b78f294395a5c74f37edc1c09dd6de0909d966a9b7fd948291c8dd803f7106aa`.  All non-latent
arrays are bit-exact between them.

## Promotion rule

If no candidate passes, the current multi-trajectory route stops and the paper reports the
feasibility boundary.  If one passes, export its GMR WBT reference and SNMR latent, then register
one 8,000-iteration seed-0 specialist with the unchanged E67 1,024-rollout gate.  Only a passing
specialist permits a fresh student experiment in a new output root.  No E67 student artifact may
be reused.

## Frozen specialist gate

Before observing a `walk1_subject1` policy result, train exactly one specialist for 8,000 PPO
iterations with the E67 recipe (512 environments, seed 0, same reward override and simulator).
Evaluate only its final `model_07999.pt` on 1,024 phase-stratified 10-s rollouts at seed 404.
The unchanged gate is completion >=0.80, survival >=9.0 s, and finite joint RMSE.  A failure
stops E69; a pass permits registration of E70, but does not itself permit reuse of any prior
student.

## Frozen endpoint (2026-08-09)

The one registered `model_07999.pt` endpoint passed on all 1,024 rollouts at seed 404:

| Metric | Result | Gate |
| --- | ---: | ---: |
| Completion | **0.9873** | >=0.80 |
| Mean survival | **9.9376 s** | >=9.0 s |
| Joint RMSE | **0.1815 rad** | finite |

The checkpoint SHA-256 is
`8f884cfbf742bfd106766ed95114a3389d7e36c18008b11f850ca2a4a8f0d9fd`; the persisted
report SHA-256 is
`60a151c007f1fa5f806120a684110dac5c3e991ed42e6d6b68abe9a78cca8f86`.
E69 is closed as a pass.  It licenses registration of E70 but is not itself evidence about
SNMR command content.
