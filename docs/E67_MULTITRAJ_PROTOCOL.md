# E67 — Multi-Trajectory Ambiguous-Time Interface Test

**Preregistered:** 2026-08-08, before training either E67 student or the second-clip
specialist teacher.  **Amended before any E67 policy result:** the selector now requires ten
seconds of reference after every start, preventing a planned 10-s rollout from crossing a clip
boundary.  **Status (2026-08-08): stopped at the preregistered specialist-teacher gate, with no
representation conclusion.**

The `walk1_subject5` specialist passed (completion 0.8467, mean survival 9.1138 s,
joint RMSE 0.1807 rad), but `walk3_subject1` did not (completion 0.5615, mean survival
7.2881 s, joint RMSE 0.2233 rad).  Both were evaluated on the frozen 1,024-start, seed-404
grid after 8,000 PPO iterations.  Therefore the original E67 representation arms are not a
valid experiment and must not be interpreted.

An orchestration bug masked the failing evaluator's process status through a Bash pipeline and
allowed the explicit positive-control student to train for 2,000 rounds.  It received no general
or ambiguity-start behavioral evaluation.  That checkpoint is quarantined as an invalid
post-gate artifact and is excluded from every analysis.  The launcher now independently reads
the persisted `passes_gate` value and returns nonzero before creating a teacher manifest or
starting any student; it also creates student log directories before launching `tee`.

## 1. Research question

Does the frozen SNMR latent provide control-usable motion information beyond current
proprioception and an absolute time-index code when the same time code and a similar current
robot state can require different future actions?

The claim is scoped to this command interface and controller class.  A null does not imply that
the frozen latent contains no motion information; it means this interface did not extract a
control advantage over time under the tested conditions.

## 2. Clip selection and ambiguity gate

The candidate pairs were fixed before running the check:

1. `walk1_subject5 + walk3_subject1`
2. `walk1_subject5 + run1_subject2`

`scripts/precheck_e67_ambiguity.py` uses only the GMR reference pair files.  It does not read an
SNMR checkpoint, latent, controller, or teacher action.  The perfect-tracking current-state proxy
is the 90-d vector `[previous_dof_target, root_angular_velocity, dof_pos, dof_vel]`.  Robot-goal
futures are `[dof_pos, dof_vel]`.  Features are standardized once over both clips in a candidate
pair.

Frozen thresholds:

- 100 normalized-time bins;
- current-state RMS distance <= 0.75 pooled standard deviations;
- future-trajectory RMS distance >= 0.75 pooled standard deviations;
- future horizon 1.0 s, sampled at 11 evenly spaced offsets;
- at least 10.0 s of the same clip remains after every selected start;
- selected pairs separated by >=0.5 s in both clips;
- feasibility floor: >=20 selected windows spanning both clips.

Result (`autoresearch/iterate-260808-0338/e67_ambiguity_precheck.json`):

| Candidate pair | Threshold-eligible | Non-overlapping windows | Gate |
| --- | ---: | ---: | --- |
| walk1 + walk3 | 1,082 | **66** | pass |
| walk1 + run1 | 30 | 5 | fail |

`walk1_subject5 + walk3_subject1` is therefore frozen for E67.  Selection used ambiguity
feasibility only, before any E67 student result.

## 3. Specialist-teacher ensemble

Train or reuse one explicit-command PPO specialist per selected clip.  Each specialist must pass
all of:

- completion >=0.80;
- mean survival >=9.0 s;
- finite/stable joint RMSE;
- 1,024 phase-stratified 10-s rollouts at evaluation seed 404.

The current Holosoma revision no longer includes the historical `wbt_metrics` callback.
`scripts/eval_e67_teacher.py` therefore realizes the same exact phase grid through the local
body/evaluation patch and computes terminal-aware completion, survival, and joint RMSE directly
from simulator state.  A four-environment, 100-step plumbing smoke must write a finite report
before either full teacher evaluation.

During distillation, Holosoma's `motion_ids` route each environment to its clip's frozen teacher
for action labels.  Motion ID is not an input to the student, its code encoder, its action
decoder, or its normalization.  The first specialist supplies one shared actor/critic
normalization for student inputs; each teacher uses its own frozen actor normalization only to
produce its label.

If either teacher misses its gate, E67 stops without a representation conclusion.

## 4. Student arms

Every arm uses the same deterministic 64-d command architecture, optimizer, replay, budget,
checkpoint selection, normalizers, seeds, and motion pool.

| Arm | Code-encoder input beyond proprioception | Environment settings | Role |
| --- | --- | --- | --- |
| C | explicit 64-d robot goal | `c_prior_explicit`, `E52_DET=1` | positive control |
| A | frozen SNMR latent at t and t+0.1 s | `a_prior_snmr`, `E52_DET=1` | main hypothesis |
| T | shared absolute time-index code | A + `E52_PHASE_ONLY=1` | time null |
| B | nothing | `b_prior_proprio`, `E52_DET=1` | proprio control |
| S | another clip's latent at matched normalized time | A + `E52_SHUFFLE_LATENT=1` | causal/content control |

The time code uses one frequency bank and one fixed projection and resets identically at each
clip boundary.  Equal local frame indices have bit-identical codes across clips.  It receives no
filename, motion index, concatenated global frame offset, or per-clip normalization statistic.

SNMR latents use one mean and standard deviation pooled over the full two-clip training tensor.
Arm S cyclically swaps clip latents and nearest-neighbor samples at matched normalized time before
that global normalization.

The action decoder receives exactly `[90-d proprioception, 64-d z_cmd]`.  Structural isolation,
same-time shuffling, time reset, and label-only motion-ID routing are regression-tested in
`tests/test_distillation.py`.

## 5. Stabilized online distillation

Seed-0 uses the deterministic E62 architecture first: no posterior, KL, sampling noise, or latent
smoothness loss.  The CVAE is not introduced unless this positive control is stable.

Fixed settings:

- 2,000 rounds; 1,024 environments; 24 collection steps per round;
- five optimizer epochs, minibatch 4,096, Adam 3e-4;
- replay of the current and previous three rounds (`E52_REPLAY_ROUNDS=4`);
- teacher mixture anneals linearly over 200 rounds but never below 0.1;
- fixed validation set: 4,096 teacher-driven samples held out from round 0;
- validation action loss every 50 rounds; select the lowest-loss checkpoint after round 50;
- periodic checkpoints every 50 rounds; final checkpoint retained separately.

Abort immediately on:

- any nonfinite loss or gradient norm;
- KL >10,000 (for any later CVAE arm);
- mean command-code norm >100;
- temporal smoothness >100 times its rounds-0-to-200 median (for any later CVAE arm).

For a later CVAE run, temporal smoothness recomputes both sides of each consecutive pair under
the current parameters and averages over latent dimension and valid transitions.  No stored
old-parameter code is a loss target.

## 6. Evaluation

General evaluation macro-averages the two clips and reports completion, survival, joint RMSE,
and routed teacher-student action error.  Every cell uses 1,024 10-s rollouts with identical
start grids and evaluation seed 404.

Primary ambiguity evaluation starts rollouts at the 66 preregistered frame pairs, with repeated
seeds balanced over both sides of every pair.  Report:

- ambiguity-start completion and survival;
- action error against the routed specialist teacher at and after the branch window;
- per-clip values;
- paired SNMR-minus-time intervals over matched starts;
- failure timing around the selected windows.

Whole-rollout completion is never assigned to a frame post hoc; ambiguity completion comes from
rollouts explicitly initialized at ambiguity starts.

## 7. Gates and interpretation

Run seed 0 in order C, A, T, B, S.  Stop after C if:

- macro completion <0.80 and more than 5 percentage points below the teacher ensemble; or
- validation diverges or an abort gate fires.

Only after a passing seed-0 smoke, run training seeds 0, 1, and 2 for all five arms.

Positive content gate:

- A exceeds T by >=10 completion points on ambiguity starts;
- paired 95% interval excludes zero;
- S loses the gain;
- the direction holds on both clips.

Interpretation:

1. **A > T and all gates pass:** the frozen SNMR latent provides control-usable information
   beyond time under this interface; describe the minimum supported content (for example clip
   identity versus future trajectory) from the action-error analysis.
2. **A approximately equals T and C passes:** scoped interface-extraction null; rewrite the paper
   as the definitive controlled measurement result.
3. **C fails:** trainer/task invalid; no latent-versus-time conclusion.
4. **B approximately equals T:** inspect state/reset leakage and ambiguity-start realization
   before interpretation.

No threshold, clip pair, arm, or stopping rule changes after observing an E67 student result.
