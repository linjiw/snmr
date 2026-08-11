# E70 — Fresh Two-Walk Interface Test

**Preregistered:** 2026-08-09, after the E69 specialist endpoint passed and before any E70
student was trained.  E70 uses a new output root and may not load the invalid E67 student.
**Amended before any E70 behavioral evaluation:** added the already implemented
matched-marginal command destruction alongside zero and shuffle; it is a validity readout and
does not alter the primary representation gate.

## 1. Question and scope

Does the frozen SNMR latent provide control-usable information beyond proprioception and an
absolute within-clip time code when two walk trajectories contain similar current robot states
but require different futures?

The conclusion is about the tested exclusive 64-d command interface and deterministic student,
not about all information in the SNMR latent.  The two motions are both LAFAN1 walks, so even a
positive result is evidence for trajectory-disambiguating content within this pair, not broad
motion-category understanding.

## 2. Frozen, policy-independent selection

E69 searched all eligible LAFAN1 clips using GMR references only and selected
`walk1_subject1` against the fixed `walk1_subject5` anchor.  It found 69 non-overlapping
ambiguous windows, a median eligible future distance of 1.113 pooled SD, and a reference
difficulty ratio of 0.852.  No student, latent value, or policy result entered the selection.

The simulator loads `*.npz` files lexically.  The registered loaded order is therefore:

1. `walk1_subject1`
2. `walk1_subject5`

E69 recorded windows as anchor first, candidate second.  `scripts/prepare_e70_protocol.py`
swaps every frame, seconds, and normalized-time side into the loaded order and emits
`autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json`.  Its SHA-256 is
`3c03b89e2bda939e9a8c5a6dd58caeb771944bb9e5755b14eb37ab75ed9c502e`.
The evaluator rejects any mismatch between that ordered clip list and the teacher manifest.

Frozen motion artifacts, also in loaded order:

| Clip | Aligned GMR + SNMR motion SHA-256 |
| --- | --- |
| `walk1_subject1` | `b78f294395a5c74f37edc1c09dd6de0909d966a9b7fd948291c8dd803f7106aa` |
| `walk1_subject5` | `d8de93425c14e90dce2930450d722d3eb2b6fcbb09e9c4ff3d59725025424f51` |

## 3. Frozen specialist ensemble

Both specialists already passed the unchanged E67 gate on 1,024 phase-stratified, 10-s
rollouts at seed 404:

| Clip | Checkpoint endpoint | Completion | Survival | Joint RMSE |
| --- | --- | ---: | ---: | ---: |
| `walk1_subject1` | 7,999 | 0.9873 | 9.9376 s | 0.1815 rad |
| `walk1_subject5` | 8,000 | 0.8467 | 9.1138 s | 0.1807 rad |

Their macro completion is 0.9170.  The checkpoint SHA-256 values are respectively
`8f884cfbf742bfd106766ed95114a3389d7e36c18008b11f850ca2a4a8f0d9fd` and
`ec11cb2fca18bc1e522cad90c12fceb0496932802b2771c04b8209781c282193`.
Motion ID routes teacher labels only; it is not a student, encoder, decoder, or normalization
input.  One pooled latent normalization and the first teacher's student-input normalization are
shared across motions.

## 4. Frozen students and validity order

All arms use the deterministic E62 64-d command architecture, 2,000 online-distillation
rounds, 1,024 environments, seed 0, four-round replay, Adam 3e-4, a 200-round teacher-mixture
anneal with floor 0.1, and the same fixed validation/checkpoint selection rules as E67.

| Tag | Encoder input beyond proprioception | Role |
| --- | --- | --- |
| C / `explicit` | explicit robot goal | positive control |
| A / `snmr` | frozen SNMR latent at t and t+0.1 s | hypothesis |
| T / `time` | shared absolute within-clip time code | time null |
| B / `proprio` | none | goal-free control |
| S / `shuffled` | other clip's latent at matched normalized time | content control |

Run C first.  It passes if general macro completion is >=0.80 **or** no more than 0.05 below
the specialist macro.  If it fails, E70 stops with no representation conclusion.  A passing C
is additionally evaluated after (i) zeroing its entire 64-d command, (ii) batch-shuffling
that command, and (iii) independently sampling every command dimension from its observed
batch marginal.  These interventions do not tune the primary gate; they establish whether
the clean explicit student's exclusive command channel is causally used.

Only after seed-0 C passes run A, T, B, and S.  Only after all five seed-0 cells are complete may
training seeds 1 and 2 be requested; each replicate also runs C and its validity gate before the
four representation arms.  All evaluation uses seed 404 and 1,024 paired 10-s rollouts.

## 5. Frozen analysis and interpretation

The primary endpoint is ambiguity-start completion.  A paired cluster bootstrap weights each
of the 69 frame pairs equally.  The positive content gate requires all of:

- C passes its general positive-control gate;
- A minus T is at least 0.10 on ambiguity starts;
- the paired 95% interval for A minus T excludes zero;
- the paired 95% interval for A minus S excludes zero; and
- A minus T is positive on both clips.

Outcome language is fixed in advance:

1. **Gate passes:** “Under this exclusive controller and two-walk assay, the frozen SNMR code
   provides control-usable trajectory information beyond absolute within-clip time.”
2. **C passes but the content gate does not:** “The interface extracts no demonstrated control
   advantage over time on this two-walk assay.”  This is not an information-theoretic statement
   about the frozen latent.
3. **C fails:** trainer/task invalid; report no A-versus-T conclusion.
4. **B approaches T or S retains A's gain:** audit reset, state, ordering, and content leakage
   before interpretation.

No clip, threshold, checkpoint, arm, seed-0 stopping rule, or primary statistic changes after
an E70 student result.  Generated outputs live only under
`/data/robotixx/snmr-research/e70/`.

## Seed-0 endpoint ledger

All seed-0 cells completed after registration.  C reaches 0.9248 general completion (9.5846 s
survival, 0.1807 rad joint RMSE), versus the two-teacher macro 0.9170, so the positive-control
gate passes.  At the 69 ambiguity pairs C reaches 0.9785.  Destroying its clean exclusive command
collapses general completion from 0.9248 to 0.000 under zero, batch shuffle, and matched-marginal
independent resampling.  The respective survival times are 0.852, 0.672, and 0.927 s.  The
structurally exclusive channel is therefore causally necessary in this positive-control student.

| Arm | General completion | Ambiguity completion | Ambiguity survival | Teacher-action RMSE |
| --- | ---: | ---: | ---: | ---: |
| C / explicit | 0.9248 | 0.9785 | 9.912 s | 0.0179 |
| A / SNMR | 0.6846 | 0.7646 | 8.870 s | 0.4923 |
| T / time | 0.6045 | 0.6084 | 7.591 s | 0.5901 |
| B / proprio | 0.4873 | 0.5176 | 7.293 s | 0.6947 |
| S / shuffled | 0.5547 | 0.5762 | 7.492 s | 0.6688 |

The equally weighted, 69-cluster paired bootstrap estimates A minus T as +0.1544 with 95% CI
[0.0932, 0.2147], and A minus S as +0.1871 with CI [0.1368, 0.2358].  A minus T is positive
on each clip: +0.1517 [0.0851, 0.2223] for `walk1_subject1` and +0.1571
[0.0639, 0.2461] for `walk1_subject5`.  Every preregistered seed-0 content gate passes.  Under
the scoped outcome language above, the frozen SNMR code provides control-usable trajectory
information beyond absolute within-clip time in this exclusive two-walk assay.  Training seeds
1 and 2 are now licensed confirmation runs; the seed-0 reports remain frozen regardless of their
outcome.

## Confirmation interruption ledger

The seed-2 proprio training process received external termination (exit 143) after logging round
index 1,550 and before writing a canonical student or running any behavioral evaluation.  A
partial continuation would not be equivalent because the periodic checkpoint omits simulator and
RNG state, the four-round replay contents, and the frozen validation tensors.  The entire partial
directory is therefore preserved at
`/data/robotixx/snmr-research/e70/interrupted/seed2_proprio_round1550_20260809T125601`, and the
cell must restart from round 0 with the unchanged E70 command.  `scripts/run_e70_cell.sh` records
the isolated one-cell recovery command.
Hashes and the exact decision are recorded in
`autoresearch/iterate-260809-0351/e70_seed2_proprio_interruption.json`.  No result from the
interrupted run is eligible for the analyzer.  Because an unrelated CUDA process remained
resident, `scripts/run_e70_recovery_supervisor.sh` was launched in a detached terminal on
2026-08-09.  It waits for the same 26,000-MiB capacity gate and then invokes the unchanged full
launcher with `E70_FULL_SEEDS=1`; completed cells are skipped, so seed-2 proprio restarts at round
0, followed by seed-2 shuffled and the frozen analyzer.  The supervisor changes neither the
student recipe nor the analysis contract.  Pre-recovery hashes of the trainer, model helpers,
launcher, analyzer, and ambiguity grid are frozen in
`autoresearch/iterate-260809-0351/e70_confirmation_code_hashes.json`; post-confirmation rendering
changes must not be confused with the code that produced E70.
