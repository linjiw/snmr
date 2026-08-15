# E75 — Command destruction on the SNMR arm (preregistration)

**Written:** 2026-08-15, **before any E75 evaluation was run.**
**Type:** evaluation-only on the three frozen E70 SNMR students. No training, no tuning, no
selection. Zero new code paths — this is the frozen explicit-arm destruction protocol applied,
unchanged, to a different arm.
**Authority:** `docs/PLAN_2026-08-14.md` Track C. Owner authorized the plan in-session 2026-08-14.

## 1. Why this exists

The paper's behavioral-necessity rung is currently established **only on the explicit controller** —
the control arm. `ls /data/robotixx/snmr-research/e70/students/*/*destroy*` returns files under
`seed{0,1,2}_explicit` and nowhere else. The abstract says *"Destroying the seed-0 explicit
student's exclusive command collapses completion from 0.925 to zero."* A reviewer will ask why the
same intervention was never applied to the arm that carries the headline claim, and reading that as
an evasion would be reasonable.

E75 closes that flank. It is the cheapest item in the whole remaining program: no training, no new
code, one GPU hour.

## 2. Exact protocol — a replication, not a variant

Identical in every registered respect to the frozen explicit-arm destruction
(`scripts/run_e70_multitraj.sh:202-216`, `run_explicit_destruction`), changing only the arm:

| Field | Explicit (frozen, already run) | SNMR (this preregistration) |
| --- | --- | --- |
| arm | `c_prior_explicit` | `a_prior_snmr` |
| tag | `explicit` | `snmr` |
| `E52_PHASE_ONLY` | 0 | 0 |
| `E52_SHUFFLE_LATENT` | 0 | 0 |
| modes | zero, shuffle, marginal_random | zero, shuffle, marginal_random |
| start grid | general (no `E52_EVAL_STARTS_JSON`) | general (no `E52_EVAL_STARTS_JSON`) |
| evaluation seed | 404 | 404 |
| rollouts | 1024 | 1024 |
| `E52_DET` | 1 | 1 |
| seeds | 0, 1, 2 | 0, 1, 2 |

Nine evaluations. The destruction itself is the frozen `destroy_command_code` in
`snmr/integration/distillation.py`, which is inside the frozen science manifest and is **not
edited**.

Note the general grid, not the ambiguity grid: the frozen explicit destroy reports are named
`c_prior_explicit_eval_destroy_*.json` with no `ambiguity` component, so they were produced on the
general start grid. E75 matches that so the two arms are directly comparable.

## 3. Frozen inputs

Read-only. SHA-256 of the three SNMR student checkpoints, which must match the values already
registered in `docs/E71_COMMAND_SWAP_PROTOCOL.md` §3:

| Seed | `a_prior_snmr_student.pt` SHA-256 |
| ---: | --- |
| 0 | `f88984971c3435e3c377f038ed2ef5abef788aa1a0f68a80ad7011b23bb9b93a` |
| 1 | `185ac3991cb6bcdd451d719c72d5d72273d4351a45bd93e7f532ebeaec730d38` |
| 2 | `6d23363133df4ba30f6ddb12887aa22b932f6255a9e27031bdf57daf683e42c1` |

Motions and teacher manifest: the frozen `/data/robotixx/snmr-research/e70/motions/` and
`teacher_manifest.json`.

## 4. Isolation

Outputs go to a **new root**, `/data/robotixx/snmr-research/e75_snmr_destruction/`. Nothing is
written, moved, or deleted under `/data/robotixx/snmr-research/e70/`. The frozen checkpoints are
reached through symlinks, and the launcher verifies each symlink target's SHA-256 against the table
above before running.

## 5. Registered reference values (from frozen artifacts, before any E75 run)

Undestroyed general completion, `a_prior_snmr_eval.json`:

| Seed | completion | mean survival (s) |
| ---: | ---: | ---: |
| 0 | 0.684570 | 8.2737 |
| 1 | 0.702148 | 8.3177 |
| 2 | 0.708984 | 8.3375 |
| **mean** | **0.698568** | |

Two reference floors, both three-seed means on the same general grid:

- **goal-blind floor**: the proprioception arm, trained with no goal channel at all, reaches
  **0.430664** (per-seed 0.487305 / 0.555664 / 0.249023).
- **explicit arm under destruction**: **0.000000** in all nine cells.

## 6. Registered interpretation — written before the numbers are read

Let `Y_d` be the three-seed mean completion after destruction mode `d`.

| Branch | Condition | Registered reading |
| --- | --- | --- |
| **Total collapse** | `Y_d ≈ 0.000` in all three modes | The exclusive channel is behaviorally necessary for the SNMR controller, at the same strength already shown for the explicit controller. The abstract's necessity sentence generalizes from the control arm to the paper's own subject. |
| **Partial residual** | `0 < Y_d`, and `Y_d` at or below the goal-blind floor 0.431 | Necessity holds for the goal-directed component. The residual is what proprioception and closed-loop dynamics sustain without a usable command. **This is a publishable, informative necessity result, not a failure**, and it is reported as such. |
| **Weak dependence** | `Y_d` materially above the goal-blind floor | The SNMR controller does **not** depend on its exclusive channel to the degree the explicit controller does. This is a genuine negative for the paper's necessity rung on its own subject arm, and it is reported plainly, prominently, and without softening. |

Two things are registered explicitly so they cannot be decided after the fact:

1. **A non-zero residual is not a failure.** Destroying `z_cmd` puts the input far outside the
   student's training distribution; a policy that still stumbles forward for some fraction of
   rollouts is expected behavior, not a defect in the assay.
2. **This measures necessity, not content.** No E75 outcome says anything about *what* the channel
   carries. It cannot separate clip identity from within-clip trajectory state, and no E75 result
   may be cited toward that question. That is E72's job.

## 7. Stop rules

- Stop if any checkpoint SHA-256 differs from §3.
- Stop if any realized report has `destroy_zcmd` not equal to the requested mode, `evaluation_seed`
  not 404, or `num_rollouts` not 1024.
- Stop if anything under `/data/robotixx/snmr-research/e70/` changes mtime during the run.
- Report all nine cells regardless of outcome. No cell may be dropped, re-run for a better number,
  or excluded after inspection.

---

# RESULT — 2026-08-15 (appended after execution; nothing above this line was edited)

**Registered branch: TOTAL COLLAPSE.** All nine cells returned `completion_rate` exactly `0.000000`.

| mode | seed 0 | seed 1 | seed 2 | mean completion | mean survival (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `zero` | 0.000000 | 0.000000 | 0.000000 | **0.000000** | 0.535 / 0.618 / 0.620 |
| `shuffle` | 0.000000 | 0.000000 | 0.000000 | **0.000000** | 0.593 / 0.536 / 0.581 |
| `marginal_random` | 0.000000 | 0.000000 | 0.000000 | **0.000000** | 0.832 / 0.780 / 0.844 |

Every realized report carries `arm=a_prior_snmr`, `eval_z=prior`, `evaluation_seed=404`,
`num_rollouts=1024`, and all three noise fields at `0.0`. No stop rule fired.

## What this licenses

The behavioral-necessity rung now holds for **the arm the paper is about**, not only for the
control arm, and at the same strength:

| arm | intact general completion | under destruction (3 modes × 3 seeds) |
| --- | ---: | ---: |
| explicit (control) | 0.923 | **0.000** |
| SNMR (subject) | 0.699 | **0.000** |

Both collapse to exactly zero in all nine cells. The abstract's necessity sentence can now be
stated for the SNMR interface directly rather than scoped to the explicit student.

One observation worth reporting, because it is not obvious: destruction drives the SNMR controller
**below** the goal-blind proprioception arm's 0.431. A policy that never had a goal channel walks
better than one whose goal channel is destroyed. That is expected — destruction puts the input far
outside the training distribution, so this is not evidence that the channel is *more* than
necessary — but it does mean the destroyed arm is not a stand-in for a goal-free controller, and
the paper should not present it as one.

## What this does NOT license

E75 measures **necessity only**. It says nothing about *what* the channel carries. It cannot
separate clip identity from within-clip trajectory state, and no E75 number may be cited toward
that question. Both `shuffle` and `marginal_random` destroy clip identity and trajectory content
together, exactly as the frozen E70 shuffled arm does.

## Provenance

- Preregistration written before execution; this section appended after.
- Launcher: `scripts/run_e75_snmr_destruction.sh`, a replication of the frozen
  `run_explicit_destruction` changed only in the arm.
- Outputs: `/data/robotixx/snmr-research/e75_snmr_destruction/students/seed{0,1,2}_snmr/`.
- Frozen checkpoints reached by symlink and SHA-256-verified before the run; nothing under
  `/data/robotixx/snmr-research/e70/` was written, moved, or deleted (verified by mtime scan
  after the run).
- Paper macros: `scripts/render_e70_destruction_values.py --snmr-destroy-root ...` emits
  `\EDestroySnmrBaselineCompletion{0.699}`, `\EDestroySnmrCompletion{0.000}`,
  `\EDestroySnmrMaxSurvival{0.844}` alongside the explicit family, every input SHA-256 stamped.
