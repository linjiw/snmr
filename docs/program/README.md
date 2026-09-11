# The methods program: decisions and evidence, 2026-09-10

Response to the external advisor review of 2026-09-10. This directory holds the
audit results and the specification they produced. **Nothing here launched a
training job.**

## The five-day question first

**ICRA 2027 closes 2026-09-15, 11:59 p.m. Pacific.** The recommendation is
**submit Paper 1, treating Sep 14 as the deadline.**

The decisive fact: **not one revision item requires an experiment.** Every
displayed quantity is emitted by a hash-stamped renderer from frozen artifacts,
so the whole revision list is prose. The two non-negotiable edits
([R1](#r1) and [R2](#r2)) are **already applied** — the rebuild is 8 pages with
0 overfull boxes and every renderer-generated macro file is byte-identical, so
no number moved.

Also verify the PaperPlaza video window. The repo records Aug 5 – Sep 9 and
Sep 17 – 22; the first is closed and the second falls **after** the paper
deadline. The repo additionally records the deadline time **two different ways**
(23:59 PST and 11:59 PST). Confirm against the live CFP; a 12-hour error is fatal.

## What the audit changed

### The one real error, and why the honest version is stronger

The shuffled arm (S) was described in the manuscript as destroying clip identity.
It does not. Its donor map is the deterministic bijection
`(destination + 1) mod n` — a pure swap on E70's two-clip pool — so identity is
recoverable **at every pool size**. Adding clips would not have fixed it.

Measured, and reproducible via `scripts/measure_arm_identity_separation.py`:

| Arm | Between-clip separation |
| --- | ---: |
| A — frozen SNMR latent | 8.0415 SD |
| S — misaligned reference | **8.0415 SD** |
| T — time code | **0.0000 SD** |

S carried *exactly* as much linearly decodable clip identity as the arm that
wins — and scored at the identity-free clock's level. **Identity was maximally
available and bought nothing.** The corrected sentence is a stronger result than
the incorrect one.

Scope: the registered primary contrast is **A−T**, against a null that is
identity-free by construction and by regression test. It is unaffected. Only the
secondary A−S contrast is reinterpreted. Full audit:
[`../E70_SHUFFLE_CONTROL_AUDIT_2026-09-10.md`](../E70_SHUFFLE_CONTROL_AUDIT_2026-09-10.md).

### Control concerns that were checked and cleared

Per-arm standardization is a *necessary* matching step (raw code energy differs
163×), and is a diagonal affine the first layer can invert, so it cannot change
capacity. Observation normalizers are frozen and are the same object across arms.
Evaluation resets are exact-state and seed-matched with a realized-grid
assertion. No clip identity reaches the actor — and in E70 the privileged critic
observations never reach the student at all, because `E52_DET=1` disables the
posterior. Capacity runs *against* the explicit arm: it and the latent arm have
identical prior parameter counts, with `z_proj`'s 82k parameters live only in the
latent arms.

### Claims that had to be softened

**Destruction.** Split into the provable half — every bit of *goal* information
reaches the action only through the channel, which is architecture and needs no
experiment — and the measured half: the decoder has no proprioception-only
fallback. It cannot separate goal content from state-coupled content. E77's
frozen dose-response argues the state-coupled part is substantial: *holding* the
command for 0.1 s drops the explicit student from 0.926 to 0.223.

**Throughput.** "≈4× faster than GMR" is not a controlled measurement —
`runs/bench_g1_v2.json` records `teacher_timing: "not measured by this
evaluator"`. Both pages now mark the teacher figure as an external estimate.

**Legacy vs proposed.** `benchmark.html` presented legacy-architecture numbers
under "No per-robot IK configs, no per-robot tuning" — the proposed method's
value proposition — on five robots that were all **in training**, with no mention
of the one leave-one-robot-out result. Fixed, with the PM01 failure (29.59 cm
zero-shot vs 5.72 cm in-training) as its counterweight.

## The novelty finding that sets the program

Of GMR, NMR, ReActor, OmniTrack, GenTrack, AdaMorph, G-DReaM and KDMR —
**none evaluates on a held-out robot.** Every one generalizes over held-out
*motions*. The cross-embodiment papers are structurally unable to: AdaMorph
learns a per-robot prompt bank *and* a per-robot output head; G-DReaM retrains
with the new skeleton in the training set; ReActor re-runs bilevel RL per
embodiment; OmniTrack's references *are* that robot's own rollouts. NMR states in
its own limitations that morphology-conditioned architectures remain future work.

That makes one property load-bearing, so it was verified rather than asserted:
**an unseen robot costs zero new parameters.** One model, 847,361 parameters,
7 real humanoids spanning 19–29 DoF, identical parameter count; only the output
width tracks the robot (`scripts/verify_zero_new_parameters.py`, pinned by
`tests/test_morpho_model.py`).

**Caveat that must travel with it:** zero new *parameters* is not zero new
*work*. A new robot needs a hand-declared `SemanticManifest`, and one exists only
for G1. The defensible wording is *"no target-specific retargeter fitting, given
a robot model and declared semantic correspondences."*

Two threats the review did not name: **X-Morph** (arXiv:2606.30290) already holds
spec conditioning + physics-aware correction + closed-loop tracking, stopping
just short of held-out-robot transfer; and **H-Zero** (arXiv:2512.00971) already
normalizes leave-one-robot-out, so a reviewer will call the protocol standard
practice rather than contribution.

## A closed line worth reopening

"Every deployable contact mask fails" is an **aggregate** hiding a clean split. On
`walk1_subject5` the deployable mask is good — precision 0.711, F1 0.799, oracle
stance prevalence 0.482. On `fight1_subject3` the oracle finds **zero** stance
frames, so precision is 0 by construction; `dance2`, `run2` and `aiming2` sit at
0.3–2.5% prevalence. A fight clip with no ground contact is not a property of the
motion — the *oracle* is broken on six of seven clips.

This does not overturn the closure (these are precision numbers, not corrector
outcomes), but the evidence used to rule the line out does not support ruling it
out, and a **CPU-only, locomotion-scoped** repairability pilot is available today.
Details: [`../CONTACT_MASK_AGGREGATE_ARTIFACT_2026-09-10.md`](../CONTACT_MASK_AGGREGATE_ARTIFACT_2026-09-10.md).

The counterweight is already on disk: **E50-A** ran tracker-guided correction and
**passed physics while failing fidelity** at 9.7 cm against a 5 cm gate —
execution improved because the movement became an easier movement. That is the
degenerate solution, already observed once.

## The documents

| File | What it is |
| --- | --- |
| [`METHOD_SPEC_TRANSFERABLE_CORRECTIONS_2026-09-10.md`](METHOD_SPEC_TRANSFERABLE_CORRECTIONS_2026-09-10.md) | The minimal method: bounded spline corrections, the intent-preservation constraint set, the acceptance rule, and the falsification conditions |
| [`EXPERIMENT_PROGRAM_2026-09-10.md`](EXPERIMENT_PROGRAM_2026-09-10.md) | Splits, baselines, the zero-shot protocol, the crossed downstream matrix, statistics, cost accounting |
| [`PAPER_STRATEGY_2026-09-10.md`](PAPER_STRATEGY_2026-09-10.md) | Paper 1 revision list with line anchors; Paper 2 outline with empty result slots; the submit/don't-submit argument |
| [`novelty_audit_2026-09-10.json`](novelty_audit_2026-09-10.json) | Primary-source check against the eight named works |
| [`control_audit_2026-09-10.json`](control_audit_2026-09-10.json) | The five control-matching concerns, checked against code |

## The design commitment worth restating

In the method spec, intent preservation is **not a loss term**. Two of the three
degenerate failure modes are removed by the *parameterization*: there is no time
channel, so the correction cannot slow the motion down; and corrections are
zero-clamped at interval endpoints with no global multiplicative parameter, so
they cannot shrink it. Only "delete a hard transition" survives, and the
constraint set catches that. A candidate that violates intent is rejected before
it is ever scored for execution, so intent can never be traded for trackability.

## Ordered next actions, none of which need a GPU

1. **Confirm the CFP deadline time and the video window.**
2. **Write the tracker qualification floor** — a number, a reference set, a
   horizon, a rollout count. The only mention in the repo defines no value, and a
   completed checkpoint has been sitting unqualified since 2026-09-02. Writing it
   after observing a rollout makes it outcome-conditioned, which this project's
   own governance treats as disqualifying.
3. **Commit the working tree.** Every frozen iteration-2 artifact records
   `dirty: true`, so no provenance hash in that partition is verifiable from a
   clean checkout.
4. **Fix the inverted `model_amplitude_collapse` label** — 1.510 against a 1.50
   *upper* bound is amplification, and the error has propagated into a governing
   status document.
5. **Run the mismatched-spec negative control** before committing to any framing.
   If held-out performance does not degrade under a shuffled RobotSpec, the
   conditioning is inert and the gains are just a better average retargeter —
   which is what a hostile reviewer will allege.
6. **Run the scaling-in-N curve** before committing to any writing. A positive
   slope is the one result no per-robot method can produce. **If the slope is
   flat, the transfer claim is not real — learn that early.**
