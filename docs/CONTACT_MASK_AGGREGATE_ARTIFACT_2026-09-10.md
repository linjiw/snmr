# The deployable contact mask was not measured fairly: the aggregate is near-degenerate

**Date:** 2026-09-10
**Status:** finding, from re-reading a frozen artifact. No experiment was re-run.
**Consequence:** the Gate 1 / Gate 1b closure rests on an aggregate that is
uninformative on 6 of 7 clips. The corrector line should be reopened for the
locomotion family.

## The received conclusion

The program's standing summary is that every corrector family passes with an
**oracle** teacher-height contact mask and fails with any **deployable** mask, so
mask precision — not the solver — is the binding constraint. Gate 1 and Gate 1b
were closed as failures on that basis, and foot skate (0.289 m/s against the
teacher's 0.052) has been carried as unsolved ever since.

The aggregate that supports it, for the deployable `source_contact` mask:
**precision 0.132, recall 0.945, F1 0.359.**

## What the per-clip numbers say

From `runs/gate1b/mask_audit.json`, the same artifact, per clip:

| Clip | Precision | Recall | F1 | **Oracle stance prevalence** |
| --- | ---: | ---: | ---: | ---: |
| `walk1_subject5` | **0.711** | 0.941 | **0.799** | **0.482** |
| `sprint1_subject4` | 0.070 | 0.878 | 0.154 | 0.051 |
| `jumps1_subject2` | 0.057 | 0.890 | 0.493 | 0.036 |
| `run2_subject1` | 0.046 | 0.980 | 0.101 | 0.025 |
| `aiming2_subject3` | 0.035 | 1.000 | 0.180 | 0.025 |
| `dance2_subject4` | 0.004 | 1.000 | 0.044 | 0.003 |
| `fight1_subject3` | 0.000 | `NaN` | `NaN` | **0.000** |

The last column is the finding. **Precision is being scored against a positive
class the oracle has left almost empty.** On `fight1_subject3` the oracle labels
*zero* frames as stance across every window, so recall is undefined and precision
is 0 by construction — that clip contributes noise, not evidence. On `dance2`,
`run2` and `aiming2` the oracle finds stance in 0.3%–2.5% of frames.

A fighting motion with no ground contact is not a property of the motion. The
oracle — teacher-height hysteresis at enter 0.03 m / exit 0.05 m with a clip-local
ground estimate — is failing to identify stance on motions without clean flat
stance phases. So on six of seven clips the audit measures the oracle's failure,
not the deployable mask's.

On the one clip where the oracle behaves (48% stance prevalence, as a walk should
have), **the deployable mask is good: F1 0.799 at recall 0.94.**

## What this does and does not license

**It does not overturn the closure.** These are precision/recall numbers, not
corrector outcomes. Nobody has yet shown that a corrector driven by the deployable
mask succeeds on `walk1_subject5`; the claim here is only that the evidence used
to rule it out does not support ruling it out.

**It does license reopening the line, scoped.** A repairability pilot restricted
to the walk/locomotion family has a working deployable mask today. That is a
CPU-only experiment against existing correctors and an existing evaluation
harness.

**It also implies a reporting fix.** Prevalence must be reported beside
precision and recall, and per clip rather than aggregated, in any future mask
audit. An aggregate F1 over clips whose positive-class prevalence spans
0.000–0.482 is not a meaningful quantity.

## The other half of the warning, already measured

There is a second, independent reason to be careful here, and it is already on
disk. **E50-A ran the tracker-guided correction** — roll out a tracker, record the
simulated state, stitch it back as the reference — and it *passed* physics (stance
speed 3–6x below the teacher, zero penetration) while *failing* fidelity at 9.7 cm
heading-local MPJPE against a 5 cm gate.

That is precisely the degenerate solution to guard against: execution improved
because the movement became a different, easier movement. Any correction
experiment must therefore carry an intent-preservation **constraint**, not an
intent-preservation loss term — and the ruler for it already exists
(`snmr/semantic_metrics.py` scores a candidate against the *human* motion rather
than against the teacher, with two-sided amplitude/energy/jitter gates).

## Recommended next step

Scope a CPU-only repairability pilot to the locomotion family:

1. Re-run `scripts/audit_contact_masks.py` reporting prevalence per clip, and
   confirm the split above holds beyond the 6-window sample.
2. Diagnose the oracle on non-locomotion clips before using it as a reference
   anywhere else. A ground estimate or threshold that yields 0.000 stance on a
   fight clip is broken, and it is used elsewhere.
3. Run the corrector arms (no-op / smoothing / geometric-contact repair) on the
   walk family with the deployable mask, scored on execution *and* on the
   source-intent ruler, with E50-A's failure as the explicit negative control.

None of this needs a GPU or a qualified tracker, which makes it the cheapest
informative experiment currently available to the project.
