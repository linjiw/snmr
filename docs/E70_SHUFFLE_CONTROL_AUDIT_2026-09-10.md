# E70 shuffle-control audit: the shuffled arm is not an identity-erasing null

**Date:** 2026-09-10
**Trigger:** external advisor review flagged the public `same_phase_shuffled_latents`
implementation as potentially identity-preserving.
**Verdict:** **CONFIRMED as an implementation fact. Interpretation must change.
The primary vs-time result is unaffected.**
**Status of prior results:** preserved. Nothing is retracted or re-run.

## 1. The finding

`snmr/integration/distillation.py:220` selects the donor clip as:

```python
source = (destination + 1) % num_motions
```

The docstring calls this "a derangement for every pool with more than one clip," which is
true and was the property being aimed at — no clip receives its own latent. But a derangement
is not the property the control needs. This map is a **deterministic bijection**, so the donor's
identity determines the destination's identity exactly:

```
destination = (donor - 1) mod n
```

Clip identity is therefore **recoverable in principle at every pool size**. This is not a
two-clip artifact and adding clips does not fix it. What the two-clip case adds is that the
cyclic shift becomes an *involution* — a pure swap — which is the most legible instance of the
same defect.

E70 used exactly two clips:

```
scripts/run_e70_multitraj.sh:19    CLIPS=(walk1_subject1 walk1_subject5)
```

so the frozen assay ran `0 -> 1, 1 -> 0`.

## 2. The frozen revision matches the public code

`snmr/integration/distillation.py` has been modified exactly once, in `061cdec`
("E67-E70: two-walk ambiguity assay pipeline, protocols, plans, and audits", 2026-08-11) —
the commit that introduced the E70 pipeline. There is no divergence between the reviewed
public implementation and the revision that produced the frozen students. The concern applies
directly to the frozen results.

Affected consumers: `scripts/train_e52_dagger.py` (the E70 shuffled arm),
`scripts/train_e78_masked_fusion.py` (the `mShf` arm), `scripts/eval_e70_video.py`.

## 3. What the arm actually is

It is a **phase-matched, misaligned-reference control**: the student receives a real motion
latent, sampled at matched normalized time, belonging to a different clip than the one it is
being asked to track. That is a meaningful and useful control. It is *not*:

- an identity-erasing null,
- a content-free null,
- a lower bound on "what a command with no task information can achieve."

The content-free null in this assay is the **time code**, and that one is verified
identity-free by construction and by test: `shared_time_index_latents` gives bit-identical
codes at equal local frame indices regardless of clip, pinned by
`tests/test_distillation.py::test_shared_time_index_resets_without_motion_identity_leak`.

## 4. Did the students exploit the recoverable identity? Measured: no

This is the empirically load-bearing question, and the frozen numbers answer it.
From `reproducibility/reports/e70_seed0-1-2_analysis.json`, ambiguity completion:

| Arm | Identity available? | Phase available? | Ambiguity completion |
| --- | --- | --- | ---: |
| explicit reference | yes (full reference) | yes | 0.9733 |
| SNMR latent | yes (own clip) | yes | 0.7542 |
| time index | **no** (verified) | yes | 0.5622 |
| **shuffled latent** | **yes, recoverable** | yes | **0.5524** |
| proprioception only | no | no | 0.4365 |

An arm carrying recoverable clip identity plus matched phase scored **0.5524**, which is
*below* the identity-free clock's 0.5622 and far below what identity would buy if used: on two
deterministic clips, identity plus phase is in principle sufficient to select the correct
future at an ambiguous start.

So the students **did not invert the fixed map**. That is unsurprising in hindsight — nothing
in the objective rewards learning "when I see clip B's latent, produce clip A's action," and
the donor latent is a 64-d motion code, not a clean identity label — but it is a measurement,
not a guarantee. The arm is safe to report because of what was observed, not because of how it
was constructed.

## 5. Scope: what changes and what does not

**Unaffected — the primary result.** The headline is the paired contrast against the **time**
null: `+0.191 [0.124, 0.274]`. The time code is identity-free by construction and by test. This
comparison, and every conclusion resting on it, stands unchanged.

**Reinterpreted — the secondary result.** The paired contrast against the shuffled arm,
`+0.199 [0.127, 0.279]`, must now be read as "the correctly-aligned latent beats a
phase-matched *misaligned* reference," not "beats a content-free null." It remains a real and
useful control — arguably a harder one, since the misaligned arm receives genuine motion
content — but it does not license identity-erasure language.

**Strengthened, unexpectedly.** The shuffled arm doubles as a weak, unplanned instance of the
clip-ID-plus-phase diagnostic that E71 was designed to run and never did: identity was present
and recoverable, and it bought nothing. That is evidence against the memorization reading of
the assay. It is weak evidence — the identity was encoded, not labelled — and should be
reported as such rather than leaned on.

## 6. Required wording changes

Any text describing the shuffled arm as destroying, erasing, or removing clip identity, or as a
content-free / no-information control, is wrong and must be corrected. Replace with
"phase-matched, misaligned-reference control." The claim that the assay's *content-free* null
is the time code should be made explicit wherever both nulls are introduced together.

## 7. A replacement control, if one is built

A donor draw independent of target identity must not admit a single lookup table from donor to
destination. Concretely: draw the donor per episode (or per rollout) from a distribution that
does not depend on the destination, verify that the donor-to-destination map is not a bijection,
and retain enough temporal coherence that the arm remains a *misaligned reference* rather than
degenerating into command jitter — otherwise it stops being matched to the treatment in the way
that makes it informative. With only two clips, "the other clip" is precisely what must be
avoided, so a replacement control needs a larger pool to be meaningful at all.

`tests/test_distillation.py` now pins these properties:

- `test_same_phase_shuffle_leaks_recoverable_clip_identity` (n = 2, 3, 5)
- `test_same_phase_shuffle_is_an_involution_for_the_two_clip_e70_pool`
- `test_an_identity_independent_donor_must_break_the_donor_to_target_map`

The third states the property a replacement must satisfy, so the requirement is executable
rather than prose.

## 8. Related control matching, recorded here for completeness

**Lookahead.** `Z_OFFSETS = (0, 5)` — current plus 0.1 s at 50 Hz. The latent, time-code and
shuffled arms all route through the same `z_window()` / `_latent_at_offsets` path, so those
three are **matched**. The explicit arm reads `full[:, GOAL_SLICE]`, the current frame only, and
therefore receives *less* information than the latent arm — and still reaches 0.9733 against
0.7542. The asymmetry disadvantages the ceiling control and does not threaten any conclusion
drawn, but it must be disclosed, and it means the explicit ceiling as reported is a *lower*
bound on what a matched explicit arm would achieve. (E78's `mGf` arm exists precisely to give
the explicit command a matched future window.)
