# ICRA Paper Skeleton — "Act Through the Latent"

**Date:** 2026-07-28. **Status:** working skeleton from the paper-craft study (10 papers
dissected: ReActor, OmniRetarget, BeyondMimic, UniTracker, PULSE, HOVER, GMT, MaskedMimic,
HoST, VideoMimic). Full style guide with verbatim exemplars in the study report; this file
holds the decisions + drafts we iterate on. Companion: PAPER_DRAFT.md (claim/evidence C1–C7).

## Title (decision: #2, fallback #3)

1. ~~The Interface Is the Model: A Learned Latent Command Between Retargeting and Tracking~~
2. **Act Through the Latent: Closing the Retarget-to-Track Interface with a
   Goal-Conditioned Command Prior** ← submission candidate
3. From Retargeting to Tracking: One Latent as the Only Motion Command ← fallback

Rules applied: no architecture words (CVAE/DAgger/SNMR) in the title; differentiator as the
mechanism phrase; award papers drop system names entirely.

## Abstract v0 (~181 words; needs integrity-fix pass, see flags)

> The dominant humanoid motion pipeline is two-stage: retarget human motion to the robot,
> then train a tracking policy on the result. The interface between the stages is a stack of
> robot-space joint targets — a representation chosen for convenience, not for control. We
> ask what a *learned* interface carries instead. We first distill a per-frame IK retargeter
> into one multi-embodiment network with a shared latent, then audit that latent with
> falsification-first probes: it is content-rich (75% cross-embodiment clip retrieval at 1%
> chance, CKA 0.91), embodiment-aligned but not invariant (0.28 linear probe vs 0.91
> nonlinear attacker), physically impoverished, and — when frozen — inert as a control
> input, a null we replicate three ways and that UniTracker's ablation independently
> confirms. Making it load-bearing requires co-training: a goal-conditioned
> residual-to-prior CVAE, DAgger-distilled from an RL teacher, whose 64-d latent is the
> tracking policy's *only* motion command reaches 0.952±0.002 task completion against a
> 0.98 explicit-command teacher over three seeds, where the frozen-latent interface managed
> 0.72. We release the negative-results ledger, and a framework defect whose repair cut
> tracking error ~45%.

## The insight sentence (variant 1, recommended)

> Our key insight is that the retargeting-to-tracking boundary should not be a
> representation the two stages *agree on*, but one the tracking policy *learns to act
> through*: a latent becomes useful for control exactly when it is the policy's only channel
> to the goal, and is inert whenever an explicit reference is available beside it.

(It predicts our own nulls — the negatives become confirmations. Variant 2 in the study
report if reviewers skew representation-learning.)

## Contribution bullets (5): learned interface / falsification-first audit /
multi-embodiment retargeter / defect found-fixed-released / negative-results ledger.
Full drafted text in the study report §E — bolded-label + noun-phrase style (BeyondMimic),
pre-emptive qualifier clauses (HoST).

## Figure 1 concept (HOVER interface diagram × VideoMimic before/after)

(a) "The interface today": grey pipe labeled `29 joint angles + root pose / frame`, icons
for what's dropped (contact reasoning, uncertainty, cross-embodiment structure).
(b) "The interface learned": 64-d latent channel; goal arrow enters PRIOR only, visible
barrier before the decoder. (c) audit badge grid + 3-bar chart (0.72 / 0.952±0.002 / 0.98).
Caption drafted in study report §F — claim-bearing, panels lettered.

## Section map (6 pages)

I Intro (0.9) → II Related, 3 subsections ending on "the interface is the unexamined
variable" (0.7) → III The Two-Stage Interface: formalize + exclusivity contract; retargeter
compressed to infrastructure (0.8) → IV Auditing the Latent: 4 claim-headed probe families,
severity-ordered; E48 contact-BCE closes it (1.0) → V Making it Load-Bearing: method + the
v1→v2→v3 "the prior needs the goal" arc as a finding (0.9) → VI Experiments as questions
Q1–Q4 (E52 3-seed / additive-value incl. E53+E54 / B1-C6 with CI as-is / E50-A) (1.4) →
VII A Defect in the Measurement Substrate — own subsection, NOT appendix (0.5) →
VIII Limitations, VideoMimic stage-headed form, 5 heads (0.5) → IX Conclusion (0.15).

Negative results in 3 tiers: load-bearing (→ §IV & intro ¶4), scoping (→ §VI inline,
concession→mechanism→bounded-scope form), closed-line (→ §VIII one sentence + appendix).

## INTEGRITY FLAGS (blocking items before any submission)

- [ ] **FLAG-1: 0.952-vs-0.72 is cross-regime.** L1's 0.72 predates DEFECT-1's fix; E52's
  0.952 is post-fix. Fix (a): re-run one L1 frozen-latent seed post-bodyfix (~2.5h GPU;
  REGISTERED as E52-L1R, queued behind E53). Fix (b) if (a) unaffordable: report gaps to
  contemporaneous explicit baselines (pre-fix: 0.72 vs 0.88 = −16pp; post-fix: 0.952 vs
  0.98 = −2.8pp). Never print raw 0.72 beside 0.952 without one of these.
- [ ] **FLAG-2: "cut tracking error ~45%"** — pin metric + baseline explicitly: joint RMSE
  0.263→0.142 rad (−46%) fix-only, →0.122 with joint reward; heading-local MPJPE
  9.7→6.8 cm (−30%). Never say "halved".
- [ ] **FLAG-3: §VI-Q2 hole** — E53 16k in flight. If gate fails again, reframe Q2 around
  cross-embodiment (dimensional unshareability = structural value) with multi-clip
  reported as in-progress.

## Writing-craft rules to hold ourselves to (from the study)

1. Every table gets a reading paragraph: point → claim → mechanism → concession.
2. Counting-based honesty (HOVER): report denominators ("7 of 12 metrics").
3. Concessions in 3 beats: concession → mechanism → bounded scope. Never bare.
4. Question-headed experiment subsections (UniTracker/HOVER).
5. Limitations: mechanistic, stage-headed, closing with the agenda sentence. Never
   scope-statements ("we didn't do X").
6. Zero-penetration is a simulator property — never claim it.
7. Figure captions are complete claim-bearing sentences, panels lettered and named.
