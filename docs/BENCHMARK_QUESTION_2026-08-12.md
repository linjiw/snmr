# The Benchmark Question — Why "Explicit + Latent" Cannot Beat the Baseline, and What "Stronger" Actually Means Here

**Date:** 2026-08-12 · Owner question: should we add an arm that gives the student the explicit
reference AND the learned SNMR latent, to beat the explicit baseline and sell the paper on a
capability win?

## We already ran that experiment — twice

1. **E52 v3/v4 (the direct test).** Arm C = explicit-only prior. Arm D = explicit + SNMR latent
   window in the same prior — exactly the proposed combination. Seed 0 suggested +2.1pp; the
   preregistered three-seed replication returned the honest verdict: **null — the paired deltas
   straddle zero**. "On single-clip walk the explicit command already saturates the goal
   channel." (One real secondary: arm D's across-seed variance was 4× tighter — the latent may
   act as a stabilizer. One sentence of optional paper text, not a claim.)
2. **E36-era observation screens.** Latent preview added beside the explicit command in the
   actor **degraded** tracking at that scale; no arm promoted. UniTracker's published ablation
   independently replicates the mechanism. This is already a row in our negative-results ledger.

## Why this is not bad luck but arithmetic

- The explicit student is **already at teacher parity**: E70 general completion 0.923 vs the
  teacher's macro 0.917 (and ambiguity 0.973). A distilled student cannot exceed the teacher it
  imitates; the only way to raise the explicit number is a better RL teacher — a different
  research program, not a different command channel.
- The latent is a **compression of the same retargeting pipeline output** that produces the
  explicit reference. On clips the explicit reference describes completely, the latent has no
  independent information to add; it can only add variance (E36) or nothing (E52 v4).
- On the ambiguity assay the explicit arm sits at 0.973 — it is the assay-validity **ceiling
  control by design**. Headroom above it is ≤0.027 and bounded by the teacher. No combined arm
  can produce a sellable margin there.

## Why "beat the benchmark" is the wrong battlefield for this paper

- There is **no external benchmark in this game**. ULTRA, RoboGhost, GMT, SONIC play the
  capability game with orders of magnitude more compute; entering that race with one GPU and 34
  days guarantees a loss and — worse — reframes our paper as a weak capability paper instead of
  a strong measurement paper.
- Reviewers of a measurement paper do not ask "did you beat SOTA?" They ask "is the instrument
  valid, are the controls matched, is the effect real?" Every one of those doors is already
  closed (five arms, paired CIs excluding zero on both clips and under the robustness analysis,
  causal destruction to 0.000 at three seeds, preregistration).
- Our credibility asset is precisely that the instrument **ruled against us** twice (E63 clock >
  latent; E52 v4 additive null) and we published the verdicts. Bolting on a post-hoc arm to
  chase a positive would spend that asset.
- The capability story we *can* sell truthfully: teacher-parity distillation through a
  production interface (0.923 ≈ 0.917), now **sim2sim deployment-qualified** on the target CPU
  MuJoCo platform (Phase E). "It is real enough to deploy" — without a single SOTA claim.

## What "gain stronger" means with the time we have

| Priority | Work | Status |
| --- | --- | --- |
| 1 | B4 video → human review → B5 PaperPlaza submission | supervisor waiting on GPU gate |
| 2 | Watch-list rechecks (scoop scan, GMT venue, arXiv revisions) week of Sep 8 | scheduled |
| 3 | Optional one-line paper adds: D-arm stabilizer sentence; teacher-parity + deployment-qualified sentence in reproducibility paragraph | owner's choice |
| post-submission | C1 held-out multi-trajectory generalization (the named next study) — the *right* way to get a bigger result | Phase C |
| post-submission | Richer pretrained motion priors (BFM-Zero-style semantics) as the latent source — the owner's "pretrained human understanding" instinct, done as its own preregistered program | Phase C+ |

**Recommendation (decisive):** do not add a combined-command arm before submission. The
experiment exists, its answer is null, and repeating it post-hoc weakens the exact property
that makes this paper sellable. Spend the remaining window on B4/B5 and ship; put the
semantic-prior idea at the head of the post-submission queue where it can win on its own terms.
