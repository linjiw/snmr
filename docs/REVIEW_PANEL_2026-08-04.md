# Internal review panel — 2026-08-04 (4 of 5 reviews in; stats/repro reviewer pending)

Five simulated reviewers with distinct backgrounds reviewed paper/main.tex.
Raw reviews in session transcripts; this file is the synthesis + action ledger.

## Score board

| Reviewer (lens) | Novelty | Sound | Rigor | Clarity | Overall (1-6) | Conf |
|---|---|---|---|---|---|---|
| Character animation (SIGGRAPH) | 2 | 3 | 3 | 3 | 3 (marginal reject) | 4 |
| Real-robot practitioner | 3 | 4 | 3 | 3 | 3 (borderline) | 4 |
| Representation learning (ICLR) | 3 | 2 | 3 | 4 | 3 (weak reject/borderline) | 4 |
| Adversarial verifier (numbers) | 3 | 3 | 4 | 3 | 3 (borderline; 4 with E63+fixes) | 5 |
| Stats/reproducibility | — | 4 | 4 | (repro 3) | **4 (weak accept)** | 5 |

Consensus: **3,3,3,3,4 — borderline+**, high confidence, every reviewer naming a
concrete path up. The strictest numbers audit scored HIGHEST — "the most reproducible
numbers set I have audited; every headline statistic reproduces to the digit" — i.e.,
the weaknesses are scope/controls, not integrity.

## Convergent findings (≥3 reviewers) → actions

1. **Single-clip/single-robot scope is the binding constraint.** All four. The paper
   motivates multi-clip/cross-embodiment and tests neither.
   → E53-2048 running (tier-deciding); cross-embodiment named the single best change
   by 2 reviewers (E54 after E53).
2. **The 0.656 clock confound (phase-vs-content) must be discharged, not disclosed.**
   Rep-learning (decisive form) + verifier (#1 fatal-if-unresolved).
   → E63 built, smoke-tested, queued FIRST post-E53 (3 seeds). Paper carries the
   caveat inline until the number lands.
3. **"Bottleneck" was unearned.** Rep-learning (dimension count) + verifier.
   → CONCEDED + measured: E64 (rank 14/64; z→goal R² 0.48 < proprio 0.62). Renamed
   to "learned command interface under an exclusivity contract" everywhere; §III
   explains why. DONE.
4. **CVAE machinery unattributed.** Animation (W5) + rep-learning (W2: posterior
   collapsed, l_kl≈0.003) + verifier (#3).
   → E62 (deterministic encoder) scripted + queued post-E63.
5. **Audit-section soft misreports.** Verifier found 4: retrieval range dropped the
   toddy outlier (57%), CKA low end 0.86, category claim contradicted logged E23
   supersession, scale-1% claim contradicted logged E11 caveat. Animation + rep also
   flagged the category/linear-probe asymmetry.
   → ALL FIXED in text (rounds 1-3 commits).
6. **0.656 rhetoric ("two-thirds") needs survival context; corruption = necessity not
   mediation.** Practitioner + rep + verifier.
   → FIXED: survival 7.9/10 s in abstract; "necessary, not bypassed" phrasing;
   Q1 rewritten with clock caveat.

## Divergent (single-reviewer) findings worth acting on

- Practitioner: completion metric undefined → FIXED (termination thresholds in §VI).
- Practitioner: latency/hold-z degradation test → registered idea (E65 candidate),
  cheap eval-only; not blocking.
- Animation: t-SNE/jargon/space allocation → partially fixed (jargon sweep);
  defect section kept at section level deliberately (it is a claimed contribution).
- Verifier writing nits → all applied except Fig-1c teacher-bar footnote (caption
  already states the protocol difference after round-3 fix... verify) and
  Table II unit label (pending Table II rewrite from E55-R anyway).

## Standing verdicts the panel did NOT overturn

- Number fidelity: verifier confirmed EVERY load-bearing number matches artifacts
  (30+ checks, zero mismatches outside the audit ranges now fixed).
- The exclusivity contract is real post-DEFECT-2 (rep-learning verified slices).
- Gate discipline + defect disclosure = the paper's strongest asset (all four).

## Path to 4-5 (explicitly stated by reviewers)

1. E63 clock control lands low → central claim established (verifier: 3→4;
   rep-learning: →accept with substitution test).
2. E53-2048 passes → multi-clip factorial → animation: 3→5; practitioner: →4/5
   (with DR); verifier: "solid 4".
3. E62 → attribution of the CVAE machinery (animation Q2, rep Q3).
4. E54 cross-embodiment (post-E53) → practitioner's and animation's single change.

## Round-4 fixes applied (stats reviewer)
- Fig-1c teacher-bar protocol note (1 seed x 100); Table I RMSE-censoring disclosure;
  E60 n=1 disclosure; E63 "running"->"queued"; reproduction recipe paragraph added to
  SVI preamble (teacher recipe, DAgger hyperparams, architectures, goal composition).
- Their decisive ask = same as verifier's: E63 clock control (already queued first).
- Teacher-bound precision ask (1024-rollout teacher re-eval) = cheap eval-only,
  added to post-E53 queue.

## Queue (GPU, in order)
E53-2048 (running, ~13.4k/16k) → E63 ×3 seeds → E62 → E57-B → [E53 students if gate
passed] → E54. CPU: E55-R specialists finishing → Table II rewrite.
