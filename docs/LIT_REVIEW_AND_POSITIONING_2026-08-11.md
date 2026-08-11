# Literature Review & Positioning Report — 2026-08-11

**Scope:** full sweep of arXiv Jan–Aug 2026 + CVPR/ICML/ICLR/RSS/SIGGRAPH 2026 for scoop
threats and must-cites; methodology-lineage citation research; verification of all 22
existing bibliography entries. Every citation below was verified against its primary
source (arXiv abstract page, ACL Anthology, dblp, OpenReview, or publisher page) on
2026-08-11.

---

## 1. Headline verdict

**No scoop exists.** As of 2026-08-11, no paper (a) causally measures what a learned
humanoid command latent carries, (b) enforces an encoder-exclusive goal-routing contract,
(c) runs matched-control assays (time-index / proprioception / phase-matched shuffle) on
command representations, or (d) names the time-index confound in tracking evaluation.
arXiv metadata searches for "latent command" return zero papers. No prior work
preregisters a humanoid tracking assay — and no canonical robotics-preregistration
methodology paper exists at all.

The paper may therefore state, scoped: *"To our knowledge, no prior work causally
measures the content of a learned retarget-to-track command code."*

**Framing gift:** three independent 2026 threads gesture at the problem without solving
it — HANDOFF calls the command space "crucial" but picks one by engineering judgment;
HumanoidArena finds hierarchical results are tracker-conditioned; the
imitation-error/difficulty paper shows tracking metrics conflate difficulty with policy
quality. The community is discovering the symptom; this paper supplies the instrument.

## 2. Threat matrix (top findings, all verified)

| Paper | ID | Rating | Differentiation |
|---|---|---|---|
| Amadio & Mingo Hoffman, "What Matters in Humanoid General Motion Tracking? An Empirical Study" | 2607.19903 (Jul 22, 2026) | **SCOOP-RISK (moderate; the only one)** | Controlled study of the command channel, but varies *which explicit signals* cross; never touches learned codes, exclusivity, time-nulls, or causal destruction. Their finding that proprioceptive history cannot recover reference phase/velocity *supports* our time/content decomposition. Cite prominently. |
| AnyBody (S. Li, S. Li, J. Li, M. Ding) | 2606.29209 (Jun 2026) | MUST-CITE (scoop-adjacent) | Teacher→student distillation into a keypoint-addressable latent — architecturally similar pipeline, zero measurement. They engineer the latent for capability; we measure what a code carries under a causal contract. |
| HANDOFF (L. Yang, …, A. Ames) | 2606.06493 (Jun 2026) | MUST-CITE | Explicitly says interface choice is crucial, then engineers a 10-D explicit one. Use as motivation: they pick; we measure. |
| SONIC (Z. Luo et al., NVIDIA) | 2511.07820 (Nov 2025) | MUST-CITE | 700h foundation tracker with a unified token interface — exactly the kind of boundary object our instrument measures. |
| Humanoid-GPT (Z. Qi et al.) | 2606.03985, **CVPR 2026** | MUST-CITE | 2B-frame scaling generalist. Scaling the tracker doesn't answer what the channel carries. |
| OmniTrack (Y. Li et al.) | 2602.23832 (Feb 2026) | MUST-CITE | Intervenes on the boundary (launders references into dynamics-consistent motions) — engineered, not measured. |
| GenTrack (Z. Ling et al.) | 2608.01410 (**Aug 2, 2026**) | MUST-CITE | Generator–tracker co-training to close the "executability gap" at the same boundary. Very fresh — re-check for revisions before submission. |
| Meng et al., "Distinguishing Imitation Error from Intrinsic Motion Learning Difficulty" | 2512.07248 (Dec 2025) | MUST-CITE | Nearest published critique of tracking-evaluation confounds (difficulty vs performance); ours is a different confound (time vs content) with a causal instrument. NOTE: sweep initially reported a wrong paraphrased title; the title here is the verified one. |
| HumanoidArena (T. Wang et al.) | 2606.17833 (Jun 2026) | MUST-CITE | Benchmark evidence that the mid-level interface is an uncontrolled variable in current evaluations. |
| Robust & Generalized Humanoid Motion Tracking (Y. Ma et al.) | 2601.23080 (Jan 2026) | Tier-2 | Cross-attention command encoder → compact latent; engineering only. |
| Asynchronous upper-body tracking (Y. Liu et al.) | 2606.25706 (Jun 2026) | Tier-2 (useful foil) | Uses a time index as a *feature* and celebrates surviving command corruption; a properly exclusive code must *not* survive destruction. |
| HoRD 2602.04412 / Heracles 2603.27756 / OmniXtreme 2602.23843 / Hybrid Motion Priors 2607.24083 / LIMMT 2606.06953 (ICML 2026) / PoLAR 2606.21139 / MotionBricks (SIGGRAPH 2026) / WholeBodyVLA (ICLR 2026) / Switch-JustDance 2511.17925 / Motion Turing Test 2603.06181 | — | Tier-2 / NICE | Cite only if a reviewer asks or space is free. |
| Human2Humanoid retargeting | 2606.03476 (Jun 2026) | Tier-2 | New retargeting entry beside GMR/OmniRetarget/IKMR. |

No successor versions of UniTracker, HOVER, GMT, BeyondMimic, or BFM-Zero exist as of
2026-08-11.

## 3. Methodology lineage (previously absent — now added)

The paper's identity is a measurement instrument, yet it cited zero measurement
methodology. Added CORE set (all verified):

1. **Alain & Bengio**, linear classifier probes, arXiv:1610.01644 (2016) — origin of the
   probe method used in §V.
2. **Belinkov**, "Probing Classifiers: Promises, Shortcomings, and Advances," *Comput.
   Linguistics* 48(1), 2022 — one-stop "decodable ≠ used."
3. **Elazar et al.**, "Amnesic Probing," *TACL* 9, 2021 — the intervene-and-measure-
   behavior precedent for our zero/shuffle/marginal command destruction.
4. **Xu et al.**, "A Theory of Usable Information Under Computational Constraints,"
   ICLR 2020 — theoretical anchor for "control-usable information."
5. **Nosek et al.**, "The preregistration revolution," *PNAS* 115(11), 2018 +
   **Bertinetto et al.**, NeurIPS Pre-registration in ML workshop, PMLR 148, 2021 —
   preregistration lineage (scientific canon + ML-native).
6. **Geirhos et al.**, "Shortcut learning in deep neural networks," *Nature MI* 2, 2020 —
   the general phenomenon the time-index confound instantiates.

Deliberate skips: Hewitt & Liang (subsumed by Belinkov), Ravichander et al. (same),
Vig et al. causal mediation (Elazar suffices), Geiger et al. causal abstraction (heavier
machinery; invites scope-creep questions), Holden PFNN (DeepMimic already carries the
phase-variable lineage), Adebayo sanity checks (optional).

## 4. Changes applied to `paper/main.tex` on 2026-08-11

1. **Bibliography venue upgrades** (verification: all 22 prior entries are real and
   correctly attributed; six had newer archival venues; five applied):
   - omniretarget → ICRA 2026; gmr → ICRA 2026; unitracker → IEEE RA-L 2026 (dblp:
     11(7):8124–8131, DOI 10.1109/LRA.2026.3692091); roboghost → ICLR 2026 (OpenReview
     "Published as a conference paper"); bfmzero → ICLR 2026 (LeCAR lab page).
   - **NOT applied:** GMT → IROS 2026 rests only on a co-author's personal page and the
     conference hasn't occurred; kept as arXiv. Re-check in September.
2. **Related Work, "Tracking and its command spaces":** added foundation-scale trackers
   (SONIC, Humanoid-GPT) and extended the contemporaneous-latent passage with AnyBody and
   HANDOFF, closing with "all of these engineer the channel rather than measure it."
3. **Related Work, "Interface critiques" → "Interface critiques and measurement":**
   added Amadio & Mingo Hoffman, OmniTrack/GenTrack boundary interventions, the two
   benchmark audits (Meng et al., HumanoidArena), the "none makes the interface the
   object of a controlled causal measurement" line, and the methodology-lineage sentence
   (probes, amnesic interventions, V-information, preregistration, shortcut learning).
4. **16 new bibitems** appended; `\begin{thebibliography}{24}` → `{38}`.

All edits are confined to Related Work + bibliography. No result-bearing or
outcome-conditioned passage was touched; the three-branch macro system is unaffected.

## 5. Optional patches NOT applied (owner's choice)

- **Intro framing-gift sentence**, after "Yet the interface between these stages is
  rarely the object of study" (~line 190):
  > `Recent systems and benchmarks begin to flag the interface as decisive~\cite{handoff,humanoidarena,mds}, but treat it as a design choice or a nuisance variable rather than a measurand.`
- **Novelty sentence** (intro or conclusion): "To our knowledge, no prior work causally
  measures the content of a learned retarget-to-track command code."
- **§III V-information sentence** where "control-usable" is defined: one clause citing
  \cite{vinfo} tightens the term's theoretical footing (currently cited only in Related
  Work).

## 6. Pre-submission watch list (week of Sep 8–15)

1. Re-check arXiv v2+ of **2607.19903** (Amadio) and **2608.01410** (GenTrack) — both
   recent enough to grow analysis sections before the deadline.
2. Confirm **GMT → IROS 2026** via proceedings/dblp; upgrade citation if confirmed.
3. Re-run a quick "latent command / exclusivity / time confound" arXiv search for
   anything posted after 2026-08-11.
4. Verify BFM-Zero's ICLR 2026 status on OpenReview (currently sourced from lab page).

## 7. Reviewer-anticipation Q&A (rebuttal-ready)

- **"Only two walks / one robot / simulation."** Preregistered scope; the contribution is
  the instrument and the causal contract, demonstrated end-to-end; breadth is the named
  next study (held-out multi-trajectory generalization). Analogy: an assay paper
  validates the assay on known substrates.
- **"No comparison against ULTRA/UniTracker/SONIC."** Those are capability systems; this
  paper measures a representation property. A capability leaderboard would not answer
  what crosses the interface — and none of those systems exposes an exclusive channel
  that *could* be measured this way without re-architecting them.
- **"Isn't this just probing?"** No — probing measures decodability; the paper's own
  probes show the goal is *less* decodable from the code than from proprioception
  (R² 0.48 vs 0.62), yet the closed-loop assay shows the code is causally load-bearing.
  That dissociation (Belinkov's "decodable ≠ used", Elazar's amnesic logic, Xu's usable
  information) is precisely the point.
- **"The windows overlap; the CI is optimistic."** Preregistered primary analysis is
  clustered over frame pairs with seed as the outer level; a preregistered secondary
  temporal-block analysis addresses dependence (Plan step A2).
- **"AnyBody/HANDOFF already learn command interfaces."** They engineer interfaces;
  neither measures content, enforces exclusivity, nor runs matched controls. Cited and
  differentiated in §II.
- **"Time is a strawman baseline."** Time is a *matched null*, not a baseline: on a
  single clip it *beats* the latent (0.754 vs 0.656) — the paper's own retired claim —
  which is exactly why the two-walk ambiguity design exists.

## 8. Full verified bibliography corrections reference

All 22 original entries verified real, correct first authors/IDs/years. Optional
precision upgrades not applied (space-neutral, cosmetic): maskedmimic → ACM TOG 43(6)
(Proc. SIGGRAPH Asia 2024), DOI 10.1145/3687951; reactor full ref: ACM TOG 45(4),
Art. 97, 2026 (SIGGRAPH 2026), arXiv:2605.06593. ExBody2 correctly remains arXiv-only
(RSS 2025 workshop spotlight is non-archival). IKMR arXiv comments now say "RSS 2026
Workshop" (non-archival; keep arXiv).
