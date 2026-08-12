# Positioning & Benchmark Strategy — From One Paper to the Standard Instrument

**Date:** 2026-08-12 · Synthesized from three verified literature sweeps (prior-art/novelty,
2025–26 humanoid interface systems, benchmark-paper design patterns). Companion docs:
`docs/BENCHMARK_QUESTION_2026-08-12.md` (why not a capability race),
`docs/E72_NOISE_TITRATION_PREREG_DRAFT_2026-08-12.md` (first supporting experiment),
`docs/LIT_REVIEW_AND_POSITIONING_2026-08-11.md` (base threat matrix).

**Hard constraint:** `paper/main.tex` is re-frozen for the B4 video (manifest amendment
2026-08-12). Every paper edit proposed here is a *patch list* to apply only after the video
composes (or under a further recorded amendment). Nothing below touches E70's frozen claims.

---

## 1. The problem statement, precisely located

The verified novelty gap (2026-08-12 sweep): probing/intervention analyses exist for RL agent
*internals* in toy domains (Bush et al., ICLR'25 oral, arXiv:2504.01871; Zhang, arXiv:2603.21546);
probing exists as a correlational *quality proxy* for representations (AtariARI; RLC'24
light-weight probing); humanoid latent command spaces are validated *only by downstream
capability plus t-SNE* (ASE, CALM, PULSE, BFM-Zero, and the whole 2025–26 retarget-to-track
wave). **No existing work measures what a deployed robot command interface carries**, and none
combines matched nulls + ceiling control + causal destruction + preregistration into one
measurement contract.

**The named failure mode: interface opacity.** Modern humanoid stacks route control through
learned bottleneck interfaces whose content nobody can state. End-to-end success cannot
attribute competence to the interface: a policy can track a clip because the command carries
the motion — or because the command is a legible clock and the clip is memorized. The field has
already observed command-ignoring in the wild (Bogdanovic et al. 2022: velocity-command-ignoring
reward exploitation) without instrumenting it. de Haan et al.'s causal confusion says this
*should* happen; our E63 clock result caught it happening *to us* (time-index 0.754 beat the
latent 0.656 on a single clip — a favored claim retired by our own instrument).

**Candidate problem statements for the intro (pick one, owner's choice):**

- (P1, controls-first — ObjectNet pattern) "Experiments have controls; interface evaluations
  don't. We give the learned humanoid command interface its first controlled experiment."
- (P2, attribution) "Capability benchmarks measure whether a humanoid tracks; they cannot say
  what the command channel contributed. We build the instrument that can."
- (P3, opacity) "Every recent humanoid stack narrows to a learned command bottleneck that no
  one can read. We make its content a measured quantity with confidence intervals."

Recommendation: P2 as the opening problem sentence, P1's rhythm for the contribution sentence.

## 2. The named object and the named scalar

Benchmark-paper analysis (CheckList, ObjectNet, HANS, SentEval, RB2, RoboArena) yields five
adoption ingredients: a named failure mode + citable scalar; controls as the artifact's
identity; fixed probe / variable subject; near-zero marginal cost; shipped nulls for cross-lab
comparability.

- **The scalar: CNM — Content-over-Null Margin.** The paired completion advantage of the real
  command over its strongest matched null on ambiguity-controlled starts (our A−T = **+0.191
  [0.124, 0.274]**). One number a lab can put in a table; zero is the honest floor; the
  explicit-reference ceiling contextualizes it (0.973 vs 0.754 vs 0.562).
- **The assay name (owner's choice):**
  1. **CLAIM** — Command-Level Assay of Interface Meaning ("CLAIM your interface") — memorable,
     verb-able.
  2. **NullSpec** — matched-null specification testing for learned interfaces — controls-first
     identity, reads like a tool.
  3. **ICA** — Interface Content Assay — sober, precise, robotics-reviewer-safe.
  Recommendation: **ICA** in the ICRA paper (one mention, low risk), decide the branded name
  when the standalone benchmark paper is written.

## 3. The assay kit — what we already have vs what to build

Fixed probe, variable subject: any retarget-to-track stack with a command bottleneck plugs in.
We already own every component; the kit is packaging, not research:

| Kit element | Exists today as | Gap to "kit" |
| --- | --- | --- |
| Exclusivity contract check | E52/E70 architecture + DEFECT-2 leak detector story | write the 1-page contract spec |
| Ambiguity pair construction | E69 reference-only screening (no student in loop) | document the recipe; ship the 69 frozen pairs |
| Matched-null battery | time / phase-shuffle / proprio arms (frozen recipe) | expose as config, not code edits |
| Ceiling control | explicit-reference arm | already generic |
| Causal destruction battery | zero / shuffle / marginal-random (`E52_EVAL_DESTROY_ZCMD`) | already env-var driven |
| Sensitivity curve | **E72 noise titration (drafted)** | run post-video |
| Paired analysis + gates | hierarchical cluster analyzer + temporal-block bootstrap | already fail-closed, hash-stamped |
| Assay card | — | 1-page reporting template (Pineau-style) |
| Preregistration template | our own E70 prereg docs | genericize |

## 4. Supporting experiments, ranked by cost-to-strength

1. **E72 noise titration (drafted, eval-only, ~2–4 GPU-hours).** Turns the binary contrast into
   a dose–response curve → "the assay measures a graded, causally manipulable quantity."
   Run after the B4 video; freeze the prereg first.
2. **E73 known-content calibration (post-submission, ~1 GPU-day).** Train students on
   PCA-k compressions of the explicit reference (k ∈ {4, 16, 64}, 3 seeds, frozen recipe):
   channels whose content is known *by construction*. A monotone assay-vs-k curve is the classic
   instrument-validation move (measure standards of known concentration). This is the heart of
   the standalone benchmark paper.
3. **E74 second-interface application (post-submission).** Run the unchanged assay on one
   external latent (e.g., a public CVAE retargeter or BFM-Zero-style latent adapted to G1) —
   "fixed probe, variable subject" demonstrated across subjects, which is what makes it a
   benchmark rather than an experiment.

## 5. What lands where

**ICRA paper (after video; minimal, page-safe patch list — DO NOT APPLY while main.tex is frozen):**
- Intro: one problem-statement sentence upgrade (P2 pattern) + one contribution phrase naming
  the assay ("an interface content assay with matched nulls, a ceiling control, and causal
  destruction") + the CNM number in the contribution list.
- Related work: 2–3 sentences / citations: Bush et al. (2504.01871), Zhang (2603.21546) as
  method-adjacent probing+intervention in toy RL ("we import this rigor to a deployed command
  interface"); Bogdanovic et al. (2107.06629) as observed-in-the-wild command ignoring; Liu et
  al. humanoid football (2105.12196) as the only prior humanoid-latent probing (agent knowledge,
  not interface content).
- Limitations/future: one sentence announcing the assay kit + calibration program (E72/E73/E74)
  as released artifact direction.
- If E72 completes in time and the page budget allows: one small dose–response figure, clearly
  labeled non-preregistered-with-E70 / separately preregistered.

**Standalone benchmark paper (RSS/CoRL 2027 target):** the kit (§3) + calibration (E73) +
second subject (E74) + assay card + public results table. That is where "build a new benchmark"
fully lands, with this ICRA paper as the founding measurement.

## 6. Systems-contrast: the 2025–26 wave asserts what we measure

Verified sweep 2026-08-12 (all primary sources checked; no scoop found through today,
including a dedicated post-Aug-1 sweep). The literature's three problem-statement patterns:
(1) "one policy, all motions" — GMT (CoRL 2025), UniTracker (RA-L 11(7):8124–8131, 2026),
SONIC, Humanoid-GPT (CVPR 2026), GenTrack; (2) "the retargeting pipeline is broken" —
OmniRetarget (**won ICRA 2026 Best Paper**), GMR, NMR, LIMMT (ICML 2026), RoboGhost (ICLR
2026); (3) "what should planners talk to" — HANDOFF, ULTRA, BFM-Zero (ICLR 2026), AnyBody.

**The differentiation line the evidence supports:** the field designs, scales, or cleans
command interfaces — and when it makes claims about what those interfaces carry, the evidence
is a t-SNE plot or an outcome ablation:

- **ULTRA** (arXiv:2603.03279; same G1, same 64-d distilled-from-tracker latent): *asserts*
  "leaving the latent to capture mainly ambiguity and multimodality under sparse goals" —
  evidence: t-SNE + KMeans coloring. **They claim exactly the quantity our assay measures.**
- **UniTracker**: claims its CVAE latent captures "inherent ambiguity in the mapping from
  observations to actions" — no empirical validation of the claim at all.
- **BFM-Zero**: "objective-centric, explainable, and smooth latent representation" — asserted.
- **HANDOFF**: opens with "the choice of command space (i.e., the interface between task
  planning and whole-body control) is crucial" — then *designs* one by argument; never
  measures. Ideal opening quote for our interface-matters paragraph.
- **Amadio & Mingo Hoffman** (2607.19903, still v1 as of today): compares explicit command
  *representations by tracking outcome* — the "controlled study" precedent, but never asks what
  any representation contains.
- **Chamachot** (2603.21268): the only quantitative humanoid-latent probing found (probe R²,
  DCI/MIG on locomotion latents, sim-only, auxiliary-supervision setting) — methodological
  neighbor, wrong object; cite and distinguish.
- **LIMMT** (ICML 2026) claims "first data-centric study" for humanoid tracking — our "first"
  must therefore be worded precisely: *first controlled content assay of a learned
  retarget-to-track command interface*.

**The intro move this licenses (patch list, post-video):** quote the assertions ("captures
ambiguity and multimodality", "inherent ambiguity", "explainable and smooth"), then: *we give
these assertions their first controlled measurement — and for one such interface, the assertion
is true and quantifiable: +0.191 [0.124, 0.274] of ambiguity-resolving completion beyond a
matched clock, erased to zero by channel destruction.* The biggest positioning threat (ULTRA)
becomes the paper's motivation.

**Watch-list updates from this sweep:** GMT verified CoRL 2025 (upgrade our arXiv citation at
patch time); UniTracker verified RA-L 11(7):8124–8131 2026; OmniRetarget won ICRA 2026 Best
Paper (strengthen the citation); BFM-Zero/RoboGhost ICLR 2026 confirmed; 2607.19903 still v1;
GenTrack v2 (Aug 5) framing unchanged; new low-severity neighbors ω-0 (2608.06375), LUCID
(2608.07746).

## 7. Discipline notes

- All new citations must be verified against primary sources before entering main.tex (standing
  rule after the 2512.07248 misreport incident).
- E72 launches only after `POSTPROCESS_COMPLETE`; E73/E74 are post-submission.
- The E70 analyzer, its numbers, and its interpretation sentence never change. The benchmark
  framing *wraps* the frozen result; it does not reopen it.
