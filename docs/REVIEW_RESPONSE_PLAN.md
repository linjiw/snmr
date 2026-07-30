# Review triage — paper draft v0.2 (internal review, 2026-07-31)

**Status:** ACCEPTED AS SUBSTANTIALLY CORRECT. The review's central finding is right and
verified against our own code: **the successful prior consumes the 58-d robot joint
command** (`train_e52_dagger.py:47,81` — `cmd` is `actor_obs[:58]`), so the honest current
claim is an *actor-side information bottleneck distilled from an explicit teacher*, not a
replacement of the retarget-to-track interface. Every blocker is triaged below as
FIX-NOW (writing), EXPERIMENT (queued with design), or REBUT (with evidence).

## Verification of the review's factual claims (done 2026-07-31)

- B1 prior-consumes-cmd: **VERIFIED** (`prior_input` concatenates `cmd`).
- B4 root-pinned eval: **VERIFIED** (`train_e55a_twoteacher.py:180` FK uses GT root).
- B8 audit numbers: **VERIFIED** — retrieval is 98 windows → chance 1.02% not 1.8%;
  window- not clip-level; E2 motion probe 0.151 vs 0.125 chance (near-chance);
  full-budget control contact F1 is 0.088 (0.023 was the 30k undertrained point).
- B5 regime mismatch: **VERIFIED** — L1R is PPO 512env×8k×1seed; students are DAgger
  1024env×2000rounds×3seeds. "One regime" as written is false.

## Blocker-by-blocker disposition

| # | Verdict | Action |
|---|---|---|
| B1 interface not replaced | **CORRECT** | Reframe paper: the current result is "a learned 64-d actor bottleneck distilled from an explicit-command teacher (0.94–0.95, 3 seeds)". The interface-replacement claim moves to a clearly-labeled "toward" section backed by E58 (below). Title keeps "Act Through the Latent" but the subtitle changes: "a goal-conditioned command bottleneck for humanoid tracking" until E58 lands. |
| B2 "same latent" wrong | **CORRECT** | Global terminology pass: z_snmr (128-d, frozen, retargeting) vs z_cmd (64-d, co-trained, control) distinguished everywhere incl. abstract; drop every "the same latent" phrasing. |
| B3 no single multi-X model | **CORRECT** | Contribution bullet rewritten: "a multi-embodiment retargeter (5 robots, one checkpoint) AND a two-teacher G1 specialist (separate checkpoint, 1.14M params)" — two artifacts, not one. |
| B4 Table II overclaim | **CORRECT** | Relabel as "preliminary articulation-fitting under GT root" in caption + text; E55-R redo queued (full root loss, group-held-out splits, no shared-human variants across split, object/terrain conditioning arms, specialist baselines). |
| B5 0.65→0.952 not same-regime | **CORRECT** | (a) Immediate: reword to "under the repaired reward stack" and enumerate the differences in a footnote; the honest defensible contrast is student-vs-teacher (0.952 vs 0.98, same algorithm family) and frozen-vs-cotrained *within DAgger* — which needs E59: L1-style frozen-z_snmr command trained under the IDENTICAL DAgger recipe/budget/seeds. Queued. |
| B6 v2→v3 confounded | **CORRECT** | Factorial ablation E60 queued: {goal-conditioning} × {prior-path action-loss samples} (4 cells, 1 seed each ≈ 8h). Paper wording until then: "we changed two things; the combination fixed it" — no "load-bearing" attribution. Also fix: the arm uses the CURRENT 58-d command, not a future window — correct §III/§V text. |
| B7 scope untested | **CORRECT** | Title/claims narrowed (B1); E53-2048 + E54 remain the unlock for the original title. |
| B8 audit numbers | **CORRECT** | All three fixed in paper + site: 1.02% window-retrieval chance; F1 0.088 full-budget control; "content-rich" → "instance-aligned (semantic category near chance: 0.151 vs 0.125)" — which actually SHARPENS our inertness story. |
| B9 Q3 rerun post-defect | **CORRECT in part** | E57-A (positive control) was already registered and unrun — promoted to next GPU slot. Post-fix GMR-vs-SNMR matrix (E57-B') queued behind it. Old B1-confirmatory stays in paper as "pre-repair, paired" with explicit caveat. |
| B10 defect scope | **CORRECT** | Remove "and, we believe, others'"; scope to our stack + pinned rev; file upstream issue with repro (queued as artifact task); keep the 1-seed caveat explicit (though R/A/S = 3 runs share the fix direction). |

## New experiment queue (order after current GPU jobs)

1. **E57-A** positive control (registered, unrun — reviewer caught this) — 3h.
2. **E59** frozen-z_snmr-only command under identical DAgger recipe (the correct
   within-regime frozen baseline) — 3 seeds × 2h.
3. **E60** v2→v3 factorial {goal} × {prior-path loss} — 4 cells × 2h.
4. **E58** the real interface arm: prior consumes ONLY [proprio, z_snmr window] — no
   58-d robot command anywhere in the student; tracking gradient optionally into the
   SNMR encoder (two sub-arms: frozen vs fine-tuned encoder). THIS is the experiment the
   original title needs. Risk: known-hard (E52 v1/v2 showed goal-free priors fail) — but
   z_snmr IS goal information (it encodes the reference motion), so the hypothesis is
   live; if it fails, the honest paper is the bottleneck paper.
5. **E55-R** two-teacher redo per B4 (full root, clean splits, conditioning arms).
6. E53-2048 / E54 as before.

## Writing changes applied immediately (this commit)

- Abstract, Fig-1 caption, §I, §III, §VI-Q1: reframed per B1/B2/B5; numbers per B8.
- Statistical units: "3 training seeds (SD shown); eval seeds pooled" stated once in
  §VI preamble; n=3 variance claim deleted (B-recommendation 5).
- "preregistered" → "pre-specified" throughout (protocols are in-repo, timestamped by
  git, but not on an external registry).
- Defect section: scope narrowed, upstream-issue plan stated.
- PAPER_DRAFT_ICRA.md marked as superseded pointer to paper/main.tex (single source).

## What survives untouched

The strongest defensible result (reviewer's own words): a goal-conditioned 64-d actor
bottleneck distilled from an explicit-command teacher reaches 0.94–0.95 completion on
single-clip G1 walking across 3 seeds, where naive alternatives fail (frozen-concat
harmful; posterior-path collection 0.001; goal-free priors 0.02–0.05). Plus: the audit
(with corrected numbers), DEFECT-1 (scoped), the two-teacher absorption (relabeled
preliminary), and the negative-results ledger.
