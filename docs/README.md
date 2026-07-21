# SNMR Documentation Index

Last organized 2026-07-21. Files never move (cross-references are everywhere); this index is
the map. **Reading order for a newcomer: ①→②→③.**

## ① Live plan — start here

| Doc | Role |
|---|---|
| `RESEARCH_PROPOSAL_RETARGET_TO_TRACKING.md` | **The active research program**: goal, hypotheses H1–H4, staged plan (B1 matrix → E49 → SSL refinement → act-through-latent CVAE → multi-clip/cross-embodiment), kill criteria |
| `PROGRAM_CONSOLIDATION.md` | Evidence map of everything built/learned/analyzed through 2026-07-20; the through-line; ranked open questions |
| `E49_ACT_THROUGH_LATENT_PROTOCOL.md` | Next registered experiment (config-only co-trained latent readout; Stage-0 wiring gate; stop rules) |
| `EXPERIMENT_LOG.md` | **Append-only ground truth** for every experiment E1–E48-100k. Newest last. All claims trace here |

## ② Main-line retargeting paper (C-claims)

| Doc | Role |
|---|---|
| `20260716-1931-sol.md` | Frozen 2026-07-16 research audit: claims table, priority order, Track A/B roadmap (partially superseded by ① for direction, still authoritative for C-claim status) |
| `NEURAL_RETARGETING_RESEARCH_sol.md` | 2026-07-13 audit with decision gates 0–6 (frozen record) |
| `NEURAL_RETARGETING_RESEARCH_fable.md` | Gate-1b/Gate-2 v2 redesign (2026-07-14 review; Gate 1b since closed — see log E40–E42) |
| `CURRENT_ARCHITECTURE.md` | Model/pipeline/code map with mermaid diagrams (status snapshot 2026-07-14) |
| `PAPER_DRAFT.md` | Paper skeleton |
| `SHARING_COST_SCREEN_PROTOCOL.md` | Frozen protocol for the C3 sharing-cost screen (registered, not yet run) |
| `DATA.md` | Pair-generation pipeline (LAFAN1 × GMR × 5 robots) |

## ③ Literature (adversarially verified)

| Doc | Role |
|---|---|
| `FORWARD_RESEARCH_MEMO.md` | Round 3 (2026-07-20): flow-matching verdict, data/supervision verdict, SSL ranking, v-next design + evidence ledger. **The most current literature synthesis** |
| `FLOW_RETARGETING_LITERATURE.md` | Rounds 1–2 (2026-07-17): guidance theory (Dirac/covariance collapse), substrate regularization, MSE conditional-mean, PULSE/MotionBERT/HuMoR |
| `WBT_LATENT_LITERATURE_REVIEW.md` | Track-B latent-command prior art (pre-E49) |

## Closed programs (kept as records; do not reopen without new evidence)

| Doc | Program | Outcome |
|---|---|---|
| `FLOW_RETARGETING_SIDE_PROJECT.md` | Flow v1 (E43–E45) | F1 pass, F2 fail; guidance structurally inert |
| `FLOW_RETARGETING_V2_PROTOCOL.md` | Flow v2 (E46) | Closed at entry (substrate + zr-decode retrofits fail) |
| `FLOW_RETARGETING_V3_PROTOCOL.md` | Flow v3 (E47) | Closed at entry; **local-uniqueness result** (Dirac is a data property) |
| `FLOW_MATCHING_RETROSPECTIVE.md` | Whole flow program | Built/learned/critique synthesis; §6 = final resolution of all open questions |
| `WBT_LATENT_PLAN_v2.md`, `WBT_LATENT_INTEGRATION_STUDY.md` | Track-B frozen-latent screens (E36–E39) | L1 feasibility only; S3 non-replication; concatenation inert |
| `WBT_LATENT_PHASE3_PROTOCOL.md` | Phase-3 multi-clip latent arms | BLOCKED on B-multi budget calibration (§3A); do not launch the arm matrix |

## Standing conclusions (the five-line summary of two weeks)

1. **Reference quality transfers to tracking** (B1: SNMR-ref 0.86 vs GMR 0.88) — confirmatory matrix promoted.
2. **Frozen-latent concatenation is inert; retrofit generation on the frozen latent is dead** (E36–E47, theory-grade).
3. **The deployable contact signal is the retargeting bottleneck** — five corrector families agree; the mask, not the solver.
4. **Co-trained auxiliaries structure the latent** (E48/E48-100k: contact F1 0.227–0.257, 1.8–2× deployable baseline; BCE weight 0.25 is the registered default going forward).
5. **The open lever is act-through-latent co-training** (UniTracker/PULSE-verified) — hypotheses H1–H4 of the live proposal.
