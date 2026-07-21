# Research Proposal: Can Learning From Retargeting Benefit Motion Tracking?

**Date:** 2026-07-21. **Status:** PROPOSAL (synthesizes the full E43–E48 program, three verified
deep-research rounds, and the B1 downstream gate; supersedes nothing — it sequences what the
program docs already registered). Companions: `docs/PROGRAM_CONSOLIDATION.md` (evidence map),
`docs/FORWARD_RESEARCH_MEMO.md` (verified forward literature), `docs/E49_ACT_THROUGH_LATENT_PROTOCOL.md`
(first registered experiment of this proposal).

---

## 1. Research goal (one sentence)

> **Make the retargeter's shared motion latent *load-bearing for control*: a causal,
> co-trained command interface through which one tracking policy acts — so that what SNMR
> learned about motion (structure, contact, cross-embodiment alignment) measurably improves
> physical tracking, rather than riding alongside it as frozen decoration.**

The original side-project question was "can flow matching from retargeting benefit tracking?"
After ~6 experimental rounds and 251 verification agents across three literature passes, the
answer decomposes cleanly:

| Sub-question | Verdict | Evidence |
|---|---|---|
| Flow matching retrofitted onto the frozen retargeting latent? | **NO — closed, theory-grade** | E43–E47: Dirac conditional (a *data-generating-process* property, E47), guidance provably annihilated (Feng ICML'25), substrate dead in 3 variants |
| Flow matching per se the wrong tool? | **No — but it's not the axis that matters** | Verified: substrate regularization + target multimodality decide; both positive precedents (PULSE, MaskedMimic) are CVAEs; family choice is secondary |
| Can SSL/auxiliary objectives structure the retargeting latent? | **YES — proven + replicated at full budget** | E48: co-trained contact BCE → z-linear contact F1 0.257 @30k, 0.227 @100k (durable); 1.8–2× deployable baseline; real fidelity price at weight 0.5 → use 0.25 |
| Is SNMR-retargeted data as *trackable* as the IK teacher's? | **YES at dev-gate level** | B1 (2026-07-20): SNMR-ref 86% vs GMR 88% completion, RMSE slightly *better* |
| Does a frozen latent, concatenated, help tracking? | **NO — replicated** | E36–E39: S3 non-replication, C3 explains it; UniTracker (verified) explains *why*: z is ignored when the actor sees the explicit reference |
| Does a latent-only command work at all? | **Feasibility yes, value not yet** | E39 L1: 72–77% completion, 8–18 pp below explicit |

So: **not flow matching into tracking, and not frozen-latent concatenation — the verified,
still-open lever is co-training the latent with the controller as the actor's only command**
(the UniTracker/PULSE "act-through-latent" mechanism), with E48-style co-trained auxiliaries
supplying physics structure. That is this proposal.

## 2. Why this is the right question now (what 6 rounds bought us)

1. **The negative space is fully mapped.** Inference-time correction of the frozen model is
   closed across five corrector families; the bottleneck (deployable contact signal) is
   localized; the Dirac result says no generative head fixes it from below. We will not
   rediscover these dead ends.
2. **The two positive levers have never been combined.** E48 proved co-trained auxiliaries
   structure z *locally* (representation metric). B1 proved SNMR references train competent
   policies (control metric). Nothing yet connects them: no arm has ever trained the
   *latent* under the *control* objective.
3. **The verified literature is unusually prescriptive here.** UniTracker (CONFIRMED,
   arXiv:2507.07356): co-trained CVAE latent beats the same pipeline without it, SR 91.82 vs
   88.03, gain widening under noise, and z must be the actor's only command. PULSE (CONFIRMED):
   the *learned conditional prior* is the load-bearing piece (93.4→45.6→18.1% ablation).
   MaskedMimic (CONFIRMED): temporally-structured masking, not i.i.d. noise; multimodality via
   goal under-specification. These are design constraints, not vibes.
4. **The infrastructure exists.** Pinned holosoma WBT stack, latent export/obs interfaces,
   phase-stratified evaluator with bootstrap CIs, frozen driver pattern with provenance —
   E49's Stage-0 wiring probe is the only unproven mechanic.

## 3. Central hypotheses (falsifiable)

- **H1 (mechanism):** a *learned, co-trained readout* over a raw SNMR-latent window
  outperforms L1's frozen hand-designed tangent features as an actor command.
  *(E49, registered; config-only; the cheapest discriminating test.)*
- **H2 (act-through-latent):** a causal encoder + residual-to-prior CVAE command latent,
  co-trained inside the tracking loss (DAgger from the 86% B1 teacher), closes L1's 8–18 pp
  gap to the explicit command — and the SNMR-latent prior beats a from-scratch prior
  (i.e., *retargeting knowledge transfers*). *(Stage 3; the core claim of the proposal.)*
- **H3 (physics structure transfers):** adding E48's co-trained contact BCE (+ MaskedMimic-style
  span masking instead of i.i.d. corruption) to the encoder improves tracking robustness
  (completion under push/noise randomization), not just probe F1. *(Stage 3 ablation.)*
- **H4 (generalization is where the value shows):** the co-trained latent's advantage over
  explicit commands appears at multi-clip and cross-embodiment scale, not single-clip walk
  (which is near-null by construction — clip phase ≈ explicit command). *(Stage 4.)*

**What would falsify the whole program:** if Stage-3's co-trained latent cannot match the
explicit-command baseline on multi-clip data (H2 fails beyond single-clip), then the
retargeting latent adds nothing to tracking that explicit references don't already carry, and
the honest conclusion is "retargeting benefits tracking only through reference *quality*
(B1 result), not through representation" — publishable as the negative arm of the C6 story.

## 4. Plan

### Stage 0 — resolved this week
- **E48-100k replication** (`runs/e48_ssl_100k/`, COMPLETE + analyzed, logged): the fidelity
  cost is REAL, not convergence lag (A 2.18 cm — exact reference reproduction — vs D 3.68 cm at
  100k, flat from ~60k), so Stage-2/3 arms register BCE weight 0.25. The representation claim
  REPLICATES at full budget: D z-linear contact F1 0.227 (vs 0.257 @30k) = 1.8× the deployable
  baseline and 2.6× the matched control (0.088) — durable structure, and the cost knob is the
  weight, not the mechanism.
- **B1 confirmatory matrix** (frozen `run_wbt_reference_confirmatory.sh`, promoted, needs
  clean tree): 6 policies (GMR/SNMR × seeds 0/1/2) × eval {404,405}, ≈13 h GPU. The main-paper
  C6 claim. *Runs first; everything below queues behind it.*

### Stage 1 — E49: config-only co-trained readout (registered, ~1 GPU-day)
As registered in `docs/E49_ACT_THROUGH_LATENT_PROTOCOL.md`: arms `l1_frozen` (control) /
`e49_window_mlp` (raw latent window, plain MLP — primary) / `e49_window_enc` (MLPEncoder,
wiring probe). Promote iff ≥ L1+5 pp AND ≥80%; informative null closes the config-only line
and *strengthens* Stage 3. Stop rules preregistered; no hyperparameter fishing.

### Stage 2 — SSL objective refinement on the retargeter (≈2 GPU-days, interleaves)
The E48 follow-ups the memo ranks: (a) temporally-coherent **span masking** replacing i.i.d.
corruption (MaskedMimic-verified; kill: doesn't beat E48 i.i.d. numbers on contact F1); (b)
contact-BCE **weight 0.25** at full budget (from the 100k fidelity price); (c) optional
FLD-style multi-step future-prediction head (thin evidence — must earn its place, kill fast).
Deliverable: the encoder recipe Stage 3 initializes from, plus the paper's latent-analysis
section (E38 negative ↔ E48 positive).

### Stage 3 — act-through-latent CVAE co-training (the core; ~1–2 weeks)
The `algo._target_` custom-PPO route (scout-confirmed, no clone edits): a **causal window
encoder** (25 past / 5 future frames, UniTracker optimum) over the reference producing z_cmd,
a **residual-to-prior** structure with the *SNMR latent as the prior mean*, policy acting
through z_cmd (no explicit reference in the actor), DAgger-distilling the B1 teacher, PPO
fine-tune. Ablations (one factor each): prior = SNMR-z vs from-scratch (isolates H2's
"retargeting knowledge transfers"), ± contact-BCE co-training (H3), ± residual structure
(the PULSE-verified load-bearing piece). Primary readout: walk1 completion vs the three
anchors — explicit 88%, L1 frozen 72–77%, E49 result. Kill: cannot beat L1 at matched budget,
or ablating z leaves completion unchanged (z redundant again).

### Stage 4 — where the value must show (multi-clip, then cross-embodiment; ~2 weeks, gated)
Only after Stage 3 beats L1 on single-clip: (a) **multi-clip** — the existing 6-clip
walk-family set + held-out run2_subject1 (the cleaner primary), B-multi budget calibration
first (the standing Phase-3 blocker); explicit-command baseline trained identically; claim
requires held-out-clip completion parity + any robustness margin. (b) **Cross-embodiment
pilot** — one additional robot (T1) through the same co-trained interface; the shared-latent
promise (one command space, N robots) is SNMR's unique asset and the differentiating claim
vs UniTracker-style single-embodiment work. (c) If (a) passes, the deferred **physics-repair
loop** (Track A5) gets its calibrated teacher: repaired rollouts → retraining targets — the
teacher-level multimodality E47 demands, closing the loop back to retargeting.

### Explicit non-goals (kept dead)
Flow/diffusion retrofits on the frozen latent; per-step guidance of any sampler; contact-mask
/ solver iteration (Gate 1b); more clips (verified task-dependent, can hurt); a PHC-style
physics imitator per robot (~1 A100-week each — not on an A10G, and single-solution anyway);
teacher upgrades for flat walking (verified ~zero value until loco-manip/terrain).

## 5. Resources & schedule (A10G, single-GPU discipline)

| Item | GPU | Wall | Gate |
|---|---|---|---|
| B1 confirmatory matrix | ~13 h | this week | frozen, promoted |
| E49 Stages 0–2 | ~3 h | this week | registered |
| Stage-2 SSL arms | ~2 d | interleaved | E48 follow-up registered |
| Stage-3 build + train | ~3–4 d GPU inside 1–2 wk | weeks 2–3 | E49 result (promote or informative null) |
| Stage-4 multi-clip | ~3 d GPU | weeks 4–5 | Stage-3 beats L1; B-multi calibration |
| Stage-4 cross-embodiment pilot | ~2 d GPU | week 6 | Stage-4a parity |

## 6. Expected contributions (paper-shaped)

1. **C6 (main paper):** SNMR references train physically non-inferior tracking policies
   (B1 + confirmatory matrix) — retargeting quality transfers to control.
2. **Negative-space result:** generative retrofits on regression latents are structurally
   dead (Dirac/covariance-collapse, five-corrector convergence, E47 local-uniqueness) — a
   cautionary, theory-anchored section others will cite before repeating our mistake.
3. **Representation result:** contact is absent from distillation latents (E38) and co-trained
   auxiliaries put it there (E48, 100k replication) — the constructive counterpart.
4. **The headline, if H2/H4 hold:** a co-trained, SNMR-initialized command latent that matches
   explicit commands on held-out clips with better robustness, and extends to a second
   embodiment through one shared interface — *learning from retargeting, benefiting tracking*.
