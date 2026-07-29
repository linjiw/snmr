# Interaction-heavy data: risk register + BFM-Zero lessons

**Date:** 2026-07-29. **Trigger:** PI flags (1) OmniRetarget interaction data may not help
retargeting/training if we lack the structure/framework/simulation to handle interaction;
(2) status of transformer/flow-matching latent experiments; (3) read BFM-Zero (2511.04131).

## 1. Interaction-data risk register (honest, verified against our stack)

The concern is correct. Three distinct places interaction can bite, with our exposure:

| layer | risk | our current exposure | mitigation in flight |
|---|---|---|---|
| **Retargeting (encoder/decoder)** | interaction poses unexpressible | LOW — E55-B inversion probe: residual 0.05-0.067 rad, no cliff; E55-A twoteach interim: omni-val 26→5.3 cm | none needed yet |
| **Retargeting (conditioning)** | the SAME human motion maps to DIFFERENT robot motions depending on terrain/object state the encoder cannot see (hidden variable h). Without h, the decoder averages siblings → mode-averaged mush *specifically on interaction clips* | **HIGH — this is the real risk.** Quantified: 4,659 identical-human sibling pairs, dof divergence 0.031-0.068 rad mean. Our per-frame z has no terrain/object input | **E56-D is exactly this test** (variant-code conditioning); object_pose is already preserved in our converted pairs; terrain height is recoverable from filenames (z_scale) |
| **Tracking (simulation)** | holosoma WBT asserts `Object is only supported in IsaacSim` (wbt.py:593) — object-interaction TRACKING cannot run on our MuJoCo-Warp stack; terrain is flat-ground-only in our current envs | **HARD LIMIT for downstream training** on robot-object clips; terrain clips partially usable (no terrain geometry in sim → climbing motions will fail in tracking even with perfect references) | scope decision below |
| **Benchmark metrics** | our foot-contact metrics assume ground plane z=0; climbing clips break the penetration/skate readouts (feet legitimately at 0.76 m) | MEDIUM — affects reporting, not training | flag interaction clips in benchmark; report contact metrics only on flat-ground clips |

**Scope decision (registered):** interaction data enters the program at the RETARGETING
level (E55-A: representation absorbs it; E56-D: conditioning explains it) — where we have
measurements showing it works. Interaction TRACKING (object/terrain in sim) is explicitly
OUT OF SCOPE for this paper: our simulator path cannot support objects, and building
terrain envs is a project of its own. The paper says this in Limitations. The interaction
data still earns its place twice over: (a) it is the multimodality source that unblocks
generative decoders, (b) it stress-tests skeleton-agnosticism (52/53-joint skeletons).

## 2. Transformer / flow-matching status (direct answer)

**Honest answer: NOT yet run — deliberately sequenced, and the order is evidence-based.**
- **Transformer (E56-A):** registered, not run. Preceded by E56-B (GAT scaling sweep)
  because three verified findings say a transformer swap at 1.5M params / 2.5M frames is
  expected to LOSE (MolGPS data-efficiency, Graphormer's 139x-params-for-nothing, graph
  scaling collapse at 10M params on small data). The experiment that must come first is
  DATA scaling (E55-A, running) — SONIC shows scale helps *with* 100M+ frames.
- **Flow matching (E56-C):** was BLOCKED by our own E43-E47 theory-grade negative (Dirac
  conditional from a single deterministic teacher). Now UNBLOCKED by the OmniRetarget
  multimodality finding (0.031-0.068 rad divergence on identical inputs = 16-36x the E47
  floor) — but gated behind **E56-D** (teacher/variant-code conditioning), because the
  multimodality is latent-variable-explained: if conditioning on h explains it, a flow
  decoder has nothing left to model and dies again (a stronger, cheaper negative).
  When run, E56-C uses **MeanFlow** (2505.13447) in output space — chosen because at a
  Dirac conditional its bootstrap term vanishes identically → collapses to exact
  regression → downside bounded at our baseline.
- **What HAS run on the latent/pretraining side:** E48/E48-100k (contact-BCE co-training
  structures z durably), E43-E47 (flow retrofits on frozen latent — closed with theory),
  E55-B (inversion probe), E55-A (two-teacher pretraining, in flight), plus the full E52
  CVAE line (which IS a latent-generative model — residual-to-prior Gaussian CVAE —
  trained and replicated at 0.952).

## 3. BFM-Zero (2511.04131, CMU+Meta, Nov 2025) — what it is and what we take

**Mechanics (read from the paper):** unsupervised RL (no tracking rewards!) with
Forward-Backward representations on a real Unitree G1. Learns F(o_hist,s,a,z) and B(s)
such that successor measure M^{π_z}(ds'|s,a) ≈ F^T B(s')ρ(ds'). One latent space Z embeds
motions (z = mean of B over trajectory states), goals (z = B(s_goal)), and rewards
(z = Σ B(s_i) r_i) → one promptable policy, zero-shot tracking/goal-reaching/reward-opt,
few-shot adaptation by sampling-based search in Z. Key training choices: off-policy
actor-critic, history-dependent asymmetric learning (actor sees obs history, critics see
privileged state), FB-CPR-style latent-conditioned discriminator regularizing toward a
mocap dataset, auxiliary safety critic, massive parallel envs + high UTD, standard DR.
Deployed on real G1.

**Relation to us (important contrasts):**
1. BFM-Zero's z is trained BY control from day one (F/B are successor features — pure
   control semantics). Ours starts as a retargeting representation and is made
   control-bearing by distillation. OPPOSITE ENTRY POINTS into the same target: "one
   latent space where motions, goals, and rewards live." Their Fig-2 pipeline has no
   retargeting at all — the mocap enters through the discriminator, human-shaped, never
   robot-shaped. **Our niche survives: nobody, including BFM-Zero, feeds RETARGETING
   knowledge (cross-embodiment, contact structure) into the control latent.**
2. **z = mean of B(s) over trajectory** for imitation — permutation-invariant averaging,
   exactly what BFMTrack (2606.25056) later criticizes for washing out fast transitions.
   Our per-frame z stream + goal-conditioned prior is the non-averaging alternative.
3. Their Z unifies THREE prompt types (motion/goal/reward). Our z_cmd handles motion
   only. Their reward-as-latent trick (z = Σ B(s)r) is the genuinely new capability we
   lack — but it requires the FB structure (B must be a *state* feature with successor
   semantics), not bolt-on-able to a CVAE.

**What we adopt (concrete, cheap):**
- **A) Latent-space arithmetic eval for E52/E53 students** (their few-shot adaptation is
  CEM in Z): add a zero-cost eval that perturbs z_cmd = μ_prior + Δ and measures
  completion vs ‖Δ‖ — a smoothness/robustness certificate for our command latent that
  BFM-Zero popularized and reviewers will expect. Fold into the E53 student eval.
- **B) History-dependent actor**: BFM-Zero (and YAHMP, and ExBody2) all use obs history;
  our student actor is single-frame. Already queued as a post-E53 lever; BFM-Zero adds
  real-robot evidence.
- **C) Their eval vocabulary** for the paper: report our latent's "promptability
  boundary" honestly — motion-prompted only — and cite BFM-Zero as the
  reward/goal-promptable frontier our retargeting-first entry point complements.
- **NOT adopted:** FB/successor-feature training (a different research program; would
  replace our thesis rather than test it).

## 4. Queue impact

E56-D moves UP (it doubles as the interaction-conditioning risk test AND the
flow-matching gate). Order: E55-A twoteach verdict (~4h) → E56-D (needs twoteach ckpt,
~4h) → E56-C MeanFlow (only if E56-D leaves residual variance) → E56-B scaling sweep.
E53 2048-env retry waits for external GPU job to clear. E52-L1R running (178/8000).
