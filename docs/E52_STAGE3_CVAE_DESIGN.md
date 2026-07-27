# E52 — Stage-3 act-through-latent CVAE: literature-locked design

**Date:** 2026-07-27. **Status:** DESIGN (registration of the build; run gated on E51 verdict —
the DAgger teacher should be the best available explicit-command policy).
**Source:** full-text literature pass (UniTracker, PULSE, MaskedMimic, ControlVAE, VMP,
BFM/Scaling-BFM, BFMTrack, LeVERB, NCP, Won'22, NPMP, IKMR, Dexplore + 2026 successors),
verified against primary sources 2026-07-27. Companion: `RESEARCH_PROPOSAL_RETARGET_TO_TRACKING.md` §Stage-3.

## 0. Positioning fact (from the pass)

**No published work co-trains a multi-embodiment retargeting latent under the control
objective.** Dexplore (2509.09671) is single-loop but dexterous-hands; One-Policy-Fits-All
(2603.14522) is manipulation; IKMR feeds rollout *data* not gradients. Stage-3 as specified is
unoccupied ground. Also: UniTracker's Table II(b) *independently replicates our E36–39
finding* — with the explicit reference in the actor obs, "the influence of the latent variable
z vanishes" (88.20 vs 91.82 SR). Our null is their ablation; cite it as third-party
replication, and treat C1 below as settled.

Cautionary datapoint: the UniTracker group's own successor (Scaling-BFM, 2607.15163) dropped
the CVAE entirely (hypersphere-normalized latent shaped by tracking PPO alone). CVAE+DAgger is
the *proven* recipe at our scale; it is not sacred.

## 1. Locked design constraints (each with its strongest evidence)

- **C1 — z is the actor's ONLY motion command.** UniTracker II(b) 88.20 vs 91.82; our E36–39.
- **C2 — residual-to-prior encoder mean** (μ_enc = μ_prior + residual): PULSE 18.1→93.4%,
  MaskedMimic 21.1→96.9%, UniTracker 85.39→91.82. Largest effect sizes in the literature.
- **C3 — learned state-conditional prior, never N(0,I)**: PULSE 45.6→93.4%. **Ours: the SNMR
  latent window is the prior mean** (the thesis: retargeting knowledge = the prior).
- **C4 — pure DAgger from the explicit-command teacher; NO RL mixed into distillation**
  (PULSE R4: mixing costs 22pp; RL-from-scratch through latent collapses: BFM 400mm mpkpe,
  UniTracker 72.2 SR). Loss = ||a_student − a_teacher||² + β·KL.
- **C5 — RL only afterwards as a frozen-backbone bounded residual** (UniTracker st.3, BFM
  st.3, Won'22 helper Δa = σ⊗Tanh(H(s,z))).
- **C6 — ~5-frame (0.1 s) lookahead in the prior's goal window** (UniTracker: 5 optimal of
  {1,5,10,20}; BFMTrack independently L=5). Posterior may see 1 privileged future frame.
  Optional Scaling-BFM stochastic far offset (one frame ~U[5,32]) for latency robustness.
- **C7 — KL is mandatory and the optimum is sharp** (Won'22: no KL → falls <1s; UniTracker:
  ±10× around 0.1 costs ~10pp). Budget a β sweep {0.03, 0.1, 0.3}.
- **C8 — episodic reparameterization noise** (ε fixed per episode; MaskedMimic) — one line.
- **C9 — asymmetric critic with full unmasked reference** — ALREADY true in holosoma's WBT
  config; keep it; never feed the explicit reference to the actor.
- **C10 — masking deferred to Stage-3b**; if added, temporally structured (MaskedMimic
  98%-repeat/2%-resample; or BFM Bernoulli(0.5)+cold-start curriculum) — never i.i.d.
  per-frame (0% success ablation).
- **C11 — PULSE latent-smoothness regularizer** L_regu = ||μ_e(t) − μ_e(t−1)||², α = 0.005.
- **C12 — engineering:** custom-PPO subclass via `algo._target_` (add CVAE params to
  actor_optimizer, ppo.py:245-282; extend grad-clip, ppo.py:505-506). PPOActorEncoder is
  broken in the pinned clone (E49 Stage-0) — this route bypasses it entirely.

## 2. Architecture (first run)

- **Prior** ρ(z_cmd | proprio, z_snmr-window): inputs = deployable proprio (existing actor
  terms minus motion_command) + SNMR latent at offsets (0, +5) [C6; 50 Hz steps]. Output:
  μ_prior. **z_snmr enters ONLY here.**
- **Posterior/encoder** ε: prior inputs + privileged state (critic terms incl. explicit
  reference @ t and t+1) → residual μ; **σ fixed, tied: σ_q = σ_p = 0.3 (ControlVAE)** →
  closed-form KL = ||μ_residual||²/(2σ_p²); no σ heads, one collapse mode removed (D2-A).
- **Decoder** D(a | proprio, z_cmd): MLP 512/256/128 (teacher-matched). No goal input (BFM).
- **z_cmd dim: 64 via a learned 128→64 projection from the SNMR window** (D4): keeps the
  retargeter's z=128 fidelity optimum (E17) while matching the control optimum (UniTracker
  64 > 128). If the two optima genuinely diverge, that is itself a publishable finding.
- **SNMR encoder: co-trained WITH a live fidelity anchor** (retargeting reconstruction loss
  at low weight; report the fidelity price like E48) (D6). Freeze is the fallback arm.
- Deployment: z = μ_prior (deterministic; UniTracker v1 ablation 91.83 vs 91.77).

## 3. Training

DAgger from the best explicit-command policy (E51 winner if promoted, else confirmatory
GMR seed0 0.90): student rolls out, teacher labels visited states, L = ||Δa||² + β·KL +
α·L_regu (+ λ_fid·L_recon on the SNMR branch). β = 0.1 first (C7), anneal-UP fallback
(MaskedMimic 1e-4→1e-2; three papers support up, one down). Same walk1 single-clip dev gate
as B1/E49, then multi-clip — **H4 says the latent's value shows at multi-clip scale;
single-clip parity (≥0.85 completion) is the promotion gate, not the claim.**

## 4. Preregistered arms & gates (dev, walk1, seed 0)

| arm | question | gate |
|---|---|---|
| A: full recipe above | does act-through-latent close E39's gap? | completion ≥0.85 (L1 was 0.72; explicit 0.88/0.90) |
| B: prior WITHOUT z_snmr (proprio-only prior, PULSE-style) | is retargeting knowledge in the prior load-bearing? | A − B = the H2 transfer claim |
| C: frozen SNMR encoder (vs A's co-trained) | does control-gradient shaping matter? | A − C isolates co-training |

Kill: A < 0.80 after β sweep → the CVAE recipe does not transfer to our stack at this scale;
fall back to VMP-style hybrid (z + raw current frame, NO future explicit) — evidence: VMP LM
beats L on every category; lands ≥ the 0.72 band rather than a new failure mode (D1).

## 5. Deferred (Stage-3b+)

Masking (C10), pink-noise latent exploration (BFMTrack; unverified transfer to PPO training),
RL residual finetune (C5), cross-embodiment arms (H4), 128-vs-64 ablation beyond the
projection head.
