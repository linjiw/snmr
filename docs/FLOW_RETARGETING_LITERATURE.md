# Literature Review: Why v1 Guidance Failed and What the Field Says (2026-07-17)

Deep-research pass (107 agents, ~1000 tool calls; every claim below survived 3-vote adversarial
verification against primary sources unless marked otherwise). Companion to
`docs/FLOW_MATCHING_RETROSPECTIVE.md` (our own evidence) and
`docs/FLOW_RETARGETING_V2_PROTOCOL.md` (what we do about it).

## Q1 — Training-free guidance for flow matching

**The E45 findings are theoretically predicted, not incidental.**

1. **Dirac conditionals annihilate guidance (root cause #1).** In the unified guidance
   framework of Feng et al., ICML 2025 spotlight (arXiv:2502.02150), gradient guidance equals
   the cost gradient **preconditioned by the posterior covariance** of p(z1 | x_t):
   `g ∝ Σ_{1|t} ∇J`. Our transport learned p(z1|c) = δ_c, so Σ_{1|t} → 0 and the effective
   guidance is squeezed to zero. Exact energy reweighting of a point mass is likewise inert
   (e^{-J}·δ = δ). [3-0 verified] → *No guidance schedule can work until the conditional target
   distribution is non-degenerate.* This is our E45 contraction finding, derived independently.
2. **Strong/deterministic couplings also break guidance direction (root cause #2).** Same
   paper, §5: "when the coupling is strong, the guidance VF no longer has the correct
   direction" — and rescaling preserves direction, so *no normalization fixes it*. Caveat: the
   proof needs strong coupling AND a rapidly-varying cost; our decoder+FK stance-velocity cost
   is plausibly rapidly varying, so both conditions co-hold. [1-1 split: quote verified,
   inference sound] This matches E45's cos(guidance, descent) ≈ 0.05 orthogonality.
3. **SafeFlow-style guidance was never justified in our regime.** Diffusion-ported gradient
   guidance (DPS/LGD/ΠGDM — the family SafeFlow's Eq. 5 belongs to) is only theoretically
   grounded for Gaussian-source, uncoupled affine-Gaussian paths; outside it, "exact diffusion
   guidance ... shows a largely distorted generated distribution." [3-0]
4. **Training-free cost guidance is intrinsically weak.** Shen et al. NeurIPS 2024
   (arXiv:2403.12404) + TFG, NeurIPS 2024 spotlight (arXiv:2409.15761): predictors trained
   only on clean samples lose smoothness on noisy inputs, admit adversarial gradients, and
   "fail to produce satisfactory samples for label guidance even on CIFAR10" (32–50% vs 85%
   for trained guidance). [3-0, merged×3]
5. **Fixed scalar strength is a documented failure mode.** Optimal step scales like
   `sqrt(α_t)/(L_f(1+L_p))` (Shen et al. Prop 3.1; FreeDoM's `sqrt(α_t)` weights match).
   Concrete fix: **Polyak adaptive step** `η·‖ε_θ‖/‖g‖²·g` — normalize the push by the model's
   own score/velocity norm. TFG (Thm 3.2) subsumes DPS/LGD/FreeDoM/MPGD/UGD in one searchable
   design space. [3-0, merged×3] (E45's norm-matched probe implemented the first half of this
   and showed scale alone is insufficient here — consistent with findings 1–2.)
6. **Guidance gradients must be manifold-projected, not just rescaled.** MPGD, ICLR 2024
   (arXiv:2311.16424): guidance on x_t pushes samples off the noisy-data manifold and corrupts
   the score net; the fix applies the gradient to the *clean estimate* (we did this) **and
   projects it through a pretrained autoencoder** (we could not — our latent has no
   noise-trained decoder to project with). Manifold projection "improves sample fidelity by a
   large margin." [3-0, merged×3]
7. **Deterministic ODE sampling has no error correction.** EDM (arXiv:2206.00364): the
   Langevin term in SDE sampling "actively correct[s] for any errors made in earlier sampling
   steps"; FreeDoM (arXiv:2303.09833) needs time-travel resampling to avoid condition-unrelated
   outputs. BUT: stochastic sampling corrects toward the *model's own marginal* — it cannot
   manufacture diversity a degenerate transport lacks. [3-0, merged×3]
8. **Guidance should live in a mid-t interval.** Kynkäänniemi et al., NeurIPS 2024
   (arXiv:2404.07724): constant-weight guidance is harmful at high noise ("oversteering",
   mode-dropping — intrinsic, shown with ideal denoisers), unnecessary at low noise, beneficial
   only in the middle; interval restriction permits much higher weights. [3-0, merged×3]
   (Our `skip_final=0.9` was backwards relative to this: we guided early — where oversteering
   lives and where our contractive field erased it — and stopped late.)
9. **The principled upgrade path is optimal control, not velocity nudging.** OC-Flow, ICLR
   2025 (arXiv:2410.18070) formulates guided flow sampling as optimal control; FlowGrad and
   D-Flow are special cases. This family optimizes a control objective and *does not depend on
   the collapsed posterior* — consistent with our z-descent control beating guidance 5×.
   [3-0, merged×2] → If v2 keeps a steering step, it should be D-Flow-style optimization of
   the source noise z0 through the flow (or direct trajectory control), not per-step nudges.

## Q2 — Regression latents as generative substrates

10. **Deterministic AE latents are brittle where guidance needs smoothness.** Decoding convex
    combinations of deterministic-AE latents "often leads to artifacts or unrelated output"
    while VAE interpolation is coherent (arXiv:2412.04755; corroborated by ACAI ICML 2018 and
    RAE ICLR 2020 — deterministic AEs need explicit regularization to match VAE smoothness).
    [2-1 medium] **Transparency:** the stronger claims — that LDM's slight KL/VQ is a
    *prerequisite* for a good diffusion substrate, and a stratified-manifold explanation — were
    **REFUTED** in verification; treat "VAE strictly required" as folklore. The defensible
    statement: *some* latent regularization (KL, noise augmentation, or contractive penalties)
    demonstrably improves off-manifold decoder behavior, which is exactly where guidance pushes.

## Q3/Q4 — SSL motion latents & retargeting supervision: verification came back EMPTY

No claims about masked motion modeling (MotionBERT/MoMask), motion VAEs/MLD, contrastive
multi-embodiment alignment, physics pretext tasks, or one-teacher-IK supervision bottlenecks
survived adversarial verification in this pass. **Everything we believe on those topics is
currently unverified prior knowledge** and is treated as exploratory (not evidence) in the v2
protocol. Two internal results partially stand in: E38 (contact not linearly decodable from z)
and E16 (latent content-specific, not semantically organized) — our latents demonstrably do
not carry physics, whatever the literature says about how to get it in.

## What this changes about our read of v1

- E45's three findings (scale, contraction, orthogonality) are all *predicted* by finding 1–2:
  we independently rediscovered the covariance-preconditioning collapse.
- The z-descent 5× advantage is not a hack — it is the OC-Flow family in embryo (finding 9).
- The v1 fix ordering is forced: **fix the substrate and the conditional distribution first**
  (Q2 + our F0), and only then consider steering — implemented as interval-restricted,
  norm-scheduled, manifold-projected guidance (findings 5, 6, 8) or as source-point/control
  optimization (finding 9). Sampler-side stochasticity (finding 7) is a quality knob, never a
  diversity source.
