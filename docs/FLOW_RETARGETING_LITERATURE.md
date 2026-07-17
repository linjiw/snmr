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

---

# Round 2 (2026-07-17, second deep-research pass): SSL motion latents, MSE failure, supervision

106 agents; the four legs that returned zero verified claims in round 1. All claims below
survived 3-vote adversarial verification against primary PDFs unless marked.

## Q1 — What verifiably structures a motion latent

11. **Physics-teacher distillation through a variational bottleneck (PULSE, ICLR'24 spotlight,
    arXiv:2310.04582).** Physically-valid latents come from online-DAgger distillation of a
    near-perfect physics-based imitator's ACTIONS (executed in simulation) through a VAE
    bottleneck with a *jointly learned proprioception-conditioned prior*. The learned prior is
    quantitatively essential (VR-tracking success 93.4% full → 45.6% no learnable prior →
    18.1% no residual prior-action), and "training with RL objectives alone is not sufficient
    to form a good latent space." No kinematic reconstruction loss anywhere. [merged 2-0/3-0/3-0]
    → *Our latent was never going to carry physics: its supervision signal (IK qpos MSE)
    contains none. Physical validity comes from simulator-executed supervision — exactly the
    calibrated-WBT-teacher prerequisite of main-line A5.*
12. **Masked+noised denoising pretext (MotionBERT, ICCV'23, arXiv:2210.06551).** Corruption is
    the separately-ablated ACTIVE ingredient (stepwise 39.2 → 38.8 vanilla pretrain → 38.1
    +noise → 37.4 +mask mm MPJPE; adding more clean data alone *worsened* it). Beats identical
    architecture from scratch on all three downstream tasks. [merged 3-0×3] Caveat: downstream
    metrics are pose/action accuracy, not physics/contact.
13. **Co-trained contact auxiliary heads (HuMoR, CVPR'21/ICCV'21, arXiv:2105.04668).** CVAE
    decoder jointly predicts per-joint contact probabilities (BCE) plus a contact-weighted
    velocity loss `L_vel = Σ_j c_j‖v_j‖²`, reaching 97% contact accuracy. [merged 3-0×3]
    → Two local notes: (a) this is the CO-TRAINED pattern — our failed M3 was a frozen-backbone
    RETROFIT, a different intervention; (b) HuMoR's L_vel is structurally our failed E10
    EDGE-style loss, but it sits inside a *generative* CVAE, not an MSE regressor — consistent
    with the E10 lesson that the loss family alone doesn't transfer.
14. **The VQ / masked-token line (T2M-GPT arXiv:2301.06052, MoMask arXiv:2312.00063) reports
    ZERO physics or contact metrics** — FID/R-Precision only. [merged 3-0×4] Deprioritized.

## Q2 — Conditional-mean failure and generative heads (our C5 diagnosis, now verified)

15. **MSE = MLE under a unimodal isotropic Gaussian; it provably interpolates across modes
    regardless of capacity** (BeT arXiv:2206.11251; IBC arXiv:2109.00137 expressiveness
    theorem), and neural-animation foot skating is explicitly attributed to "a regressive
    model converging to the mean pose" (ACM IMX'23). [merged 2-1/3-0×3] → The C5 skate
    signature is a literature-documented failure mode of our exact training setup.
16. **Swapping to a generative head fixes the averaging artifact, with big effects:** IBC
    83.3% vs 6.7% (explicit MSE) on real-robot insertion; Diffusion Policy +46.9% average
    over LSTM-GMM/IBC/BeT with clean per-rollout mode commitment. [merged 3-0×5] Caveat: all
    verified head-swap evidence is manipulation-domain; none reports foot-contact metrics.

## Q4 — Latent vs output space

17. **MLD's own ablation (arXiv:2212.04048): a diffusion head on an *unregularized* AE latent
    collapses (FID 0.473 → 5.033)** — the decisive, mechanistic confirmation of our E43/E44
    frozen-regression-latent failure from the original latent-diffusion literature. The
    MLD-vs-MDM latent-vs-output comparison itself is confounded (architecture, steps, VAE)
    and physics-metric-free. [merged 3-0×2]

## Q3 — Physics-repaired supervision: STILL unverified after two rounds

18. No PhysCap/SimPoE/UHC-cleaning/RobotMDM/self-distillation claim survived either pass.
    Closest verified adjacency: PULSE's supervision is simulator-executed actions (valid by
    construction). [low] → Main-line A5 remains supported by our own Gate-1b oracle result +
    PULSE adjacency, not by direct verified pipeline literature. Treat as first-principles
    territory; our W1/E47 probe machinery is the in-house prototype.

## Round-2 synthesis for the roadmap

Every verified thread converges on the same design for any future generative retargeter:
**output-space (or freshly co-trained latent) generative head + co-trained contact/velocity
auxiliaries + masked-denoising encoder pretext + simulator-executed supervision targets** —
and E47 adds our own constraint that the multimodality must enter at the teacher level.
Registered consequence: the v3 §2 bounded SSL arm is now UNLOCKED (findings 12–13 are the
verified triggers it required).
