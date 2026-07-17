# Flow-Matching Side Project: Retrospective and Critique (E43–E47)

Written 2026-07-17 after closing v1 at gate F2 and the E45 post-mortem; §6 extends it same-day
with the v2/v3 entry-gate results (E46/E47) and the round-2 verified literature. Companions:
`docs/FLOW_RETARGETING_SIDE_PROJECT.md` (v1), `docs/FLOW_RETARGETING_V2_PROTOCOL.md` and
`docs/FLOW_RETARGETING_V3_PROTOCOL.md` (both closed at entry), and
`docs/FLOW_RETARGETING_LITERATURE.md` (two verified deep-research rounds).

## 1. What we built

| Piece | File | State |
|---|---|---|
| Conditional rectified-flow net over frozen 128-d SNMR latents (temporal transformer, AdaLN-zero flow-time conditioning, zero-init head) | `snmr/flow.py` | works; F1-grade |
| CondOT conditional flow-matching loss | `snmr/flow.py` | works |
| Euler sampler with SafeFlow-style endpoint guidance (clamped ∇C, linear α) | `snmr/flow.py` | works mechanically; ineffective scientifically |
| Differentiable physics cost through frozen decoder + FK (soft/gated/oracle stance) | `snmr/flow.py` | works |
| Manifest-tracked trainer on frozen SNMR | `scripts/train_latent_flow.py` | works (20k steps ≈ 15 min on the A10G) |
| Gate evaluator: raw / z_r-oracle / z-descent control / flow / guided grid under frozen Gate-1b guards | `scripts/eval_latent_flow.py` | works |
| Guidance post-mortem probes (norms, contraction, diversity, norm-matched push, direction cosine) | `scripts/diagnose_latent_flow_guidance.py` | works |
| Tests | `tests/test_flow_retarget.py` | 10 tests, suite green |

## 2. What we learned (evidence-backed)

1. **E43/F0 — the frozen latent is not a two-sided interface.** Dec(Enc(teacher robot motion))
   = 50 cm MPJPE. Phase-2's `L_z` aligns latents but nothing ever trained *decoding from* robot
   encodings (`zr_decode_prob` deferred). Consequence: the only trainable conditional target was
   `z1 = z_h` — the transport degenerated to an identity map.
2. **E44/F1 — a generative latent head is free.** The flow reproduces the deterministic decode
   exactly (3.49 vs 3.48 cm). Infrastructure and fidelity are not the obstacle.
3. **E44/F2 — guidance inert; adaptive descent 5× better; both insufficient.** Best oracle cell
   0.400→0.339 m/s vs the 0.08 endpoint; z-descent 0.221 m/s at +0.15 cm MPJPE.
4. **E45 — the failure is structural, three ways:**
   - the clamped push was ~0.15% of the velocity norm (600× below parity) — a *scale* bug in
     the adapted recipe, but fixing it is not enough, because
   - the learned transport is *contractive* (perturbations at u=0.2 keep 9% of their magnitude;
     endpoint diversity 0.4% of ‖z_h‖) — a Dirac conditional leaves no distribution to steer
     within; and
   - even norm-matched pushes (ρ up to 1.0) make skate *worse*, and the guidance direction is
     ~orthogonal (cos 0.05) to the productive Adam-descent direction — the instantaneous
     decoder+FK gradient is ill-conditioned in this regression latent.
5. **Cross-project convergence (with Gate 1b, E41/E42):** four corrector families — windowed
   projection, DLS lock, latent descent, guided flow — all pass with the oracle contact mask
   and all fail deployably. The binding constraint is the *deployable contact signal*, upstream
   of any corrector.

## 3. Were we "using flow matching the correct way"? — honest critique

**What was correct:** CondOT path, velocity target, CFM loss, Euler integration, endpoint-
prediction guidance (arguably more correct than SafeFlow's raw `Dec(z_u)` cost for a decoder
never trained on noisy latents), preregistered gates with a null control.

**What was not, in hindsight:**

- **We fit a generative model to a deterministic dataset.** With exactly one teacher solution
  per window and target = condition, `p(z1|c) = δ_c`. Flow matching learns exactly what it is
  given: a contraction onto a point. Guidance of a Dirac transport is a category error — there
  is no conditional *distribution* in which to move probability mass. SafeFlow's setting is the
  opposite: text→motion is massively one-to-many.
- **Wrong substrate.** SafeFlow guides in a KL-regularized VAE latent whose decoder is trained
  to behave under noise. Ours is a raw regression latent: off-manifold z has undefined decoder
  behavior, which is exactly where guidance pushes. The E45 orthogonality (cos ≈ 0.05) is the
  measurable symptom.
- **Scale transplant without dimensional analysis.** SafeFlow's α ∈ [500, 10000] with ±0.2
  clamps was tuned for their latent/cost scales; in ours the same clamp bound produced pushes
  600× below the velocity norm. Any future guidance must be *relative* (normalized to ‖v‖ or
  Adam-preconditioned), never absolute.
- **The one defensible v1 reading:** F1 + the oracle-pass lineage says the machinery is sound;
  v1 failed on *representation and supervision*, not on the flow-matching math.

## 4. Is our data/supervision good enough? — critique

- **One-teacher regression is the root limitation.** GMR gives one IK solution per human frame;
  MSE distillation of a conditional mean over residual ambiguity is our best explanation of the
  C5 skate signature (smooth xy oscillation of correctly-placed stance feet). A generative model
  cannot recover modes the dataset never contained.
- **The teacher itself is imperfect** (skate 0.047–0.052 m/s — below our 0.08 endpoint but not
  clean), so even perfect distillation inherits residual physical error.
- **The latent was never asked to carry physics.** E38: contact is not linearly decodable from
  z (probe F1 0.044 vs 0.127 baseline). E16: content-specific, not semantically organized.
  Nothing in the training objective (distill + limits + smooth + L_z) rewards physical structure
  in z, and the encoder has no ordering signal (no positional encoding by default).
- **No noise-robustness anywhere:** encoder sees clean mocap; decoder sees clean encoder
  outputs. Both a generative substrate and any guidance/repair loop need off-manifold
  robustness that pure supervised regression never provides.
- **What self/unsupervised learning could add** (see literature review for citations):
  (a) a *VAE-style noise-regularized latent* (KL or simple Gaussian latent augmentation +
  decoder finetune) — the minimal change that makes the substrate guidance-compatible;
  (b) *masked motion modeling / denoising pretext tasks* to make z structurally informative
  beyond the distillation target; (c) *physics-aware pretext heads* (contact, foot velocity)
  trained jointly so z carries the very signal every corrector is starved of — note this is a
  different lever from Gate-1b M3, which bolted a contact head onto a frozen backbone;
  (d) *multi-solution supervision* (physics-repaired or simulator-vetted variants of GMR
  targets) so the conditional distribution is genuinely non-degenerate before any flow is fit.

## 5. Open questions carried into v2

Q1. Does noise-regularizing the latent (and finetuning the decoder on noisy z) alone make
    inference-time physical correction work deployably — even without a flow?
Q2. Is there real multi-modality to model — i.e., can we manufacture a valid multi-solution
    dataset (physics-repaired + jitter-vetted GMR variants) at acceptable cost?
Q3. With (Q1) and (Q2) in place, does flow + guidance then beat Adam z-descent, or is descent
    on a good latent all we need? (The v1 control says: always run both.)
Q4. Does a physics-pretext latent finally yield a deployable contact signal — the single
    upstream blocker every corrector shares?

## 6. How the open questions resolved (E46/E47 + round-2 research, same day)

**Q1 — NO, three ways.** Decoder-only noise finetune (V0): fidelity fine, descent transfer nil
(0.224 vs 0.221 m/s). Encoder-joint (W0): same (0.219). The substrate hypothesis is dead in
every variant; substrate geometry never was the binding constraint here.

**Q2 — NO, and this is the deepest finding of the whole line (E47).** Jittered oracle-mask
projections around one decoded window nearly coincide: config-jitter diversity/edit ratio
0.29 (solver converges to one fixed point), anchor-jitter 0.74 but with absolute dof diversity
0.0019 rad — 5× below the preregistered floor, on edits that are themselves only ~0.0026 rad.
Manufacturability also fails (3/21 windows; the oracle mask has empty support on 18/21 at
64-frame scale — E41's ~9% support problem again). *Locally corrected retargeting solutions
around a single IK teacher are near-unique.* The Dirac conditional is a property of the DATA
GENERATING PROCESS, not of our model, our latent, or our correction machinery. Genuine
multimodality must enter at the teacher level (multiple IK configs, RL/simulator repair with
behaviorally distinct strategies) — which is main-line Track A5.

**Q3 — mooted by Q2** (nothing non-degenerate to model). The corrector-formulation insight
(OC-Flow/D-Flow ≻ per-step nudging) stands and transfers to any future steering problem.

**Q4 — open; now running as E48 with verified backing.** Round 2 verified the two triggers the
v3 §2 arm required: masked+noised denoising pretexts structure motion latents (MotionBERT —
corruption is the ablated active ingredient), and CO-TRAINED contact heads carry contact into
the latent (HuMoR, 97% contact accuracy) — a different intervention from our failed
frozen-backbone M3 retrofit. Also verified: our C5 conditional-mean diagnosis is
literature-standard (MSE = unimodal-Gaussian MLE, provably interpolates modes; animation foot
skate explicitly attributed to regression-to-the-mean; IBC/Diffusion-Policy head swaps fix it
with large effects), and MLD's own ablation (unregularized-AE latent: FID 0.473 → 5.033) is
the mechanistic twin of our E43/E44 failure. Physics-repaired-supervision literature remains
UNVERIFIED after two targeted rounds — main-line A5 proceeds on our own oracle-pass evidence
plus PULSE adjacency (physically-valid latents from simulator-executed supervision), largely
first-principles.

**Net for the paper:** the side project contributes (a) a five-family corrector convergence on
the deployable-contact-signal bottleneck, (b) a theory-verified account of why generative
retrofits on regression latents fail (Dirac + covariance-preconditioning + brittle substrate),
(c) the E47 local-uniqueness result reframing "add a generative head" as "fix the teacher",
and (d) the E48 representation study feeding the latent-analysis section.
