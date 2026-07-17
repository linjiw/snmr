# Flow-Matching Side Project: Retrospective and Critique (E43–E45)

Written 2026-07-17, after closing the v1 side project at gate F2 and running the E45
post-mortem. Companion to `docs/FLOW_RETARGETING_SIDE_PROJECT.md` (frozen v1 protocol) and the
literature review in `docs/FLOW_RETARGETING_LITERATURE.md`. The v2 protocol (if any) lives in
`docs/FLOW_RETARGETING_V2_PROTOCOL.md`.

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
