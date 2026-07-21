# SNMR Forward Research Memo: Flow Matching, Data/Supervision, and SSL (2026-07-20)

> Third deep-research pass (38 agents, 391 tool calls, ~718k tokens; 27/28 claims survived 1-vote
> adversarial verification against primary sources). Deliberately pushes past the two prior verified
> literature rounds (`docs/FLOW_RETARGETING_LITERATURE.md`), especially the physics-repaired-supervision
> leg that returned zero verified claims twice. Companion: `docs/PROGRAM_CONSOLIDATION.md` (our own
> evidence). Verification caveats are flagged inline; the evidence ledger at the end lists every verdict.
>
> Provenance note: one finder hallucinated a repo miscitation (LAFAN1 arXiv id); the adversarial
> verifier caught it (REFUTED) and a direct repo grep confirms the repo cites LAFAN1 correctly.
> This is logged as a dead-end row, not a fix.

## Q1. Are we using flow matching the right way — and is a generative retargeter worth revisiting?

**Verdict: The *retrofit* was wrong; a generative retargeter is worth revisiting, but only if rebuilt on a regularized, sampleable substrate and trained end-to-end — not bolted onto the frozen regression latent.**

Reconciliation with the verified Dirac/covariance-collapse finding: our failure was over-determined by two independent causes, and neither is a property of flow matching per se.

1. **Wrong substrate.** We diffused on a latent that was never regularized to be a smooth, sampleable density. MLD's ablation is the direct analogue: diffusing on an *unregularized plain-AE latent* gives FID 5.033 vs **0.473** on a KL-regularized VAE latent (CONFIRMED, arXiv:2212.04048, Appendix B.3/Table 8). A frozen regression latent trained by MSE distillation is the AE case, not the VAE case — generation on it is expected to be poor.
2. **No target multimodality to learn.** E47 already established (given) that per-frame IK solutions around a single teacher are near-unique (0.0019 rad). If the conditional p(z1|c) is genuinely a point mass in the data, the Dirac collapse Feng ICML'25 describes is the *correct* fit, not a training pathology. Guidance is preconditioned to zero because there is no spread to guide.

**Correct recipe if revisited:**
- **Substrate:** a variational / regularized latent, not the frozen regression latent. PULSE is the existence proof — a 32-d variational information-bottleneck latent distilled from a physics tracker, with a **learnable prior conditioned on proprioception**, reaching 99.8%/97.1% imitation (CONFIRMED, arXiv:2310.04582). Its ablation shows the *prior*, not the autoencoder, carries the benefit: removing the learnable prior drops VR success 93.4% → 45.6%; removing the residual prior-action → 18.1% (CONFIRMED).
- **Supervision:** the one-to-many distribution must exist in the *targets*. MaskedMimic manufactures it by **under-specifying (masking) the goal**, so the physics-valid solution set is genuinely one-to-many, then models it with a C-VAE prior (CONFIRMED, arXiv:2409.14393). This is the concrete mechanism for injecting the multimodality E47 says must enter above the latent.
- **Guidance vs control-optimization:** classifier guidance on a collapsed conditional is a dead end (verified, given). The field's working alternative is **sampling a learned prior and letting a physics/RL loop enforce validity** (PULSE, MaskedMimic) rather than steering a score field.

**Does flow-matching vs diffusion vs CVAE matter here?** No — the axis that matters is *substrate regularization + target multimodality*, not the generative family. The two positive precedents (PULSE, MaskedMimic) are both **CVAEs**, and MLD's lesson is about latent regularization, not the diffusion sampler. Recommendation: if we revisit generation, a proprioception/prior-conditioned CVAE is the lower-risk choice; flow matching buys nothing until the substrate and targets are fixed.

## Q2. Is our data / supervised learning good enough — or is the one-IK-teacher the bottleneck?

**Verdict: The single-IK-teacher is the binding limitation for *diversity/contact*, but NOT for our current walking objective. Scaling clip count is not the fix. Physics-repaired supervision is real and near-lossless but single-solution; genuine multi-solution supervision requires deliberate under-constraining, not just physics repair.**

**What our teacher is.** GMR is confirmed to be a two-stage warm-started per-frame IK (Mink), **with no explicit contact constraint** and only post-hoc FK foot-height subtraction (CONFIRMED, arXiv:2510.02252). Per-frame IK is the field standard (GMR, OmniRetarget per-frame; PHC/VideoMimic trajectory-wise; IMMA multi-stage — CONFIRMED). So we are on-paradigm, but on the contact-agnostic end of it.

**Is it the bottleneck?** Two verified facts bound this precisely:
- Kinematic IK injects foot-slide/penetration/infeasibility from the embodiment gap, and these *significantly reduce policy robustness, particularly for dynamic/long sequences* (CONFIRMED, arXiv:2510.02252). A single GMR teacher bakes those artifacts into targets for all 5 robots.
- **But** contact preservation predicts RL success *only for loco-manip/terrain* — OmniRetarget 82.20%/94.73% vs GMR 50.83%/78.94% on object/terrain, while on flat robot-only LAFAN1 both hit **100%** (CONFIRMED, arXiv:2509.26633, Tab.II). SNMR already tracks flat walking at near-parity (86% vs 88%, given). So a better teacher pays off *when we move to manipulation/terrain*, and does little for the current walking benchmark.

**What the field does about multi-solution supervision (the leg unverified for two rounds — this pass's finding):**
- Physics repair at scale is **real and near-lossless**: PHC imitates 98.9%/96.4% of AMASS (CONFIRMED); PULSE distills it to a low-dim latent at 99.8%/97.1% (CONFIRMED). Feasible in principle.
- **Cost:** ~1 A100-week / ~10B samples in Isaac Gym (1536 parallel) *per embodiment* (CONFIRMED). For SNMR's 5 robots that is ~5 A100-weeks-equivalent — and we have A10G, not A100, so wall-clock is worse. A full physics-imitator-per-robot is a multi-week program, not a quick swap.
- **Critical catch:** PHC produces *one* tracked solution per reference; its "primitives" fuse multiplicatively into a single output (CONFIRMED). So **physics repair alone still yields a unimodal target set per input** — it cleans artifacts but does *not* supply the teacher-level multimodality E47 requires. The only verified route to genuine one-to-many targets is **under-specifying the goal (MaskedMimic masking)** so the physics-valid solution set is intrinsically multi-valued.

**Data-quality interventions, ranked:**
1. **Contact/collision-aware teacher (stance-foot anchoring, interaction-mesh, hard constraints)** — highest-value, but only once we target loco-manip/terrain. OmniRetarget's approach reduces foot-skate/penetration to near-zero (PARTIALLY_CONFIRMED — "significantly reduce" is a paraphrase, but near-zero penetration/foot-skate numbers are confirmed).
2. **Height/scale consistency** — improper source scaling is named the most impactful artifact source; uniform root-translation scaling is crucial to avoid foot-slide; PHC hit 60 cm penetration on Dance (CONFIRMED). Cheap to add to our IK stage.
3. **Do NOT just add clips.** LAFAN1 is small (5 subjects, 77 seq, 496,672 frames, ~4.6 h; we use all 77 — CONFIRMED), but MotionBERT shows adding data is task-dependent: in-the-wild 2D data *hurt* pose lifting 37.4 → 37.5 mm while helping mesh/action (CONFIRMED). Scaling clip count alone will not improve retargeting fidelity and can trade against auxiliary heads. (LAFAN1 jitter propagation is plausible but only PARTIALLY_CONFIRMED — an inference, not established.)

## Q3. Do we need unsupervised / self-supervised learning for the latent — and which?

**Verdict: Yes for structuring the latent, and E48 already proved the local version works — but the highest-leverage SSL/auxiliary objectives are the ones with *control-metric* (not pose-metric) evidence, and the design detail matters more than the mask.**

E48 (given) is the positive anchor: co-trained contact BCE puts contact into the latent (z-linear contact F1 0.257 = 2× deployable baseline), and denoising corruption alone gives 4× control. This pass refines *how* to spend that structure:

**Ranked by verified downstream leverage (control > contact > pose):**
1. **Act-through-latent + residual-to-prior structure (control-metric).** The strongest, most control-relevant result. UniTracker co-trains a CVAE latent inside the tracking loop and beats the same DAgger pipeline without it: SR **91.82 vs 88.03**, and the gain *widens under noise* (86.78 vs 79.58 at noise level 2) (CONFIRMED, arXiv:2507.07356). PULSE and MaskedMimic ablations converge on the same load-bearing piece: the **residual prior** (MaskedMimic no-residual-prior 21.1%/57.4 cm; PULSE no-residual-prior-action 18.1%), not the VAE wrapper (MaskedMimic no-VAE only 93.2%) (CONFIRMED).
2. **Multi-step future prediction (control-metric, but thin).** FLD's decay-weighted N-step prediction over Fourier primitives yields a latent whose distance metric supports better OOD tracking (general 0.72 → 0.80) and survives 60%-unlearnable-data stress where random sampling fails (CONFIRMED, arXiv:2402.13820). *Flag: evidence is thin* — the 0.72→0.80 gain is confounded (unstructured RANDOM sampling also reaches 0.80).
3. **Temporally-coherent masked denoising (refines E48).** MaskedMimic shows the control benefit comes from **structured (temporally-consistent) masking**, not i.i.d. corruption: per-step re-masking generalizes worse; no-structured-masking collapses to 0%/274 cm (CONFIRMED). Implication: our E48 denoising should mask temporally-coherent spans, not per-frame i.i.d. noise.
4. **Contact BCE head (contact-metric).** Already ours (E48). Note MaskedMimic reaches contact-correct behavior with **no explicit contact loss** — physics-in-the-loop instead (CONFIRMED). So a BCE head and physics-in-loop distillation are alternatives, not both required.

**Invariance vs alignment for the cross-embodiment latent.** Our current symmetric latent-alignment loss enforces *alignment* (shared code across robots). The verified precedents suggest the more valuable structure is a **proprioception-conditioned learnable prior** (PULSE, UniTracker) — i.e., make the latent *invariant to the target's redundant DOF* while remaining *decodable per embodiment through a control-trained decoder*, rather than merely aligning frozen codes. Alignment alone is what E39 showed is inert when the controller can bypass it.

## Recommended v-next design (buildable on A10G, existing train_phase1.py + holosoma WBT)

**#1 — Move z into the WBT loop as an act-through-latent (highest leverage).**
- *Intervention:* Stop concatenating a frozen z beside the explicit reference. Instead, co-train z with the holosoma WBT controller as a residual-to-prior CVAE where the policy **acts through z** (z decoded by a control-trained decoder), distilling the existing 86% teacher via DAgger. Feed a short causal window (UniTracker's 25 past / 5 future frames).
- *Mechanism (verified):* UniTracker CONFIRMED — co-trained latent improves SR 88.03 → 91.82 and robustness under noise; and directly explains E39 — "when the actor receives the reference directly, the influence of z vanishes" (actor-with-explicit-reference 88.20 vs 91.82 without). PULSE CONFIRMED — downstream policies benefit *because they act through the frozen decoder*, not from concatenation.
- *Cost:* Reuses the WBT stack; A10G-feasible (a distillation loop onto an existing teacher, not 10B-sample physics training). ~days.
- *Kill criterion:* If WBT walk completion does not exceed the frozen-latent-concat baseline (and ideally approaches the 88% GMR reference) within one training budget, or if z is again shown redundant (ablating z leaves SR unchanged), kill.

**#2 — Switch E48 denoising to temporally-coherent span masking + add residual prior.**
- *Intervention:* Replace i.i.d. per-frame corruption with temporally-consistent masked spans; model the encoder as a residual to a proprioception-conditioned prior.
- *Mechanism (verified):* MaskedMimic — structured masking is load-bearing (no-structured-masking → 0%); residual prior is load-bearing (removing it → 21.1%). Refines E48's 4×-control denoising result.
- *Cost:* Small; a data-augmentation + loss change to train_phase1.py. ~1 day.
- *Kill criterion:* If contact F1 and WBT control do not beat the E48 i.i.d.-denoise numbers, revert to E48 config.

**#3 — Add a self-supervised multi-step future-prediction auxiliary head.**
- *Intervention:* Add FLD-style decay-weighted N-step prediction (in motion space) to the encoder alongside per-frame distillation.
- *Mechanism (verified, thin):* FLD — multi-step prediction structure improves OOD tracking and stress-robustness. Flagged thin (confounded gain).
- *Cost:* Moderate (new decoder head + horizon rollout). ~few days.
- *Kill criterion:* If it does not improve OOD-clip WBT tracking over #1 alone, drop — the base evidence is weak, so it must earn its place empirically.

**#4 (conditional, deferred) — Contact/scale-aware teacher upgrade.**
- *Intervention:* Add uniform root-scaling + stance-foot anchoring / collision constraints to the per-robot IK (OmniRetarget-style), regenerating targets.
- *Mechanism (verified):* Contact preservation drives RL success up to ~40 pts *for loco-manip/terrain*; ~0 pts for flat walking (OmniRetarget, 100%/100% on flat LAFAN1).
- *Cost:* Moderate IK-stage work; a full physics-imitator-per-robot (PHC-style) is ~1 A100-week/robot (CONFIRMED) and is **not** recommended now.
- *Kill criterion:* **Do not start** until the benchmark includes a manipulation/terrain task — on flat walking it is verified to be near-zero-value.

**Explicitly deprioritized:** generative flow-matching / diffusion retrofit on the current latent (verified dead end unless #1's substrate is first made variational and targets made multimodal via masking).

## Evidence ledger

| Claim | Verdict | Source |
|---|---|---|
| PHC is a single-solution tracker/imitator, not a retargeter (no teacher multimodality) | CONFIRMED | arXiv:2305.06456 |
| PHC 98.9%/96.4% AMASS imitation | CONFIRMED | arXiv:2305.06456, Tab.1 |
| Physics imitator cost ~1 A100-week / 10B samples per embodiment | CONFIRMED | arXiv:2305.06456 |
| PULSE = 32-d VIB latent, 99.8%/97.1%; prior carries benefit (no-prior 45.6%, no-residual 18.1%) | CONFIRMED | arXiv:2310.04582 |
| PULSE prior diversity is *unconditional*, not one-to-many for fixed input | CONFIRMED | arXiv:2310.04582 |
| MaskedMimic = CVAE, one-to-many via masking; residual prior + structured masking load-bearing (0% w/o) | CONFIRMED | arXiv:2409.14393 |
| MaskedMimic uses NO explicit contact loss (physics-in-loop) | CONFIRMED | arXiv:2409.14393 |
| GMR = 2-stage warm-started per-frame IK, no contact constraint, post-hoc foot fix | CONFIRMED | arXiv:2510.02252 |
| Per-frame IK is field standard | CONFIRMED | arXiv:2509.26633 / 2510.02252 |
| Improper scaling = top artifact source; PHC 60 cm penetration on Dance | CONFIRMED | arXiv:2510.02252 |
| Contact preservation drives RL success (~40 pts) for loco-manip/terrain, ~0 for flat walking | CONFIRMED | arXiv:2509.26633, Tab.II |
| Kinematic IK injects artifacts; hurts robustness for dynamic/long seq | CONFIRMED | arXiv:2510.02252 |
| UniTracker co-trained latent improves tracking (91.82 vs 88.03), esp. under noise; z ignored if actor sees explicit reference | CONFIRMED | arXiv:2507.07356 |
| UniTracker: short causal window (25 past / 5 future) best | CONFIRMED | arXiv:2507.07356 |
| PULSE/E39: frozen latent helps only under act-through-latent, not concatenation | CONFIRMED | arXiv:2310.04582 |
| MLD: unregularized-AE latent generates poorly (FID 5.033) vs KL-VAE (0.473) | CONFIRMED | arXiv:2212.04048 |
| GMT: 128-d "latent" is just future-window compression, not a control latent; immediate frame dominates | CONFIRMED | arXiv:2506.14770 |
| LAFAN1 small: 5 subj, 77 seq, ~4.6 h; SNMR uses all 77 | CONFIRMED | Ubisoft repo / arXiv:2102.04942 |
| MotionBERT: more (2D) data slightly hurts pose lifting (37.4→37.5), task-dependent | CONFIRMED | arXiv:2210.06551, Tab.5 |
| FLD multi-step prediction latent aids OOD control (0.72→0.80) | CONFIRMED (thin — gain confounded, RANDOM also 0.80) | arXiv:2402.13820 |
| OmniRetarget interaction-mesh "significantly reduces" foot-skate/penetration | PARTIALLY_CONFIRMED (phrase paraphrased; near-zero numbers confirmed) | arXiv:2509.26633 / 2005.05732 |
| Optical mocap has occlusion/mislabel/jitter; artifacts may persist through IK | PARTIALLY_CONFIRMED (failure modes real; "persist" is inference) | arXiv:1904.03278 |
| **Dead end:** finder claimed SNMR miscites arXiv:2005.08526 (audio GAN) for LAFAN1 | **REFUTED** — repo cites LAFAN1 correctly by name/URL; nothing to fix | grep of repo; arXiv:2102.04942 |

**Thin-evidence flags:** FLD control gain is confounded (unstructured sampling matches it); OmniRetarget artifact-reduction magnitude is paraphrased; LAFAN1-jitter-propagation is a hypothesis. The physics-repair leg is now verified as *real and near-lossless but single-solution* — multimodality still requires deliberate goal under-specification (masking), the one confirmed mechanism to satisfy E47 at the teacher level.
