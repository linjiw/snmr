# E55–E57 — Data scaling, model scaling, and downstream discrimination

**Date registered:** 2026-07-28. **Status:** REGISTERED; Stage-0 feasibility on CPU while
E53 holds the GPU. Motivated by three PI questions: (1) how much data trained SNMR and can
we expand; (2) can the latent/distillation model be stronger (transformer / flow matching);
(3) the point-equal downstream result (86.7% = 86.7%) "seems weird" — test it harder.

## 0. Current facts (audited 2026-07-28)

- **Data:** 77 LAFAN1 clips = 496,672 frames = **4.6 h** per robot @30 fps; × 5 robots =
  **2.48 M paired frames**. Teacher = GMR IK (~160 fps CPU → the full pair set regenerates
  in ~1 CPU-day). Held-out split: 7 clips, fixed since Phase 1.
- **Model:** GAT encoder/decoder, z=128, hidden 256, **1.54 M params** (the "large" config;
  the small default is 0.41 M). Temporal transformer exists but E08 found it HURTS at this
  scale (3.95 vs 4.71 cm without/with) — we currently train content-only.
- **Downstream point-equality:** per-seed completions are NOT identical — GMR {0.90, 0.85,
  0.85} vs SNMR {0.85, 0.84, 0.91}; the aggregate means coincide at 0.867. Walk1 only,
  and the matrix was trained under DEFECT-1 (orientation+velocity tracker).

## E55 — Data scaling (teacher expansion)

**Hypothesis:** retargeting fidelity and (more importantly) the latent's downstream value
scale with clip diversity; 4.6 h is small by 2026 standards (UniTracker: 8k+ AMASS motions;
OmniRetarget: 8.4 h released).

Expansion sources, ranked by feasibility:
1. **100STYLE** (~4 M frames, 100 locomotion styles, BVH; site reachable). ReActor uses it
   as its generalization set. Stage-0: verify GMR retarget path for its skeleton (GMR
   supports `bvh_lafan1` + Nokov; 100STYLE's skeleton differs → may need a joint-map
   config or BVH conversion; budget 1 day. If conversion is heavy, defer).
2. **OmniRetarget released dataset** (HF `omniretarget/OmniRetarget_Dataset`, MIT, G1
   trajectories incl. loco-manipulation). Two distinct uses: (a) **second teacher** — gives
   SNMR interaction-rich motions GMR cannot produce; (b) **teacher multimodality** — two
   teachers for the same source motion create a genuinely multimodal retargeting
   conditional, which is EXACTLY the missing precondition E47 identified for generative
   decoders (see E56). Stage-0: download sample, map schema → our pair format (their qpos
   convention vs ours), measure overlap with LAFAN1 clips.
3. **AMASS** — still blocked on registration in this environment; keep as handoff item.

**Gates:** E55-A (data-scaling curve): retrain G1 specialist at {25%, 50%, 100%, 100%+new}
of clips at matched steps; promote data expansion iff held-out MPJPE improves ≥10% at
100%+new vs 100%. Kill: flat curve → capacity-bound, run E56 first, then re-test.

## E56 — Model scaling (stronger latent/distillation models)

Three candidate upgrades, with the evidence we already hold:

1. **Transformer encoder/decoder (skeleton tokens instead of GAT):** PROMISING, untested.
   AdaMorph (our closest competitor) is transformer-based; our GAT was chosen for
   skeleton-agnosticism, but a per-node-token transformer with learned skeleton embeddings
   preserves that property. E56-A: swap GAT stack → 4-layer transformer encoder at matched
   params (~1.5 M) and at 4× (6 M); same training budget; readouts = held-out MPJPE +
   the E48 probe suite (does a stronger encoder change what the latent carries?).
2. **Scale within the GAT family:** cheapest. E56-B: hidden 256→512, layers 4→6 at 100k
   steps (LR ≤3e-4 per the E17/lr8e4 lessons). This is the control for E56-A — if plain
   scaling matches the transformer, architecture is not the bottleneck.
3. **Flow-matching / diffusion decoder: NOT until multimodality exists.** We hold
   theory-grade negative evidence (E43–E47): the retargeting conditional from a SINGLE
   deterministic IK teacher is locally near-Dirac — a generative decoder has nothing to
   model, and guidance vanishes in the Dirac limit (corollary OURS, from Feng ICML'25's covariance-preconditioning identity — their paper has no deterministic-limit theorem). The fix is upstream: **E55's two-teacher
   data (GMR + OmniRetarget) creates real conditional multimodality**, after which a flow
   decoder becomes a live hypothesis (E56-C, gated on E55 source-2 landing). This ordering
   converts our own negative result into the experiment design.

**Gates:** E56-A/B promote iff held-out MPJPE ≤ 3.3 cm (−10% vs BENCH-v2's 3.66) at
matched budget, with skate not worse. E56-C registers only after E55-2.

## E57 — Is downstream point-equality real? (discriminative-power test)

The doubt is legitimate; the reassuring reading ("distillation is faithful, so equality is
expected — walk1 SNMR-vs-teacher MPJPE is 3.4 cm, tiny vs what a tracker can absorb") is
itself an untested hypothesis. Three preregistered probes:

- **E57-A (positive control — can our assay detect differences at all?):** train one
  tracker on a deliberately degraded walk1 reference (joint noise σ=0.05 rad, the scale of
  known retargeting artifacts) with the post-fix recipe. If completion does NOT drop ≥5pp
  vs the clean-reference tracker, the assay is insensitive and ALL equality claims are
  suspended (this also retro-validates C6). If it drops, the assay works and equality is
  informative.
- **E57-B (harder clips):** the C6 matrix used walk1 — our EASIEST clip (teacher skate
  0.054, SNMR MPJPE 3.4 cm). Repeat matched GMR-vs-SNMR tracking (1 seed each, post-fix
  recipe) on dance2 + fight1, where reference divergence is larger. Prediction under the
  faithful-distillation reading: still ≈equal; prediction under "walk1 hides differences":
  a gap appears on the harder clips.
- **E57-C (divergence-response curve):** correlate per-clip GMR↔SNMR kinematic distance
  (MPJPE between the two references) with the downstream completion delta across E57-B +
  E53's per-clip evals. A flat curve at small divergences with a knee = the honest,
  quantified version of "same is expected below X cm".
- **Metric upgrade (folds into all arms):** add the GMR paper's metric family to our WBT
  eval exports — E_g-mpbpe (global body pos), E_mpbpe (root-relative), E_mpjpe (joint) —
  computed from the repair-recording FK path we already have. Success-only completion is a
  coarse instrument; the GMR paper's Tab. II shows tracking-error families discriminate
  where success rates saturate.

**Order:** E57-A first (one 2.5 h run — it gates everything), then B (two runs/source),
C is free analysis. All post-DEFECT-1 recipe (arm A). Queue: after E53 completes.

## Sequencing vs GPU budget

E53 (16k teacher + students + per-clip evals, in flight) → E52-L1R (FLAG-1 integrity re-run)
→ E57-A → E57-B → E56-B → E56-A → E55-A (needs pair regeneration first, CPU-parallel).
CPU-side now: E55 Stage-0 feasibility (OmniRetarget schema, 100STYLE retarget path).

## Addendum (2026-07-28, scaling-law literature via research agent)

Verified findings that re-order E56:
- **SONIC (2511.07820)**: humanoid motion tracking scales 1.2M -> 42M params monotonically
  (MPJPE 27.7 -> 23.8 mm) — but on 100M+ frames. Scale helps WITH data.
- **MolGPS (2404.11568, NeurIPS'24)**: message-passing GNNs are MORE parameter- and
  data-efficient than graph transformers at small width/depth — direct evidence for our
  regime (1.5M params).
- **"Neural Scaling Laws on Graphs" (2402.02054)**: model-scaling COLLAPSE from overfitting
  at ~10^7 params on small graph datasets — a warning for E56 at 2.5M frames.
- **ScaMo (2412.14559, CVPR'25)**: motion scaling laws only fit from 44M params on 30M
  frames — our regime is below the scaling-law-visible range.
- **No motion-retargeting scaling law exists** — running E56-B as a small scaling sweep
  (0.4M/1.5M/6M at fixed data) is itself a small novel contribution.

**Revised E56 ordering:** E56-B (GAT scaling sweep, expect the 2402.02054 collapse pattern
at 6M unless E55 data lands first) BEFORE E56-A (transformer — now expected to LOSE at this
scale per MolGPS; run it as a confirmatory data-efficiency comparison, not as an upgrade
bet). E56-C (flow) unblocked by the E55-A terrain multimodality finding but sequenced after
E55-A training verdict.

## Addendum 2 (2026-07-28, model-variants literature + local data audit — full report in agent log)

**Local audit findings (verified on disk):**
- Two-teacher multimodality QUANTIFIED: identical-human/different-robot siblings are
  bit-identical on the human side (max abs diff 0.0) with robot DoF divergence mean 0.077
  rad (max-joint 1.36) — **12-40x above E47's 0.0019 rad multimodality floor**, and
  leg-dominated (0.110 vs 0.076 rad arms) = lands on our foot-skate axis.
- BUT the multimodality is **latent-variable-explained** (terrain z-scale / object pose =
  hidden variable h; p(robot|human) multimodal, p(robot|human,h) near-Dirac again).
  Conditioning and generation are CONFOUNDED explanations — E56-D separates them.
- v1 E55-A pilot pool was missing ALL rot/trans siblings (the richest 197 pairs) +
  z_scale 1.0 → killed at <4k steps, full pool converted (1,938 clips / 365k frames /
  3.38 h), v2 relaunched.
- Skeleton compatibility is the REVERSE of assumed: 53-joint mocap names overlap LAFAN
  21/22; 52-joint OMOMO overlaps SMPL-H only 3/52.

**E56-D (NEW, promoted to first — teacher-conditioned decoder):** 2-way teacher embedding
concat to embodiment code feeding AdaLN (broadcast per-frame, Kobus'17: feature > prefix
token), dropped to null with p=0.5 (HOVER Bernoulli(0.5) precedent). Eval: decode under
each code + null. **This is the instrument that decides whether the 0.077 rad divergence
is explainable (conditioning suffices → E56-C dead again, publishable negative) or
residual (generative head justified).** Cost ~4h. Kill: per-teacher-code MPJPE not >=10%
better than pooled no-code on omni clips AND null-code not between the two.

**E56-C narrowed to MeanFlow (2505.13447), output-space head:** at a Dirac conditional the
MeanFlow bootstrap term vanishes IDENTICALLY and the model becomes noise-independent
regression (derivation ours from their Eqs. 3+12) — downside bounded at our current
baseline, the property that makes it safe post-E43-47. Gates: (i) NFE=1 matches baseline
within +0.3cm (implementation check); (ii) resampling reproduces >=50% of the 0.077 rad
spread (else close the generative line permanently). ACT's ablation (35.3%->2% human data,
no effect deterministic data) is the closest published prediction. Alternative if GPU
tight: 2-head relaxed-WTA decoder (eps=0.05, WINDOW-level assignment per Seo'20 — per-frame
WTA would thrash mid-clip).

**Two-teacher theory:** pool, don't compose (RRR 2302.11552: mixtures unreachable by score
composition; products select the intersection = wrong target). Averaging destroys
complementarity (CA-MKD: -1.67%; Fukuda'17 switched training). Teacher-ID conditioning
w/ dropout = genuine literature gap (zero hits) — E56-D is novel.

**Doc corrections applied:** Feng attribution (Dirac corollary is ours), AdaMorph zero-shot
scope (unseen MOTIONS not robots — LORO setting remains open for everyone). Cite SAME/NSM/
DeepPhase/LMP by DOI (no arXiv IDs exist).

## E61 — REGISTERED (2026-07-31): Why leverage the retargeting latent? The noise-redundancy test

**Hypothesis (the paper's "why" story):** the z_ret channel is computed from the HUMAN
side of the pipeline, so it is statistically independent of corruption in the robot-space
reference. Arm D (prior sees cmd + z_ret) therefore holds redundant goal information that
arm C (cmd only) structurally cannot have. Under reference corruption — the real-world
failure mode (retargeting artifacts, video-mocap jitter, network dropout in teleop) —
D should degrade more slowly than C. Precedent: UniTracker's latent gain widens under
noise (+3.58pp clean -> +7.20pp at their noise level 2).

**Design (eval-only, zero training cost — existing C/D checkpoints, 3 seeds each):**
sweep eval-time Gaussian corruption of the NORMALIZED 58-d command at sigma in
{0, 0.1, 0.25, 0.5, 1.0} x {C, D} x 3 seeds x 100 rollouts (knobs E52_EVAL_NOISE_CMD;
also E52_EVAL_NOISE_ZRET for the symmetric control — corrupt z_ret instead, D should
converge to C's clean performance, proving the redundancy is bidirectional).

**Pre-specified readouts/gates:**
- Primary: completion-vs-sigma curves; PROMOTE the "retargeting latent as robustness
  channel" story iff D-C >= +5pp at some sigma with non-overlapping seed ranges.
- Secondary: the D-C gap as a function of sigma (monotone widening = clean story).
- Control: noise on z_ret only — if D degrades toward C-clean (not below), redundancy
  confirmed; if D collapses, D was actually leaning on z_ret more than on cmd.
- Null reading: D==C at all sigma -> z_ret adds nothing even under corruption; the
  additive-value claim is dead on single-clip in ALL conditions (still a publishable
  boundary for the paper).

Cost: 30 eval runs x ~4 min = ~2 h GPU. Queue: immediately after E58 seeds finish
(same checkpoints directory layout).
