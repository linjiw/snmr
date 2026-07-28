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
   model, and guidance is provably annihilated. The fix is upstream: **E55's two-teacher
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
