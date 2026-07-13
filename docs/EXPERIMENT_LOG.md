# SNMR experiment log

One entry per training/eval run: artifact paths, exact provenance, headline numbers, and what we
concluded (including corrections). Newest last. Conventions: MPJPE = whole-body mean per-body
position error vs GMR teacher on the 7 held-out clips; "4-window eval" = in-training eval
(4 windows/clip), "16-window" = denser final eval — the two are NOT directly comparable.

---

## 2026-07-09

### E01 — Phase-1 G1, 60k, z=64 (`runs/phase1_g1/`)
- cmd: `train_phase1.py --steps 60000` (defaults; scaled-anchor root frame)
- result: 5.2 cm (4-win) / 7.0 cm (16-win); dof 0.10 rad. Baseline ckpt `ckpt_60k_final.pt`.
- concluded: capacity-bound (train loss still falling); scale up.

### E02 — Phase-1 G1, 100k, z=128, lr 8e-4 (`runs/phase1_g1_large_lr8e4_diverged/`)
- result: DIVERGED at ~12k (loss 0.37→0.66 plateau); final 18.9 cm. **Negative result kept.**
- concluded: this architecture needs lr ≤ 3e-4.

### E03 — Phase-1 G1, 100k, z=128, lr 3e-4 (`runs/phase1_g1_large/ckpt_100k_final.pt`)
- result: **2.18 cm** (4-win) / 3.12 cm (16-win); dof 0.05. **GATE G1 PASSED.**
- benchmark vs teacher (`benchmark.json`): MPJPE 3.6 cm; penetration/limits/joint-jumps ≈ teacher
  or better; limit-proximity 0.29 vs 0.57 (better); **skate 0.25 vs 0.05 m/s (open gap)**;
  skate decomposition: dof-caused (teacher-root substitution: 0.42; teacher-dof: 0.24).

## 2026-07-09/10

### E04 — Phase-2 all-5 shared, 100k, K=2, L_z=1.0 (`runs/phase2_all5/ckpt_100k_final.pt`)
- result (16-win): G1 6.67 / T1 5.80 / N1 5.54 / PM01 5.72 / Toddy 3.16 cm (mean 5.38); Lz→0.001.
- ⚠ CORRECTION HISTORY: an initial "positive transfer" reading compared shared@100k
  (complete cosine schedule) vs specialist@40k (mid-schedule) — invalid; schedule completion alone
  moves the specialist 6.30→2.18 cm. See E07 for the confound-free verdict.
- latent analysis (`latent_analysis_final.json`, code post-bugfix): CKA 0.910; retrieval
  human→robot R@1 0.749 / MRR 0.838 (chance 0.010); E1 linear probe 0.278 (chance 0.167);
  E3 MLP attacker 0.909, k-class proxy-A 1.78; E2 motion probe 0.151 (window-mean caveat).

### E05 — N6 latent polish on E03 ckpt (no artifact dir; numbers in design doc §0.4b)
- skate-heavy window: 0.160→0.118 m/s (~25 %) at +1–2 cm MPJPE; clean window: no gain.
- concluded: weak, window-dependent lever; training-time contact loss is primary (→ E10).

### E06 — LORO, holdout engineai_pm01, 100k (`runs/phase2_loro_pm01/`)
- result (16-win): trained robots ≈ E04 (Δ ≤ 0.04 cm each); **zero-shot PM01 29.59 cm vs 5.72
  in-training = 5.2× — FAILS ≤2× criterion.** Negative result.
- concluded: MJCF-derived embodiment code alone doesn't generalize at 5-robot diversity;
  embodiment augmentation (synthetic MJCF variants) promoted to critical path.

### E07 — Ablation `base`: G1 specialist, z=128, 50k complete schedule (`runs/ablations/base/`)
- result: 3.75 cm (4-win). **The matched baseline for E04's ~40k effective G1 steps.**
- verdict: sharing costs ~1.4 cm at 1.6M params (5.12 vs 3.75). Gate-G2 part 1 strictly FAILS;
  honest claim = "5 robots in one net at modest per-robot cost; scale-outlier Toddy best (3.16)".
- benchmark: 50k has worse skate than 100k (0.386 vs 0.255) → part of skate is undertraining,
  but the curve plateaus far above teacher 0.052.

### E08 — Ablation `no_temporal` (`runs/ablations/no_temporal/`)
- result: 3.07 cm val (vs base 3.75); benchmark MPJPE 3.95 vs 4.71 cm, skate 0.358 vs 0.386,
  dof jerk 720 vs 614 (worse).
- concluded (SURPRISE): the temporal transformer is NOT the noise-suppressor we assumed — removing
  it *improves* MPJPE and skate slightly and only hurts jerk. The NMR-style
  bidirectional-context rationale does not transfer at this scale. Candidate explanations:
  temporal mixing over a 64-frame window smooths *across* poses (hurting per-frame fidelity)
  more than it removes jitter. Revisit after the contact retrain: if contact loss needs temporal
  context, this trade may flip.

### E09 — Ablations `z32`/`small`/`contact`/`lr8e4` — RUNNING (`runs/ablations/`)
- driver: `scripts/_run_ablations_nogate.py` (GPU-busy gate bypassed; foreign 1.4 GB job co-resident).
- note: benchmark loader fixed mid-grid to reconstruct optional modules from the state dict
  (no_temporal row was failing to load); backfilled.

### E18 — trackability proxy (open-loop PD replay) — DONE (`runs/trackability/comparison.md`)
- method: PD-torque replay in MuJoCo (holosoma gains/effort limits), survival before divergence +
  dof error while alive, 3 deterministic windows/clip, matched SNMR-vs-GMR exports (N8 package).
- validity: +0.3 rad reference noise cuts survival 0.98→0.67 s — the metric discriminates.
- result: survival SNMR 0.87 s vs GMR 0.82 s (mixed sign per clip); dof err 0.212 vs 0.210 rad.
  **Measured equivalence: SNMR data is not less trackable than teacher data under this proxy.**
  Caveat: deltas within between-window spread; supports "not worse", not "better". IsaacSim WBT
  stays the ground truth. Side-finding: holosoma G1 MJCF has permanent ~2 cm knee↔ankle
  self-penetration (rest-pose contact forces move the root ~1 cm) — relevant to the WBT handoff.

### E20 — WBT-on-MuJoCo feasibility scout — DONE, **GO-WITH-SHIM (major unblock)**
- Verdict (verified with file:line evidence): robot-only `g1-29dof-wbt` can train on the holosoma
  **MuJoCo/Warp backend TODAY** — the "IsaacSim-only" gates don't fire for our case:
  `wbt_manager.py:16` blocks IsaacGym only; `command/terms/wbt.py:592` fires only for object
  motions; `:1251` only for default-pose prepend/append (both off by default). The MuJoCo wrapper
  was deliberately prepped for WBT (`mujoco.py:444` "_body_list ... for compatibility with
  whole_body_tracking"); every API the WBT managers call is implemented (full table in the scout
  report; `refresh_sim_tensors` populates per-body buffers; WarpBackend gives zero-copy GPU views).
- Required: CLI overrides only — simulator target swap + `--randomization.ignore_unsupported=True`
  (skips one IsaacSim-only material randomizer). Needs `warp-lang==1.10.0` + `mujoco_warp` in a
  separate env (compatible with cu124).
- Consequence: **E14 (the decisive SNMR-vs-GMR tracking comparison) no longer needs an external
  IsaacSim machine** — schedule as the next GPU block after E10. Caveat: MuJoCo-vs-PhysX physics
  differences are fine for a *comparative* experiment (same sim both arms).

### E19 — latent exploitation suite — DONE, mixed + one actionable failure (`runs/phase2_all5/latent_exploitation.json`)
- **Interpolation (latent vs qpos-blend baseline, a=0.5):** the qpos baseline scores BETTER on
  plausibility (G1 skate 0.045 vs 0.087; body jerk 49 vs 136). Interpretation: frame-wise nlerp of
  two smooth motions is smooth-but-semantically-meaningless "average pose" — plausibility metrics
  alone cannot demonstrate latent-interpolation superiority. Don't claim it in the paper without a
  semantic metric (e.g. retrieval of blends, or user study). Honest reframe: latent blends stay
  *valid motions* (limit_viol 0, low penetration) but are not smoother than naive blending.
- **Arithmetic:** z_fight − z_walk + z_run retrieves fight1 as nearest clip (composition dominated
  by the added component — sane), decodes plausibly (skate 0.148, limit_viol 0). Works as a
  qualitative demo, matches SAME's emergent-arithmetic claim.
- **Robot→robot transfer (the unique capability): FAILS at usable fidelity** — G1→T1 44.7 cm,
  T1→G1 38.2 cm, G1→Toddy 24.0 cm vs 2.9–4.7 cm for the human→robot path on the same window.
  **Root cause (structural, not mysterious): the decoder was only ever trained on HUMAN encodings
  z_h; robot encodings z_r enter training solely through L_z (aligned to ~0.001 MSE), and that
  residual gap is out-of-distribution for the decoder.** Fix is cheap and concrete: **decode-from-
  z_r augmentation** — with probability p, decode a sampled robot's teacher encoding instead of
  z_h and distill as usual (also strengthens L_z gradients). Scheduled as E21; enables the
  "retarget robot A's library to robot B without human data" capability the design promises.

### E12-engine — embodiment-variant generator — BUILT & VALIDATED (subagent, reviewed)
- `scripts/make_embodiment_variants.py` + 11 tests: 6 scaled G1 variants (legs/arms/uniform),
  FK-verified ~1e-8, limb ratios exact to 1e-7, base untouched; `--retarget` patches GMR's
  ROBOT_XML_DICT and emits standard pair NPZs (walk1 direction-check passed: legs_0.85 pelvis
  0.751 < base 0.781 < legs_1.15 0.819). Variants live beside the source model (documented in
  THIRD_PARTY.md). Ready for the E12 augmented retrain.

### E09/N9 — Ablation grid COMPLETE (`runs/ablations/SUMMARY.md`); + skate diagnosis
Full 50k G1 table (MPJPE / skate m/s vs teacher 0.052):
  base 4.71/0.386 · **no_temporal 3.95/0.358 (best fidelity)** · z32 5.51/0.435 (bottleneck hurts)
  · small 4.78/0.412 (0.41M ≈ base) · **contact(w=0.1) 4.87/0.402 (skate barely moved)** ·
  lr8e4 32.0/0.829 (diverged — negative control).
Diagnoses:
1. **Skate tracks MPJPE** (ratio ~7–9 across all G1 ckpts: E03 3.62cm/0.255, base 4.71/0.386,
   no_temporal 3.95/0.358) — foot sliding is the *same* dof error, not a separate failure mode.
2. **E22 filter test (post-hoc Gaussian low-pass on decoded dof, E03 ckpt, 4-clip subset):**
   σ=0 → 0.093 m/s, σ=1 → 0.081, σ=2 → 0.086 at ~unchanged MPJPE. **Low-pass barely helps → the
   residual foot motion is SMOOTH systematic error, not high-frequency jitter.** Corrects the
   earlier "dof-jitter" wording. Implication: filtering/architecture won't fix it; the
   contact-velocity loss (or inference foot-lock) is the right tool, and w=0.1 (teacher-mask only,
   no EDGE head — train_phase1 lacks predict_contact) was too weak → E10 must test the EDGE head
   at higher weight.
3. no_temporal beating base persists at 50k (confirmed E08) — temporal transformer trades per-frame
   fidelity for jerk; keep it only if the contact/tracking interplay later justifies it.

### E23 — latent figures (t-SNE/UMAP dual-color + CKA heatmap) — DONE (`runs/figures/`)
- `scripts/make_latent_figures.py`; 672 window-mean latents (112/encoder × 6), held-out clips.
- **Dual embedding: embodiment-colored plots are thoroughly intermixed (6 embodiments overlap
  everywhere); motion-colored plots show clean category clusters (aiming/walk/run/sprint/jumps
  each a distinct region) in BOTH t-SNE and UMAP.** The visual headline of the shared space.
- **Reconciles the E16 "null" result:** motion category IS strongly organized in the latent —
  the linear probe (E16, 0.15) just couldn't read it because the structure is NONLINEAR
  (manifold-organized), which t-SNE/UMAP surface. Correct paper phrasing: "motion is nonlinearly
  organized (visible under t-SNE/UMAP; a linear probe underreads it)", NOT "not semantically
  organized". Supersedes E16's conclusion.
- **CKA heatmap** matches morphology: adult humanoids cluster tight (n1↔pm01 0.98, t1↔{n1,pm01}
  0.96); the weakest pair is **human↔toddy 0.82** — the two morphological extremes (adult human
  vs 0.34 m toddler). Sensible, publishable structure.

## Queued / planned
- E10 — contact-weight sweep (EDGE head) on G1 — RUNNING (w={0.5,2.0}, 100k each). (EDGE head + world-frame losses,
  post-review). Decisive for C5. Accept: skate ≤ 0.08 m/s at MPJPE ≤ 1.5× current. AUTO-CHAINED
  after the ablation grid.
- E21 — decode-from-z_r augmentation (fix for E19's robot→robot failure): `--zr_decode_prob`
  wired into train_phase2 (smoke-tested); fold p≈0.3 into the next shared retrain (can combine
  with E10's contact run or E12's augmented run to save GPU blocks).
- E12 — augmented retrain: generate variant pairs for a clip subset (engine ready), add variants
  as extra "robots" in train_phase2, re-run LORO → the zero-shot fix experiment.
- E14 (UPGRADED by E20): SNMR-vs-GMR WBT tracking comparison **locally on MuJoCo/Warp backend** —
  no external IsaacSim needed; separate env (warp-lang 1.10 + mujoco_warp), CLI-override command
  in the E20 scout report. Sequence after E10.
### E11 — scale-leak probe — DONE (`runs/phase2_all5/scale_leak_probe.json`)
- setup: robot-only latents (5 classes, chance 0.2), E04 ckpt; features height-normalized by
  standing root height (positions+velocities /h; rot6d untouched) vs raw.
- result: MLP attacker **0.943 raw → 0.933 normalized — scale explains only ~1% of the leak.**
  **H-deep confirmed:** the embodiment signature is structural/stylistic (limb proportions,
  per-robot IK habits), not amplitude. Curious secondary: the *linear* probe **rose** 0.35→0.68
  under normalization — height division seems to make proportional differences more linearly
  separable (worth a sentence in the paper, not a headline).
- consequence: a domain-confusion (GRL/uniform-CE) term on the encoder is now *motivated by
  measurement*, not speculation → schedule as E15 after the contact sweep. Also: "leak ≠ scale"
  strengthens the analysis-section contribution.
- E12 — embodiment augmentation (synthetic MJCF variants) → LORO revisit.
- E13 — capacity scale-up of the shared model (does the 1.4 cm sharing cost shrink?).
- E14 — WBT tracking comparison (N8 package, needs IsaacSim machine).
- E15 — domain-confusion term on the encoder (motivated by E11's H-deep verdict): GRL or
  uniform-CE adversary on embodiment id; measure attacker drop vs MPJPE cost.

### E17 — Ablation `z32` — DONE (`runs/ablations/z32/`)
- result: 4.4 cm val / 5.51 cm benchmark MPJPE (vs base 3.75/4.71); skate 0.435 (worse than base).
- concluded: the per-frame latent bottleneck is load-bearing — SAME's z=32 (enough for stylized
  characters) underfits whole-body robot retargeting; validates the §5 risk-table prediction
  ("single z per frame too coarse → scale d"). Keep z=128.

### E16 — temporal-statistics motion probe — DONE, NULL RESULT (`runs/phase2_all5/e2_temporal_probe.json`)
- hypothesis (from E04): E2's weak motion probe (0.151) is a window-MEAN artifact; richer readout
  (mean ⊕ std ⊕ |Δz| mean, 3× features) should recover category structure.
- result: **0.152 for both readouts** (chance 0.125, control 0.133). Hypothesis REJECTED.
- reframed conclusion: the latent carries *instance-level* motion detail (exact-clip retrieval
  R@1 0.75 at 1% chance) but motion-*category* structure is not linearly separable in it — the
  space is organized around reproducing specific trajectories (what distillation trains for),
  not semantic classes (which nothing trains for). Honest paper phrasing: "content-specific, not
  semantically organized"; a nonlinear probe or contrastive category loss would be needed to say
  more. Correct the earlier "window-mean caveat" wording in the draft.
