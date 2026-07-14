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

### E10a — Phase-2 contact sweep (bundled BCE + self-consistency + contact head) — DONE
`runs/e10_contact_w{0.05,0.2,1.0}/` (5-robot, 60k, --contact_weight only). Final G1 MPJPE:
w0.05 8.34 / **w0.2 7.93** / w1.0 9.07 cm (all-5 no-contact baseline E04 was 6.67) — contact loss
at 60k costs a little MPJPE (undertrained vs 100k), Toddy still best (3.5–3.9). Skate benchmark
pending to answer whether it reduced sliding (the actual point).
⚠ PROVENANCE CORRECTION (2026-07-13 audit): the original entry said "NO EDGE head" — that was
WRONG. train_phase2 builds the contact head whenever contact_weight>0 (predict_contact wiring
predates this run, commit 95c7b2c), and the checkpoints contain `decoder.contact_head.*` weights
(verified directly). E10a therefore tested the *bundled* objective (BCE + EDGE self-consistency +
teacher-mask velocity at ONE shared weight), so its failure does not attribute blame to any single
component. See docs/NEURAL_RETARGETING_RESEARCH_sol.md P0 for the factorized re-test plan.

### E10a-skate — CRITICAL: contact loss (current form) does NOT reduce skate
Benchmarked phase2 contact ckpts: w0.2 → skate **0.497 m/s** (WORSE than no-contact ~0.39);
ablation w0.1 was 0.402 (no gain). Mask is verified sane (E10a-diag: contact frac 0.58 walk /
0.20 run; teacher feet near-stationary under their own mask, 0.034 m/s). So the loss TARGET is
right but it isn't being learned — hypothesis: at w≤0.2 the contact term is dominated by distill;
and the distill loss itself pulls toward teacher dof whose tiny errors already skate, so the two
objectives conflict. w1.0/w0.05 skate benchmarks running to complete the weight curve → decides
whether E10b (higher weight + EDGE head) can work or whether we need a different mechanism
(inference foot-lock, or hard contact constraint) for C5.

### E24 — ROOT CAUSE of foot skate found (reframes C5)
Chased why both the contact loss (E10a) and an inference foot-lock (E24-footlock: skate
0.160→0.153, negligible) fail. Decoded-vs-teacher foot diagnostic on walk1 (E03 ckpt):
- decoded foot HEIGHTS match teacher (mean z 0.067 = 0.067) — feet reach the ground correctly;
- but **contact-mask agreement is only 58%; decoded contact frac 0.33 vs teacher 0.74** — the
  decoded foot is at the right height yet moving TOO FAST in xy during stance (fails the 0.3 m/s
  speed test), so contact under-fires.
**So the skate is xy oscillation of a correctly-placed foot, not drift-from-a-planted-point.**
That explains everything: (a) foot-lock (pins position) can't fix a velocity problem; (b) low-pass
(E22) can't fix it because the oscillation is correlated across the leg chain (small dof errors ×
long lever arm), not per-joint iid jitter; (c) the contact-velocity loss SHOULD address it but
loses to distill at usable weights (E10a). **Correct C5 fix candidates, reframed:** (i) match
teacher FK foot VELOCITY directly as a distill term (not just dof/pos), so the objective that wins
also enforces low stance velocity; (ii) contact loss with a schedule that ramps weight as distill
converges; (iii) accept the gap and report it honestly (skate ~2-3× teacher) — it is a known
limitation of pure kinematic distillation that the downstream RL tracker is expected to absorb
(GMR paper: the tracker cleans residual artifacts). Leaning (i) — cheap, fits the existing loss
framework, doesn't conflict. E25 = add FK-foot-velocity distill term.

### E10b — G1 contact sweep WITH EDGE head — ABANDONED mid-run
`runs/e10_edge_w{0.5,2.0}/` via train_phase1 --edge_contact, 100k. w0.5 has only a 5k-step log
point (19.3 cm); w2.0 never started. Superseded: (a) the 2026-07-13 audit showed E10a already
bundled the EDGE head, so E10b was not the clean test it was framed as; (b) E26 (below) identified
a strong source-mask correction lever, changing what a contact-loss retrain is for. If revisited, use the
factorized C0–C6 matrix in docs/NEURAL_RETARGETING_RESEARCH_sol.md instead of a bundled weight.

## 2026-07-13/14

### E25 — foot-velocity distill sweep — w2.0 DONE (`runs/e25_footvel_w2.0/`), w10.0 running
w=2.0, G1, 100k, z=128: val MPJPE 2.11 cm (4-win) / **2.99 cm final (16-win)**, benchmark 3.01 cm
— BETTER than the E03 no-footvel baseline (3.12/3.62 cm): the term costs nothing and slightly
helps fidelity. Skate: legacy metric **0.225 m/s vs E03 0.255** — improved, but the skate/MPJPE
ratio (~7.5 s⁻¹) is UNCHANGED, i.e. the improvement is the fidelity improvement's shadow, not a
stance-specific fix. Verdict: **keep the term as a free regularizer; it is NOT the C5 fix.**
Consistent with E24's diagnosis: matching teacher foot velocity everywhere mostly reweights the
same regression error; the stance oscillation survives because it is the *residual* of that
regression, not a separately-supervised quantity. Literature context (verified 2026-07-13 review):
Aberman'20 (2005.05732) uses EE-velocity matching at their LARGEST loss weight (anti-slide, with
residual IK cleanup still needed); T2M-GPT shows weight sensitivity (α=0.5 helps, α=1 hurts) —
w=10.0 (running) tests whether a much larger weight shifts the trade-off or destabilizes.

### E26 — foot-lock driven by SOURCE contact mask — promising pilot, not a gate result
(`runs/skate_structure/diagnosis_v2.json`, `scripts/diagnose_skate_structure.py`, E03 ckpt, 3
held-out clips.) E24's foot-lock failed because contact DETECTION fails on the decoded motion —
the stance xy oscillation (~2.9 cm RMS, autocorr τ≈0.10 s) exceeds the 0.3 m/s speed gate, so
detected contact frac is 0.03 vs teacher 0.29 and most stance frames were never locked. The
literature-standard fix (Harvey'20/PFNN/Villegas'21/UnderPressure: labels must come from a CLEAN
signal or a trained head, never from thresholding the noisy output) applies directly: the HUMAN
source motion's contact flags are always available at inference and agree with teacher stance at
0.80 (vs 0.74 decoded). Pipeline: human mask → dilate (merge gaps ≤6, extend 2) → per-interval
pin (median xy, min z) with 2-frame blend ramp → 12-iter damped leg IK.
Results (mean of walk1/run2/dance2, pilot with normalized-gradient solver): **skate 0.489 →
0.064 m/s (7.6×, BELOW the ≤0.08 target) at ~+0.1 cm MPJPE**. Ablation chain: decoded-mask lock
0.27 (detection is the bottleneck) → human-mask 0.18 (coverage) → +dilation 0.064. Jerk cost
observed on dance (raw 1540 → locked 1572 dof jerk; the raw motion is already jerky there).
NOTE on σ-smoothing: smoothing the IK *correction* trades the skate reduction back
(σ=1: 0.19; σ=2: 0.28 vs σ=0: 0.05 on walk1) because the correction is NOT constant within
stance — it tracks the oscillation it cancels; low-passing it re-admits the oscillation. Keep
σ=0 as default; the jerk mitigation must come from interval-level blending, not correction
smoothing. The production implementation (`snmr.footlock.foot_lock_masked`) was then upgraded to
a true damped-least-squares solver (Jacobian via autograd, line search, limit clamping) replacing
the pilot's normalized-gradient step; E26b re-runs the full 7-clip × windows × σ grid with the
Gate-0 bootstrap protocol on the DLS solver — its numbers are the citable ones.

### E26b — full source-mask DLS grid — strong partial result, Gate 1 still open
`runs/skate_structure/footlock_dls_grid_full.json`: 42 fixed windows, seven held-out clips,
source-contact mask, four DLS iterations, σ∈{0,1,2}, clip bootstrap. An evaluator bug was found
while expanding the pilot: windows with no active teacher-height samples previously returned zero,
which biased the first aggregate downward. `contact_motion_metrics` now returns undefined values
for empty masks and `_aggregate_rows` excludes them; regression coverage protects this contract.

The best arm is **σ=0**. Window-weighted teacher-height stance speed falls
**0.502→0.119 m/s** (clip mean 0.419→0.094, 95% clip-bootstrap CI
`[0.043, 0.161]`), while source-contact speed falls **0.341→0.077 m/s**. MPJPE changes
3.66→3.86 cm, DOF jerk rises 13%, limits remain valid, and penetration/MPJPE/jerk guards all pass.
However, the preregistered teacher-height target is `<=0.08 m/s`, so **E26b does not pass Gate 1**.
σ=1/2 reduce the jerk cost but weaken teacher-height speed to 0.142/0.167 m/s. A source-height
mask over-locks badly (8.47 cm MPJPE), while a sparse teacher-height oracle reaches only
0.110 m/s. The remaining issue is stance coverage/temporal constraints, not DLS convergence or
smoothing. This heuristic also holds root pose fixed and solves frames independently, so it is
not the full windowed C6 projection specified by the audit.

### E26c — converged DLS arms (12 iters, extend 3) — **primary endpoint reached; jerk guard open**
(`runs/skate_structure/footlock_dls_i12_e3_{source,teacher}.json`, same 42-window protocol.)
E26b's residual was under-convergence (4 iters) + stance under-coverage (extend 2):
- **source_contact mask (deployable): teacher-height stance speed 0.502 → 0.068 m/s — PASSES the
  ≤0.08 preregistered endpoint** (legacy skate 0.289→0.030); cost MPJPE 3.66→4.08 cm (relative
  guard +0.5 cm passes; absolute ≤4.0 cm misses by 0.08) and dof jerk 693→939 (+35%, fails the
  1.2× guard) with joint jumps 0.016→0.066 — the aggressive full-strength pinning trades contact
  for smoothness.
- **teacher_height mask (oracle upper bound): 0.502 → 0.047 m/s with MPJPE 3.70, jerk 727 —
  ALL GUARDS PASS.** The oracle-vs-source gap (0.047 vs 0.068 + the jerk difference) is mask
  precision: the dilated source mask over-covers swing frames, whose blended pinning adds edits
  the oracle never makes.
- **E26c-2 blend 5 completed:** the longer ease-in/out restores the jerk guard (693→780,
  `1.13×`) and keeps MPJPE within the absolute target (3.66→3.98 cm), but teacher-height speed
  rebounds to **0.132 m/s** and misses the endpoint. The framewise heuristic therefore has a
  measured skate-vs-jerk Pareto: blend 2 passes speed and fails jerk; blend 5 passes jerk and
  fails speed. The oracle row shows that mask precision/transition handling, not the DLS solve,
  is the remaining gap. A trained contact head is one possible mask lever, but C0-C4 attribution
  and the full temporal C6 baseline come first.

### E27 — WBT-on-MuJoCo/Warp SMOKE — **PASSED (2026-07-13)**; paired pilot QUEUED
Dedicated env `.venv-wbt` (py3.11, torch 2.6 cu124, holosoma editable, mujoco-warp 3.10.0.2,
warp-lang 1.15 — holosoma's ==1.10.0 pin conflicts on paper but everything imports and runs).
Smoke: `exp:g1-29dof-wbt simulator:mjwarp logger:disabled --training.num_envs=256
--randomization.ignore_unsupported=True` with OUR exported GMR walk1 NPZ — 20 PPO iterations,
~4k steps/s at 256 envs, reward terms populate, checkpoint saved. **E20's GO-WITH-SHIM verdict is
now executed reality; C6 no longer has an infrastructure blocker.** Syntax gotchas: logger and
simulator are tyro SUBCOMMANDS (`logger:disabled`), not flags.
**Queued (auto-chained after E25 w10 frees the GPU):** paired pilot — 3 clips (walk/dance/fight)
× {gmr, snmr} × seed 0, 1024 envs, 1000 iters each, identical config
(`$CLAUDE_JOB_DIR/tmp/wbt_pilot.sh` → `runs/wbt_pilot/`). Per audit Gate 2: pilot detects
catastrophic regressions only; the confirmatory claim needs more clips + seeds.

### E28 — Gate-3 sharing-gradient diagnostic — DONE (`runs/phase2_all5/sharing_gradient_diagnosis_eval.json`)
Per-robot distill gradients on the shared parameters of the E04 checkpoint (12 windows × 5 robots,
`scripts/diagnose_sharing_gradients.py`, model in eval mode): mean cosine is **+0.30** for the
encoder, **+0.14** for the decoder trunk, and +0.12 for the embodiment encoder. Mean-pair conflict
is limited to 1/10 pairs in the latter two groups, but per-window conflict is not negligible:
20.8%/28.3%/41.7% of observations are negative, with negative means
−0.18/−0.18/−0.27. Gradient-norm imbalance is 2.94×/1.88×/1.89×.
**Verdict:** there is no pervasive mean directional conflict and no severe decoder imbalance, so
PCGrad/GradNorm are not first-line arms. Sporadic conflicts remain a measured secondary
hypothesis; test S2 width and S3 lightweight adapters first, then use matched training to decide
whether S4 is warranted. The earlier diagnostic used training mode and only pair means; this
revision removes dropout contamination and preserves per-window distributions.

### E29 — full temporal C6 projection — **source mask fails; teacher-mask oracle passes**
Implementation commits `7903994` and `f6ed96a`; clean 42-window artifacts:
`windowed_c6_{source,teacher_oracle}_velocity_full.json`. This is materially different from E26:
one L-BFGS solve jointly optimizes the full 192-frame window over both leg chains and bounded root
XYZ/yaw, with interval anchors, exact joint limits, and correction deviation/velocity/acceleration.
The endpoint-aligned revision also supports one-frame intervals and directly penalizes the
non-circular, mask-gated XY displacement scored by the evaluator.

- **Source-contact mask:** teacher-height speed **0.502→0.099 m/s**, source-contact speed
  **0.341→0.020**, MPJPE **3.66→4.28 cm**, DOF jerk **693→741**. Limits and penetration/jerk
  guards pass; the primary speed and relative MPJPE guards fail.
- **Teacher-height oracle:** teacher-height speed **0.502→0.0056 m/s**, MPJPE
  **3.66→3.82 cm**, DOF jerk **693→697**; **all guards pass**.

All 42 source-mask windows reached the iteration cap and most saturated the 4 cm root bound.
Loosening bounds is not justified because MPJPE already fails. The controlled oracle result
localizes the remaining deployable C6 gap to mask precision/coverage. Do not call C6 or Gate 1
closed; proceed with C0-C4 and prioritize learned/physics-repaired contact labels over another
post-process parameter sweep.

### GPU queue (2026-07-13 night)
1. E25 w10.0 (running, ~35k/100k) — note new diagnostics.jsonl shows w10 foot-vel term reaches
   grad-norm ~0.08 on output heads vs distill 0.22 with cosine −0.11 (mild conflict, not the
   w≤0.2 swamping of E10a).
2. WBT pilot (6 runs, auto-chained).
3. Next block: factorized contact calibration C0–C4 (Gate 1), from one fixed clean revision.
   `zr_decode_prob` remains a later, separate Gate 4 experiment.
CPU: E26b DLS grid complete; see above.

## Queued / planned
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
- result: MLP attacker **0.943 raw → 0.933 normalized**; the linear probe rises 0.35→0.68.
- audit correction: this evaluation-time normalization is out of distribution because the encoder
  was trained on raw features. It shows that this intervention does not erase identity, but it
  does **not** quantify scale's causal share or prove a structural/stylistic cause. Use a
  normalized-input retrain, matched-scale subsets, or conditional morphology probes before a
  domain-confusion intervention.
- E12 — embodiment augmentation (synthetic MJCF variants) → LORO revisit.
- E13 — capacity scale-up of the shared model (does the 1.4 cm sharing cost shrink?).
- E14 — WBT tracking comparison (N8 package; local MuJoCo/Warp smoke passed, pilot queued).
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
