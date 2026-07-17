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

### E25 — foot-velocity distill sweep — COMPLETE
w=2.0, G1, 100k, z=128: val MPJPE 2.11 cm (4-win) / **2.99 cm final (16-win)**, benchmark 3.01 cm
— BETTER than the E03 no-footvel baseline (3.12/3.62 cm): the term costs nothing and slightly
helps fidelity. Its original benchmark predates the corrected teacher-height aggregation and must
not be compared with the final w=10 artifact. Re-evaluation with the same current evaluator
(`benchmark_matched.json`) gives w=2 teacher-height stance speed **0.417 m/s**, source-contact
speed **0.289 m/s**, and DOF jerk **682 rad/s^3**.

w=10.0 completed 100k with 1.91 cm four-window validation MPJPE and 2.66 cm final held-out MPJPE.
On the matched 42-window evaluator it improves MPJPE **3.01 -> 2.68 cm**, teacher-height stance
speed **0.417 -> 0.304 m/s** (27%), source-contact speed **0.289 -> 0.221 m/s** (24%), and DOF
jerk **682 -> 648 rad/s^3**. This is a real favorable tradeoff, but `0.304 m/s` is still 3.8x the
Gate-1 `<=0.08 m/s` endpoint and 2.3x the matched teacher (`0.133 m/s`). Over the final 26
diagnostic events, the legacy all-phase displacement term's shared-trunk gradient ratio has median
0.069 and p90 0.131. It does not isolate stance and its median is below the factorized movement
band.

Verdict: **use E25 as evidence that stronger velocity matching helps, not as the contact fix.**
The phase-balanced C4 arm remains required, with all legacy weights zero in Gate 1 so attribution
is preserved. Literature context (verified 2026-07-13 review):
Aberman'20 (2005.05732) uses EE-velocity matching at their LARGEST loss weight (anti-slide, with
residual IK cleanup still needed); T2M-GPT shows weight sensitivity (α=0.5 helps, α=1 hurts) —
w=10.0 confirms that a much larger weight shifts the tradeoff without destabilizing.

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

### E27 — WBT-on-MuJoCo/Warp smoke + seed-0 paired pilot — **PASSED**
Dedicated env `.venv-wbt` (py3.11, torch 2.6 cu124, holosoma editable, mujoco-warp 3.10.0.2,
warp-lang 1.15 — holosoma's ==1.10.0 pin conflicts on paper but everything imports and runs).
Smoke: `exp:g1-29dof-wbt simulator:mjwarp logger:disabled --training.num_envs=256
--randomization.ignore_unsupported=True` with OUR exported GMR walk1 NPZ — 20 PPO iterations,
~4k steps/s at 256 envs, reward terms populate, checkpoint saved. **E20's GO-WITH-SHIM verdict is
now executed reality; C6 no longer has an infrastructure blocker.** Syntax gotchas: logger and
simulator are tyro SUBCOMMANDS (`logger:disabled`), not flags.

The auto-chained paired pilot completed three clips (walk/dance/fight) × {GMR, SNMR} × seed 0,
1024 environments, 1000 iterations. `runs/wbt_pilot/analysis.json` passes the frozen analyzer:
all six runs contain exactly 1000 finite events at steps 0–999, final checkpoints, and resolved
configs identical except motion path and run name.

Final-100 effects are small and mixed. Relative to GMR, SNMR reward is −2.9% on walk, +3.1% on
dance, and +3.3% on fight; episode length is −1.2%, +2.7%, and +0.02%. Joint-position error is
3.9% worse on walk and 5.2%/6.0% better on dance/fight. Reference-position error is consistently
0.5–1.8% higher. **Verdict:** no catastrophic tracking regression is visible, and this is enough
to justify replication, but one training seed and training curves cannot establish
non-inferiority or benefit. Complete seeds 1–2 for these clips before calling Stage B complete;
the confirmatory study still needs independent evaluation rollouts and deployment endpoints.

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

### E26c-3 — blend 3 midpoint (`footlock_dls_i12_e3_b3_source.json`)
0.082 m/s @ jerk 857 (1.24×), MPJPE 4.04 — the Pareto is smooth in the blend knob (blend 2:
0.068/1.35×; blend 3: 0.082/1.24×; blend 5: 0.132/1.13×); no free lunch inside the framewise
family. Converges with E29 below on the same conclusion: **mask precision is the bottleneck for
every correction method** → a trained contact head (decoder predict_contact supervised on
teacher-height labels, read out at inference) is the highest-leverage next step — it serves the
Gate-1 C1/C2 arms AND replaces the human-flag mask in both the framewise lock and the E29
projection.

### E26d — foot-lock from the E10a PREDICTED contact head — negative, but not the final word
(`runs/skate_structure/footlock_predicted_mask_e10a.json`; lock_mask=predicted_contact,
threshold 0.5, E10a w0.2 ckpt — the only existing checkpoint with a trained contact head.)
Teacher-height speed 0.874→0.314 m/s: better than nothing, far worse than the human-source mask.
Two confounds keep this from killing the trained-head hypothesis: (a) the E10a base model is much
weaker (raw MPJPE 7.25 cm vs E03 3.66 — its raw stance speed is 0.874 vs 0.502), so the head was
trained on/reads a noisier decoder; (b) the head was supervised with the LEGACY speed-gated mask
(the circular one) at bundled weight w=0.2, exactly the objective E10a showed was mis-scaled.
A head BCE-trained with the corrected mask/normalization (Gate-1 C1-style, weight 0.25 per E30
calibration) on the E03-class specialist is the right test before abandoning the lever.

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

### GPU queue (2026-07-14)
1. E25 w10.0 complete; matched result and verdict recorded above.
2. WBT seed-0 pilot complete; protocol passed and the descriptive verdict is recorded above.
3. Gate 1 three-seed replication complete: C4/C3 pass relative guards but fail the endpoint.
4. WBT pilot seeds 1-2 and 5,400 independent rollouts complete; the GMR control is undertrained.
5. Calibrate GMR seed-0 training horizon at 2k/4k/8k before extending the matched policy matrix.
   `zr_decode_prob` remains a later, separate Gate 4 experiment.
CPU: E26b DLS grid complete; see above.

### E30 — Gate 1 factorized calibration — **COMPLETE WITH C2 DROPPED**
Frozen revision `4929924`, G1, seed 0, 5k steps, ten diagnostics per arm; decision uses the final
five shared-trunk gradient ratios. C0 passed. C1 BCE `1.0` failed (`median=1.704`, `p90=5.628`);
its deterministic retry at `0.25` passed (`0.368`, `0.606`). C3 stance velocity `0.03` passed
(`0.399`, `0.757`). C4 phase-balanced teacher velocity `0.05` passed (`0.414`, `0.568`).

C2's initial BCE/EDGE ratios were `3.212/18.015` and `0.423/2.114`. Its only retry used BCE
`0.25` and EDGE `0.021268901529295597`; EDGE passed (`0.222/0.588`) but BCE still failed
(`0.729/2.008`). Per protocol, **drop C2 and do not try the newly suggested BCE `0.0625`**.
The accepted seed-0 screen is C0, C1 BCE `0.25`, C3 `0.03`, and C4 `0.05`.

### E31 - Gate 1 seed-0 screen - **C4 AND C3 PROMOTE**
Frozen trainer revision `4929924`, evaluator hash `76f73644...28cdd`, G1, seed 0, 50k steps and
the fixed 42-window benchmark. All manifests and benchmarks pass the screen artifact contract.

| Arm | Teacher-height speed | MPJPE | Source-contact speed | DOF jerk | Verdict |
|---|---:|---:|---:|---:|---|
| C0 | `0.709 m/s` | `4.71 cm` | `0.442 m/s` | `614` | control |
| C1 BCE | `0.754 m/s` | `5.75 cm` | `0.487 m/s` | `641` | negative control |
| C3 stance | `0.489 m/s` | `5.00 cm` | `0.299 m/s` | `588` | **promote** |
| C4 phase-balanced | `0.270 m/s` | `3.02 cm` | `0.200 m/s` | `562` | **promote** |

C3 reduces the primary endpoint by `31.0%` and C4 by `62.0%`; both improve every one of the six
measurable clips, remain within the MPJPE guard, improve source-contact speed and jerk, preserve
zero limit violations, and pass both penetration guards. `fight1_subject3` has zero teacher-height
support in all arms, so its undefined speed counts conservatively as no improvement while the
absolute five-clip rule remains unchanged. The analyzer's original all-finite assumption rejected
this valid evaluator output; the preserved pre-fix report and regression-tested zero-support fix
record that correction.

This is a promising causal screen, not Gate 1 completion: neither seed-0 candidate reaches the
`<=0.08 m/s` endpoint. Retrain C0, C3, and C4 from scratch at seeds 1 and 2 before making a contact
claim. The replication analyzer is frozen before launch: aggregate physical guards and MPJPE use
arithmetic means over seeds 0-2, while endpoint and direction requirements count matched seeds.

### E32 - Gate 1 three-seed replication - **FAILS ENDPOINT**
Frozen trainer revision `4929924`, evaluator hash `76f73644...28cdd`, G1, seeds 0-2, and the same
42-window benchmark. All six new manifests and benchmarks pass the frozen artifact contract; no
arm hit the gradient early-stop rule.

| Family | Teacher-height speed seed 0/1/2 | Mean | Mean MPJPE | Mean source speed | Mean jerk |
|---|---|---:|---:|---:|---:|
| C0 | `0.709 / 0.733 / 0.662` | `0.701 m/s` | `4.70 cm` | `0.453 m/s` | `609` |
| C3 stance | `0.489 / 0.418 / 0.502` | `0.469 m/s` | `4.99 cm` | `0.298 m/s` | `567` |
| C4 phase-balanced | `0.270 / 0.244 / 0.278` | `0.264 m/s` | `3.09 cm` | `0.198 m/s` | `566` |

C3 and C4 improve speed in all three matched seeds and pass every relative MPJPE, source-contact,
jerk, limit, and penetration guard. C4 is a robust favorable regularizer: it cuts stance speed by
`62.4%`, improves MPJPE by `1.61 cm`, and passes the absolute `4.0 cm` product target. However,
neither candidate reaches `<=0.08 m/s` in any seed, versus the required two of three. **Gate 1
fails its primary endpoint.**

Decision: retain C4 as evidence and as a possible initialization component for a separately
preregistered physics-aware method, but do not call it contact-consistent and do not tune its
weight again. The deployable C6 projection and the factorized soft arms have now both failed; move
to physics-repaired targets or joint retargeting/control after completing WBT replication.

### WBT replication protocol - FROZEN
Extend E27 with seeds 1 and 2 only: the same three clips, GMR/SNMR sources, MuJoCo/Warp backend,
1024 environments, and 1000 PPO iterations. The combined artifact analyzer requires 18 complete
runs and reports paired final-100 and curve-AUC effects across nine clip/seed pairs. This completes
the Stage B training-seed pilot, not non-inferiority; independent fixed-seed policy rollouts remain
mandatory.

### WBT independent rollout protocol - FROZEN
Before reading the three-seed replication result, freeze evaluation of all 18 final policies with
Holosoma `eebdcf4`: evaluation seeds `101/202/303`, 100 phase-stratified 10-second windows per
seed, identical starts and keyed randomization for each GMR/SNMR pair, and terminal-step metrics
captured before reset. Coprimary endpoints are completion (SNMR-minus-GMR margin `-0.05`) and
joint-position RMSE (relative margin `+0.10`). A 10,000-replicate paired hierarchical bootstrap
over clips, training seeds, evaluation seeds, and rollout ids must pass both confidence bounds.
Pooled GMR completion must also be at least `50%`; otherwise the control is undertrained and the
study cannot return a non-inferiority verdict. This 5,400-rollout experiment can support a
conclusion only for the three pilot clips; Stage C still requires 15-21 clips. Protocol:
`runs/wbt_independent_eval_protocol.sh`.

### E33 - WBT three-seed training replication - **COMPLETE, DESCRIPTIVE ONLY**
All 18 matched GMR/SNMR runs pass the frozen artifact contract: 1000 finite scalar events,
`model_00999.pt`, and resolved configs differing only in run name, motion source, and training
seed. Across nine clip/seed pairs, pooled final-100 means are:

| Metric | GMR | SNMR | Favorable pairs |
|---|---:|---:|---:|
| Reward | `0.655` | `0.658` | `6/9` |
| Episode length | `39.37` | `39.63` steps | `6/9` |
| Joint-position error | `1.680` | `1.615 rad` | `8/9` |
| Reference-position error | `0.267` | `0.272 m` | `1/9` |
| Body-position error | `6.942` | `6.945 m` | `2/9` |

SNMR has a consistent joint-space advantage and small reference-space disadvantage, without a
large aggregate reward or episode-length shift. These are training curves, so E33 completes the
training-seed pilot but makes no non-inferiority or benefit claim. Run the frozen independent
policy evaluation next.

### E34 - WBT independent policy evaluation - **UNDERTRAINED CONTROL**
Frozen Holosoma `eebdcf4`, 18 final policies, evaluation seeds `101/202/303`, 100 matched
phase-stratified 10-second windows per policy/seed, and 5,400 total rollout rows. Every artifact
and pairing check passes.

| Endpoint | GMR | SNMR | SNMR effect |
|---|---:|---:|---:|
| 10 s completion | `0.0%` | `0.0%` | `0.0 pp`, CI `[0.0, 0.0]` |
| Survival | `0.934 s` | `0.956 s` | `+2.3%` descriptive |
| Joint-position RMSE | `0.296 rad` | `0.286 rad` | `-3.30%`, CI `[-5.43%, +0.26%]` |
| Root-position error | `0.250 m` | `0.254 m` | `+1.5%` descriptive |
| Absolute mechanical power | `187.6 W` | `177.3 W` | `-5.5%` descriptive |

Both numerical relative bounds satisfy the stated margins, but the GMR control misses the
preregistered `50%` completion floor by 50 points. The formal verdict is **undertrained GMR
control**; do not claim non-inferiority or tracking benefit. Errors and costs are conditional on
the roughly one second before termination.

Next, calibrate GMR seed-0 controls only at total PPO iterations `2k/4k/8k`, using development
rollout seed `404`. Promote the earliest horizon with pooled completion `>=50%` and every clip
`>=25%`; if none passes at 8k, stop before spending the full 18-policy matrix and reassess the
default 30k schedule/config.

### E35 - WBT GMR horizon calibration - **PARTIAL: WALK CALIBRATED; FULL GATE INCOMPLETE**
Use Holosoma `9fb2b57` and `runs/wbt_horizon_calibration_protocol.sh`. Two prerequisite PPO
bookkeeping defects were fixed in Holosoma: resume now starts after the saved iteration
(`1fb2840`), and periodic checkpoints are emitted after a completed update interval (`9fb2b57`).
Without those fixes, continuation would duplicate iteration 999 and mislabel the requested
milestones.

For each clip, continue the existing GMR seed-0 `model_00999.pt` once for 7,000 additional PPO
updates with optimizer and normalization state restored. Retain `model_01999.pt`,
`model_03999.pt`, and `model_07999.pt`; evaluate each with seed `404`, 100 phase-stratified
10-second windows, and the E34 terminal-aware endpoint set. The independent analyzer requires
three finite 7,000-event training runs, exact checkpoint iteration/config/hash contracts, all
nine rollout reports, and the frozen promotion thresholds. Seed `404` is development-only.

The walk arm is complete and shows the intended budget effect: completion rises from `3%` at 2k
to `48%` at 4k and `88%` at 8k. Dance reaches only `0%/2%/10%`; fight produced no final 8k
checkpoint or reports. The registered pooled/per-clip gate therefore has no valid final verdict,
and E35 cannot select one budget for all three clips. Its valid walk result supports 8k for the
single-clip E36-E39 studies and current reference-source development gate only. Calibration
selects a training budget; it cannot establish non-inferiority or tracking benefit.

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

### E36 - WBT latent observation screen (S1/S2/S3) - **COMPLETE, NO ARM PROMOTES; S3 DIRECTIONALLY POSITIVE**
Frozen protocol `docs/WBT_LATENT_INTEGRATION_STUDY.md`; artifacts `runs/wbt_latent_pilot/`
(driver + analyzer sha256'd; COMPLETE 2026-07-15T12:22Z). One clip (walk1_subject5), one
training seed (0), 8k PPO iters, 1024 envs, eval seed 404, 100 stratified windows, vs the E35
GMR seed-0 8k baseline (88% completion / 8.953 s / 0.2543 rad RMSE). Literature predictions
pre-registered in `docs/WBT_LATENT_LITERATURE_REVIEW.md` §6 before unblinding.

- S1 (GMR command + z_t, actor+critic): completion 78% (-10 pp, CI95 [-18,-3]), survival
  -6.7%, joint RMSE +3.7% (CI [+1.3%,+6.2%] rel), undesired contacts +72%. HARMFUL.
- S2 (+ latent tangent preview, actor+critic): completion 80% (-8 pp, CI [-17,+1]), RMSE +2.8%,
  root-position error -11.4% (CI [-22%,-1%]), contacts +51%. Harmful on primaries.
- S3 (preview to critic only, actor = baseline): completion 91% (+3 pp, CI [-4,+11]), survival
  +2.7%, joint RMSE **-4.4%** (CI [-6.3%,-2.6%] rel — bootstrap CI excludes 0), undesired
  contacts -40% (CI excludes 0). Floor passed; misses the frozen ≥5% promotion margin (RMSE
  -4.4% vs -5% required). NOT promoted under the frozen rule.

Verdict matches the pre-registered predictions in direction for all three arms (S1 predicted
neutral-to-negative → negative; S2 predicted best-of-actor-arms → still negative on primaries;
S3 predicted ~null-to-small-positive → the only arm with any significant improvements). Reading:
adding the frozen latent to the ACTOR hurts on a single deterministic clip (redundant input,
optimization cost — consistent with UniTracker's interference result and the frozen-features
literature); giving the latent preview to the CRITIC only is cost-free at deployment and
directionally beneficial but under-powered/under-sized at one seed. Interpretation limits: one
clip, one seed; no cross-motion or cross-seed claim.

Consequences for `docs/WBT_LATENT_PLAN_v2.md`: Phase 1 runs as designed (C1 explicit-preview
control + L1 latent-only command, both unconditional). S1/S2 actor-concatenation line is closed
absent new evidence. S3 sits 0.6 pp short of the margin: treat as "near-miss, replicate-eligible
alongside Phase 1" — add S3 to the Phase-2 multi-seed replication set (seeds 1,2 + baseline
seeds 1,2) rather than discarding it on a margin that is within one-seed noise. An explicit-
preview-to-critic-only control (C1-critic variant) becomes the natural attribution partner for
S3 in Phase 2 if S3 replicates.

### E37 - WBT latent Phase 1 (C1 explicit preview, L1 latent-only command) - **L1 PROMOTES (72% >= 70% floor); C1 NULL/HARMFUL**
Frozen driver `runs/wbt_latent_phase1/protocol.sh` (arms.json sha256'd; COMPLETE
2026-07-15T17:03Z); same clip/seed/budget/eval as Phase 0; plan `docs/WBT_LATENT_PLAN_v2.md`
§5.2 with pre-registered predictions. vs E35 GMR baseline (88% / 8.953 s / 0.2543 rad).

- C1 (GMR command + explicit joint_pos preview +0.2/+0.5 s, actor+critic, 212-d actor):
  completion 86% (-2 pp, ns), survival -1.1% (ns), joint RMSE **+7.9% WORSE** (CI [+5.4%,+10.8%]
  excludes 0). NOT promoted. Prediction ("C1 >= baseline, gain >= S2's") WRONG in an informative
  direction: raw robot-space preview concatenated into this 3-layer MLP actor hurts joint
  tracking. Attribution consequence: S3's critic-only latent RMSE gain (-4.4%, E36) is NOT
  explained by generic "future information helps" — actor-side preview of either kind (latent S2
  or explicit C1) degrades RMSE at this scale, and the remaining open contrast is latent-vs-
  explicit preview *to the critic only*.
- L1 (latent-only command: [z, dz+0.2s, dz+0.5s], NO explicit joint command in the actor
  (480-d); critic privileged with explicit command + latent preview): completion **72%** >= the
  pre-registered 70% absolute floor -> PROMOTED for multi-seed replication. vs baseline:
  -16 pp completion, survival -12.6%, RMSE +6.1%, contacts +73.8% — as predicted, worse than the
  explicit-command policy, but the first demonstration that a WBT PPO policy can track a clip
  from the frozen SNMR latent alone (rewards/termination still robot-space GMR). Root-position
  error -9.6% (CI [-22%,+3%], ns) mirrors S2's root-error signal. L1 >= 85% fast-track NOT met.
- Verdict: promote_for_multiseed_replication (L1). Interpretation limits: one clip, one seed.

Reading against the lit review: consistent with UniTracker (explicit-reference concatenation
interferes; latent-only command viable) and with GMT only at the encoder level (their preview
gain used a *trained* conv encoder; frozen/raw concatenation is not the same mechanism).
Phase 2 (next): multi-seed replication set = S3 (E36 near-miss) + L1 (promoted) + baseline
seeds {1,2}; plus c3_explicit_preview_critic seed-0 screening arm (explicit preview to critic
only) as S3's attribution partner, per the plan's C1-critic variant note.

### E38 - latent contact probe (Phase-4 probe b) - **NEGATIVE: contact not deployably decodable from z**
`scripts/probe_latent_contact.py`, `runs/latent_contact_probe/probe_val7.json`. Held-out-clip
logistic probes on frozen Phase-2 z (7 G1 val clips; labels = teacher-height hysteresis oracle
0.03/0.05, the Gate-1b reference). Result: aggregate held-out F1 0.044 (z), 0.052 (z with
+-1-frame context) vs 0.127 for the existing human-source height mask; on walk1_subject5 — the
only clip with balanced label support (pos rate 0.37-0.53; all other clips <=5%, the known
teacher-mask support problem) — z-linear AUROC is 0.51-0.64 (chance-ish) vs source-mask F1
0.57-0.72. Mean AUROC 0.73 across cells is inflated by tiny-support clips where rare contact
frames are posturally extreme. Temporal context adds ~nothing (0.731 -> 0.743).

Conclusions: (i) consistent with E16 ("content-specific, not semantically organized") — the
distillation-trained latent does not expose binary contact events linearly; (ii) Gate-1b should
NOT wait on a z-derived mask; the contact-head routes (M2/M3) remain the lever; (iii) for the
paper, this is the honest counterpoint to any "latent carries physics-relevant structure" claim:
no evidence beyond decoded kinematics at linear readout. A nonlinear probe could be tried later
but is not a deployable mask candidate under Gate-1b guards. CPU-only; no GPU contention with
Phase 2.

### E36/E37 addendum - per-window and cross-seed context (pre-Phase-2 verdict)
Paired per-window analysis of the existing seed-0/eval-404 reports (same 100 stratified starts):

- S3's RMSE gain is BROAD, not outlier-driven: improved in 83/100 windows vs baseline
  (73/82 restricted to windows where both complete; mean delta -0.0086 rad there). Completion
  gain concentrates in the last phase quintile (baseline 0.90 -> S3 1.00) and S3 removes 5 of
  the baseline's 12 failure windows while adding 2 new ones.
- L1's failures (28) cluster in early/mid-clip windows and are a superset-ish of baseline's;
  RMSE is uniformly ~0.02 rad worse across phase quintiles — a global gap, not a phase-local
  artifact. Consistent with a policy that tracks from a lossier command signal.
- Baseline seed variance (training curves, mean reward at iter 7000): seed0 9.02, seed1 8.97,
  seed2 7.98 — nontrivial spread; validates Phase 2's insistence on baseline seeds before any
  claim.

Multi-clip eval mechanics confirmed for Phase 3: holosoma MotionConfig.motion_dir takes
precedence over motion_file when non-empty, so multi-clip training uses
--...motion-config.motion-dir <dir>; held-out evaluation overrides motion-dir "" +
motion-file <clip>.npz on top of the checkpoint's saved config (eval_agent tyro override path
verified in source). The wbt_metrics report records the resolved motion_file for validation.

### E39 - WBT latent Phase 2 multi-seed replication - **L1 FEASIBILITY REPLICATES; S3 BENEFIT DOES NOT**
Frozen artifacts `runs/wbt_latent_phase2/` (COMPLETE 2026-07-16T08:12Z): train seeds
{0,1,2}, eval seeds {404,405}, 100 paired windows/cell. Baseline, S3, and L1 each have six
cells; C3 is a seed-0 attribution screen. The frozen analyzer passes every report/schema check.
A post-completion source-analyzer audit aggregated eval seeds within each training seed, as the
registered plan intended; it produces the same verdict.

- **S3 critic-only latent preview does not replicate the promotion effect.** Per-train-seed
  mean RMSE effects are -4.6%, -0.1%, and -12.0%; completion effects are +2.5 pp, +0.5 pp,
  and +5.5 pp. The median training seed therefore misses every 5%/5 pp improvement threshold
  while all completion-floor checks pass. The large seed-2 gain and null seed-1 result make this
  an optimization-variance signal, not a stable latent benefit.
- **L1 latent-only command robustly clears its registered feasibility floor.** All six cells
  complete 72-77% (median 75%; none below 70%). Relative to the explicit-command baseline,
  completion remains 8-18 pp lower and survival 5-14% lower; RMSE is worse for seeds 0/1 and
  approximately tied for seed 2. This establishes that the frozen offline SNMR latent can carry
  enough information for single-clip tracking across seeds. It does not establish a tracking
  improvement.
- **C3 removes a latent-specific reading of the seed-0 critic gain.** Explicit future joint
  positions in the critic yield RMSE effects -3.7%/-5.1% on eval seeds 404/405, versus
  -4.4%/-4.9% for S3. Direct paired S3-C3 RMSE differences are -0.66% and +0.16%, with
  per-window bootstrap intervals crossing zero. Generic privileged future information is a
  sufficient explanation at this resolution.

Verdict: `replicated:l1`, interpreted strictly as a command-feasibility result on
`walk1_subject5`. S1/S2 actor concatenation remains closed; S3 is not a paper claim. Phase 3
must not launch the full arm matrix until a baseline-only multi-clip budget calibration passes.
If S3 is nevertheless retained as a secondary method question, C3 is mandatory.

### E40 - Gate-1b M1b plumbing smoke - **EMPTY MASK SUPPORT; NOT A VALID SOLVER SMOKE**
`runs/gate1b/smoke_m1b.json` used the registered clip-local decoded-ground mask on the first
192-frame `walk1_subject5` window. It produced zero contact samples and exercised only the
projection solver's `empty_contact_support` return. Direct reproduction explains the result:
the decoded per-foot full-clip minima are 5.1/6.9 cm below the first-window minima, beyond the
3 cm contact-on threshold. This is evidence that decoder vertical drift can make the simple
clip-minimum mask under-detect contact; it is not evidence of a code crash and not validation of
the optimization path. Before the full M1b run, audit support over all 42 frozen windows and use
a support-bearing window for the solver plumbing smoke. The registered full-study semantics do
not change: zero-support windows count as mask failures.

### E41 - Gate-1b M1b support audit and full projection - **FAILS BY UNDER-SUPPORT**
`runs/gate1b/m1b_support_audit.json` and
`runs/gate1b/windowed_c6_m1b_decoded_clip_height_full.json`. Clip-local decoded ground selects
only 58/16,128 samples across 8/42 windows; 34 windows have zero support. Precision is 0.259 but
recall is 0.0105 (F1 0.0201). A support-bearing smoke validates the accepted L-BFGS path and all
correction bounds, but the full unchanged projection is effectively identity on most windows:
teacher-height speed `0.479 m/s`, source-contact speed `0.339 m/s`, and MPJPE `3.66 cm`.
M1b therefore fails both coprimaries and triggers the already registered M3 arm.

### E42 - Gate-1b M3 frozen contact-head retrain - **VALID FAILURE; CLOSE MASK ITERATION**
`runs/gate1b/m3_teacher_height_head_seed0/analysis.json`, 50k G1 steps from C4 seed 0, with only
the four new `decoder.contact_head.*` tensors trainable. The artifact contract passes: all 98
inherited C4 tensors are byte-identical, the checkpoint and mask/projection hashes agree, and all
42 registered windows are present.

The learned mask reproduces the density failure instead of fixing it: prevalence `0.715` versus
oracle `0.0887`, precision `0.0963`, recall `0.6587`, F1 `0.3236`. Projection reduces raw stance
speeds but remains far above both endpoints (`0.430 m/s` teacher-height, `0.254 m/s`
source-contact), raises MPJPE `3.66 -> 6.51 cm`, exceeds the 1.2x jerk guard, and fails the
penetration-fraction guard. Verdict: `fail_close_mask_iteration`.

The long-running shell encountered a non-scientific recovery event after training: its source
driver had been committed in place while Bash was still reading it, causing a shifted command
fragment after the valid 50k manifest was written. The exact frozen audit and projection commands
were resumed from the staged protocol; `DRIVER_RECOVERY.md` records the details. This does not
change the checkpoint, registered code paths, endpoints, or verdict.

Decision: no M3 replication, no additional threshold/label/solver tuning, and no
`contact-consistent` claim. Gate 1b is closed. The oracle-mask pass supports the diagnosis that
correct constraints are sufficient; deployable mask precision/density remains unresolved.
Proceed to physics-repaired supervision only after calibrated WBT provides a competent simulator
teacher.

### E43 - Latent-flow side project registered; F0 target viability - **z_r TARGET NOT VIABLE; FALLBACK TO z_h**
Side project (SafeFlow-inspired physics-guided rectified flow in the frozen SNMR latent space)
registered in `docs/FLOW_RETARGETING_SIDE_PROJECT.md` with preregistered gates F0-F3 and an
NFE-matched z-descent null control. Code: `snmr/flow.py`, `scripts/train_latent_flow.py`,
`scripts/eval_latent_flow.py`, `tests/test_flow_retarget.py` (10 tests; full suite 194 passing).

F0 (`runs/latent_flow/f0_viability.json`, CPU, frozen `phase1_g1_large/ckpt_100k_final.pt`,
7 val clips x 6 windows of 64): decoding the frozen encoder's TEACHER-ROBOT latent z_r gives
MPJPE `50.06 cm` vs raw `3.48 cm` — far above the 3 cm viability bar. The Phase-1 G1 encoder was
never trained to make robot encodings decodable (no L_z on this checkpoint, and even Phase-2's
L_z only aligns latents; `zr_decode_prob` remains deferred). Decision per protocol: the flow
target degrades to `z1 = z_h` (identity transport + guidance mechanism).

The z-descent control at deployable soft-decoded cost (25 iters, lr 3e-3, trust 10) already
moves stance speed 0.400 -> 0.221 m/s (teacher-height mask) at +0.15 cm MPJPE with all
non-speed guards passing — a real bar for F2 to beat.

F1 training run `runs/latent_flow/f1_zh` (20k steps, z_h target) launched 2026-07-17; F1/F2
evaluation pending. This project does not alter main-line priorities (M1b/M3 closed Gate 1b;
B1 confirmatory matrix remains next).

### E44 - Latent-flow F1/F2 - **F1 PASSES; F2 FAILS; SIDE PROJECT CLOSED PER STOP RULE**
`runs/latent_flow/f1_zh` (20k steps, z_h target, CFM 0.0035) and
`runs/latent_flow/f2_screen.json` (7 val clips x 6 windows of 64, NFE 25, frozen Gate-1b guards).

F1 PASS: unguided flow samples decode to MPJPE `3.49 cm` vs raw `3.48 cm` (≤ raw + 0.5 cm) —
the flow reproduces the deterministic decode with no fidelity loss. Generative-head viability
is established, but only as an identity transport (F0 forced z1 = z_h).

F2 FAIL, and informatively so. Across the full preregistered grid (alpha_end ∈ {1,3,10} x
{soft_decoded, source_gated, teacher_height}):
- Guidance is nearly inert: best cell (a10, ORACLE teacher-height mask) moves teacher-height
  stance speed only `0.400 -> 0.339 m/s` (endpoint ≤ 0.08); deployable cells move ≤ 0.012 m/s.
  MPJPE/jerk/penetration guards all pass — the edits are tiny, not destructive.
- The NFE-matched z-descent control (Adam on z, deployable soft-decoded cost, trust region)
  reaches `0.221 m/s` at +0.15 cm MPJPE — 5x the guided-flow improvement — but still misses
  both speed endpoints by ~2.5x. No cell passes; control also fails.

Reading: (1) clamped raw cost-gradients through the frozen decoder+FK are weak in latent space;
adaptive (Adam-normalized) descent extracts more, so SafeFlow-style velocity steering does not
transfer to this 128-d regression latent at preregistered strengths. (2) Even unconstrained
latent descent under a deployable soft stance signal plateaus near 0.22 m/s — consistent with
Gate 1b: the binding constraint is the deployable contact signal, not the corrector class
(projection, DLS lock, latent descent, or guided flow all agree). No alpha/threshold tuning
beyond the registered grid; project CLOSED at F2 with F3 never armed. Main-line priorities
unchanged.

### E45 - Latent-flow guidance post-mortem - **FAILURE IS STRUCTURAL: DIRAC TRANSPORT + ILL-CONDITIONED GRADIENT, NOT TUNING**
`runs/latent_flow/guidance_diagnostics.json` (`scripts/diagnose_latent_flow_guidance.py`,
3 clips x 2 windows, frozen F1 checkpoint, descriptive). Three probes separate the E44 hypotheses:

- **H1 scale — confirmed:** at E44 settings the clamped push is median `0.15%` of the velocity
  norm (grad-to-velocity ratio 0.0015; clamp saturation 0.0 — raw gradients are simply tiny,
  median norm 0.17 against velocity norms ~100). E44's guidance was ~600x below parity.
- **H2 contraction — confirmed:** perturbations injected at u=0.2/0.5/0.8 retain only
  9%/29%/67% of their magnitude at the endpoint; endpoint diversity across 8 noise draws is
  0.4% of ||z_h||. The z_h-target flow learned a near-Dirac, contractive transport that
  actively annihilates early guidance — expected, since its training target was literally
  deterministic (z1 = cond).
- **Decisive third finding:** norm-matched guidance (push rescaled to rho*|v|, rho up to 1.0)
  does NOT rescue it — rho=1 makes skate WORSE (0.074 -> 0.275 m/s) while degrading MPJPE,
  and the guidance displacement direction is ~orthogonal to the Adam z-descent displacement
  (mean cos 0.05). The instantaneous decoder+FK cost gradient is not a productive descent
  direction in this regression latent; Adam only wins through iterative adaptive rescaling.

Verdict: the F2 failure is structural, not a hyperparameter miss. Two necessary conditions for
SafeFlow-style guidance were both absent in v1: (a) a *genuinely multimodal* conditional
distribution (ours collapsed to Dirac after the F0 z_r fallback — there is no distribution to
steer within), and (b) a *noise-regularized latent* whose decoder behaves smoothly off the
data manifold (ours is a raw regression latent; SafeFlow guides in a KL-bottlenecked VAE
latent). Any v2 must change the representation/supervision, not the sampler constants.

### E46 - Flow v2 gates V0/V1 - **BOTH FAIL; DECODER-ONLY INTERVENTIONS INSUFFICIENT; V2 CLOSES AT ENTRY**
Protocol `docs/FLOW_RETARGETING_V2_PROTOCOL.md` (registered this morning, incl. the pre-data
50/50 noise-mixture amendment). Trainer `scripts/train_v0_noise_decoder.py` (frozen encoder;
decoder+embodiment finetune, 20k steps, lr 1e-4, latent noise z + sigma*eps*std(z) on 50% of
steps; --zr_decode_prob adds the Phase-2-deferred robot-encoding decode arm).

- **V0 sigma=0.1** (`runs/latent_flow/v0_sigma0.1`): V0a fidelity guard PASS (clean 2.29 vs
  baseline 2.18 cm). **V0b descent-transfer gate FAIL** (`v0b_sigma0.1_descent.json`): z-descent
  on the noise-finetuned decoder reaches 0.224 m/s teacher-height stance speed vs the v1 frozen
  decoder's 0.221 — no transfer at all (gate required <=0.166). sigma=0.3 fails V0a outright
  (2.49 cm). The "substrate brittleness is binding" hypothesis is dead at this decoder scale:
  noise-robust decoding neither helps nor hurts latent descent.
- **V1 zr_decode_prob=0.5 + sigma=0.1** (`runs/latent_flow/v1_zr_decode`): z_r decodability
  improves 50.06 -> 19.07 cm (`v1a_zr_decodability.json`) — the decoder CAN partially learn to
  answer from robot encodings — but misses the 3 cm V1a bar by 6x, and clean-z fidelity
  regresses 2.18 -> 3.11 cm, failing the +0.3 cm V1b guard. The human/robot latent gap is too
  large for a decoder-only bridge; closing it requires touching the ENCODER (joint L_z +
  zr-decode training), i.e. a different SNMR, not a bolt-on.

Verdict: per the registered dependency order (V3 runs only after V0 or V1 passes), **flow v2
closes at its entry gates**. Combined with E44/E45, the inference-time latent-correction line
is now closed end-to-end: frozen-latent guidance (structurally inert), adaptive descent
(insufficient), noise-regularized substrate (no transfer), and decoder-side conditional
de-degeneration (too costly). The surviving levers are upstream and main-line: (a)
physics-repaired supervision of SNMR itself (Track A5; the calibrated WBT teacher is a
prerequisite), and (b) if a generative retargeter is ever revisited, it must be trained
end-to-end with a noise-regularized (VAE-style) latent and genuinely multi-solution targets —
not retrofitted onto a frozen regression model. Total v2 cost: ~35 GPU-minutes.
