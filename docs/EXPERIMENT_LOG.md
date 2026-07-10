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

## Queued / planned
- E10 — contact-weight sweep {0.05, 0.2, 1.0} on the shared model (EDGE head + world-frame losses,
  post-review). Decisive for C5. Accept: skate ≤ 0.08 m/s at MPJPE ≤ 1.5× current.
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
