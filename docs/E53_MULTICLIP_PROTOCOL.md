# E53 — Multi-clip act-through-latent (H4's real test)

**Date registered:** 2026-07-28. **Status:** REGISTERED; queued behind the E52 multi-seed
replication. **Depends on:** E52 v3 promotion (C 0.935 / D 0.955 vs 0.85 gate, single clip).

## 1. Question

H4 (RESEARCH_PROPOSAL §3): *the co-trained latent's advantage over explicit commands appears
at multi-clip scale* — single-clip walk is near-null by construction (clip phase ≈ explicit
command). E52 v3 found D − C = +2pp even on single-clip; E53 tests whether the gap grows
when one policy must cover 8 heterogeneous clips (walk1/walk3/run1/dance1/fight1/jumps1/
push1/sprint1 — the E52 walk1 clip plus 7 clips spanning 5 motion categories).

## 2. Data

`runs/wbt_latent_gmr_multi8/` (8 clips, GMR-teacher references + frozen phase-1 G1 SNMR
latents at 50 Hz, exported by `export_wbt_gmr_latent_batch.py`, manifest with schema checks;
consumed as a holosoma `motion_dir` — MultiMotionLoader + adaptive sampler; latents
concatenated per-loader by `snmr.integration.wbt_latent`).

## 3. Arms & stages

**Stage 0 — multi-clip explicit teacher** (prerequisite, ~2.5h): the E51 arm-A recipe
(bodyfix + joint reward w1.0 σ0.5) on the 8-clip motion_dir, 8k iters @1024 envs, seed 0.
This is both the DAgger teacher and the "explicit command" baseline at multi-clip scale.
Gate to proceed: completion ≥ 0.6 (multi-clip is harder; GMT/BeyondMimic-class recipes hold
up at thousands of clips, 8 should be comfortable — if < 0.6, iterate budget/16k first).

**Stage 1 — students** (2 arms × ~2h, DAgger from the Stage-0 teacher, E52 v3 recipe
unchanged): arm C (prior = proprio + explicit cmd) and arm D (prior = + SNMR z window).
Same 2000 rounds @1024 envs, prior-path collection, posterior-z action loss, β=0.1.

## 4. Readouts & gates (preregistered)

Eval = 1024 phase-stratified 10-s rollouts over the concatenated timeline (spans all clips),
deterministic z = μ_prior; plus per-clip completion breakdown (start-step → clip mapping
from the loader's motion boundaries).

- **Primary: D − C completion at multi-clip scale**, compared against the single-clip
  D − C (+2.0pp). H4 PASSES if the multi-clip delta ≥ +4pp (double the single-clip delta)
  OR D ≥ teacher − 2pp while C < teacher − 5pp.
- **Secondary:** per-clip patterns (does z help most on dynamic clips — sprint/fight/jumps —
  as E48's per-clip contact-probe pattern predicts?); student-vs-teacher RMSE parity.
- **Null reading:** D ≈ C at multi-clip too → the frozen z window adds no goal information
  beyond the explicit command at any single-embodiment scale; the additive-value claim
  moves entirely to cross-embodiment (E54), where the explicit command is dimensionally
  impossible to share. (This does NOT kill act-through-latent — C passing at multi-clip is
  itself the "latent command scales" result.)
- **Kill (for this stage only):** both C and D < teacher − 15pp → the DAgger recipe does
  not scale past single clip at this budget; diagnose (rounds? mixing schedule?) before E54.

## 5. Next after E53

E54 — cross-embodiment command (the flagship): export a second robot's references + the
SAME shared z (phase-2 all5 checkpoint), train one tracker per robot with arm-D architecture,
demonstrate one z stream commanding both; quantify with per-robot completion vs per-robot
explicit baselines. Design doc to follow E53's verdict.
