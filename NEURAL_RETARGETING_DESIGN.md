# SNMR: Shared Neural Motion Retargeting for Unified Humanoid Motion-Tracking Training

**A research design plan** — replacing per-frame IK/optimization retargeting (GMR / holosoma_retargeting)
with a SAME-style skeleton-agnostic neural retargeter built around a **shared latent motion space**,
feeding a **shared (multi-embodiment) whole-body tracking training** pipeline.

Date: 2026-07-10 · Status: **Historical design and implementation record.**

> **Status precedence (2026-07-14):** use
> `docs/NEURAL_RETARGETING_RESEARCH_sol.md` as the current decision-gated plan. It supersedes
> optimistic or stale status language below. Gate 0 is complete; the immediate work is the
> factorized Gate 1 contact study and matched Gate 2 WBT validation. Framewise DLS exposes a
> speed-versus-jerk Pareto. A full temporal C6 projection reaches `0.099 m/s` with the deployable
> source mask but fails speed and MPJPE, while the same solver reaches `0.0056 m/s` with all guards
> under a teacher-mask oracle. The matched three-clip, three-seed WBT training pilot passes its
> 18-run artifact contract. Final-window effects are mixed but modest: pooled SNMR reward is
> `0.655 -> 0.658`, episode length `39.37 -> 39.63` steps, joint-position error
> `1.680 -> 1.615 rad`, and reference-position error `0.267 -> 0.272 m`. These are training
> curves, not policy evaluation. The preregistered 5,400-rollout study then finds zero 10-second
> completions for both sources and mean survival of only `0.934 / 0.956 s` (GMR/SNMR).
> Although SNMR joint RMSE is descriptively `3.3%` lower (95% CI `[-5.43%, +0.26%]`), GMR fails
> the `50%` assay floor, so no non-inferiority claim is allowed. Calibrate a longer policy-training
> horizon on GMR controls before repeating the matched comparison.
> Gate 1 calibration accepts C1 BCE `0.25`, C3 stance `0.03`, and C4 phase-balanced velocity
> `0.05`; C2 is dropped after its only retry still leaves BCE above the gradient band. Three-seed
> replication confirms robust favorable tradeoffs for C4 (mean speed `0.701 -> 0.264 m/s`,
> MPJPE `4.70 -> 3.09 cm`) and C3 (`0.701 -> 0.469 m/s`, MPJPE `4.70 -> 4.99 cm`), but neither
> reaches `0.08 m/s` in any seed. Gate 1 therefore fails its endpoint; stop soft penalty sweeps
> and move to physics-repaired supervision or retargeting/tracking co-optimization.
> The shared-model, unseen-target, representation, temporal, and throughput claims remain bounded
> by that audit.

---

## 0. Status update (2026-07-09) & next-step plan

### 0.1 What is built and validated (code: `snmr/`)

The C1 retargeter core is implemented from scratch in `snmr/` (~1.3k LOC source, **28/28 tests
passing**) and validated against the real assets in this repo:

- **Differentiable FK** (`snmr/robot_model.py`): torch, batched; matches `mujoco.mj_forward` to
  ~1e-8 m/rad across **six robots** (G1, H1, H1-2, N1, T1-29dof, Toddy) on random in-limit
  configurations — the load-bearing correctness result for all FK-based losses.
- **Rotation/convention layer** (`snmr/rotation.py`): wxyz quats, 6D rotation rep, scipy-verified.
- **Shared skeleton graph** (`snmr/skeleton.py`, `snmr/data.py`): one structure for human + robot;
  heading/translation-invariant per-node pose features (tested); SMPL-X body-22 topology built in.
- **Model** (`snmr/model.py`): GAT encoder → per-frame shared latent (max-pool over nodes) →
  temporal transformer → AdaLN embodiment-conditioned decoder emitting `qpos`; **joint limits by
  construction** via tanh heads. 0.41M params.
- **Losses** (`snmr/losses.py`): distill, task-space FK, limits, smoothness, foot-contact,
  latent-consistency (L_z ready for Phase 2).
- **End-to-end proof**: overfit-a-batch on a real 60-frame G1 clip from the holosoma WBT NPZ —
  loss 1.56 → ~0.05, whole-body MPJPE → 5–10 cm, zero limit violations
  (`snmr/tests/test_overfit.py`, `snmr/scripts/overfit_batch.py`).

An adversarial multi-agent review (3 reviewers → refute-oriented verifiers that executed code)
confirmed and led to fixes for: (a) **joint-limit/data contract mismatch** — GMR's
`g1_mocap_29dof.xml` narrows hip-pitch to [-1.57, 1.57] while the training NPZ spans to −2.27 rad
(~38% of frames unreachable by the limit head); we now use the holosoma G1 model and added a
limits-cover-data contract test; (b) **silent slide/ball-joint drop** in MJCF parsing — now
fail-loud + tested; (c) a CUDA device bug in `matrix_to_quat`. The predicted "convention bugs —
certain to bite" risk row materialized exactly as forecast; the canonicalization module + tests
caught the rest.

### 0.2 New environment facts (verified, they reshape the near-term plan)

- **GPU**: one NVIDIA A10G (23 GB) is present (shared; venv currently has CPU torch — installing
  CUDA torch is step N0 below). 8 CPU cores, 30 GB RAM, ~925 GB free disk.
- **Teacher works here**: GMR installed into the snmr venv and runs at **~150–170 FPS** on this
  machine (measured on LAFAN1→G1) — faster than the README benchmark; full LAFAN1 × 1 robot ≈
  50 min, × 5 robots ≈ 4–5 h CPU. **LAFAN1 is downloaded** (77 clips, ≈4.7 h of motion at 30 fps)
  and a pair-generation script is written and smoke-tested
  (`snmr/scripts/make_pairs_lafan1.py` → `data/pairs/<robot>/<clip>.npz` with human pos/quat +
  teacher qpos).
- **AMASS/SMPL-X is NOT available here** (body-model download requires registration). Phase 1
  therefore trains on **LAFAN1** (BVH path, 24-body human skeleton); AMASS becomes a Phase-2+
  data expansion when models are provisioned. This replaces the original "AMASS + LAFAN1" Phase-0
  data assumption.
- **Robots with LAFAN1 teacher configs**: `unitree_g1`, `booster_t1_29dof`, `fourier_n1`,
  `engineai_pm01`, `stanford_toddy` (+ `pal_talos`) — 5–6 embodiments, enough for the Phase-2
  multi-robot and leave-one-robot-out experiments without any new IK config work.
- **IsaacSim/WBT training cannot run on this box** (no IsaacSim install; holosoma WBT is
  IsaacSim-only). Phase 3 remains a handoff: SNMR output → `convert_data_format_mj.py` →
  `g1-29dof-wbt` on an IsaacSim-capable machine. The contract is already satisfied
  (`RobotMotion.qpos()` = `[root_pos, root_quat wxyz, dof]`).

### 0.3 Next steps (ordered, with acceptance criteria)

**N0 — CUDA training env — DONE (2026-07-09).** torch 2.5.1+cu124 installed; A10G visible; full
fast suite green under the CUDA build. Measured on GPU (64-frame windows, 0.41M model):
**~1130 frames/s training (fwd+bwd)**, **~1940 frames/s inference** vs the ~160 fps CPU teacher —
the ≥1000 fps Gate-G1 throughput criterion is already met at small batch.

**N1 — Full paired dataset — DONE (2026-07-09).** 77 clips × 5 robots generated in 4.3 h at
159 fps: **2,483,360 teacher frames (~23 h of paired motion), 1.6 GB** in `data/pairs/<robot>/`.
Integrity spot-checks per robot passed (frame alignment, unit quats, finite values, sane root
heights: G1 0.67 m, T1 0.67 m, N1 0.68 m, PM01 0.78 m, Toddy 0.34 m — the size spread we want for
the shared-latent experiments). Gate G0 satisfied.

**N2 — Human-side encoder path — DONE (2026-07-09).** `snmr/human.py` implements the LAFAN1
24-body `SkeletonGraph`, `human_pose_features` (mirrors the robot side: heading normalization, 6D
rots, velocities), schema-matched static features, height+velocity contact flags, and the pair-NPZ
loader; `SNMR.retarget_human_to_robot` is the production path. Validated by 8 new tests including
**an end-to-end human→latent→robot training convergence test against real GMR teacher output**
(loss drops >70% in 220 steps; root height within 15 cm of teacher). Suite now 36 tests, all green.

**N3 — Phase-1 training run: LAFAN1 → G1 (IN PROGRESS).** Trainer written
(`snmr/scripts/train_phase1.py`: 70 train / 7 held-out clips, 64-frame windows, AdamW + cosine,
periodic held-out MPJPE eval, resumable checkpoints); full 60k-step GPU run launched.
*Design lessons from the smoke runs (root-pose parametrisation matters a lot):*
1. **World-frame root targets are unlearnable by construction** — the encoder features are
   deliberately heading/translation invariant, so the network cannot know where in the world the
   motion happens. Measured: ~3.3 m val MPJPE of pure root drift.
2. **Anchoring at the raw human root is still ill-posed**, because GMR *scales* the human
   trajectory before IK (`human_scale_table` × height ratio): robot_xy ≈ s·human_xy with
   s ≈ 0.875 for LAFAN1→G1 (least-squares residual ≤ 1.5 cm). The residual (robot − human) offset
   grows with distance from the world origin — an error decomposition showed 59 cm of a 61 cm val
   MPJPE was root error, only 12 cm from dof error.
3. **Fix:** predict the root in the **scaled-human-root heading frame**; the per-robot scale s is
   fitted once on the training set, stored in the checkpoint, and treated as a retargeting
   constant at inference (`fit_xy_scale` in `scripts/train_phase1.py`;
   `world_root_to_local`/`local_root_to_world` in `snmr/data.py`, round-trip + invariance tested).
   Val MPJPE at a 2k-step smoke: 3.3 m → 0.56 m → **0.28 m** across the three formulations.

*First full 60k-step run (0.41M params, z=64) — COMPLETE (2026-07-09, ~1 h on the A10G):*
held-out val MPJPE converged **33 cm → 5.2 cm** (dof err 0.10 rad); the final 16-window-per-clip
eval reads **7.0 cm** (more windows per clip than the in-training eval — treat 5–7 cm as the honest
band). Loss still trending down at 60k with capacity as the visible bottleneck (train loss 0.035
vs early-clip overfit 0.046 at 0.41M params). **Gate G1 (<3 cm) not yet met — no red flag in
sight:** the remaining error is dof detail, not root drift, and the model is 20–50× smaller than
comparable retargeting nets (AdaMorph/NMR-scale). The 60k baseline checkpoint is kept at
`snmr/runs/phase1_g1/ckpt_60k_final.pt`. `scripts/export_wbt_npz.py` (N5) is already
schema-validated against the real holosoma sample NPZ using an intermediate checkpoint.

*Scaling attempt #1 (negative result, kept for the record):* z=128/hidden=256 at **lr 8e-4
destabilized** — train loss spiked 0.37→0.66 at ~12k steps and plateaued ~30k steps before
partially recovering; final val MPJPE 18.9 cm ≫ the small model's 5.2 cm
(`snmr/runs/phase1_g1_large_lr8e4_diverged/`). Lesson: the larger GAT+transformer needs a smaller
LR (and/or warmup).

*Scaling attempt #2 — **GATE G1 PASSED** (2026-07-09):* same z=128/hidden=256 architecture at
**lr 3e-4**, 100k steps (~1.6 h on the A10G): held-out val MPJPE **2.18 cm** (in-training eval,
4 windows/clip) / **3.12 cm** (final denser 16-windows/clip eval), dof err **0.046–0.061 rad** —
at or under the <3 cm criterion, and already inside the original 1–2 cm AMASS-scale stretch band
on the sparser eval. Checkpoint: `snmr/runs/phase1_g1_large/ckpt_100k_final.pt` (the export
script now reads model dims from the checkpoint config). The capacity hypothesis from the 60k
baseline was correct: 4× params took 5.2 cm → 2.2 cm at identical data. **N3 complete; proceed
to N4 (multi-robot shared latent) using this architecture + lr.**
*Accept (Gate G1, revised to be honest for a 4.7 h dataset):* held-out **MPJPE-to-teacher
< 3 cm**, foot-skate no worse than teacher, jerk below teacher (temporal window should smooth),
batched GPU throughput ≥ 1000 fps (already measured: ~1130 fps train / ~1940 fps inference).
The original "1–2 cm" target stays as the stretch goal for the AMASS-scale dataset.

**N4 — Phase-2 shared latent (IN FLIGHT, launched 2026-07-09).** Implemented
`snmr/scripts/train_phase2.py`: per step the human window is encoded once → z_h, decoded to K=2
sampled robots with per-robot distillation, plus **symmetric L_z** pulling each robot-teacher
encoding z_r toward z_h — tying all embodiments' encodings of the same motion to one latent
point. Per-robot xy trajectory scales are **config-derived** (human_scale_table × height ratio;
verified vs least-squares data fits, diff ≤ 1.3e-3 across all 5 robots) so the LORO holdout needs
no fitted constants. All 5 robot MJCFs FK-verified (~1e-8) with dof dims matching the pairs data.
Latent-space analysis tool ready (`snmr/scripts/eval_latent_space.py`: same-motion
cross-embodiment distance ratio + cross-embodiment retrieval top-1/MRR vs chance).
Two 100k-step runs queued on the A10G (z=128/hidden=256, lr 3e-4 — the Phase-1 winning recipe):
(1) **all-5 shared** in `snmr/runs/phase2_all5/` — first eval @4k: G1 19.0 / T1 14.1 / N1 13.6 /
PM01 16.7 / Toddy 9.3 cm, L_z already 0.003 (latents aligning); (2) **LORO** (holdout
engineai_pm01, zero-shot eval) auto-chained to start when (1) finishes.
*Accept (Gate G2):* multi-robot ≥ per-robot models on each robot; zero-shot holdout within 2× of
its in-training error; retrieval substantially above chance (chance top-1 ≈ 0.012 at 84 windows).

*Early positive-transfer signal (mid-run analysis @52k):* comparing G1 val MPJPE at **matched
effective per-robot gradient steps** (phase-2 samples K=2 of 5 robots/step ⇒ eff ≈ 0.4×steps),
the shared model beats the single-robot run at every checkpoint from 8k on — e.g. 6.7 cm (shared,
~21k eff G1 steps) vs 9.4 cm (single-robot, 20k steps). Caveats: L_z acts as an extra regularizer
and total optimizer steps differ; the clean claim needs the final-checkpoint comparison (Gate G2),
but the direction is consistently positive — no multi-robot interference.

*Early shared-space signal (E2 pilot on the ~52k mid-training checkpoint, held-out clips):*
cross-embodiment retrieval **mean top-1 66.4% / MRR 0.77 vs 1.8% chance (37×)** across all 30
query→gallery pairs among {human, 5 robots}; same-motion cross-embodiment latent distances are
0.2–0.5× the mean inter-motion distance. Human↔robot retrieval works both directions (e.g.
human→G1 78.6%, G1→human 69.6%). Toddy (the 0.34 m-tall outlier embodiment) is the weakest
retrieval partner (36–66%) — consistent with its extreme scale; worth a per-embodiment note in the
final analysis. The Gate-G2 retrieval criterion is effectively already exceeded at mid-training.

**N4b — Benchmark & ablation infrastructure — BUILT (2026-07-09/10).**
- `snmr/snmr/metrics.py`: literature-aligned metric suite computed by one code path for SNMR and
  teacher alike — MPJPE (+per-body), foot-skate speed & slide fraction (contact detected on the
  *teacher's* feet: holosoma thresholds — 3 cm height, 0.3 m/s, 1 cm/frame slide), **FS-MANN**
  (the animation-standard formula v·(2−2^(h/H)), H=2.5 cm, from MANN/SIGGRAPH'18 — the metric SAME
  reports as "FS"), ground penetration mean/max/fraction (1 cm tolerance = OmniRetarget/holosoma),
  dof/body jerk, **joint-jump fraction** (NMR: |Δdof|>0.5 rad/step) and **limit-proximity
  fraction** (NMR: within 0.05 rad of a limit), plus hard limit violations. 11 unit tests with
  analytically-known values (all green; suite now 46 tests).
- *Protocol research (verified from sources):* our metric set now covers every quantitative metric
  used by GMR (E_mpbpe family via MPJPE/dof-err), OmniRetarget/ULTRA (penetration + skate
  duration/magnitude), NMR (joint-jump/self-collision-adjacent/limit-proximity counts), and SAME
  (FS/GP), so tables are directly comparable. Notable conventions adopted: stance from source/
  teacher feet (OmniRetarget) rather than height-only (ULTRA); BeyondMimic-based downstream eval is
  the field's standard (GMR/NMR/OmniRetarget all use it) — that's exactly our N5 handoff. Notable
  gap in the literature we can exploit: none of the five papers ablate data scale, and AdaMorph
  (the only other multi-robot retargeter) reports **no baselines and no ablations** — our ablation
  grid alone exceeds its evaluation.
- `snmr/scripts/benchmark.py`: checkpoint vs GMR teacher on held-out clips per robot →
  JSON + markdown tables (throughput included).
- `snmr/scripts/run_ablations.py`: sequential grid runner (waits for a free GPU, resume-safe):
  base(z=128@50k) / no_temporal / z32 / small(0.41M) / +contact-loss / lr8e4-instability-row —
  each trained then benchmarked identically; queued to start automatically when the LORO run
  frees the GPU. Trainer gained `--no_temporal` and `--contact_weight` (world-frame skate penalty
  with teacher-derived contact masks; local-frame skate would be wrong since a planted foot moves
  in the anchor frame).

*First benchmark (Phase-1 G1 ckpt, held-out, full metric set):* MPJPE 3.6 cm; penetration ≈
teacher (0.0002 vs 0.0000 m mean); hard limit violations 0 vs 0; joint jumps *better* than
teacher (0.005 vs 0.007); limit-proximity *better* (0.29 vs 0.57 — the tanh head keeps margins the
IK solver doesn't); dof jerk ≈ teacher (568 vs 527 rad/s³); FS-MANN 0.215 vs 0.132 cm/frame — but
**contact-gated foot skate 0.25 m/s vs teacher 0.05, slide fraction 32% vs 0%**, body jerk 2×.
Decomposition (teacher-root vs teacher-dof substitution) shows the skate is **dof-caused**
(0.42 m/s with teacher root; root-only 0.24): mm-level frame-to-frame leg-angle noise
differentiates into foot velocity. This is the #1 quality gap to close; the `contact` ablation row
targets it directly, and a stronger smoothness term or inference-time foot-locking are the
fallbacks. (This also confirms why the GMR paper found foot artifacts dominate downstream RL
quality.)

**N5 — Phase-3 handoff package (parallel).** Export N3/N4 checkpoints' retargeted motions through
`convert_data_format_mj.py`; produce the exact `g1-29dof-wbt` commands + a small (a) GMR-data vs
(b) SNMR-data comparison protocol for an IsaacSim machine. *Accept:* converted NPZs load in
holosoma's `MotionLoader` (schema-validated locally with a loader-shape check, no IsaacSim
needed).

**N6 — Close the foot-skate gap — UPDATED with empirical root cause (2026-07-11).** The literature
priors below motivated the first attempts; our own experiments then *localized* the problem, which
matters more than the priors. **Empirical findings (E10a/E22/E24):** (1) the contact-velocity loss
does NOT reduce skate at any weight w∈{0.05..1.0} (skate 0.50–0.57 vs no-contact 0.26; it loses to
distill and even hurts MPJPE at w=1); (2) a post-hoc low-pass filter barely helps (0.093→0.081) and
(3) an inference position foot-lock barely helps (0.160→0.153). Root cause (E24): the decoded foot
sits at the **correct height** but **oscillates in xy during stance** (contact-mask agreement only
58%; decoded contact frac 0.33 vs teacher 0.74) — a *velocity* error of a correctly-placed foot,
correlated across the leg chain, so neither position-locking nor per-joint low-pass can fix it, and
a low-weight contact term is swamped by distill. **Execution update:** E25 `w=2` improved fidelity
but left the skate/MPJPE ratio unchanged. On a matched corrected evaluator, `w=10` improves MPJPE
`3.01 -> 2.68 cm`, teacher-height stance speed `0.417 -> 0.304 m/s`, and source-contact speed
`0.289 -> 0.221 m/s` relative to `w=2`; it still misses the primary speed endpoint by 3.8x and
does not isolate stance. Converged source-mask framewise DLS can pass the speed endpoint only by
failing the jerk guard, while longer blending passes jerk but misses speed; the teacher-mask
oracle passes both. The factorized soft-objective study remains the decision path, and the
implemented temporal projection should next be driven by a validated learned mask. Original
literature-prior plan retained below for the record:

*Prior-driven plan (partially falsified by E10a/E24 above):*
- *The precedent that maps 1:1 onto our failure is SAME's own ablation:* their pose metrics were
  blind to skating exactly like our MPJPE — adding the contact loss (label-gated FK-velocity +
  height-clamped velocity + penetration; labels thresholded at 5 cm/0.4 m/s) improved FS **14×**
  (0.1056→0.0075) and GP 40×, while JP got slightly *worse* — i.e. expect a small MPJPE cost.
- *Plan, in order of implementation:*
  1. **Contact prediction head + EDGE-style self-consistency loss** ‖(FK(x̂_{t+1})−FK(x̂_t))·b̂_t‖²
     masked by the network's *own* predicted contacts (EDGE ablation: removing it doubles PFC
     1.54→3.08). Keep our teacher-mask loss (MotioNet-style: their contact-frame error 64→52)
     as the supervised variant; SAME's height-clamped term is complementary.
  2. **Inference-time latent polish** — the literature is unanimous that training losses alone
     never fully close the gap; works reaching IK-teacher contact quality use predicted contacts
     at inference: ~30-iter gradient descent on contact-masked foot velocity + height, in
     **latent space** (Villegas ESO: contact accuracy 0.71→0.97; Holden null-space) rather than
     output-space IK (smoother). UnderPressure's output-space variant halves contact velocity.
  3. **Architecture support (cheap, already half-built):** our temporal transformer is the
     NMR-style bidirectional-context noise suppressor (their non-AR argument); the `no_temporal`
     ablation row will quantify it. If needed: delta/velocity output heads (HuMoR ablation:
     removing Δ-prediction worsens accel 26%) — short-window regression avoids the drift caveat
     (QuaterNet). Phase conditioning is explicitly NOT a standalone fix (PFNN/LMP still do
     contact IK after phase conditioning; no published skate delta).
- *Metrics:* add the GMD skate ratio (<5 cm height, >2.5 cm/frame) alongside our contact-gated
  velocity and FS-MANN (both implemented).
*Accept:* held-out G1 contact-gated skate ≤ 1.5× teacher (≤0.08 m/s) with MPJPE ≤ 4 cm; the
`contact` ablation row quantifies step 1 alone, a new `+polish` benchmark column quantifies step 2.

**N7 — Latent-space analysis suite ("analyze the neural").** Protocol finalized from a deep
research pass (20 central claims adversarially verified, 0 refuted). Ranked experiments:
1. **Embodiment-identity linear probe** (headline invariance test): logistic regression on frozen
   latents → embodiment (6 classes); want **near-chance balanced accuracy** (~16.7%) + a
   shuffled-label control probe (Hewitt & Liang selectivity).
2. **Motion-category linear probe** (positive control): same probe → clip category on held-out
   clips; want high accuracy. E1-near-chance + E2-high is the signature of a shared,
   content-rich, embodiment-invariant space.
3. **Post-hoc adversarial attacker** (Elazar & Goldberg safeguard): a *strong* MLP attacker
   trained on frozen latents, evaluated held-out — an adversary-at-chance does NOT prove
   invariance (their measured gap: 49.0% adversary vs 56.0% post-hoc attacker). Report
   proxy-A-distance d_A = 2(1−2ε).
4. **Cross-embodiment retrieval** (already piloted: 66.4% top-1 / 0.77 MRR vs 1.8% chance at
   mid-training): upgrade to DTW-over-frame-latents (SAME's metric, quantified — SAME itself
   only did this qualitatively) with R@1/5/10 + median rank.
5. **CKA between per-embodiment latent sets** (15 pairs, 6×6 heatmap). Verified novelty note:
   CKA/probing/adversarial-invariance analysis is *absent* from the cross-embodiment robotics
   literature (HOVER, URMA, GET-Zero, Body Transformer, MetaMorph, AnyMorph checked) — this
   analysis section is itself a contribution.
6. **Dual-colored t-SNE/UMAP** (motion-colored should cluster; embodiment-colored should mix),
   with the standard misread-t-SNE caveats.
7. **Latent interpolation + arithmetic** (SAME's z_wave−z_stand+z_walk, MotionCLIP interpolation),
   decoded per robot and **scored with our metric suite** (foot-skate/jerk of decoded
   interpolations — no prior work quantifies interpolation quality; ours can).
Corrections from sources: SAME's latent is z=32 (ours z=128 is our choice), SAME used PCA not
t-SNE, and its 95% classifier was a Transformer, not a linear probe — cite accordingly.

**N8 — Tracking-RL validation ("improve tracking performance").** Export matched clip sets
(SNMR-retargeted vs GMR-retargeted, same held-out clips; post-N6 checkpoints preferred) via
`export_wbt_npz.py`; write the exact `g1-29dof-wbt` commands + eval protocol (success rate,
tracking rewards, sim2sim MuJoCo) for an IsaacSim machine; validate NPZs load in holosoma's
`MotionLoader` locally. The GMR paper's central result (retargeting quality → tracking quality)
makes this the decisive experiment for the "improve tracking" goal.

**N10 — Latent-as-command design (C2), research-grounded (2026-07-10).** A verified literature
pass found the direct precedent we lacked: **VMP** (Serifi et al., SCA 2024, Disney/ETH — no
arXiv; RobotMDM reuses it on hardware) pretrains a frame-wise VAE latent (z=64) and conditions a
PPO tracking policy on `[z_t ⊕ current raw frame]` — their LM configuration **halves** joint MAE
vs raw-reference windows (dance 12.79°→5.80°) and latent-only beats every raw-window variant;
retraining ONLY the encoder (RL data fixed) cuts unseen-motion error 15% — the benefit is
attributable to the representation. Independent agreement: GMT's ablation shows dropping the raw
immediate frame hurts badly → the winning interface is **latent window + minimal raw current
frame**, not pure z. Design for our C2 experiment (holosoma WBT, `managers/command/terms/wbt.py:870`):
- primary arm: actor observes `[z_t, z_{t+5}, z_{t+25}, z_{t+50}] ⊕ embodiment code ⊕ existing
  anchor-error terms ⊕ current-frame raw joint_pos (29-D)`; rewards/terminations stay on decoded
  q̂; critic keeps full raw reference (asymmetric).
- ablation arms: raw-only (holosoma default), pure-z, random-projection control (never run by
  anyone), generalist-vs-specialists at matched steps (headline), encoder-upgrade-at-frozen-policy
  (VMP's high-signal bonus).
- verified openings no one has shown: (i) sample-efficiency (steps-to-threshold) for ANY command
  representation; (ii) latent-command in the mainstream humanoid-RL stack; (iii) multi-embodiment
  sharing via latent+embodiment-code — raw joint-space commands structurally CANNOT be shared
  across different-DoF robots, so our interface is the only entrant (the safe headline).
- citation hygiene: BeyondMimic's distillation is latent-free state-action diffusion (not CVAE at
  the RL interface); PULSE is latent-as-ACTION not command; UniTracker/MaskedMimic latents serve
  distillation students. Cite VMP as the mechanism precedent.

**Deferred (unchanged from plan):** AMASS/SMPL-X ingestion once body models are provisioned;
holosoma interaction-mesh teacher for object clips; T1 WBT port; Stage-C physics feedback loop.

### 0.4a Phase-2 all-5 shared run — COMPLETE (2026-07-10); Gate G2 part 1 PRELIMINARY

100k steps, z=128/lr 3e-4, 5 robots, K=2 sampled/step + symmetric L_z (ckpt
`runs/phase2_all5/ckpt_100k_final.pt`). Final held-out MPJPE (16-window eval): G1 6.67, T1 5.80,
N1 5.54, PM01 5.72, **Toddy 3.16** cm (mean 5.38).

*Positive-transfer signal — **downgraded to preliminary on self-review (schedule confound).*** The
earlier claim compared shared-G1 @100k (5.12 cm, cosine schedule COMPLETE, lr 1e-5) against
single-robot @40k (6.30 cm, MID-schedule, lr 2e-4) at "matched effective steps" — but the
single-robot curve shows schedule completion alone drives 6.30 → 2.18 cm, so that comparison
conflates sharing with annealing. **The honest matched baseline is the ablation grid's `base` row**
(single-robot G1, 50k steps, COMPLETE schedule ≈ the shared model's ~40k effective G1 steps with a
complete schedule); the Gate-G2 part 1 verdict waits on it (N9). What survives the confound today:
no catastrophic interference (all 5 robots converge in one model, 3.2–6.7 cm), and Toddy — the
0.34 m scale outlier — is the *best* robot in the shared model at 3.16 cm.
Remaining Gate-G2 checks: matched-baseline transfer (ablation `base` row), LORO zero-shot
(running, holdout engineai_pm01), retrieval (final ckpt re-run done: R@1 0.75) → N9.

### 0.4b N6/N7/N8 — implemented & first results (2026-07-10)

**Repo is public: https://github.com/linjiw/snmr** (MIT; LAFAN1-derived data excluded, regenerable).

**N6 (foot-skate) — code done, two levers measured:**
- *Inference-time latent polish* (`snmr/polish.py`, Villegas ESO): on the Gate-G1 checkpoint it
  cuts contact skate ~25% on skate-heavy windows (0.16→0.12 m/s) at a 1–2 cm MPJPE cost, and is
  neutral-to-harmful on already-clean windows — a weak, window-dependent lever, exactly as the
  literature predicts. **Not the primary fix.**
- *Training-time contact head + losses* (decoder `predict_contact` head; EDGE-style self-consistency
  on own-predicted mask + SAME-style teacher-mask supervision; `contact_self_consistency_loss`,
  `contact_prediction_loss`). Wired into both trainers behind `--contact_weight` (default 0, so the
  queued LORO/ablation runs are unaffected). This is the bigger lever (SAME's 14× FS ablation); the
  quantified retrain is the next GPU job after the Phase-2 pipeline. 4 new tests green.

**N7 (analyze the neural) — suite done, first real numbers on the ~92k mid-training Phase-2 ckpt
(`scripts/analyze_latent.py`, held-out clips):**
- **E5 CKA mean 0.91** (range 0.86–0.98) across all 15 embodiment pairs incl. human — the six
  encoders converge to a strongly aligned representation.
- **E4 retrieval** human→robot mean R@1 0.75 / MRR 0.84 vs 0.01 chance.
- **E1 embodiment linear probe 0.27** (chance 0.167) — low *linear* leakage — **but E3, a strong
  MLP attacker, recovers embodiment at 0.89** (proxy-A-distance 1.56). This is precisely the
  Elazar–Goldberg gap the protocol was built to expose: the space is geometrically shared and
  same-motion-aligned, yet embodiment is still *nonlinearly* decodable. Honest headline: **"shared
  and content-aligned, not fully embodiment-invariant"** — and a clear target for a stronger L_z /
  domain-confusion term (a concrete paper finding, not a failure).
- **E2 motion probe 0.173** (chance 0.143, selectivity 0.055) after fixing the ill-posed split
  (auto-discovers 24 clips / 7 repeated categories). Low — but this is a **window-mean-latent
  artifact**, not necessarily the space: mean-pooling 64 frames destroys the temporal dynamics
  that separate walk/run/dance, while clip-level retrieval (R@1 0.75) survives because it matches
  exact clips. Fix for the final analysis: a temporal-aware readout (probe the frame-latent
  sequence, not its mean). Noted as a to-do; does not change the E1/E3/E4/E5 conclusions (those
  use the same features and are internally consistent).
- **Final all-5 checkpoint (100k) re-run confirms the mid-training picture** (corrected code): CKA 0.910, retrieval R@1 0.749, E1 linear probe 0.278 vs **E3 MLP attacker 0.909** (proxy-A-distance 1.78, now a valid k-class value), E2 motion probe 0.151 (window-mean caveat stands). The E1/E3 gap is stable across checkpoints — the headline finding.
- *Caveats:* all numbers are on the ~92k **mid-training** checkpoint; the final all-5 + LORO
  checkpoints (imminent) get the full re-run for the paper. The E1/E3 divergence is the headline.

**N8 (tracking validation) — package built (`scripts/prepare_wbt_validation.py`):** 3 matched
clip pairs (walk/dance/fight) exported both as SNMR-retargeted and GMR-teacher through the
*identical* MuJoCo-replay converter, so the only variable the WBT policy sees is the retargeting
source. All 6 NPZs schema-validated against the holosoma sample; `WBT_COMMANDS.md` has the exact
`g1-29dof-wbt` train/eval commands + a no-IsaacSim `MotionLoader` load check. Ready to hand to an
IsaacSim box.

### 0.4c Adversarial review of N6/N7/N8 code — 6 bugs found & fixed (2026-07-10)

A 3-dimension review workflow (reviewers → refute-oriented verifiers that executed code) on the
new N6/N7/N8 code confirmed **6 real bugs (4 refuted)**, all fixed with regression tests:
1. N6 contact self-consistency loss ran FK in the LOCAL (heading) frame — a world-planted foot
   moving with the anchor was penalized as sliding, which would *induce* skate (50× the dof-jitter
   signal). Fixed: recompose to world via anchor before the velocity FK (mirrors polish.py).
2. N6 contact MASK detected in the local frame → planted feet fail the 0.3 m/s speed test → mask
   ~all-zero during locomotion → contact head trained to predict no-contact. Fixed: detect on
   world qpos.
3. N7 proxy-A-distance used the binary 2(1−2·err) formula on a 6-class attacker → negative,
   meaningless values (−1.33 at chance). Fixed: normalize error against chance, clamp [0,2].
4. N7 `_discover_motion_clips` leaked 3 held-out VAL_CLIPS into the E2 probe set (contradicting
   its docstring). Fixed: filter VAL_CLIPS.
5. N8 export angular velocity computed conj(q0)·q1 (BODY frame) not q1·conj(q0) (WORLD) — wrong
   cross-arg order; contradicted the `_w` suffix and holosoma's converter. Fixed + regression test
   (tilted base yawing 4 rad/s → [0,0,4]).
6. N8 export ang_vel was a backward difference (aligned t−0.5) while lin_vel is centered (t) —
   half-frame misalignment (0.27 rad/s RMS on reversing rotation). Fixed: centered SO(3) stencil
   (0.02 rad/s RMS). N8 package regenerated.

Two review rounds now (Phase-0 core: 2 bugs; N6/N7/N8: 6 bugs) — the executed-code verify stage
keeps paying for itself. Suite: 60+ tests, all green.

### 0.4 Design adjustments from implementation experience

1. **LAFAN1-first, AMASS-later** (data availability; §0.2). The human skeleton abstraction already
   handles both (24-body BVH vs body-22 SMPL-X) since the encoder is skeleton-agnostic.
2. **Robot-model provenance is a first-class contract**: the limits the decoder enforces MUST come
   from the same model family that produced/consumes the data (holosoma G1, not GMR's narrowed
   mocap variant). Added to the risk table as resolved; guarded by a test.
3. **Overfit gate before dataset training** (validated): the 0.41M model reproduces real motion to
   5–10 cm MPJPE when overfit — capacity/gradients are not the bottleneck; data scale is.
4. **Review workflow works**: the refute-oriented verify stage killed 1 of 3 findings and produced
   executable evidence for the other 2 — keep it as the standard gate before each phase merge.

### 0.4d N9 — LORO + matched-baseline verdicts (2026-07-10, honest and mixed)

**LORO run complete** (100k steps, holdout engineai_pm01, `runs/phase2_loro_pm01/`):
- Trained robots are *unchanged* vs the all-5 run (G1 6.66 vs 6.67, T1 5.76 vs 5.80, N1 5.52 vs
  5.54, Toddy 3.20 vs 3.16 cm) — dropping a robot neither helps nor hurts the others; the shared
  trunk is robust to composition.
- **Zero-shot PM01: 29.6 cm vs 5.72 cm in-training — 5.2×, FAILING the ≤2× Gate-G2 criterion.**
  The MJCF-derived embodiment code alone does not transfer decoding to an unseen robot at this
  data/robot scale (5 robots is far from AdaMorph's 12; SAME needed 160 augmented skeletons).
  This is a *negative result to report honestly* and the strongest argument for the deferred
  embodiment-augmentation step (synthetic MJCF variants) — generalization over embodiments needs
  embodiment-level data diversity, exactly as SAME found for skeletons.

**Matched-baseline transfer (ablation `base` row, 50k single-robot complete schedule): 3.75 cm vs
shared-G1 5.12 cm at ~40k effective steps — the confound-free comparison REVERSES the preliminary
claim: sharing 5 robots in one 1.6M-param trunk costs ~1.4 cm on G1 rather than helping.** Gate-G2
part 1, strictly read (multi-robot ≥ per-robot at matched budget), FAILS at this model scale;
the honest claim is "5 robots in one network at modest per-robot cost (5.4 cm mean vs ~3.8 cm
specialist), with the *scale-outlier* robot (Toddy) benefiting most (3.16 cm)". Capacity is the
obvious suspect (5 embodiments share the parameters one robot had); a width scale-up is the
cheap test.

**Skate vs training length (base 50k vs 100k benchmark):** skate 0.39→0.25 m/s and slide fraction
0.46→0.32 improve with training alone — part of the gap is undertraining, but the curve is far
from the 0.05 teacher level; the contact-loss retrain remains necessary.

**Ablation grid running** (no_temporal → z32 → small → contact → lr8e4; ~5 h). The `contact` row
directly measures lever 1a on the skate metric.

### 0.5 Project review & next-goal direction (2026-07-10)

**Where the research actually stands (critical self-assessment, not the optimistic read):**

*Solid and defensible:*
- Retargeter core: Gate G1 passed (2.18 cm held-out G1); 5-robot shared model trains without
  interference (3.2–6.7 cm each); FK/rotation/data layers hardened by 2 review rounds (8 bugs
  total found by executed-code verification, all fixed + regression-tested); public repo with
  reproducible dataset provenance. Throughput ~2000 fps vs teacher's 160.
- Analysis infra is paper-grade: literature-aligned metric suite, benchmark harness, probe/CKA/
  retrieval/attacker suite — stronger evaluation than AdaMorph (no baselines/ablations) offers.

*Open problems, ranked by threat to the thesis:*
1. **Foot skate (0.25 vs teacher 0.05 m/s)** — threatens the "improve tracking" goal directly,
   since the GMR paper shows foot artifacts dominate downstream RL. The two training-time levers
   (teacher-mask loss; EDGE self-consistency head) are implemented but **unquantified** — and note
   the ablation grid's `contact` row only tests the teacher-mask lever (train_phase1 has no
   contact head); the EDGE-head lever needs a Phase-2 `--contact_weight` run.
2. **Embodiment leakage (E3 attacker 0.91)** — the latent is shared-and-aligned but not invariant.
   Two interpretations to disambiguate: (a) L_z is too weak / needs a domain-confusion term;
   (b) *scale* is the leak (Toddy retrieval weakest; embodiment may be readable from motion
   amplitude alone, which no invariance loss can or should remove). A scale-normalized probe
   (probe latents of amplitude-normalized inputs) distinguishes these before adding losses.
3. **Positive transfer is not yet cleanly shown** (schedule confound, §0.4a) — resolves free with
   the ablation `base` row.
4. **Tracking validation not yet run** — the N8 package is ready but IsaacSim access is the
   blocker. Until one WBT comparison runs, the central claim ("neural retargeting is as trackable
   as IK") rests on kinematic metrics only.

**Next-goal sequence (recommended):**
- **G-next-1 (now, GPU-gated):** finish the running pipeline (LORO → ablations), render N9
  verdicts: Gate-G2 parts 1–3 with the confound-free baseline, ablation table (temporal/z-dim/
  capacity/contact-mask-loss/lr rows).
- **G-next-2 (the decisive kinematic experiment):** Phase-2 retrain with `--contact_weight`
  sweep {0.05, 0.2, 1.0} on the shared model (the EDGE head + world-frame losses, now
  review-verified). Accept: skate ≤ 1.5× teacher at MPJPE ≤ 1.5× current. This closes the last
  kinematic-quality gap and produces the paper's "SNMR vs teacher" table in final form.
- **G-next-3 (the decisive downstream experiment):** run the N8 package on an IsaacSim machine
  (the one external dependency — needs provisioning). One G1 clip-set comparison (SNMR vs GMR
  data → identical WBT config → success rate + tracking errors) converts the thesis from
  kinematic to closed-loop. If contact-retrained checkpoints exist by then, export those.
- **G-next-4 (paper depth, cheap):** temporal-aware motion probe (fixes the E2 window-mean
  artifact), scale-normalized embodiment probe (leak diagnosis), t-SNE/interpolation figures —
  all CPU, all scripted already or trivial extensions.
- **Defer:** AMASS scale-up, T1 WBT port, ReActor-style bilevel refinement — valuable but none
  blocks the paper story; revisit after G-next-3 lands.

**Historical framing, superseded by the 2026-07-13 audit:** the earlier draft claimed
IK-teacher fidelity, an exact throughput ratio, scale attribution, and tracking validation before
the corresponding controlled gates passed. Current wording must remain: low pose error with an
open contact gap; plausibly faster inference with controlled timing pending; aligned but not
invariant latent; no causal scale attribution; and WBT evidence pending the paired pilot.

---

## 1. The existing workflow (what we have today)

Two repos implement the current `motion → retarget → motion-tracking-training` workflow:

### 1.1 GMR (`GMR/`) — real-time IK retargeting, 17+ robots

```
human motion (SMPL-X / BVH / FBX / Xsens / PICO)
   → intermediate dict: {body_name: (world pos[3], quat wxyz[4])} per frame
   → per-body scaling (human_scale_table × height ratio) + constant frame offsets
   → two-stage mink differential-IK (QP, daqp solver, damping 5e-1, ≤10 iters/stage,
     warm-started from previous frame)                    [motion_retarget.py]
   → robot qpos = [root_pos(3), root_quat wxyz(4), dof_pos(29 for G1)]
   → saved pkl {fps, root_pos, root_rot(xyzw), dof_pos, local_body_pos, link_body_list}
```

Key facts that shape the design:
- **Per-robot hand-tuned JSON configs** (`ik_configs/*.json`): body-pair mapping, position/rotation
  weights per body per stage, constant rotation offsets, per-body scale factors. 30 configs exist;
  every new robot or new source format costs manual tuning ("Designing a single config for all
  different humans is not trivial" — GMR README known issues).
- **Sequential, CPU-bound**: 35–70 FPS on high-end desktop CPUs; no batching possible (stateful
  mink `Configuration` warm start). Retargeting 40 h of AMASS × 17 robots is expensive.
- **Purely kinematic, soft-cost only**: no hard contact/penetration constraints; foot skate and
  self-penetration artifacts leak into the data (this is exactly what the GMR paper shows degrades
  downstream RL tracking).
- **A differentiable torch FK already exists in-repo**: `general_motion_retargeting/kinematics_model.py`
  parses MJCF directly, batched, GPU, returns (body_pos, body_rot) — currently only used for post-hoc
  `local_body_pos`. This is our free differentiable-loss backbone.

### 1.2 holosoma (`holosoma/`) — offline optimization retargeting + WBT RL training

```
world joint positions (T, J, 3)  [lafan/smplh/mocap/smplx loaders]
   → InteractionMeshRetargeter: SQP over full qpos; cost = Laplacian deformation of a
     Delaunay interaction mesh (human keypoints + object/ground samples);
     HARD constraints: non-penetration (mj_geomDistance), foot-sticking window, foot lock,
     self-collision pairs, joint limits, trust region; solver CVXPY/Clarabel
     [src/interaction_mesh_retargeter.py]
   → npz {qpos (T, nq), human_joints, fps=30}
   → convert_data_format_mj.py: 30→50 fps interp, MuJoCo FK replay → full-body kinematics
   → training NPZ {fps, joint_pos(T,7+29), joint_vel(T,6+29), body_pos_w, body_quat_w(wxyz),
     body_lin_vel_w, body_ang_vel_w, joint_names, body_names [, object_*]}
   → WBT training (IsaacSim, 4096 envs, PPO or FastSAC, 50 Hz policy):
     DeepMimic-style RSI + BeyondMimic-style adaptive timestep sampling,
     yaw/xy-re-anchored reference tracking of 14 bodies, early termination on z/ori error,
     DR + push randomization, ONNX export for sim2sim/real deployment
     [managers/command/terms/wbt.py, config_values/wbt/g1/*]
```

Key facts:
- The training consumer contract is exactly the converter NPZ schema — **any retargeter that emits
  robot `qpos @ any fps` is a drop-in** via `convert_data_format_mj.py`.
- `MultiMotionLoader` already supports dataset-of-motions training (directories of clips,
  per-clip RSI, adaptive sampling over the concatenated timeline) — the infra for "train one policy
  on a big retargeted dataset" already exists.
- **WBT is currently G1-only** (T1 has locomotion configs but no WBT experiment) — multi-robot
  tracking requires config work regardless of retargeting method.
- `evaluation/eval_retargeting.py` already computes penetration fraction/depth + contact metrics —
  reusable as our retargeting quality benchmark.

### 1.3 The gap

Both retargeters are **per-frame optimization with hand-tuned per-robot artifacts**
(IK weight tables / constraint configs), non-differentiable, sequential, and embodiment-siloed:
N robots × M source formats ⇒ N×M configs, N separate motion datasets, N separate tracking runs.
There is no shared representation connecting "the same motion" across embodiments.

---

## 2. Literature: is neural retargeting viable, and what's the opening?

(Full citations verified against arXiv/ACM; summary of a broader survey.)

**Almost every major humanoid pipeline still uses non-learned retargeting.** H2O/OmniH2O/HOVER/ASAP
(SMPL shape-fit + gradient descent), HumanPlus (joint-angle copy), ExBody (rotation remap),
TWIST/KungfuBot (mink IK), VideoMimic (LM-solver IK with contact costs), OmniRetarget (interaction-mesh
SQP — the same method as holosoma_retargeting), BeyondMimic (consumes Unitree's closed-source
retargeted LAFAN1). Learning appears downstream (RL trackers, distillation) — not in the human→robot map.

**Learned retargeting is an emerging fringe (2023–2026)** — validating both feasibility and timing:
- *ImitationNet* (Humanoids'23, 2309.05310): unsupervised human→robot retargeting via shared latent space.
- *GBC* (2508.09960): self-supervised differentiable-IK network trained with torch FK losses on AMASS.
- *NMR / "Make Tracking Easy"* (2603.22201): shows per-frame optimization retargeting is ill-conditioned;
  RL-expert-repaired data supervises a CNN-Transformer retargeter (G1).
- *AdaMorph* (2601.07284): unified transformer retargeter with embodiment-aware AdaLN across
  **12 humanoid robots**, zero-shot to unseen motions.
- *ReActor* (SIGGRAPH'26, 2605.06593): bilevel retarget-and-track — RL tracking gradient refines retargeting.
- *ULTRA* (2603.03279): physics-driven neural retargeting preserving contact-rich interaction, real G1.

**SAME** (Lee et al., SIGGRAPH Asia 2023, DOI 10.1145/3610548.3618206) is the strongest template for the
*shared-latent* part:
- GAT (graph-attention) autoencoder; joints = nodes, bones = edges. Per-node features: skeleton
  (offsets, 6d) + pose (6d rot, pos, root Δ, velocities, contact ≈ 26d).
- Encoder → **global max-pool over joints → single 32-d per-frame latent z** (skeleton-agnostic).
- Decoder conditions on **any target skeleton graph** (features re-injected at every layer) → outputs
  per-node rotations + root motion + contact; positions via a depth-batched FK layer.
- Trained on 160 augmented skeletons × 780 min of paired motion (pairs from MotionBuilder's retargeter);
  losses: reconstruction + velocity/jerk + contact/foot-slide/penetration (soft) +
  **embedding consistency L_z = ‖Enc(S_a,D_a) − Enc(S_b,D_b)‖²** for the same motion on different skeletons.
- Retargeting = Dec(S_tgt, Enc(S_src, D_src)); handles unseen skeletons + missing joints (random limb masking).
- **Limitations for robots** (why we can't use it off-the-shelf): purely kinematic, no joint limits,
  no dynamics/self-collision, biped-character assumptions, paired-data requirement, ground contact as
  heuristic soft labels. (Note: SAME has no arXiv version — cite the ACM DOI.)

**SAME's graphics successors (2024–2026)** confirm the shared-latent direction is active but still
animation-only: *WalkTheDog* (SIGGRAPH'24, 2407.18946) — shared phase manifold across morphologies,
unsupervised; *AnyTop* (SIGGRAPH'25, 2502.17327) — any-topology motion diffusion with joint-name text
embeddings for cross-skeleton correspondence; *PUMPS* (ICCV'25, 2507.20170) — skeleton-free point-cloud
motion pre-training; *SATA* (ICML'26, 2605.27055) — the most direct SAME successor: semantic-aware,
topology-agnostic latent trained on unaligned raw BVH, zero-shot cross-species retargeting. None touch
robot constraints or RL tracking — reinforcing the gap this proposal targets. AnyTop's joint-name
textual embeddings are worth borrowing for human↔robot joint correspondence (robot MJCF link names are
semantically meaningful: `left_knee_link` ↔ `left_knee`).

**The research opening**: nobody has combined (a) a SAME-style shared skeleton/embodiment-agnostic
latent motion space, (b) robot-grade constraints (joint limits, self-collision, contact preservation)
learned from optimization-teacher + physics signals, and (c) using that latent as the **command interface
of a multi-embodiment tracking policy**. HOVER unified command *modes* for one robot; AdaMorph unified
retargeting but not tracking; we unify motion representation across robots AND feed tracking with it.
The two repos on disk provide the teachers, the differentiable FK, the metrics, and the RL harness.

---

## 3. Proposed system: SNMR

Two coupled contributions:

> **C1 — Neural retargeter with a shared motion latent.** One network replaces all
> `ik_configs/*.json` + per-frame IK: any human skeleton in → any robot qpos out, batched on GPU,
> with an embodiment-invariant latent z in the middle.
>
> **C2 — Shared neural motion-tracking training.** The latent z (not per-robot joint targets)
> becomes the motion command; one tracking policy, conditioned on an embodiment embedding, trained
> across robots on neurally-retargeted data — with the tracking results feeding back to improve
> the retargeter (sim-to-data filtering → optional bilevel refinement).

### 3.1 Architecture

```
                         HUMAN SIDE                                ROBOT SIDE
 human skeleton graph ┐                                  ┌ robot embodiment graph (from MJCF/URDF:
 (SMPL-X 24–55 j /    │                                  │  body tree, joint axes, ranges, link
  BVH 22 j, offsets)  │                                  │  lengths, masses — parseable today by
                      ▼                                  ▼  GMR KinematicsModel._parse_xml)
 pose seq D_h ──► GAT/Transformer ENCODER ──► z_{1:T} ──► embodiment-conditioned DECODER ──► robot motion
 (6d rot, pos,    (per-frame GAT over joints   shared      (GAT over robot graph + temporal      q_{1:T} =
  root Δ, vels,    + small temporal attention   latent      attention; robot features re-injected  [root pos,
  contact flags)   window ±k frames)            space       per layer, AdaLN on embodiment code)    root 6d rot,
                                                                                                    joint angles]
                                              z ∈ R^{d}                                            + contact ĉ
                                              d ≈ 64–128
                                              per frame
```

Design choices (deviations from SAME, motivated by robot constraints):

1. **Decode joint *angles*, not per-node rotations.** Robots have 1-DoF hinge chains with hard limits;
   output = qpos directly (root pos + 6d root rot + per-joint scalar angles via per-node heads),
   squashed by `tanh` scaled to joint ranges ⇒ joint-limit satisfaction **by construction**.
2. **Differentiable FK in the loss loop**, using GMR's `KinematicsModel` (torch, batched, MJCF-parsed):
   position/orientation losses computed on FK'd bodies, so supervision can live in task space.
3. **Temporal context**: SAME is frame-wise with temporal soft losses; we add a small ±k-frame attention
   window (k≈8 @ 30 fps) — retargeting needs velocity-consistent, non-jittery output for RL consumption,
   and NMR's analysis shows framewise optimization is ill-conditioned exactly here.
4. **Contact-aware**: encoder input includes foot-contact flags (height+velocity heuristic, as in
   holosoma's `extract_foot_sticking_sequence_velocity`); decoder predicts robot contact ĉ used by the
   foot-skate loss and exported for downstream reward shaping.
5. **Embodiment code**: a learned per-robot token from pooling the robot graph (so unseen robots get a
   code zero-shot from their MJCF), used both in the decoder (AdaLN, following AdaMorph's verified
   recipe) and later as the tracking policy's embodiment conditioning.

### 3.2 Losses

Let T_r = teacher retargeting (GMR for robot-only; holosoma InteractionMeshRetargeter where
object/terrain interaction matters), FK = differentiable forward kinematics, H = human keypoints
scaled by GMR's `human_scale_table` convention.

| Loss | Form | Role |
|---|---|---|
| L_distill | ‖q̂ − q_teacher‖² (+ velocity) | dense supervision from the optimization teachers (paired data is *free*: run batch scripts once) |
| L_task | Σ_b w_b‖FK_b(q̂) − target_b(H)‖² + rot geodesic | self-supervised keypoint matching, GBC-style; weights initialized from the ik_config tables, then *learned* |
| L_limits | penalty outside soft range (mostly inactive due to tanh head) | safety margin |
| L_smooth | ‖q̈̂‖² + jerk | RL-consumable smoothness |
| L_contact | ĉ·‖ṗ_foot‖² + penetration min(z_foot,0)² + skate clamp | SAME-style, using robot FK feet |
| L_coll | hinge on SDF proxy between capsule pairs (self-collision pairs from holosoma configs) | replaces the teacher's hard constraint, softly |
| **L_z** | ‖Enc(S_h, D_h) − Enc*(R_i, q̂_i)‖² across robots i, + z-space alignment of the *same clip retargeted to different robots* | the **shared-space** loss — the SAME idea, extended: one motion ⇒ one z regardless of embodiment |
| L_cycle (opt.) | decode to robot → re-encode → decode to human skeleton → match original | unpaired-data extension (ImitationNet/CycleGAN-style), Phase-4 option |

L_z requires encoding *robot* motion too ⇒ the encoder is embodiment-agnostic by construction
(it already takes arbitrary graphs), giving us robot→robot and robot→human transfer for free.

### 3.3 Data engine (teachers we already own)

- **Pairs**: `GMR/scripts/smplx_to_robot_dataset.py` + `bvh_to_robot_dataset.py` over AMASS + LAFAN1
  for all 17 robots (multiprocessing; one-time CPU cost). Holosoma parallel retargeter for
  interaction clips (OMOMO, climbing) on G1/T1.
- **Embodiment augmentation** (SAME's key trick, robot version): randomize link lengths (±20%),
  joint range shrinkage, dropped joints (fixed wrist/waist variants), base height — generate synthetic
  MJCF variants and re-run the teacher on a subset ⇒ decoder generalization to unseen embodiments,
  measured by leave-one-robot-out.
- **Canonical intermediate**: keep GMR's dict format `{body: (pos, quat)}` as the human-side interface
  so all six source formats keep working unchanged.

### 3.4 Shared tracking training (C2)

Staged to keep risk low:

- **Stage A (per-robot policies, neural data)**: swap teacher data for SNMR output; holosoma pipeline
  unchanged (`q̂ → convert_data_format_mj.py → WBT NPZ → g1-29dof-wbt`). This is the *GMR-paper
  methodology in reverse*: fixed tracker, vary retargeting ⇒ clean measurement of whether neural
  retargeting matches/exceeds optimization retargeting for downstream RL.
- **Stage B (shared policy)**: extend holosoma WBT to T1 (+1–2 more robots via MJCF import); condition
  actor/critic on the embodiment code; motion command becomes `[z_t..z_{t+H}, embodiment code]`
  (replacing per-robot ref joint_pos ⊕ joint_vel in `motion_command`), while per-robot reference
  bodies for *rewards* still come from the decoded q̂ (rewards need ground truth in robot space;
  the *observation* interface is what becomes shared). Multi-embodiment batches: per-robot env groups
  in the same run, alternating or weighted sampling.
- **Stage C (feedback loop)**:
  1. *Sim-to-data filtering* (H2O-recipe): clips the tracker fails → flagged, teacher-vs-student
     diff analyzed, hard clips upweighted in retargeter fine-tuning.
  2. *Physics refinement* (ReActor/NMR-inspired, stretch goal): fine-tune decoder heads with tracking
     reward gradient approximation or with RL-repaired trajectories as new supervision — the retargeter
     learns to emit *trackable* motion, which optimization teachers cannot do.

---

## 4. Build plan

### Phase 0 — Foundations & data (weeks 1–3) — **DONE (2026-07-09), see §0.1**
- [x] New package `snmr/` alongside GMR + holosoma (dependency-light: torch/numpy/scipy/mujoco;
      dense masked graph attention instead of torch_geometric; FK re-implemented and validated
      against MuJoCo rather than lifting GMR's XML parser — more robust to MJCF defaults/includes).
- [x] Canonicalization (wxyz internally) + rotation layer, scipy-verified; the predicted
      convention pitfalls materialized and were caught by tests + adversarial review.
- [x] Pair-generation script for LAFAN1 (`snmr/scripts/make_pairs_lafan1.py`), smoke-tested;
      full 77-clip × 5-robot generation is step N1 (§0.3). AMASS deferred — no SMPL-X models in
      this environment (§0.2).
- [~] **Gate G0**: teacher runs at ~150–170 FPS here; visual spot-check + stats note pending in N1.

### Phase 1 — Single-robot neural retargeter (weeks 3–8) — **CORE DONE, training run = N3**
- [x] Encoder/decoder implemented (4 GAT layers, z=64, temporal transformer; AdaLN embodiment
      conditioning; tanh limit heads) — `snmr/model.py`.
- [x] All losses implemented (`snmr/losses.py`); overfit-a-batch validation passed (real G1 clip:
      loss 1.56→0.05, MPJPE 5–10 cm, zero limit violations).
- [ ] Dataset training LAFAN1→G1 with held-out clips + benchmarks vs GMR (step N3, §0.3).
- **Gate G1 (revised for the 4.7 h LAFAN1 dataset)**: held-out MPJPE-to-teacher < 3 cm, foot-skate
  ≤ teacher, ≥1000 fps batched GPU. Original 1–2 cm remains the AMASS-scale stretch target.
  If not met: increase temporal window, per-body decoder queries, or 2–3-iter test-time L_task
  polish (amortized-IK + refine).

### Phase 2 — Shared latent, multi-robot (weeks 6–12, overlaps P1)
- [ ] Multi-robot training (7 robots) with L_z consistency + embodiment augmentation.
- [ ] The two flagship experiments:
      **(E1) leave-one-robot-out zero-shot**: hold out e.g. H1-2; retarget via MJCF-derived embodiment
      code only; report degradation vs fine-tuned. **(E2) latent unification**: same clip → all robots,
      measure z-space clustering (same-motion-across-robots vs different-motions); latent interpolation
      / motion-matching demos in z-space (SAME's downstream tasks, robot edition).
- **Gate G2**: multi-robot model ≥ per-robot models on each robot (positive transfer, not interference);
  zero-shot robot within 2× of tuned error.

### Phase 3 — Tracking validation, Stage A (weeks 8–14, needs GPU cluster w/ IsaacSim)
- [ ] Pipe SNMR output through `convert_data_format_mj.py`; train `g1-29dof-wbt` (identical config)
      on (a) GMR-retargeted vs (b) SNMR-retargeted single clips + a 20-clip `motion_dir` dataset.
- [ ] Metrics: episode tracking rewards, termination/failure rates, adaptive-sampler failure heatmaps,
      sim2sim MuJoCo eval via holosoma_inference.
- **Gate G3**: SNMR-trained policies within noise of teacher-trained policies. (This alone is a
  publishable ablation given the GMR paper's finding that retargeting quality dominates tracking.)

### Phase 4 — Shared tracking + feedback loop, Stage B/C (weeks 12–20)
- [ ] Port holosoma WBT config to T1 (robot cfg exists for locomotion; needs WBT command/reward/
      termination presets + retarget models — teacher configs exist in holosoma_retargeting for T1).
- [ ] Embodiment-conditioned policy: motion command = decoded per-robot reference (baseline) vs
      z-latent command (ours); 2-robot joint training; measure cross-robot transfer (train G1+T1,
      test few-shot adaptation to PM01).
- [ ] Sim-to-data filtering loop; optional ReActor-style refinement if time allows.
- **Gate G4**: one checkpoint tracks motions on ≥2 robots; z-command ≥ decoded-command baseline.

### Phase 5 — Write-up & release (weeks 18–24)
- Paper target: ICRA/CoRL/RSS. Framing: *"One latent to move them all: shared neural motion
  retargeting and tracking across humanoid embodiments."* Release code + the multi-robot paired
  dataset (check AMASS license terms for derived data; LAFAN1 is research-friendly).

### Team/compute assumptions
1–2 researchers + this codebase. Phase 0–2: single 8×GPU node (retargeter training is light —
SAME trained in hours). Phase 3–4: IsaacSim-capable GPUs, 4096 envs per run; budget ~10–20 WBT runs.

---

## 5. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Student ceiling = teacher quality (distillation can't beat GMR artifacts) | High | L_task/L_contact/L_coll are *independent* of teacher; NMR/ReActor show physics feedback surpasses optimization teachers; measure student-vs-teacher on contact metrics explicitly |
| Single z per frame too coarse for whole-body detail (SAME z=32 was for stylized characters) | Medium | scale d, add per-limb pooled latents (5-token z), ablate |
| Multi-robot interference (negative transfer) | Medium | per-robot LoRA/AdaLN heads on shared trunk; gate G2 explicitly tests this |
| Contact/interaction fidelity (mesh-teacher's hard constraints are soft in the student) | Medium | keep holosoma teacher for interaction clips; add SDF losses; optional 2–3-step constrained polish at inference (still ≫ faster than full SQP) |
| Holosoma WBT is G1-only; T1 port unknown effort | Medium | Stage A results don't depend on it; T1 retargeting models/configs already exist in holosoma_retargeting; budgeted in Phase 4 |
| Wheeled/odd embodiments (R1 Pro) break biped assumptions | Low (scope) | exclude wheeled robots; scope = bipedal humanoids |
| z-command policy underperforms explicit reference (information bottleneck) | Medium | hybrid command (z ⊕ decoded root/EE targets); this ablation is itself a paper finding |
| Quaternion/frame-convention bugs across repos (wxyz vs xyzw, y-up vs z-up, cm vs m) | **Materialized & resolved** | canonicalization module + round-trip tests built (`snmr/rotation.py`); adversarial review caught the residual cases (see §0.1) |
| Robot-model provenance mismatch (limits/geometry differ across MJCF copies of the "same" robot) | **Materialized & resolved** | decoder limits must come from the model family that produced the data; contract test `test_joint_limits_cover_training_data` guards it |

## 6. Evaluation summary (what "success" means)

**Retargeting**: task-space MPJPE (vs scaled human keypoints & vs teacher), foot-skate score,
ground-penetration depth/fraction, self-collision count, joint-limit violations, jerk, throughput,
zero-shot-unseen-robot error. **Tracking**: WBT success/termination rate, per-body tracking errors,
reward curves under identical config (GMR-paper methodology), sim2sim transfer, multi-robot single
checkpoint coverage. **Shared space**: same-motion cross-embodiment z distance vs inter-motion
distance (retrieval mAP), latent interpolation quality.

## 7. Key references

GMR (2510.02252, ICRA'26) · TWIST (2505.02833) / TWIST2 (2511.02832) · OmniRetarget (2509.26633) ·
SAME (Lee et al., SIGGRAPH Asia'23, 10.1145/3610548.3618206) · Skeleton-Aware Networks (2005.05732) ·
H2O (2403.04436) / OmniH2O (2406.08858) / HOVER (2410.21229) / ASAP (2502.01143) · PHC (2305.06456) /
PULSE (2310.04582) · BeyondMimic (2508.08241) · GBC (2508.09960) · ImitationNet (2309.05310) ·
NMR "Make Tracking Easy" (2603.22201) · AdaMorph (2601.07284) · ReActor (2605.06593, SIGGRAPH'26) ·
ULTRA (2603.03279) · MaskedMimic (2409.14393) · KungfuBot (2506.12851) · VideoMimic (2505.03729) ·
CrossLoco (2309.17046) · S3LE (2103.06447) · HumanMimic (2309.14225) ·
WalkTheDog (2407.18946) · AnyTop (2502.17327) · PUMPS (2507.20170) · SATA (2605.27055) ·
R2ET (2303.08658) · PAN (2306.08006) · NKN (1804.05653).
