# SNMR: One Latent to Move Them All — Shared Neural Motion Retargeting for Humanoid Robots

**Working paper draft.** Numbers marked [DONE] are from committed run artifacts (see
`docs/EXPERIMENT_LOG.md` for provenance); [PEND] awaits a queued experiment. This draft is kept
brutally honest — negative results are part of the contribution.

---

## Abstract (draft v1)

Human-to-humanoid motion retargeting is dominated by per-frame inverse-kinematics or
trajectory-optimization pipelines that require hand-tuned per-robot configuration, run
sequentially on CPU, and provide no shared representation connecting the same motion across
embodiments. We present SNMR, a skeleton-agnostic neural retargeter that maps human motion into a
shared per-frame latent space and decodes it onto any of five heterogeneous humanoid robots
(0.34–0.78 m root height, 22–29 DoF) as MuJoCo-ready joint trajectories, with joint-limit
satisfaction by construction. Distilled from an IK teacher (GMR) with differentiable-FK task
losses, a single 1.6 M-parameter network reaches 2.2 cm held-out whole-body error on Unitree G1 —
matching the teacher's fidelity at 12× its throughput — and 3.2–6.7 cm across all five robots
simultaneously. We contribute the first quantitative *representation analysis* in cross-embodiment
robot learning: the latent is strongly aligned across embodiments (linear CKA 0.91; 75 %
cross-embodiment motion retrieval at 1 % chance) yet embodiment identity remains nonlinearly
decodable (91 % MLP attacker vs 28 % linear probe) — a measured gap between "shared" and
"invariant" that prior work leaves unexamined. We report honest negative results with matched
baselines: at fixed parameter budget, joint training costs ~1.4 cm per robot versus specialists,
and zero-shot decoding to an unseen robot from its kinematic description alone fails (5.2× the
in-training error), quantifying how far embodiment-level generalization remains. All code, metrics
aligned with the GMR/OmniRetarget/NMR/SAME evaluation conventions, and a matched-pipeline
whole-body-tracking validation package are released.

*(~230 words. Rewrite pass pending final ablation + contact-retrain numbers.)*

## 1. Contributions (claim → evidence)

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | One skeleton-agnostic network retargets to 5 heterogeneous humanoids at IK-teacher fidelity, 12× throughput, joint limits by construction | G1 2.18 cm held-out (Gate G1); all-5: 3.2–6.7 cm; ~2000 fps GPU vs ~160 fps CPU teacher; 0 limit violations | [DONE] |
| C2 | First quantitative shared-latent analysis in cross-embodiment robotics | CKA 0.91, retrieval R@1 0.75, E1 linear 0.28 vs E3 MLP 0.91 (Elazar-Goldberg protocol) | [DONE] |
| C3 | Honest matched-baseline accounting of sharing costs | specialist 3.75 vs shared 5.12 cm at matched steps/schedule | [DONE] |
| C4 | Zero-shot embodiment transfer quantified (negative) | LORO PM01: 29.6 vs 5.7 cm (5.2×) | [DONE] |
| C5 | Foot-skate gap closed by prediction + source-mask-constrained correction | E26 pilot: skate 0.489→0.064 m/s (≤0.08 target) at +0.1 cm MPJPE; training-time losses alone verifiably insufficient (E10a bundled sweep, E25 velocity-distill) | [PILOT DONE — full-clip eval E26b running] |
| C6 | Matched-pipeline tracking validation (retargeting → RL) | N8 package ready; **WBT trains locally on holosoma's MuJoCo/Warp backend (E20 scout: CLI overrides only)** + trackability proxy already shows PD-replay equivalence (E18) | [PEND — local GPU block] |
| C7 | Robot→robot motion transfer via the shared latent (no human data) | currently FAILS (24–45 cm; decoder never trained on robot encodings — E19); decode-from-z_r augmentation (E21) is the fix | [PEND] |

## 2. Method (framework)

### 2.1 Problem
Given human motion D_h = {(p_j^t, q_j^t)} on skeleton S_h, produce robot configurations
q^t = [root_pos, root_quat, θ] for target embodiment R_i, for any i, from ONE network.

### 2.2 Architecture (algorithm overview)
```
 human skeleton graph S_h ──┐                          ┌── robot graph R_i (from MJCF: tree,
 pose features (per node:   │                          │   offsets, joint axes, ranges)
 pos, 6D rot, vel — in the  ▼                          ▼
 root heading frame)   GAT ENCODER ──► max-pool ──► z_t ∈ R^128 ──► AdaLN DECODER ──► qpos + ĉ
                       (node-shared     over nodes   (per-frame     (graph attention   (tanh heads →
                        weights, any    → skeleton-   shared         over R_i, latent    joint limits;
                        topology)       agnostic)     latent)        re-injected/layer)  root in scaled-
                                                                                         human-heading frame)
```
Root pose is predicted in the **scaled-human-root heading frame** (per-robot scale s_i from the
teacher's config; world pose recomposed at inference) — world-frame or unscaled-anchor targets are
unlearnable/ill-posed under heading-invariant encoder features (§4, lesson 1).

### 2.3 Training (Algorithm 1, informal)
```
for each step:
    sample clip window (64 frames); encode human -> z
    sample K=2 robots; for each robot i:
        decode(z, R_i) -> qpos_i, contact ĉ_i
        L_distill (teacher qpos) + L_task (diff-FK keypoints) + L_smooth + L_limits
        [+ L_contact: teacher-mask velocity + EDGE self-consistency on ĉ, world frame]
        encode robot-teacher motion -> z_i;  L_z = ||z - z_i||^2   (symmetric)
    backprop mean loss
```
Teachers: GMR IK (per-robot configs) → 2.48 M paired frames (77 LAFAN1 clips × 5 robots).

### 2.4 Analysis protocol (the "analyze the neural" contribution)
E1 embodiment linear probe (want chance) · E2 motion probe (positive control) · E3 post-hoc MLP
attacker + k-class proxy-A-distance (Elazar-Goldberg safeguard) · E4 cross-embodiment retrieval ·
E5 linear CKA · [scale-normalized probe to attribute the leak: scale vs style].

## 3. Results snapshot (running log — see EXPERIMENT_LOG.md for full provenance)

**Retargeting fidelity [DONE]:** G1 specialist 2.18 cm (100k) / 3.75 cm (50k); shared-5:
G1 6.67, T1 5.80, N1 5.54, PM01 5.72, Toddy 3.16 cm. Teacher-matched on penetration, limits,
joint-jumps; *better* than teacher on limit-proximity margins (0.29 vs 0.57).

**Foot skate [PILOT RESOLVED — E26]:** raw decoder output skates at 0.25–0.5 m/s vs teacher 0.05.
Root cause (E24/E26): smooth xy oscillation (~2.9 cm RMS, τ≈0.1 s) of a correctly-placed stance
foot — the velocity shadow of the regression error (skate/MPJPE ratio ~7–9 s⁻¹ constant across
all checkpoints), NOT high-frequency jitter (low-pass ineffective) and NOT drift from a planted
point. Training-time levers verifiably insufficient: bundled contact losses fail at every weight
(E10a), foot-velocity distill helps MPJPE but leaves the ratio unchanged (E25). The fix is the
field's standard hybrid (cf. OmniRetarget hard stance constraints → exactly-zero skate;
Villegas'21 ESO beats IK-only post-processing; production practice per Harvey'20/PFNN): a
**source-mask-driven foot-lock** — contact labels from the CLEAN human input (decoded-motion
detection under-fires 0.03 vs 0.29 frac; the known chicken-and-egg), dilated for coverage, blended
leg-IK per stance interval. E26 pilot: **0.489→0.064 m/s (below the 0.08 target) at +0.1 cm
MPJPE**; jerk cost mitigated by smoothing the IK correction (σ grid in E26b).

**Shared-latent analysis [DONE]:** CKA 0.910 (0.86–0.98 all 15 pairs incl. human); retrieval
human→robot R@1 0.749 / MRR 0.838 (chance 0.010); E1 0.278 / E3 0.909 / proxy-A 1.78. E2 motion
probe stays ~chance under both window-mean and temporal-statistics readouts (0.151/0.152) — the
space is **content-specific, not semantically organized**: it carries instance-level trajectory
detail (75 % exact-clip retrieval) but no linearly-separable category structure, consistent with
what pure distillation trains for. (Earlier "window-mean artifact" hypothesis tested and rejected.)
**Leak attribution [DONE]:** height-normalizing input features moves the MLP attacker only
0.943→0.933 — **scale explains ~1 % of the embodiment leak; the signature is structural/stylistic
(H-deep)**. This measurement (a) motivates a domain-confusion term as the principled fix and
(b) upgrades the analysis contribution: we don't just report the shared-vs-invariant gap, we
attribute it.
**Figures [DONE]** (`runs/figures/`): dual-colored t-SNE+UMAP — embodiment-colored intermixed,
motion-colored cleanly clustered — the visual proof of the shared space, and it shows motion is
*nonlinearly* organized (reconciling the weak linear motion probe). CKA heatmap tracks morphology
(adult humanoids 0.96–0.98; human↔toddy weakest at 0.82).

**Ablations [PARTIAL — grid running]:**
| variant | MPJPE (cm) | skate (m/s) | dof jerk | note |
|---|---|---|---|---|
| base z=128 (50k) | 4.71 | 0.386 | 614 | control |
| no_temporal | **3.95** | **0.358** | 720 | temporal transformer NOT the noise-suppressor we assumed — removing it *improves* MPJPE/skate, worsens jerk |
| z32 / small / contact / lr8e4 | — | — | — | running |
| teacher (GMR) | — | 0.052 | 621 | reference |

**Negative results [DONE]:** matched-budget sharing cost (3.75→5.12 cm); LORO zero-shot 5.2×.
Both reported with the confounds that earlier, wrong readings had (schedule alignment).

## 4. Lessons / design findings (paper discussion candidates)
1. Root-pose parametrization dominates early error (3.3 m → 0.28 m by frame choice alone).
2. Pose metrics are blind to contact artifacts (MPJPE fine at 5× the teacher's skate) — matches
   SAME's ablation; motivates contact-gated losses + metrics.
3. Model-provenance contract: decoder limits must come from the data-generating MJCF variant.
4. Executed-code adversarial review found 8 real bugs across 2 rounds (frame conventions,
   probe leaks, binary-formula misuse, body-vs-world angular velocity) — recommend as practice.
5. Temporal-transformer encoder mixing is not required for per-frame fidelity at this scale
   (ablation): its cost may only be justified by jerk/smoothness, pending contact-retrain interplay.

## 5. TODO before submission
- [x] C5: E26 source-mask foot-lock pilot passes the ≤0.08 m/s target — full 7-clip eval (E26b)
      + apply to WBT exports; contact sweep DONE (negative, reported as evidence)
- [ ] C6 WBT comparison — now LOCAL (holosoma MuJoCo/Warp backend, venv ready); smoke → pilot
      (3 clips × paired seeds) → confirmatory (per audit Gate 2)
- [ ] Embodiment augmentation run (synthetic MJCF variants) → revisit LORO (audit Gate 4: keep
      separate from zr_decode_prob arm)
- [ ] Sharing-cost diagnosis: per-robot gradient cosine/norm BEFORE capacity scale-up (audit Gate 3)
- [ ] Temporal: positional-encoding transformer arm before any temporal-modeling claim (audit Gate 6
      — current module is content-only attention, snmr/model.py:164)
- [ ] Figures: architecture, CKA heatmap, dual-colored t-SNE, per-robot qualitative strips
- [ ] Wording sweep per audit claim ledger: "aligned, not invariant"; no "contact-consistent"/
      "improves tracking"/"positive transfer"/"zero-shot new robots" until gates pass; throughput
      re-measured with warm-up + median/p10/p90 (benchmark.py updated)
