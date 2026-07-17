# Side Project: Latent Rectified-Flow Retargeting with Physics-Guided Sampling

Status: SIDE PROJECT, registered 2026-07-16. This document is frozen before any flow training.
**F0 outcome (E43, 2026-07-17): z_r target NOT viable (Dec(z_r) MPJPE 50.06 cm vs 3 cm bar,
`runs/latent_flow/f0_viability.json`); per §4 the flow target is `z1 = z_h`.**
**F1/F2 outcome (E44, 2026-07-17): F1 PASS (flow 3.49 cm vs raw 3.48 cm); F2 FAIL — guidance
nearly inert across the whole preregistered grid (best oracle cell 0.400→0.339 m/s vs 0.08
endpoint) and beaten 5x by the z-descent control (0.221 m/s), which itself misses both
endpoints. PROJECT CLOSED at F2 per the stop rule; F3 never armed. See EXPERIMENT_LOG E43/E44.**
It does not change the main-line priority order in `docs/20260716-1931-sol.md` (M1b audit →
calibrated WBT B1 → sharing-cost screen → B-multi calibration). No GPU training for this project
may run while a registered main-line job (currently `reference_dev_snmr_walk1_seed0_to8000`) is
on the device.

Inspiration: SafeFlow (arXiv:2603.23983) — physics-guided rectified flow matching in a VAE latent
space for text-driven humanoid control, with reflow distillation for real-time use. We adapt the
*generation mechanism* (rectified flow + inference-time physics guidance in a frozen latent
space), not the text-conditioning or the safety gate.

## 1. Why flow matching here at all

SNMR is a deterministic regressor: `Enc(human) → z_h → Dec(z_h, embodiment) → qpos`, distilled
from GMR. Three facts from the experiment record motivate a generative head:

1. **Training-time soft physics objectives are exhausted.** E10 (contact/EDGE weights),
   E25 (foot-velocity distillation), and Gate 1 (C4 phase-balanced velocity, 0/3 seeds reach
   stance speed ≤ 0.08 m/s) all failed their registered endpoints. The stop rule on soft
   penalties has fired. Any new physics lever must act at *inference time*.
2. **Inference-time correction works but is mask-limited.** The frozen windowed projection and
   DLS foot-lock both pass every guard with the teacher-height ORACLE mask (0.0056 / 0.047 m/s)
   and fail with every deployable mask (Gate 1b M1/M2 failed; M1b/M3 pending). The bottleneck is
   MASK PRECISION for a *hard, local, qpos-space* corrector.
3. **The C5 skate signature looks like conditional-mean behavior.** E24/E26: decoded stance feet
   sit at the correct height but oscillate smoothly in xy (~2.9 cm RMS, correlated across the
   leg chain). A deterministic regressor trained with MSE on residually-ambiguous windows
   predicts the conditional mean, which is exactly a smooth average of nearby modes rather than
   any single planted-foot solution.

Physics-guided flow sampling is a *different corrector class* from Gate-1b's projection: instead
of locking foot positions in configuration space under a binary mask, it steers the **latent**
along a learned transport ODE using gradients of a differentiable physics cost through the frozen
decoder + FK. Three properties make it worth one screen:

- the edit is global (root + all joints move together, on the decoder's manifold), so it does not
  need the projection's per-frame stance anchors;
- the stance signal can be **soft** (sigmoid of decoded foot height) rather than a binary mask,
  side-stepping the hard-mask precision cliff that killed M1/M2;
- if it works, the same machinery gives the tracking task two hooks (§5).

## 2. Model

All SNMR weights are FROZEN (`runs/phase1_g1_large/ckpt_100k_final.pt` unless F0 redirects).
The flow model is a separate, small network in the 128-d frozen latent space.

- **Conditioning** `c_{1:T}`: the frozen encoder's post-temporal human latent `z_h` (per frame).
- **Target** `z_1`: the frozen encoder's embedding of the *teacher robot motion*,
  `z_r = Enc(robot_pose_features(teacher qpos))` — the same quantity Phase 2's `L_z` aligns.
  The flow therefore learns the conditional transport `p(z_r | z_h)`, directly modelling the
  measured human→robot latent gap ("aligned, not invariant": CKA 0.91, MLP attacker 0.91).
- **Rectified flow**: CondOT path `z_u = u·z1 + (1−u)·z0`, `z0 ∼ N(0, I)`, velocity target
  `z1 − z0`, MSE conditional flow-matching loss. Sampling: Euler, NFE ≈ 25 for the screen
  (reflow/NFE-1 distillation is out of scope until F2 passes).
- **Network**: temporal Transformer over the noisy latent sequence (hidden 256, 4 layers,
  4 heads), sinusoidal *frame* positions, flow-time `u` injected via AdaLN-zero, per-frame
  conditioning concatenated at input and re-injected. ~1.6 M parameters.

**Fallback target (decided by F0):** if the frozen decoder cannot reproduce the teacher from
`z_r` (Dec(z_r) MPJPE far above the 2.18 cm baseline — the deferred `zr_decode_prob` question),
the target degrades to `z1 = z_h` and the flow becomes a learned identity transport whose only
value is the guidance mechanism. That is a weaker but still testable configuration.

## 3. Physics-guided sampling

During ODE integration, steer the velocity with the endpoint-predicted cost gradient:

```
ẑ1  = z_u + (1−u)·v_θ(z_u, u | c)          # one-step endpoint prediction
ṽ   = v_θ − α(u) · clamp(∇_{ẑ1} C(Dec(ẑ1)), ±g_max)
```

(SafeFlow evaluates C at `Dec(z_u)`; we use the endpoint prediction because the frozen decoder
was only ever trained on clean latents.) `α(u)` increases linearly over the trajectory;
per-element gradient clamping as in SafeFlow.

Cost `C = λ_sk·C_skate + λ_pen·C_pen + λ_sm·C_smooth`, all through the frozen decoder,
`local_root_to_world`, and differentiable FK:

- `C_skate`: stance-weighted squared foot XY velocity (m/s), stance weight per mode below;
- `C_pen`: squared ground penetration of foot bodies;
- `C_smooth`: squared dof acceleration (+0.1× jerk), guarding the known failure of correctors
  (blend-knob Pareto: skate fixes bought with jerk).

Joint limits are satisfied by the tanh head construction; no `C_lim`.

**Stance-weight modes** (the Gate-1b lesson, transposed):

| Mode | Signal | Deployable? |
|---|---|---|
| `soft_decoded` | `σ((h_thr − decoded foot height)/τ)`, self-consistent, no binary mask | yes |
| `source_gated` | soft_decoded × human-source contact mask (height+speed heuristic) | yes |
| `teacher_height` | frozen oracle mask (enter 0.03 / exit 0.05) | NO — diagnostic ceiling only |

E24 showed *hard-thresholded* contact detection on decoded motion under-fires (0.03 vs 0.29
prevalence); `soft_decoded` deliberately avoids the threshold, and whether a soft weight is
enough is precisely what F2 measures.

## 4. Preregistered gates

Evaluation protocol matches the frozen Gate-0/1b conventions: `VAL_CLIPS` (7 held-out clips),
fixed windows, full-clip contact masks, and `scripts/eval_footlock.py`'s guard set. Window
length = the flow's training window (64 frames), 6 windows/clip; the raw deterministic decode of
the SAME windows is the paired baseline. Thresholds are those of `projection_decision`:
teacher-height stance speed ≤ 0.08, source-contact ≤ 0.10, MPJPE ≤ raw + 0.005 m and ≤ 0.04 m
absolute, dof jerk ≤ 1.2× raw, zero limit violations, penetration mean ≤ raw + 0.002 /
fraction ≤ raw + 0.02.

- **F0 — target viability (no training, CPU).** Decode `z_r = Enc(teacher)` on the eval windows.
  If MPJPE(Dec(z_r), teacher) ≤ 0.03 m the flow target is `z_r`; otherwise `z_h`. Also record
  the **z-descent control**: Adam directly on `z` initialized at `z_h`, minimizing the identical
  cost `C` under a trust region, at NFE-matched compute (25 decoder+FK gradient evaluations).
  This control is the null hypothesis "you don't need a flow prior, plain latent descent
  suffices"; it runs in every later gate.
- **F1 — fidelity (unguided flow non-inferior to regression).** Sampled `z ∼ flow(·|z_h)`
  decodes to MPJPE ≤ raw + 0.005 m on the eval windows (median over 3 samples/window). Failing
  F1 closes the project (a generative head that loses fidelity cannot help downstream).
  Descriptive secondary: multi-sample spread, skate of unguided samples vs raw.
- **F2 — guided screen (the decision gate).** Grid over α-end ∈ {1, 3, 10} × stance-weight mode.
  PASS requires some deployable-mask cell to (a) meet both stance-speed endpoints with every
  guard, AND (b) beat the z-descent control on teacher-height stance speed at equal guard
  compliance. Oracle-mask cells are reported but cannot pass the gate. If only oracle passes,
  the result is "flow guidance is also mask-limited" — a real finding that merges this line
  into the Gate-1b mask conclusion, and the project stops.
- **F3 — tracking hand-off (only after F2 PASS and after B1's confirmatory matrix is resolved).**
  Export flow-guided walk1 references through the unchanged `export_wbt_npz.py` boundary and run
  ONE 8k dev policy under the existing `run_wbt_reference_dev.sh`-style protocol as a third arm
  (GMR / SNMR-raw / SNMR-flow-guided). Same non-inferiority framing as B1 (completion −5 pp,
  RMSE +10%).

Stop rules: any gate failure stops the project at that gate with a negative write-up in
`docs/EXPERIMENT_LOG.md`. No threshold tuning after seeing results; α-grid and mask modes above
are the entire search space.

## 5. Relation to the tracking task (the second half of the question)

Two hooks, strictly ordered:

1. **Reference-quality route (F3).** The only near-term, measurable path: guided-flow references
   are just NPZs to holosoma; the existing calibrated harness answers "did physics-guided
   generation make references that *train a better policy*". This is the SafeFlow Table-I claim
   (JV/SC ↓ ⇒ success rate ↑) instantiated with our teacher-relative metrics.
2. **Latent-command route (blocked, future).** E39 established L1 feasibility only (frozen z can
   carry a single-clip command at −8..−18 pp completion); Phase 3 is blocked on B-multi budget
   calibration. If both F2 and Phase 3 unblock, the flow prior gives the policy a *generative*
   latent command with physics folded in by guidance — and reflow distillation would make it
   cheap enough for closed-loop use. Caveats that stand: exported z is noncausal (full-clip
   encoding, no positional encoding), and contact is not linearly decodable from z (E38), so no
   "latent carries physics" claim without new evidence.

Also noted for later (not registered): SafeFlow's directional-sensitivity instability score R
correlates with downstream tracking MPJPE; once a flow model exists, probing R on our reference
windows against the existing E35/E39 rollout reports is a nearly free descriptive study of
"which reference windows are trackable".

## 6. Artifacts

| Concern | File |
|---|---|
| Flow net, CFM loss, sampler, guidance, soft stance weights | `snmr/flow.py` |
| Trainer (frozen SNMR, manifest, held-out fidelity eval) | `scripts/train_latent_flow.py` |
| F0–F2 evaluation (arms: raw / zr_oracle / z-descent / flow / flow-guided) | `scripts/eval_latent_flow.py` |
| Tests | `tests/test_flow_retarget.py` |
| Results | `runs/latent_flow/` (checkpoints git-ignored; manifests/metrics in git) |

Training budget: 20k steps, batch 8 windows × 64 frames, frozen-encoder features computed on the
fly; fits the A10G alongside nothing else (single-GPU-job rule applies).
