# E51 — Joint-space tracking reward (the fidelity lever for E50 Stage B)

**Date registered:** 2026-07-27. **Status:** REGISTERED; arm A launched on registration.
**Depends on:** E50-A caution-flag verdict (`docs/E50_PHYSICS_REPAIRED_TEACHER_PROTOCOL.md`).

## 1. Diagnosis this tests (from E50-A error decomposition, logged 07-24)

The 9.7 cm heading-local MPJPE / 0.25 rad joint RMSE of our 8k trackers is **not undertraining**:

1. **Joint error is flat from ~1k iterations** (`Env/motion/error_joint_pos` 1.37→1.45 norm,
   d(6k→8k) ≈ −0.003) while episode length still climbs 4×— the optimizer is improving
   survival/body-pose terms, not joint fidelity.
2. **48% of the joint MSE is a constant bias toward the default/nominal pose** (hip pitch
   −0.37 rad ≈ default −0.312; elbows −0.33 toward default 0.6; waist pitch −0.36
   compensates so torso stays on target). The policy "leans on" the PD default pose.
3. **Gait amplitude undershoot ~1/3** (hip-pitch std 0.23 vs reference 0.35); phase lag is
   NOT the issue (best-lag shift improves RMSE by only 0.001 rad at ~20 ms).
4. Root cause is structural: holosoma's g1 WBT reward (`config_values/wbt/g1/reward.py`) has
   **no joint-space term** — only 14-body pose/velocity exp-kernels. 29-DoF detail
   (esp. distal/aliased joints) is invisible to the objective, so nullspace drift toward the
   action-space default is unpenalized. (Consistent with the tracked-body set: pelvis/torso
   are best-tracked bodies at ~2 cm; ankles worst at ~17 cm.)

ReActor/DeepMimic-family rewards always include joint tracking (ReActor Tab. 6 has no direct
joint term but dense per-body rbs pos/ori at weight 5.0/2.5 over MANY bodies; DeepMimic/
BeyondMimic use explicit joint terms). E51 adds the standard term.

## 2. Arms

Injection: `scripts/train_agent_joint_reward.py` (rebuilds reward cfg pre-train; tyro cannot
add dict keys) + `snmr/integration/wbt_rewards.py` (exp-kernel terms reading
`command.joint_pos/robot_joint_pos`). No clone edits; smoke-verified (term active in TB).

- **A (primary): joint_pos w=1.0 σ=0.5**, GMR walk1, seed 0, 8k iters @1024 envs — exactly
  the confirmatory recipe ± the new term. Single-variable comparison against
  `reference_confirm_gmr_walk1_seed0` (completion 0.90, eval joint RMSE 0.25 rad,
  E50 heading-local MPJPE 9.7 cm).
- **B (conditional, only if A moves fidelity but costs completion): w=0.5.**
- **C (conditional, only if A improves both): SNMR walk1 seed 2 replication + 16k-iter arm
  to re-test the budget axis with the fixed objective.**

## 3. Readouts & gates (preregistered)

Per arm: standard 100-rollout eval (seed 404) + the E50 recording/export pipeline re-run
with the new policy (`run_e50_stage_a.sh` machinery, pointed at the E51 checkpoint).

- **Primary: E50 heading-local MPJPE** — PROMOTE the lever iff ≤ 6.5 cm (≥1/3 reduction
  from 9.7). Stage-B distillation unlocks iff ≤ 5 cm (the original H-A(iii) gate).
- **Guard: completion ≥ 0.85** (no meaningful robustness price; baseline 0.90).
- **Secondary: eval joint RMSE ≤ 0.15 rad** (vs 0.25); default-pose bias share of MSE
  (target < 25%, vs 48%); hip-pitch amplitude ratio (target > 0.85, vs 0.66).
- **Kill for the lever:** joint RMSE improves < 20% at any completion — the objective was
  not the binding constraint after all; fall back to the 16k budget axis before abandoning
  E50 Stage B.

## 4. Notes

- σ=0.5 rad chosen so the term is informative at current error levels
  (exp(−0.25²/0.5²) ≈ 0.78, gradient-rich), not saturated like σ→∞ or cliffed like σ=0.1.
- Adding reward terms changes the return scale by ≤ +1/step (bounded exp term); PPO
  advantage normalization absorbs this; no other weights retuned (single-variable).
- This also feeds C6 hygiene: if fidelity improves for BOTH sources equally, the GMR-vs-SNMR
  comparison conclusions are unchanged; E51 does not reopen the confirmatory matrix.
