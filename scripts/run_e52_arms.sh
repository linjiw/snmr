#!/usr/bin/env bash
set -euo pipefail

# E52 Stage-1 arms (docs/E52_STAGE3_CVAE_DESIGN.md §4), sequential on one GPU:
#   A': a_prior_snmr    — prior sees [proprio, proj(SNMR z window)]  (H2 transfer arm)
#   B : b_prior_proprio — prior sees proprio only                    (PULSE-style control)
# Teacher = E51-v2 arm A (bodyfix + joint reward, 0.98 completion). 2000 DAgger rounds
# @1024 envs ≈ matched interaction budget to an 8k-iter PPO run at num_steps_per_env=24
# (rounds use the same 24-step collection). Eval: deterministic z = mu_prior, 100
# phase-stratified 10-s rollouts (inline in the trainer).

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REFERENCE_Z="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/e52_act_through_latent"

export PYTHONPATH="$MAIN"
test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -f "$TEACHER"
test -f "$REFERENCE_Z"
mkdir -p "$OUT"
git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
cp "$0" "$OUT/protocol.sh"
sha256sum "$TEACHER" "$REFERENCE_Z" >> "$OUT/input_sha256.txt"

for ARM in a_prior_snmr b_prior_proprio; do
  echo "=== E52 $ARM start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  cd "$HOLOSOMA"
  E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="$OUT" E52_ROUNDS=2000 \
    nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 0 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE_Z" \
    --training.name "e52_${ARM}" --training.headless True \
    >> "$OUT/${ARM}.train.log" 2>&1
  test -f "$OUT/${ARM}_eval.json"
  echo "=== E52 $ARM done $(date -u +%FT%TZ): $(cat "$OUT/${ARM}_eval.json" | tr -d '\n') ===" \
    | tee -a "$OUT/driver.log"
done

date -u +%FT%TZ > "$OUT/COMPLETE"
echo "E52 arms complete" | tee -a "$OUT/driver.log"
