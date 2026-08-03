#!/usr/bin/env bash
set -euo pipefail
# E53 Stage-0 retry, preregistered lever: envs 1024 -> 2048 (log entry "E53 Stage-0 @16k").
# Multi-clip explicit teacher on the 8-clip motion_dir; gate >= 0.6 mean completion.
# If this also fails, the preregistered reframe applies (Q2 via cross-embodiment;
# multi-clip reported in-progress).
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
MOTION_DIR="$MAIN/runs/wbt_latent_gmr_multi8"
OUT="$MAIN/runs/e53_multiclip"
NAME=e53_teacher_multi8_16k_2048env_seed0

export PYTHONPATH="$MAIN"
mkdir -p "$OUT"
cd "$HOLOSOMA"
echo "=== E53-2048 teacher start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 nice -n 15 "$PY" \
  "$MAIN/scripts/train_agent_joint_reward.py" \
  exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
  --training.num-envs 2048 --training.seed 0 \
  --algo.config.num-learning-iterations 16000 --algo.config.save-interval 4000 \
  --randomization.ignore-unsupported True \
  --command.setup-terms.motion-command.params.motion-config.motion-file "" \
  --command.setup-terms.motion-command.params.motion-config.motion-dir "$MOTION_DIR" \
  --training.name "$NAME" --training.headless True \
  >> "$OUT/${NAME}.train.log" 2>&1
CKPT=$(ls -1t "$HOLOSOMA"/logs/WholeBodyTracking/*-${NAME}-locomotion/model_15999.pt | head -1)
echo "$CKPT" > "$OUT/teacher2048_ckpt.txt"
echo "=== E53-2048 teacher done $(date -u +%FT%TZ): $CKPT ===" | tee -a "$OUT/driver.log"
