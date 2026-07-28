#!/usr/bin/env bash
set -euo pipefail

# E53 multi-clip act-through-latent (docs/E53_MULTICLIP_PROTOCOL.md).
# Stage 0: explicit-command teacher on the 8-clip motion_dir (E51 arm-A recipe).
# Stage 1: DAgger students C (prior=proprio+cmd) and D (prior=+SNMR z), E52 v3 recipe.
# One GPU job at a time; launch only when e52_v3_seeds has COMPLETE.

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
MOTION_DIR="$MAIN/runs/wbt_latent_gmr_multi8"
OUT="$MAIN/runs/e53_multiclip"
EVAL_SEED=404

export PYTHONPATH="$MAIN"
test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -f "$MOTION_DIR/manifest.json"
test "$(ls "$MOTION_DIR"/*_mj_z.npz | wc -l)" -eq 8
mkdir -p "$OUT/reports"
git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
cp "$0" "$OUT/protocol.sh"

# --- Stage 0: multi-clip explicit teacher (bodyfix + joint reward) ---------------------
TEACHER_NAME=e53_teacher_multi8_seed0
if [[ ! -f "$OUT/TEACHER_DONE" ]]; then
  # train only if no completed checkpoint is recorded (idempotent resume: the first launch
  # trained to 8k but died in the eval step; do not retrain)
  if [[ ! -s "$OUT/teacher_ckpt.txt" || ! -f "$(cat "$OUT/teacher_ckpt.txt")" ]]; then
    echo "=== E53 Stage-0 teacher start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    cd "$HOLOSOMA"
    E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 nice -n 15 "$PY" \
      "$MAIN/scripts/train_agent_joint_reward.py" \
      exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
      --training.num-envs 1024 --training.seed 0 \
      --algo.config.num-learning-iterations 8000 --algo.config.save-interval 2000 \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-file "" \
      --command.setup-terms.motion-command.params.motion-config.motion-dir "$MOTION_DIR" \
      --training.name "$TEACHER_NAME" >> "$OUT/$TEACHER_NAME.train.log" 2>&1
    RUN_DIR=$(ls -1dt "$HOLOSOMA"/logs/WholeBodyTracking/*-"$TEACHER_NAME"-locomotion | head -1)
    TEACHER_CKPT="$RUN_DIR/model_07999.pt"
    test -f "$TEACHER_CKPT"
    echo "$TEACHER_CKPT" > "$OUT/teacher_ckpt.txt"
  fi
  TEACHER_CKPT=$(cat "$OUT/teacher_ckpt.txt")
  # teacher eval: per-clip (wbt_metrics fixed starts require exactly one motion)
  for CLIP_NPZ in "$MOTION_DIR"/*_mj_z.npz; do
    CLIP=$(basename "$CLIP_NPZ" _mj_z.npz)
    "$PY" "$MAIN/scripts/eval_agent_repair.py" \
      --checkpoint "$TEACHER_CKPT" \
      --wbt-metrics.config.enabled \
      --wbt-metrics.config.output-path "$OUT/reports/${TEACHER_NAME}_${CLIP}_eval${EVAL_SEED}.json" \
      --wbt-metrics.config.horizon-s 10.0 \
      --command.setup-terms.motion-command.params.motion-config.motion-dir "" \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$CLIP_NPZ" \
      --training.headless True --training.num-envs 100 --training.seed "$EVAL_SEED" \
      --training.max-eval-steps 500 --training.export-onnx False \
      --simulator.config.sim.max-episode-length-s 100000.0 \
      >> "$OUT/$TEACHER_NAME.eval.log" 2>&1
    test -f "$OUT/reports/${TEACHER_NAME}_${CLIP}_eval${EVAL_SEED}.json"
  done
  COMPLETION=$(jq -s '[.[].completion_rate] | add/length' \
    "$OUT/reports/${TEACHER_NAME}"_*_eval${EVAL_SEED}.json)
  echo "=== E53 teacher done: mean per-clip completion=$COMPLETION ===" | tee -a "$OUT/driver.log"
  # gate: proceed iff >= 0.6 (protocol §3)
  awk "BEGIN{exit !($COMPLETION >= 0.6)}" || {
    echo "TEACHER GATE FAILED ($COMPLETION < 0.6) — stopping per protocol" | tee -a "$OUT/driver.log"
    exit 1
  }
  date -u +%FT%TZ > "$OUT/TEACHER_DONE"
fi
TEACHER_CKPT=$(cat "$OUT/teacher_ckpt.txt")

# --- Stage 1: DAgger students ----------------------------------------------------------
for ARM in c_prior_explicit d_prior_explicit_snmr; do
  echo "=== E53 $ARM start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  cd "$HOLOSOMA"
  E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER_CKPT" E52_OUT="$OUT" E52_ROUNDS=2000 \
    nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 0 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "" \
    --command.setup-terms.motion-command.params.motion-config.motion-dir "$MOTION_DIR" \
    --training.name "e53_${ARM}" --training.headless True \
    >> "$OUT/${ARM}.train.log" 2>&1
  test -f "$OUT/${ARM}_student.pt"
  # per-clip eval (inline eval skips multi-motion; fixed starts need exactly one motion)
  for CLIP_NPZ in "$MOTION_DIR"/*_mj_z.npz; do
    CLIP=$(basename "$CLIP_NPZ" _mj_z.npz)
    E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER_CKPT" E52_OUT="$OUT" \
      E52_EVAL_ONLY=1 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
      exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
      --training.num-envs 100 --training.seed "$EVAL_SEED" \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-dir "" \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$CLIP_NPZ" \
      --training.name "e53_${ARM}_eval_${CLIP}" --training.headless True \
      >> "$OUT/${ARM}.eval.log" 2>&1
    mv "$OUT/${ARM}_eval.json" "$OUT/reports/${ARM}_${CLIP}_eval${EVAL_SEED}.json"
  done
  echo "=== E53 $ARM done $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
done

date -u +%FT%TZ > "$OUT/COMPLETE"
echo "E53 complete" | tee -a "$OUT/driver.log"
