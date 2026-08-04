#!/usr/bin/env bash
set -uo pipefail
# E57-B — GMR-vs-SNMR matched tracking on HARDER clips (dance2, fight1), post-fix recipe.
# Registered in E55_E57_SCALING_AND_DISCRIMINATION.md; the paper's SVI-D promises this
# rerun ("post-repair rerun on harder clips remains registered"). 1 seed per cell
# (paired design), E51-v2 arm-A recipe (bodyfix + joint reward), 8k @512 envs, then
# 100-rollout eval per cell. Prediction under faithful-distillation: still ~equal;
# under "walk1 hides differences": a gap appears.
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
OUT="$MAIN/runs/e57b"
export PYTHONPATH="$MAIN"
mkdir -p "$OUT"
cp "$0" "$OUT/protocol.sh"
cd "$HOLOSOMA"

for CLIP in dance2_subject4 fight1_subject3; do
  for SRC in gmr snmr; do
    TAG="${CLIP}_${SRC}"
    [ -f "$OUT/${TAG}_eval.json" ] && { echo "skip $TAG"; continue; }
    REF="$MAIN/runs/wbt_validation/$SRC/${CLIP}_mj.npz"
    test -f "$REF" || { echo "MISSING $REF" | tee -a "$OUT/driver.log"; continue; }
    echo "=== E57B $TAG train start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 nice -n 15 "$PY" \
      "$MAIN/scripts/train_agent_joint_reward.py" \
      exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
      --training.num-envs 512 --training.seed 0 \
      --algo.config.num-learning-iterations 8000 --algo.config.save-interval 2000 \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
      --training.name "e57b_${TAG}" --training.headless True \
      >> "$OUT/${TAG}.train.log" 2>&1
    CKPT=$(ls -1t "$HOLOSOMA"/logs/WholeBodyTracking/*-e57b_${TAG}-locomotion/model_07999.pt 2>/dev/null | head -1)
    [ -z "$CKPT" ] && { echo "$TAG TRAIN FAILED" | tee -a "$OUT/driver.log"; continue; }
    "$PY" "$MAIN/scripts/eval_agent_repair.py" \
      --checkpoint "$CKPT" \
      --wbt-metrics.config.enabled \
      --wbt-metrics.config.output-path "$OUT/${TAG}_eval.json" \
      --wbt-metrics.config.horizon-s 10.0 \
      --training.headless True --training.num-envs 100 --training.seed 404 \
      --training.max-eval-steps 500 --training.export-onnx False \
      --simulator.config.sim.max-episode-length-s 100000.0 \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
      >> "$OUT/${TAG}.eval.log" 2>&1
    echo "=== E57B $TAG done: $(tr -d '\n' < "$OUT/${TAG}_eval.json" 2>/dev/null | head -c 200) ===" | tee -a "$OUT/driver.log"
  done
done
date -u +%FT%TZ > "$OUT/COMPLETE"
