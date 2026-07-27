#!/usr/bin/env bash
set -euo pipefail

# E51 v2 (post DEFECT-1): re-baseline with the body-index fix, then the joint-reward arm.
# Arms (GMR walk1, seed 0, 8k iters @1024 envs, matched to the confirmatory recipe):
#   R: bodyfix only  (E51_JOINT_POS_WEIGHT=0)  — the confound-free re-baseline
#   A: bodyfix + joint_pos reward (w=1.0 σ=0.5)
# Both trained via train_agent_joint_reward.py (applies wbt_bodyfix.patch()); evals via
# eval_agent_repair.py (same patch; repair recorder inert without --recording flag).

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
REFERENCE="$MAIN/runs/wbt_validation/gmr/walk1_subject5_mj.npz"
OUT="$MAIN/runs/e51_bodyfix"
EVAL_SEED=404

export PYTHONPATH="$MAIN"
test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
mkdir -p "$OUT/reports"
git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
cp "$0" "$OUT/protocol.sh"

run_arm() {
  local NAME="$1" W="$2"
  echo "=== $NAME (joint_w=$W) train start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  cd "$HOLOSOMA"
  E51_JOINT_POS_WEIGHT="$W" E51_JOINT_POS_SIGMA=0.5 nice -n 15 "$PY" \
    "$MAIN/scripts/train_agent_joint_reward.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 0 \
    --algo.config.num-learning-iterations 8000 --algo.config.save-interval 2000 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE" \
    --training.name "$NAME" >> "$OUT/$NAME.train.log" 2>&1
  local RUN_DIR CHECKPOINT
  RUN_DIR=$(ls -1dt "$HOLOSOMA"/logs/WholeBodyTracking/*-"$NAME"-locomotion | head -1)
  CHECKPOINT="$RUN_DIR/model_07999.pt"
  test -f "$CHECKPOINT"
  echo "=== $NAME eval start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  "$PY" "$MAIN/scripts/eval_agent_repair.py" \
    --checkpoint "$CHECKPOINT" \
    --wbt-metrics.config.enabled \
    --wbt-metrics.config.output-path "$OUT/reports/${NAME}_eval${EVAL_SEED}.json" \
    --wbt-metrics.config.horizon-s 10.0 \
    --recording.config.enabled \
    --recording.config.output-path "$OUT/${NAME}_recording.npz" \
    --training.headless True --training.num-envs 100 --training.seed "$EVAL_SEED" \
    --training.max-eval-steps 500 --training.export-onnx False \
    --simulator.config.sim.max-episode-length-s 100000.0 \
    >> "$OUT/$NAME.eval.log" 2>&1
  test -f "$OUT/reports/${NAME}_eval${EVAL_SEED}.json"
  sha256sum "$CHECKPOINT" "$OUT/reports/${NAME}_eval${EVAL_SEED}.json" >> "$OUT/input_sha256.txt"
  echo "=== $NAME done $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
}

run_arm e51v2_bodyfix_baseline_seed0 0.0
run_arm e51v2_bodyfix_jointrew_seed0 1.0

# Stage-A style export for both arms (heading-local MPJPE vs reference, sim contact mask)
for NAME in e51v2_bodyfix_baseline_seed0 e51v2_bodyfix_jointrew_seed0; do
  /home/ec2-user/work/retarget/.venv-snmr/bin/python "$MAIN/scripts/export_e50_repaired_pairs.py" \
    --recording "$OUT/${NAME}_recording.npz" \
    --reference "$REFERENCE" \
    --report "$OUT/reports/${NAME}_eval${EVAL_SEED}.json" \
    --out "$OUT/${NAME}_export" | tee "$OUT/${NAME}_stage_a_summary.txt"
done

date -u +%FT%TZ > "$OUT/COMPLETE"
echo "E51v2 complete" | tee -a "$OUT/driver.log"
