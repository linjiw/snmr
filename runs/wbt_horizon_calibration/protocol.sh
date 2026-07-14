#!/usr/bin/env bash
set -euo pipefail

MAIN_REV=7326fb9
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
VAL="$MAIN/runs/wbt_validation"
SOURCE="$MAIN/runs/wbt_pilot"
OUT="$MAIN/runs/wbt_horizon_calibration"
ANALYZER="$MAIN/scripts/analyze_wbt_horizon.py"
CLIPS=(walk1_subject5 dance2_subject4 fight1_subject3)
HORIZONS=(2000 4000 8000)
TRAINING_SEED=0
EVALUATION_SEED=404
SOURCE_ITERATIONS=1000
ADDITIONAL_ITERATIONS=7000
SAVE_INTERVAL=2000
NUM_ROLLOUTS=100
HORIZON_STEPS=500
HORIZON_S=10.0

git -C "$MAIN" merge-base --is-ancestor "$MAIN_REV" HEAD
test -z "$(git -C "$MAIN" status --porcelain --untracked-files=no)"
test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
test -f "$SOURCE/COMPLETE"
jq -e '
  .passed == true
  and .interpretation == "single-seed descriptive pilot; no inferential tracking claim"
  and (.runs | length == 6)
' "$SOURCE/analysis.json" >/dev/null

if [[ -e "$OUT" ]]; then
  test ! -e "$OUT/COMPLETE"
  cmp "$0" "$OUT/protocol.sh"
  cmp "$ANALYZER" "$OUT/analyze_wbt_horizon.py"
  test "$(cat "$OUT/holosoma_revision.txt")" = "$HOLOSOMA_REV"
else
  mkdir -p "$OUT/reports"
  cp "$0" "$OUT/protocol.sh"
  cp "$ANALYZER" "$OUT/analyze_wbt_horizon.py"
  printf '%s\n' "$HOLOSOMA_REV" > "$OUT/holosoma_revision.txt"
  git -C "$HOLOSOMA" status --porcelain > "$OUT/holosoma_status.txt"
  git -C "$MAIN" rev-parse HEAD > "$OUT/launch_revision.txt"
  sha256sum "$SOURCE/analysis.json" > "$OUT/source_analysis.sha256"
  sha256sum "$OUT/analyze_wbt_horizon.py" > "$OUT/analysis_analyzer.sha256"
  cat > "$OUT/protocol.txt" <<EOF
training_source=gmr
training_seed=$TRAINING_SEED
evaluation_seed=$EVALUATION_SEED
source_iterations=$SOURCE_ITERATIONS
additional_iterations=$ADDITIONAL_ITERATIONS
candidate_total_iterations=2000,4000,8000
save_interval=$SAVE_INTERVAL
rollouts_per_clip_horizon=$NUM_ROLLOUTS
horizon_steps=$HORIZON_STEPS
horizon_s=$HORIZON_S
pooled_completion_floor=0.50
per_clip_completion_floor=0.25
EOF
fi

touch "$OUT/training_map.tsv" "$OUT/eval_map.tsv"
source /home/ec2-user/work/retarget/.venv-wbt/bin/activate
cd "$HOLOSOMA"

for CLIP_FILE in "${CLIPS[@]}"; do
  CLIP="${CLIP_FILE%%_*}"
  SOURCE_NAME="pilot_gmr_${CLIP}_seed0"
  SOURCE_RUN=$(awk -F $'\t' -v name="$SOURCE_NAME" '$1 == name { print $2 }' \
    "$SOURCE/run_dirs.tsv")
  test -n "$SOURCE_RUN"
  SOURCE_CHECKPOINT="$SOURCE_RUN/model_00999.pt"
  test -f "$SOURCE_CHECKPOINT"
  SOURCE_SHA=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')
  TRAIN_NAME="horizon_gmr_${CLIP}_seed0_to8000"

  TRAIN_ROW=$(awk -F $'\t' -v name="$TRAIN_NAME" '$1 == name { print }' \
    "$OUT/training_map.tsv")
  if [[ -n "$TRAIN_ROW" ]]; then
    IFS=$'\t' read -r _ ROW_CLIP RUN_DIR ROW_SOURCE ROW_SOURCE_SHA <<< "$TRAIN_ROW"
    test "$ROW_CLIP" = "$CLIP"
    test "$ROW_SOURCE" = "$SOURCE_CHECKPOINT"
    test "$ROW_SOURCE_SHA" = "$SOURCE_SHA"
    test -f "$RUN_DIR/model_07999.pt"
  else
    RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
      -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
      -exec test -f '{}/model_07999.pt' ';' -print | sort | tail -n 1)
    if [[ -z "$RUN_DIR" ]]; then
      echo "=== $TRAIN_NAME start $(date -u +%FT%TZ) ===" \
        | tee -a "$OUT/driver.log"
      python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt \
        simulator:mjwarp \
        logger:disabled \
        --training.num-envs 1024 \
        --training.seed "$TRAINING_SEED" \
        --training.checkpoint "$SOURCE_CHECKPOINT" \
        --algo.config.num-learning-iterations "$ADDITIONAL_ITERATIONS" \
        --algo.config.save-interval "$SAVE_INTERVAL" \
        --randomization.ignore-unsupported True \
        --command.setup-terms.motion-command.params.motion-config.motion-file \
          "$VAL/gmr/${CLIP_FILE}_mj.npz" \
        --training.name "$TRAIN_NAME" \
        >> "$OUT/$TRAIN_NAME.train.log" 2>&1
      RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
        -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
        -exec test -f '{}/model_07999.pt' ';' -print | sort | tail -n 1)
      test -n "$RUN_DIR"
      echo "=== $TRAIN_NAME complete $(date -u +%FT%TZ) ===" \
        | tee -a "$OUT/driver.log"
    fi
    RUN_DIR=$(realpath "$RUN_DIR")
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$TRAIN_NAME" "$CLIP" "$RUN_DIR" "$SOURCE_CHECKPOINT" "$SOURCE_SHA" \
      >> "$OUT/training_map.tsv"
  fi

  for TOTAL in "${HORIZONS[@]}"; do
    CHECKPOINT="$RUN_DIR/model_$(printf '%05d' "$((TOTAL - 1))").pt"
    test -f "$CHECKPOINT"
    CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
    EVAL_NAME="${TRAIN_NAME}_iter${TOTAL}_eval${EVALUATION_SEED}"
    REPORT="$OUT/reports/$EVAL_NAME.json"
    EVAL_ROW=$(awk -F $'\t' -v name="$EVAL_NAME" '$1 == name { print }' \
      "$OUT/eval_map.tsv")
    if [[ -n "$EVAL_ROW" ]]; then
      test -f "$REPORT"
      continue
    fi
    if [[ ! -e "$REPORT" ]]; then
      echo "=== $EVAL_NAME start $(date -u +%FT%TZ) ===" \
        | tee -a "$OUT/driver.log"
      python src/holosoma/holosoma/eval_agent.py \
        --checkpoint "$CHECKPOINT" \
        --wbt-metrics.config.enabled \
        --wbt-metrics.config.output-path "$REPORT" \
        --wbt-metrics.config.horizon-s "$HORIZON_S" \
        --training.headless True \
        --training.num-envs "$NUM_ROLLOUTS" \
        --training.seed "$EVALUATION_SEED" \
        --training.max-eval-steps "$HORIZON_STEPS" \
        --training.export-onnx False \
        --simulator.config.sim.max-episode-length-s 100000.0 \
        >> "$OUT/$EVAL_NAME.eval.log" 2>&1
      echo "=== $EVAL_NAME complete $(date -u +%FT%TZ) ===" \
        | tee -a "$OUT/driver.log"
    fi
    jq -e \
      --arg train_name "$TRAIN_NAME" \
      --arg clip "$CLIP" \
      --argjson seed "$EVALUATION_SEED" \
      '
        .passed == true
        and .seed == $seed
        and .training_name == $train_name
        and .num_rollouts == 100
        and .horizon_steps == 500
        and .horizon_s == 10.0
        and (.rollouts | length) == 100
        and (.motion_file | split("/")[-2]) == "gmr"
        and (.motion_file | split("/")[-1] | startswith($clip + "_"))
      ' "$REPORT" >/dev/null
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$EVAL_NAME" "$TRAIN_NAME" "$CLIP" "$TOTAL" "$REPORT" \
      "$CHECKPOINT" "$CHECKPOINT_SHA" >> "$OUT/eval_map.tsv"
  done
done

python "$OUT/analyze_wbt_horizon.py" \
  "$OUT/training_map.tsv" "$OUT/eval_map.tsv" \
  --output "$OUT/analysis.json" > "$OUT/analysis.txt"
SELECTED=$(jq -r '.selected_horizon_iterations // "none"' "$OUT/analysis.json")
if [[ "$SELECTED" = "none" ]]; then
  printf 'COMPLETE_NO_VIABLE_HORIZON_THROUGH_8000\n' > "$OUT/ANALYSIS_STATUS"
else
  printf 'COMPLETE_PROMOTE_%s_ITERATIONS\n' "$SELECTED" > "$OUT/ANALYSIS_STATUS"
fi
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT horizon calibration complete: $(cat "$OUT/ANALYSIS_STATUS")" \
  | tee -a "$OUT/driver.log"
