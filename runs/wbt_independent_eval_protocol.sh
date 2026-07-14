#!/usr/bin/env bash
set -euo pipefail

MAIN_REV=fe72c9a4f954e86777e94db5c16425c2f8d3a22a
HOLOSOMA_REV=eebdcf428d6ff6b17113c221fcc42a9e51168dc2
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
TRAINING="$MAIN/runs/wbt_pilot_replication"
OUT="$MAIN/runs/wbt_independent_eval"
ANALYZER="$MAIN/scripts/analyze_wbt_rollouts.py"
EVAL_SEEDS=(101 202 303)
NUM_ROLLOUTS=100
HORIZON_STEPS=500
HORIZON_S=10.0

git -C "$MAIN" merge-base --is-ancestor "$MAIN_REV" HEAD
test -z "$(git -C "$MAIN" status --porcelain --untracked-files=no)"
test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
test -f "$TRAINING/COMPLETE"
jq -e '
  .passed == true
  and .interpretation == (
    "three-seed paired training-curve pilot; independent policy rollouts " +
    "are required before any non-inferiority or benefit claim"
  )
  and (.runs | length == 18)
' "$TRAINING/analysis.json" >/dev/null
test "$(wc -l < "$TRAINING/combined_run_dirs.tsv")" -eq 18

test ! -e "$OUT"
mkdir -p "$OUT/reports"
cp "$0" "$OUT/protocol.sh"
cp "$ANALYZER" "$OUT/analyze_wbt_rollouts.py"
printf 'evaluation_seeds=101,202,303\nrollouts_per_seed=%s\nhorizon_steps=%s\nhorizon_s=%s\ngmr_completion_floor=0.50\n' \
  "$NUM_ROLLOUTS" "$HORIZON_STEPS" "$HORIZON_S" > "$OUT/protocol.txt"
printf '%s\n' "$HOLOSOMA_REV" > "$OUT/holosoma_revision.txt"
git -C "$HOLOSOMA" status --porcelain > "$OUT/holosoma_status.txt"
git -C "$MAIN" rev-parse HEAD > "$OUT/launch_revision.txt"
sha256sum "$TRAINING/analysis.json" > "$OUT/training_analysis.sha256"
sha256sum "$OUT/analyze_wbt_rollouts.py" > "$OUT/analysis_analyzer.sha256"

source /home/ec2-user/work/retarget/.venv-wbt/bin/activate
cd "$HOLOSOMA"

while IFS=$'\t' read -r TRAIN_NAME RUN_DIR; do
  if [[ ! "$TRAIN_NAME" =~ ^pilot_(gmr|snmr)_(walk1|dance2|fight1)_seed([0-2])$ ]]; then
    echo "invalid training run name: $TRAIN_NAME" >&2
    exit 1
  fi
  SRC="${BASH_REMATCH[1]}"
  CLIP="${BASH_REMATCH[2]}"
  TRAIN_SEED="${BASH_REMATCH[3]}"
  CHECKPOINT="$RUN_DIR/model_00999.pt"
  test -f "$CHECKPOINT"

  for EVAL_SEED in "${EVAL_SEEDS[@]}"; do
    NAME="${TRAIN_NAME}_eval${EVAL_SEED}"
    REPORT="$OUT/reports/$NAME.json"
    echo "=== $NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    python src/holosoma/holosoma/eval_agent.py \
      --checkpoint "$CHECKPOINT" \
      --wbt-metrics.config.enabled \
      --wbt-metrics.config.output-path "$REPORT" \
      --wbt-metrics.config.horizon-s "$HORIZON_S" \
      --training.headless True \
      --training.num-envs "$NUM_ROLLOUTS" \
      --training.seed "$EVAL_SEED" \
      --training.max-eval-steps "$HORIZON_STEPS" \
      --training.export-onnx False \
      --simulator.config.sim.max-episode-length-s 100000.0 \
      > "$OUT/$NAME.log" 2>&1
    jq -e \
      --argjson seed "$EVAL_SEED" \
      --arg train_name "$TRAIN_NAME" \
      --arg motion_parent "$SRC" \
      --arg clip "$CLIP" \
      '
        .passed == true
        and .seed == $seed
        and .training_name == $train_name
        and .num_rollouts == 100
        and .horizon_steps == 500
        and .horizon_s == 10.0
        and (.rollouts | length) == 100
        and (.motion_file | split("/")[-2]) == $motion_parent
        and (.motion_file | split("/")[-1] | startswith($clip + "_"))
      ' "$REPORT" >/dev/null
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$NAME" "$TRAIN_NAME" "$SRC" "$CLIP" "$TRAIN_SEED" "$EVAL_SEED" \
      "$REPORT" "$CHECKPOINT" >> "$OUT/eval_map.tsv"
    echo "=== $NAME complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  done
done < "$TRAINING/combined_run_dirs.tsv"

python "$OUT/analyze_wbt_rollouts.py" "$OUT/eval_map.tsv" \
  --output "$OUT/analysis.json" > "$OUT/analysis.txt"
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT independent evaluation complete" | tee -a "$OUT/driver.log"
