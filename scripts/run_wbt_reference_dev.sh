#!/usr/bin/env bash
set -euo pipefail

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
SNMR_REF="$MAIN/runs/wbt_validation/snmr/walk1_subject5_mj.npz"
GMR_REF="$MAIN/runs/wbt_validation/gmr/walk1_subject5_mj.npz"
GMR_REPORT="$MAIN/runs/wbt_horizon_calibration/reports/horizon_gmr_walk1_seed0_to8000_iter8000_eval404.json"
ANALYZER="$MAIN/scripts/analyze_wbt_reference_dev.py"
ANALYZER_LIBS=(
  "$MAIN/scripts/analyze_wbt_horizon.py"
  "$MAIN/scripts/analyze_wbt_latent_pilot.py"
)
OUT="$MAIN/runs/wbt_reference_dev"
TRAIN_NAME=reference_dev_snmr_walk1_seed0_to8000
EVAL_NAME="${TRAIN_NAME}_eval404"
REPORT="$OUT/reports/$EVAL_NAME.json"
SNMR_REF_SHA=2b847f97b7cf3cf6aebef9ff8a80a4231729b01ffc2d61ff8ba928fb4d617c66
GMR_REF_SHA=3b06c53a10f9933a5a715789f65f982772592f6864edea75773818d852ec01df

test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
test -z "$(git -C "$MAIN" status --porcelain --untracked-files=no)"
test "$(sha256sum "$SNMR_REF" | cut -d' ' -f1)" = "$SNMR_REF_SHA"
test "$(sha256sum "$GMR_REF" | cut -d' ' -f1)" = "$GMR_REF_SHA"
jq -e '
  .passed == true
  and .seed == 404
  and .training_name == "horizon_gmr_walk1_seed0_to8000"
  and .completion_rate == 0.88
  and .num_rollouts == 100
' "$GMR_REPORT" >/dev/null

if [[ -e "$OUT" ]]; then
  test ! -e "$OUT/COMPLETE"
  cmp "$0" "$OUT/protocol.sh"
  cmp "$ANALYZER" "$OUT/analyze_wbt_reference_dev.py"
  for LIB in "${ANALYZER_LIBS[@]}"; do
    cmp "$LIB" "$OUT/$(basename "$LIB")"
  done
  test "$(cat "$OUT/holosoma_revision.txt")" = "$HOLOSOMA_REV"
else
  mkdir -p "$OUT/reports"
  cp "$0" "$OUT/protocol.sh"
  cp "$ANALYZER" "$OUT/analyze_wbt_reference_dev.py"
  for LIB in "${ANALYZER_LIBS[@]}"; do
    cp "$LIB" "$OUT/"
  done
  git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
  git -C "$HOLOSOMA" rev-parse HEAD > "$OUT/holosoma_revision.txt"
  git -C "$HOLOSOMA" status --porcelain > "$OUT/holosoma_status.txt"
  sha256sum "$SNMR_REF" "$GMR_REF" "$GMR_REPORT" > "$OUT/input_sha256.txt"
  sha256sum "$OUT/protocol.sh" "$OUT"/analyze_*.py \
    > "$OUT/code_sha256.txt"
  cat > "$OUT/protocol.txt" <<'EOF'
question=Can a from-scratch 8k WBT policy track the frozen SNMR walk1 reference well enough for a confirmatory source comparison?
source=SNMR decoded reference
clip=walk1_subject5
training_seed=0
evaluation_seed=404
iterations=8000
save_interval=2000
num_envs=1024
rollouts=100
horizon_steps=500
horizon_s=10.0
promotion_rule=SNMR completion >= 0.70
next_if_pass=from-scratch GMR vs SNMR, train seeds 0/1/2, eval seeds 404/405
next_if_fail=stop confirmatory source comparison and diagnose SNMR reference failures
gmr_context=existing 8k continuation policy at 0.88 completion; context only, not matched inference
EOF
fi

cd "$HOLOSOMA"
if [[ -s "$OUT/training.tsv" ]]; then
  IFS=$'\t' read -r RUN_DIR CHECKPOINT CHECKPOINT_SHA < "$OUT/training.tsv"
  test "$CHECKPOINT" = "$RUN_DIR/model_07999.pt"
  test -f "$CHECKPOINT"
  test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$CHECKPOINT_SHA"
else
  RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
    -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
    -exec test -f '{}/model_07999.pt' ';' -print | sort | tail -n 1)
  if [[ -z "$RUN_DIR" ]]; then
    echo "=== $TRAIN_NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    "$PY" src/holosoma/holosoma/train_agent.py \
      exp:g1-29dof-wbt \
      simulator:mjwarp \
      logger:disabled \
      --training.num-envs 1024 \
      --training.seed 0 \
      --algo.config.num-learning-iterations 8000 \
      --algo.config.save-interval 2000 \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$SNMR_REF" \
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
  CHECKPOINT="$RUN_DIR/model_07999.pt"
  CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)
  printf '%s\t%s\t%s\n' "$RUN_DIR" "$CHECKPOINT" "$CHECKPOINT_SHA" \
    > "$OUT/training.tsv"
fi

if [[ ! -f "$REPORT" ]]; then
  echo "=== $EVAL_NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  "$PY" src/holosoma/holosoma/eval_agent.py \
    --checkpoint "$CHECKPOINT" \
    --wbt-metrics.config.enabled \
    --wbt-metrics.config.output-path "$REPORT" \
    --wbt-metrics.config.horizon-s 10.0 \
    --training.headless True \
    --training.num-envs 100 \
    --training.seed 404 \
    --training.max-eval-steps 500 \
    --training.export-onnx False \
    --simulator.config.sim.max-episode-length-s 100000.0 \
    >> "$OUT/$EVAL_NAME.eval.log" 2>&1
  test -f "$REPORT"
  echo "=== $EVAL_NAME complete $(date -u +%FT%TZ) ===" \
    | tee -a "$OUT/driver.log"
fi

cd "$MAIN"
PYTHONPATH="$OUT" "$PY" "$OUT/analyze_wbt_reference_dev.py" \
  --run-dir "$RUN_DIR" \
  --checkpoint "$CHECKPOINT" \
  --report "$REPORT" \
  --snmr-reference "$SNMR_REF" \
  --gmr-report "$GMR_REPORT" \
  --gmr-reference "$GMR_REF" \
  --output "$OUT/analysis.json" \
  > "$OUT/analysis.txt"
VERDICT=$(jq -r .verdict "$OUT/analysis.json")
printf 'COMPLETE_%s\n' "$VERDICT" | tr '[:lower:]' '[:upper:]' \
  > "$OUT/ANALYSIS_STATUS"
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT reference development complete: $(cat "$OUT/ANALYSIS_STATUS")" \
  | tee -a "$OUT/driver.log"
