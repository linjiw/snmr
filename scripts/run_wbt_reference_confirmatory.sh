#!/usr/bin/env bash
set -euo pipefail

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
DEV="$MAIN/runs/wbt_reference_dev"
OUT="$MAIN/runs/wbt_reference_confirmatory"
ANALYZER="$MAIN/scripts/analyze_wbt_reference_confirmatory.py"
ANALYZER_DEV="$MAIN/scripts/analyze_wbt_reference_dev.py"
ANALYZER_HORIZON="$MAIN/scripts/analyze_wbt_horizon.py"
ANALYZER_ROLLOUT="$MAIN/scripts/analyze_wbt_latent_pilot.py"
GMR_REF="$MAIN/runs/wbt_validation/gmr/walk1_subject5_mj.npz"
SNMR_REF="$MAIN/runs/wbt_validation/snmr/walk1_subject5_mj.npz"
SOURCES=(gmr snmr)
TRAIN_SEEDS=(0 1 2)
EVAL_SEEDS=(404 405)

test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
test -z "$(git -C "$MAIN" status --porcelain --untracked-files=no)"
test -f "$DEV/COMPLETE"
jq -e '
  .passed == true
  and .verdict == "promote_confirmatory_matrix"
  and .decision.completion_floor_passed == true
  and .decision.completion_floor == 0.70
' "$DEV/analysis.json" >/dev/null

if [[ -e "$OUT" ]]; then
  test ! -e "$OUT/COMPLETE"
  cmp "$0" "$OUT/protocol.sh"
  cmp "$ANALYZER" "$OUT/analyze_wbt_reference_confirmatory.py"
  cmp "$ANALYZER_DEV" "$OUT/analyze_wbt_reference_dev.py"
  cmp "$ANALYZER_HORIZON" "$OUT/analyze_wbt_horizon.py"
  cmp "$ANALYZER_ROLLOUT" "$OUT/analyze_wbt_latent_pilot.py"
  test "$(cat "$OUT/holosoma_revision.txt")" = "$HOLOSOMA_REV"
else
  mkdir -p "$OUT/reports"
  cp "$0" "$OUT/protocol.sh"
  cp "$ANALYZER" "$OUT/analyze_wbt_reference_confirmatory.py"
  cp "$ANALYZER_DEV" "$OUT/analyze_wbt_reference_dev.py"
  cp "$ANALYZER_HORIZON" "$OUT/analyze_wbt_horizon.py"
  cp "$ANALYZER_ROLLOUT" "$OUT/analyze_wbt_latent_pilot.py"
  git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
  git -C "$HOLOSOMA" rev-parse HEAD > "$OUT/holosoma_revision.txt"
  git -C "$HOLOSOMA" status --porcelain > "$OUT/holosoma_status.txt"
  sha256sum "$GMR_REF" "$SNMR_REF" "$DEV/analysis.json" > "$OUT/input_sha256.txt"
  sha256sum "$OUT/protocol.sh" "$OUT"/analyze_*.py > "$OUT/code_sha256.txt"
  cat > "$OUT/protocol.txt" <<'EOF'
question=Is SNMR-reference WBT non-inferior to GMR-reference WBT on calibrated walk1 tracking?
sources=gmr,snmr
clip=walk1_subject5
training_seeds=0,1,2
evaluation_seeds=404,405
iterations=8000 from scratch for every policy
save_interval=2000
num_envs=1024
rollouts_per_cell=100
horizon_steps=500
horizon_s=10.0
completion_noninferiority_margin=-0.05
joint_rmse_relative_noninferiority_margin=+0.10
gmr_assay_completion_floor=0.50
bootstrap=10000 paired hierarchical replicates over training seeds, evaluation seeds, and phase windows
decision=both noninferiority confidence bounds and GMR assay floor must pass
scope=single-clip confirmatory result; evaluation seeds are rollout variability, not trained-policy replicates
EOF
fi

touch "$OUT/training.tsv" "$OUT/manifest.tsv"
cd "$HOLOSOMA"
for SOURCE in "${SOURCES[@]}"; do
  if [[ "$SOURCE" = gmr ]]; then
    REFERENCE="$GMR_REF"
  else
    REFERENCE="$SNMR_REF"
  fi
  for TRAIN_SEED in "${TRAIN_SEEDS[@]}"; do
    TRAIN_NAME="reference_confirm_${SOURCE}_walk1_seed${TRAIN_SEED}_to8000"
    TRAIN_ROW=$(awk -F $'\t' -v src="$SOURCE" -v seed="$TRAIN_SEED" \
      '$1 == src && $2 == seed { print }' "$OUT/training.tsv")
    if [[ -n "$TRAIN_ROW" ]]; then
      IFS=$'\t' read -r _ _ ROW_NAME RUN_DIR CHECKPOINT CHECKPOINT_SHA \
        <<< "$TRAIN_ROW"
      test "$ROW_NAME" = "$TRAIN_NAME"
      test "$CHECKPOINT" = "$RUN_DIR/model_07999.pt"
      test -f "$CHECKPOINT"
      test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$CHECKPOINT_SHA"
    else
      RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
        -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
        -exec test -f '{}/model_07999.pt' ';' -print | sort | tail -n 1)
      if [[ -z "$RUN_DIR" ]]; then
        echo "=== $TRAIN_NAME start $(date -u +%FT%TZ) ===" \
          | tee -a "$OUT/driver.log"
        "$PY" src/holosoma/holosoma/train_agent.py \
          exp:g1-29dof-wbt \
          simulator:mjwarp \
          logger:disabled \
          --training.num-envs 1024 \
          --training.seed "$TRAIN_SEED" \
          --algo.config.num-learning-iterations 8000 \
          --algo.config.save-interval 2000 \
          --randomization.ignore-unsupported True \
          --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE" \
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
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$SOURCE" "$TRAIN_SEED" "$TRAIN_NAME" "$RUN_DIR" "$CHECKPOINT" \
        "$CHECKPOINT_SHA" >> "$OUT/training.tsv"
    fi

    for EVAL_SEED in "${EVAL_SEEDS[@]}"; do
      EVAL_NAME="${TRAIN_NAME}_eval${EVAL_SEED}"
      REPORT="$OUT/reports/$EVAL_NAME.json"
      if awk -F $'\t' -v src="$SOURCE" -v ts="$TRAIN_SEED" -v es="$EVAL_SEED" \
        '$1 == src && $2 == ts && $3 == es { found = 1 } END { exit !found }' \
        "$OUT/manifest.tsv"; then
        test -f "$REPORT"
      else
        if [[ ! -f "$REPORT" ]]; then
          echo "=== $EVAL_NAME start $(date -u +%FT%TZ) ===" \
            | tee -a "$OUT/driver.log"
          "$PY" src/holosoma/holosoma/eval_agent.py \
            --checkpoint "$CHECKPOINT" \
            --wbt-metrics.config.enabled \
            --wbt-metrics.config.output-path "$REPORT" \
            --wbt-metrics.config.horizon-s 10.0 \
            --training.headless True \
            --training.num-envs 100 \
            --training.seed "$EVAL_SEED" \
            --training.max-eval-steps 500 \
            --training.export-onnx False \
            --simulator.config.sim.max-episode-length-s 100000.0 \
            >> "$OUT/$EVAL_NAME.eval.log" 2>&1
          test -f "$REPORT"
          echo "=== $EVAL_NAME complete $(date -u +%FT%TZ) ===" \
            | tee -a "$OUT/driver.log"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$SOURCE" "$TRAIN_SEED" "$EVAL_SEED" "$TRAIN_NAME" "$RUN_DIR" \
          "$CHECKPOINT" "$CHECKPOINT_SHA" "$REPORT" >> "$OUT/manifest.tsv"
      fi
    done
  done
done

cd "$OUT"
PYTHONPATH="$OUT" "$PY" "$OUT/analyze_wbt_reference_confirmatory.py" \
  --manifest "$OUT/manifest.tsv" \
  --gmr-reference "$GMR_REF" \
  --snmr-reference "$SNMR_REF" \
  --output "$OUT/analysis.json" > "$OUT/analysis.txt"
VERDICT=$(jq -r .verdict "$OUT/analysis.json")
printf 'COMPLETE_%s\n' "$VERDICT" | tr '[:lower:]' '[:upper:]' \
  > "$OUT/ANALYSIS_STATUS"
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT reference confirmatory complete: $(cat "$OUT/ANALYSIS_STATUS")" \
  | tee -a "$OUT/driver.log"
