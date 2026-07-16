#!/usr/bin/env bash
set -euo pipefail

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PYTHON=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
REFERENCE="$MAIN/runs/wbt_validation/gmr/walk1_subject5_mj.npz"
AUGMENTED="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
BASELINE_REPORT="$MAIN/runs/wbt_horizon_calibration/reports/horizon_gmr_walk1_seed0_to8000_iter8000_eval404.json"
BASELINE_CHECKPOINT="$HOLOSOMA/logs/WholeBodyTracking/20260714_162242-horizon_gmr_walk1_seed0_to8000-locomotion/model_07999.pt"
OUT="$MAIN/runs/wbt_latent_pilot"
ANALYZER="$MAIN/scripts/analyze_wbt_latent_pilot.py"
TRAINING_SEED=0
EVALUATION_SEED=404
ITERATIONS=8000
SAVE_INTERVAL=2000
NUM_ENVS=1024
NUM_ROLLOUTS=100
HORIZON_STEPS=500
HORIZON_S=10.0
BASE_FUNC=holosoma.managers.observation.terms.wbt:motion_command
CURRENT_FUNC=snmr.integration.wbt_latent:motion_command_with_current_latent
PREVIEW_FUNC=snmr.integration.wbt_latent:motion_command_with_latent_preview
ARMS=(s1_current_ac s2_preview_ac s3_preview_critic)

declare -A TRAIN_NAMES=(
  [s1_current_ac]=latent_s1_current_ac_walk1_seed0_to8000
  [s2_preview_ac]=latent_s2_preview_ac_walk1_seed0_to8000
  [s3_preview_critic]=latent_s3_preview_critic_walk1_seed0_to8000
)
declare -A ACTOR_FUNCS=(
  [s1_current_ac]="$CURRENT_FUNC"
  [s2_preview_ac]="$PREVIEW_FUNC"
  [s3_preview_critic]="$BASE_FUNC"
)
declare -A CRITIC_FUNCS=(
  [s1_current_ac]="$CURRENT_FUNC"
  [s2_preview_ac]="$PREVIEW_FUNC"
  [s3_preview_critic]="$PREVIEW_FUNC"
)

test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
test -f "$REFERENCE"
test -f "$AUGMENTED"
test -f "$BASELINE_REPORT"
test -f "$BASELINE_CHECKPOINT"
test -f "$ANALYZER"

if [[ -e "$OUT" ]]; then
  cmp "$0" "$OUT/protocol.sh"
  cmp "$ANALYZER" "$OUT/analyze_wbt_latent_pilot.py"
  test "$(cat "$OUT/holosoma_revision.txt")" = "$HOLOSOMA_REV"
else
  mkdir -p "$OUT/reports"
  cp "$0" "$OUT/protocol.sh"
  cp "$ANALYZER" "$OUT/analyze_wbt_latent_pilot.py"
  printf '%s\n' "$HOLOSOMA_REV" > "$OUT/holosoma_revision.txt"
  git -C "$HOLOSOMA" status --porcelain > "$OUT/holosoma_status.txt"
  git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
  git -C "$MAIN" status --porcelain > "$OUT/snmr_status.txt"
  sha256sum "$REFERENCE" "$AUGMENTED" "$BASELINE_REPORT" \
    "$BASELINE_CHECKPOINT" > "$OUT/input_sha256.txt"
  sha256sum "$OUT/protocol.sh" "$OUT/analyze_wbt_latent_pilot.py" \
    > "$OUT/code_sha256.txt"
  cat > "$OUT/protocol.txt" <<EOF
question=Can frozen SNMR latent observations improve GMR-reference WBT?
clip=walk1_subject5
training_seed=$TRAINING_SEED
evaluation_seed=$EVALUATION_SEED
iterations=$ITERATIONS
save_interval=$SAVE_INTERVAL
num_envs=$NUM_ENVS
rollouts=$NUM_ROLLOUTS
horizon_steps=$HORIZON_STEPS
horizon_s=$HORIZON_S
arms=${ARMS[*]}
completion_floor_delta=-0.05
promotion_improvement=0.05
interpretation=one-clip one-training-seed development screen
EOF
fi

export PYTHONPATH="$MAIN"
"$PYTHON" - "$REFERENCE" "$AUGMENTED" <<'PY'
import sys
from pathlib import Path

from scripts.analyze_wbt_latent_pilot import _validate_augmented_reference

_, errors = _validate_augmented_reference(Path(sys.argv[1]), Path(sys.argv[2]))
if errors:
    raise SystemExit("\n".join(errors))
PY

touch "$OUT/training_map.tsv" "$OUT/evaluation_map.tsv"
cd "$HOLOSOMA"

for ARM in "${ARMS[@]}"; do
  TRAIN_NAME="${TRAIN_NAMES[$ARM]}"
  ACTOR_FUNC="${ACTOR_FUNCS[$ARM]}"
  CRITIC_FUNC="${CRITIC_FUNCS[$ARM]}"
  TRAIN_ROW=$(awk -F $'\t' -v arm="$ARM" '$1 == arm { print }' \
    "$OUT/training_map.tsv")

  if [[ -n "$TRAIN_ROW" ]]; then
    IFS=$'\t' read -r _ ROW_NAME RUN_DIR CHECKPOINT CHECKPOINT_SHA \
      ROW_ACTOR ROW_CRITIC <<< "$TRAIN_ROW"
    test "$ROW_NAME" = "$TRAIN_NAME"
    test "$ROW_ACTOR" = "$ACTOR_FUNC"
    test "$ROW_CRITIC" = "$CRITIC_FUNC"
    test "$CHECKPOINT" = "$RUN_DIR/model_07999.pt"
    test -f "$CHECKPOINT"
    test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA"
  else
    RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
      -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
      -exec test -f '{}/model_07999.pt' ';' -print | sort | tail -n 1)
    if [[ -z "$RUN_DIR" ]]; then
      echo "=== $TRAIN_NAME start $(date -u +%FT%TZ) ===" \
        | tee -a "$OUT/driver.log"
      OBS_ARGS=(
        --observation.groups.actor-obs.terms.motion-command.func "$ACTOR_FUNC"
        --observation.groups.critic-obs.terms.motion-command.func "$CRITIC_FUNC"
      )
      "$PYTHON" src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt \
        simulator:mjwarp \
        logger:disabled \
        --training.num-envs "$NUM_ENVS" \
        --training.seed "$TRAINING_SEED" \
        --algo.config.num-learning-iterations "$ITERATIONS" \
        --algo.config.save-interval "$SAVE_INTERVAL" \
        --randomization.ignore-unsupported True \
        --command.setup-terms.motion-command.params.motion-config.motion-file \
          "$AUGMENTED" \
        "${OBS_ARGS[@]}" \
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
    CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$ARM" "$TRAIN_NAME" "$RUN_DIR" "$CHECKPOINT" "$CHECKPOINT_SHA" \
      "$ACTOR_FUNC" "$CRITIC_FUNC" >> "$OUT/training_map.tsv"
  fi

  EVAL_NAME="${TRAIN_NAME}_eval${EVALUATION_SEED}"
  REPORT="$OUT/reports/$EVAL_NAME.json"
  EVAL_ROW=$(awk -F $'\t' -v arm="$ARM" '$1 == arm { print }' \
    "$OUT/evaluation_map.tsv")
  if [[ -n "$EVAL_ROW" ]]; then
    test -f "$REPORT"
    continue
  fi

  echo "=== $EVAL_NAME start $(date -u +%FT%TZ) ===" \
    | tee -a "$OUT/driver.log"
  "$PYTHON" src/holosoma/holosoma/eval_agent.py \
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
  test -f "$REPORT"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ARM" "$TRAIN_NAME" "$REPORT" "$CHECKPOINT" "$CHECKPOINT_SHA" \
    "$ACTOR_FUNC" "$CRITIC_FUNC" >> "$OUT/evaluation_map.tsv"
  echo "=== $EVAL_NAME complete $(date -u +%FT%TZ) ===" \
    | tee -a "$OUT/driver.log"
done

"$PYTHON" "$OUT/analyze_wbt_latent_pilot.py" \
  --baseline-report "$BASELINE_REPORT" \
  --baseline-checkpoint "$BASELINE_CHECKPOINT" \
  --reference "$REFERENCE" \
  --augmented-reference "$AUGMENTED" \
  --evaluation-map "$OUT/evaluation_map.tsv" \
  --output "$OUT/analysis.json" > "$OUT/analysis.txt"

if [[ "$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' \
  "$OUT/analysis.json")" = "promote_for_multiseed_replication" ]]; then
  printf 'COMPLETE_PROMOTE_FOR_MULTISEED_REPLICATION\n' \
    > "$OUT/ANALYSIS_STATUS"
else
  printf 'COMPLETE_NO_ARM_MEETS_FROZEN_PROMOTION_RULE\n' \
    > "$OUT/ANALYSIS_STATUS"
fi
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT latent pilot complete: $(cat "$OUT/ANALYSIS_STATUS")" \
  | tee -a "$OUT/driver.log"
