#!/usr/bin/env bash
set -euo pipefail

# Phase 1 of docs/WBT_LATENT_PLAN_v2.md: attribution + load-bearing arms.
#   c1_explicit_preview — GMR command + explicit future joint_pos(+0.2s,+0.5s); attribution
#                         control for the S2 latent preview (GMT-style).
#   l1_latent_command   — actor sees latent-only command [z, dz0.2, dz0.5], NO explicit joint
#                         command; critic keeps explicit command + latent preview (privileged).
#                         Promotion rule: absolute completion floor 0.70 (not baseline-relative).
# Same clip/seed/budget/eval as Phase 0 (runs/wbt_latent_pilot). E36 recorded Phase-0 results.

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PYTHON=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
REFERENCE="$MAIN/runs/wbt_validation/gmr/walk1_subject5_mj.npz"
AUGMENTED="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
BASELINE_REPORT="$MAIN/runs/wbt_horizon_calibration/reports/horizon_gmr_walk1_seed0_to8000_iter8000_eval404.json"
BASELINE_CHECKPOINT="$HOLOSOMA/logs/WholeBodyTracking/20260714_162242-horizon_gmr_walk1_seed0_to8000-locomotion/model_07999.pt"
OUT="$MAIN/runs/wbt_latent_phase1"
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
EXPLICIT_PREVIEW_FUNC=snmr.integration.wbt_latent:motion_command_with_explicit_preview
LATENT_COMMAND_FUNC=snmr.integration.wbt_latent:latent_preview_command
PREVIEW_FUNC=snmr.integration.wbt_latent:motion_command_with_latent_preview
ARMS=(c1_explicit_preview l1_latent_command)

declare -A TRAIN_NAMES=(
  [c1_explicit_preview]=latent_c1_explicit_preview_walk1_seed0_to8000
  [l1_latent_command]=latent_l1_latent_command_walk1_seed0_to8000
)
declare -A ACTOR_FUNCS=(
  [c1_explicit_preview]="$EXPLICIT_PREVIEW_FUNC"
  [l1_latent_command]="$LATENT_COMMAND_FUNC"
)
declare -A CRITIC_FUNCS=(
  [c1_explicit_preview]="$EXPLICIT_PREVIEW_FUNC"
  [l1_latent_command]="$PREVIEW_FUNC"
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
  cat > "$OUT/arms.json" <<EOF
{
  "c1_explicit_preview": {
    "training_name": "${TRAIN_NAMES[c1_explicit_preview]}",
    "actor_func": "$EXPLICIT_PREVIEW_FUNC",
    "critic_func": "$EXPLICIT_PREVIEW_FUNC"
  },
  "l1_latent_command": {
    "training_name": "${TRAIN_NAMES[l1_latent_command]}",
    "actor_func": "$LATENT_COMMAND_FUNC",
    "critic_func": "$PREVIEW_FUNC",
    "min_completion": 0.70
  }
}
EOF
  sha256sum "$OUT/protocol.sh" "$OUT/analyze_wbt_latent_pilot.py" \
    "$OUT/arms.json" > "$OUT/code_sha256.txt"
  cat > "$OUT/protocol.txt" <<EOF
question=Phase 1: does explicit preview explain latent-preview effects (C1), and can z carry the command (L1)?
plan=docs/WBT_LATENT_PLAN_v2.md section 5.2
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
c1_rule=frozen baseline-relative (floor -5pp, >=5% improvement)
l1_rule=absolute completion floor 0.70
predictions=C1>=baseline with gain >= S2's; L1 below baseline but >=70%; L1>=85% fast-tracks Phase 3
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
  --arms-json "$OUT/arms.json" \
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
echo "WBT latent phase1 complete: $(cat "$OUT/ANALYSIS_STATUS")" \
  | tee -a "$OUT/driver.log"
