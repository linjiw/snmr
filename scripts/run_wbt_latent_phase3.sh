#!/usr/bin/env bash
set -euo pipefail

# Phase 3 of docs/WBT_LATENT_PLAN_v2.md: multi-clip screen.
# Protocol: docs/WBT_LATENT_PHASE3_PROTOCOL.md (arm list resolved per its section 4 after E39).
#
# ARMS below must be edited ONCE, before first launch, to the E39-resolved set; the run
# directory then freezes that choice (protocol.sh cmp on re-entry). Do not edit after launch.

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PYTHON=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
TRAIN_DIR="$MAIN/runs/wbt_latent_gmr_multi/train"
HELDOUT_DIR="$MAIN/runs/wbt_latent_gmr_multi/heldout"
OUT="$MAIN/runs/wbt_latent_phase3"
ANALYZER="$MAIN/scripts/analyze_wbt_latent_phase3.py"
ANALYZER_LIBS=("$MAIN/scripts/analyze_wbt_latent_pilot.py" "$MAIN/scripts/analyze_wbt_latent_phase2.py")
TRAINING_SEED=0
EVALUATION_SEED=404
ITERATIONS=16000
SAVE_INTERVAL=4000
NUM_ENVS=1024
NUM_ROLLOUTS=100
HORIZON_STEPS=500
HORIZON_S=10.0
BASE_FUNC=holosoma.managers.observation.terms.wbt:motion_command
PREVIEW_FUNC=snmr.integration.wbt_latent:motion_command_with_latent_preview
LATENT_COMMAND_FUNC=snmr.integration.wbt_latent:latent_preview_command
EXPLICIT_PREVIEW_FUNC=snmr.integration.wbt_latent:motion_command_with_explicit_preview

# E39 resolution: L1 replicated; C3 matched S3 closely enough that the registered
# explicit-preview attribution control is required alongside S3.
ARMS=(b_multi s3_multi l1_multi c3_multi)

declare -A TRAIN_NAMES=(
  [b_multi]=latent_phase3_b_multi_seed0_to16000
  [s3_multi]=latent_phase3_s3_multi_seed0_to16000
  [l1_multi]=latent_phase3_l1_multi_seed0_to16000
  [c3_multi]=latent_phase3_c3_multi_seed0_to16000
)
declare -A ACTOR_FUNCS=(
  [b_multi]="$BASE_FUNC"
  [s3_multi]="$BASE_FUNC"
  [l1_multi]="$LATENT_COMMAND_FUNC"
  [c3_multi]="$BASE_FUNC"
)
declare -A CRITIC_FUNCS=(
  [b_multi]="$BASE_FUNC"
  [s3_multi]="$PREVIEW_FUNC"
  [l1_multi]="$PREVIEW_FUNC"
  [c3_multi]="$EXPLICIT_PREVIEW_FUNC"
)
declare -A ARM_RULES=(
  [b_multi]=baseline
  [s3_multi]=phase0_heldout
  [l1_multi]=descriptive
  [c3_multi]=phase0_heldout
)

# Evaluation clips: clip|split|npz
EVAL_CLIPS=(
  "walk1_subject5|heldout|$HELDOUT_DIR/walk1_subject5_mj_z.npz"
  "run2_subject1|heldout|$HELDOUT_DIR/run2_subject1_mj_z.npz"
  "walk1_subject2|trained|$TRAIN_DIR/walk1_subject2_mj_z.npz"
  "run1_subject2|trained|$TRAIN_DIR/run1_subject2_mj_z.npz"
)

test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
test -d "$TRAIN_DIR"; test -d "$HELDOUT_DIR"
test "$(ls "$TRAIN_DIR"/*.npz | wc -l)" -eq 6
test -f "$ANALYZER"

if [[ -e "$OUT" ]]; then
  cmp "$0" "$OUT/protocol.sh"
  cmp "$ANALYZER" "$OUT/analyze_wbt_latent_phase3.py"
  test "$(cat "$OUT/holosoma_revision.txt")" = "$HOLOSOMA_REV"
else
  mkdir -p "$OUT/reports"
  cp "$0" "$OUT/protocol.sh"
  cp "$ANALYZER" "$OUT/analyze_wbt_latent_phase3.py"
  for LIB in "${ANALYZER_LIBS[@]}"; do cp "$LIB" "$OUT/"; done
  printf '%s\n' "$HOLOSOMA_REV" > "$OUT/holosoma_revision.txt"
  git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
  git -C "$MAIN" status --porcelain > "$OUT/snmr_status.txt"
  sha256sum "$TRAIN_DIR"/*.npz "$HELDOUT_DIR"/*.npz > "$OUT/input_sha256.txt"
  {
    printf '{\n'
    FIRST=1
    for ARM in "${ARMS[@]}"; do
      [[ $FIRST -eq 0 ]] && printf ',\n'
      printf '  "%s": "%s"' "$ARM" "${ARM_RULES[$ARM]}"
      FIRST=0
    done
    printf '\n}\n'
  } > "$OUT/arm_rules.json"
  {
    printf '{\n'
    FIRST=1
    for ENTRY in "${EVAL_CLIPS[@]}"; do
      IFS='|' read -r CLIP _ NPZ <<< "$ENTRY"
      [[ $FIRST -eq 0 ]] && printf ',\n'
      printf '  "%s": "%s"' "$CLIP" "$NPZ"
      FIRST=0
    done
    printf '\n}\n'
  } > "$OUT/clip_motion_files.json"
  sha256sum "$OUT/protocol.sh" "$OUT/analyze_wbt_latent_phase3.py" \
    "$OUT/arm_rules.json" "$OUT/clip_motion_files.json" > "$OUT/code_sha256.txt"
  cat > "$OUT/protocol.txt" <<EOF
question=Phase 3: does the frozen SNMR latent help a multi-clip WBT policy on held-out clips?
protocol=docs/WBT_LATENT_PHASE3_PROTOCOL.md (arm list resolved per section 4; cite E39)
arms=${ARMS[*]}
train_motion_dir=$TRAIN_DIR (6 walk-family clips, 75184 frames)
heldout_clips=walk1_subject5 run2_subject1
trained_eval_clips=walk1_subject2 run1_subject2
training_seed=$TRAINING_SEED
evaluation_seed=$EVALUATION_SEED
iterations=$ITERATIONS
num_envs=$NUM_ENVS
rollouts_per_clip=$NUM_ROLLOUTS
horizon_s=$HORIZON_S
s3_rule=phase0 rule on held-out mean vs b_multi
l1_rule=descriptive; collapse flag at heldout completion < 0.40
interpretation=multi-clip single-training-seed screen
EOF
fi

export PYTHONPATH="$MAIN"
touch "$OUT/training_map.tsv" "$OUT/manifest.tsv"
cd "$HOLOSOMA"

for ARM in "${ARMS[@]}"; do
  TRAIN_NAME="${TRAIN_NAMES[$ARM]}"
  ACTOR_FUNC="${ACTOR_FUNCS[$ARM]}"
  CRITIC_FUNC="${CRITIC_FUNCS[$ARM]}"
  CKPT_FILE="model_$(printf '%05d' $((ITERATIONS - 1))).pt"
  TRAIN_ROW=$(awk -F $'\t' -v arm="$ARM" '$1 == arm { print }' \
    "$OUT/training_map.tsv")

  if [[ -n "$TRAIN_ROW" ]]; then
    IFS=$'\t' read -r _ ROW_NAME RUN_DIR CHECKPOINT CHECKPOINT_SHA <<< "$TRAIN_ROW"
    test "$ROW_NAME" = "$TRAIN_NAME"
    test -f "$CHECKPOINT"
    test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA"
  else
    RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
      -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
      -exec test -f "{}/$CKPT_FILE" ';' -print | sort | tail -n 1)
    if [[ -z "$RUN_DIR" ]]; then
      echo "=== $TRAIN_NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
      "$PYTHON" src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt \
        simulator:mjwarp \
        logger:disabled \
        --training.num-envs "$NUM_ENVS" \
        --training.seed "$TRAINING_SEED" \
        --algo.config.num-learning-iterations "$ITERATIONS" \
        --algo.config.save-interval "$SAVE_INTERVAL" \
        --randomization.ignore-unsupported True \
        --command.setup-terms.motion-command.params.motion-config.motion-dir \
          "$TRAIN_DIR" \
        --observation.groups.actor-obs.terms.motion-command.func "$ACTOR_FUNC" \
        --observation.groups.critic-obs.terms.motion-command.func "$CRITIC_FUNC" \
        --training.name "$TRAIN_NAME" \
        >> "$OUT/$TRAIN_NAME.train.log" 2>&1
      RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
        -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
        -exec test -f "{}/$CKPT_FILE" ';' -print | sort | tail -n 1)
      test -n "$RUN_DIR"
      echo "=== $TRAIN_NAME complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    fi
    RUN_DIR=$(realpath "$RUN_DIR")
    CHECKPOINT="$RUN_DIR/$CKPT_FILE"
    CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$ARM" "$TRAIN_NAME" "$RUN_DIR" "$CHECKPOINT" "$CHECKPOINT_SHA" \
      >> "$OUT/training_map.tsv"
  fi

  for ENTRY in "${EVAL_CLIPS[@]}"; do
    IFS='|' read -r CLIP SPLIT NPZ <<< "$ENTRY"
    if awk -F $'\t' -v a="$ARM" -v c="$CLIP" \
      '$1 == a && $2 == c { found = 1 } END { exit !found }' \
      "$OUT/manifest.tsv"; then
      continue
    fi
    EVAL_NAME="${TRAIN_NAME}_${CLIP}_eval${EVALUATION_SEED}"
    REPORT="$OUT/reports/$EVAL_NAME.json"
    if [[ ! -f "$REPORT" ]]; then
      echo "=== $EVAL_NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
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
        --command.setup-terms.motion-command.params.motion-config.motion-dir "" \
        --command.setup-terms.motion-command.params.motion-config.motion-file "$NPZ" \
        --simulator.config.sim.max-episode-length-s 100000.0 \
        >> "$OUT/$EVAL_NAME.eval.log" 2>&1
      test -f "$REPORT"
      echo "=== $EVAL_NAME complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$ARM" "$CLIP" "$SPLIT" "$EVALUATION_SEED" "$TRAIN_NAME" "$CHECKPOINT" \
      "$REPORT" >> "$OUT/manifest.tsv"
  done
done

cd "$MAIN/scripts"
"$PYTHON" "$OUT/analyze_wbt_latent_phase3.py" \
  --manifest "$OUT/manifest.tsv" \
  --arm-rules "$OUT/arm_rules.json" \
  --clip-motion-files "$OUT/clip_motion_files.json" \
  --output "$OUT/analysis.json" > "$OUT/analysis.txt"

VERDICT=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' \
  "$OUT/analysis.json")
printf 'COMPLETE_%s\n' "$VERDICT" | tr '[:lower:],:' '[:upper:]__' \
  > "$OUT/ANALYSIS_STATUS"
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT latent phase3 complete: $(cat "$OUT/ANALYSIS_STATUS")" \
  | tee -a "$OUT/driver.log"
