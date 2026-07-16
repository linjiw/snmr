#!/usr/bin/env bash
set -euo pipefail

# Phase 2 of docs/WBT_LATENT_PLAN_v2.md (section 5.3): multi-seed replication.
#   base — GMR reference, default observations. Seeds 1,2 trained from scratch (seed 0 reuses
#          the E35 continuation checkpoint; provenance caveat recorded in protocol.txt).
#   s3   — latent preview to critic only (E36 near-miss). Seeds 1,2.
#   l1   — latent-only actor command (E37 promoted). Seeds 1,2.
#   c3   — EXPLICIT preview to critic only; seed 0 screening arm: the latent-vs-explicit
#          critic-preview attribution contrast for s3.
# Every checkpoint is evaluated with seeds 404 and 405 (100 stratified 10-s windows each).

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PYTHON=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
REFERENCE="$MAIN/runs/wbt_validation/gmr/walk1_subject5_mj.npz"
AUGMENTED="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/wbt_latent_phase2"
ANALYZER="$MAIN/scripts/analyze_wbt_latent_phase2.py"
ANALYZER_LIB="$MAIN/scripts/analyze_wbt_latent_pilot.py"
ITERATIONS=8000
SAVE_INTERVAL=2000
NUM_ENVS=1024
NUM_ROLLOUTS=100
HORIZON_STEPS=500
HORIZON_S=10.0
EVAL_SEEDS=(404 405)
BASE_FUNC=holosoma.managers.observation.terms.wbt:motion_command
PREVIEW_FUNC=snmr.integration.wbt_latent:motion_command_with_latent_preview
LATENT_COMMAND_FUNC=snmr.integration.wbt_latent:latent_preview_command
EXPLICIT_PREVIEW_FUNC=snmr.integration.wbt_latent:motion_command_with_explicit_preview

# Existing seed-0 artifacts (reused, not retrained).
BASE0_NAME=horizon_gmr_walk1_seed0_to8000
BASE0_CKPT="$HOLOSOMA/logs/WholeBodyTracking/20260714_162242-${BASE0_NAME}-locomotion/model_07999.pt"
BASE0_REPORT404="$MAIN/runs/wbt_horizon_calibration/reports/horizon_gmr_walk1_seed0_to8000_iter8000_eval404.json"
S30_NAME=latent_s3_preview_critic_walk1_seed0_to8000
S30_CKPT="$HOLOSOMA/logs/WholeBodyTracking/20260715_101236-${S30_NAME}-locomotion/model_07999.pt"
S30_REPORT404="$MAIN/runs/wbt_latent_pilot/reports/${S30_NAME}_eval404.json"
L10_NAME=latent_l1_latent_command_walk1_seed0_to8000
L10_CKPT="$HOLOSOMA/logs/WholeBodyTracking/20260715_145119-${L10_NAME}-locomotion/model_07999.pt"
L10_REPORT404="$MAIN/runs/wbt_latent_phase1/reports/${L10_NAME}_eval404.json"

# Training jobs: arm|train_seed|training_name|actor_func|critic_func|motion_file
TRAIN_JOBS=(
  "base|1|latent_phase2_base_walk1_seed1_to8000|$BASE_FUNC|$BASE_FUNC|$REFERENCE"
  "base|2|latent_phase2_base_walk1_seed2_to8000|$BASE_FUNC|$BASE_FUNC|$REFERENCE"
  "s3|1|latent_s3_preview_critic_walk1_seed1_to8000|$BASE_FUNC|$PREVIEW_FUNC|$AUGMENTED"
  "s3|2|latent_s3_preview_critic_walk1_seed2_to8000|$BASE_FUNC|$PREVIEW_FUNC|$AUGMENTED"
  "l1|1|latent_l1_latent_command_walk1_seed1_to8000|$LATENT_COMMAND_FUNC|$PREVIEW_FUNC|$AUGMENTED"
  "l1|2|latent_l1_latent_command_walk1_seed2_to8000|$LATENT_COMMAND_FUNC|$PREVIEW_FUNC|$AUGMENTED"
  "c3|0|latent_c3_explicit_preview_critic_walk1_seed0_to8000|$BASE_FUNC|$EXPLICIT_PREVIEW_FUNC|$AUGMENTED"
)
# Reused checkpoints: arm|train_seed|training_name|checkpoint|motion_file
REUSED_CKPTS=(
  "base|0|$BASE0_NAME|$BASE0_CKPT|$REFERENCE"
  "s3|0|$S30_NAME|$S30_CKPT|$AUGMENTED"
  "l1|0|$L10_NAME|$L10_CKPT|$AUGMENTED"
)

test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
test -f "$REFERENCE"
test -f "$AUGMENTED"
test -f "$ANALYZER"
test -f "$ANALYZER_LIB"
test -f "$BASE0_CKPT"; test -f "$BASE0_REPORT404"
test -f "$S30_CKPT"; test -f "$S30_REPORT404"
test -f "$L10_CKPT"; test -f "$L10_REPORT404"

if [[ -e "$OUT" ]]; then
  cmp "$0" "$OUT/protocol.sh"
  cmp "$ANALYZER" "$OUT/analyze_wbt_latent_phase2.py"
  cmp "$ANALYZER_LIB" "$OUT/analyze_wbt_latent_pilot.py"
  test "$(cat "$OUT/holosoma_revision.txt")" = "$HOLOSOMA_REV"
else
  mkdir -p "$OUT/reports"
  cp "$0" "$OUT/protocol.sh"
  cp "$ANALYZER" "$OUT/analyze_wbt_latent_phase2.py"
  cp "$ANALYZER_LIB" "$OUT/analyze_wbt_latent_pilot.py"
  printf '%s\n' "$HOLOSOMA_REV" > "$OUT/holosoma_revision.txt"
  git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
  git -C "$MAIN" status --porcelain > "$OUT/snmr_status.txt"
  sha256sum "$REFERENCE" "$AUGMENTED" "$BASE0_CKPT" "$S30_CKPT" "$L10_CKPT" \
    > "$OUT/input_sha256.txt"
  cat > "$OUT/arm_rules.json" <<'EOF'
{
  "base": {},
  "s3": {},
  "l1": {"min_completion": 0.70},
  "c3": {"screening": true}
}
EOF
  cat > "$OUT/motion_files.json" <<EOF
{
  "base": "$REFERENCE",
  "s3": "$AUGMENTED",
  "l1": "$AUGMENTED",
  "c3": "$AUGMENTED"
}
EOF
  sha256sum "$OUT/protocol.sh" "$OUT/analyze_wbt_latent_phase2.py" \
    "$OUT/analyze_wbt_latent_pilot.py" "$OUT/arm_rules.json" \
    "$OUT/motion_files.json" > "$OUT/code_sha256.txt"
  cat > "$OUT/protocol.txt" <<EOF
question=Phase 2: do S3 (critic latent preview) and L1 (latent-only command) replicate across training seeds; c3 = explicit-vs-latent critic-preview screening
plan=docs/WBT_LATENT_PLAN_v2.md section 5.3 (+ E36/E37 amendments in EXPERIMENT_LOG)
clip=walk1_subject5
training_seeds=base/s3/l1: {0 reused, 1, 2}; c3: {0}
evaluation_seeds=${EVAL_SEEDS[*]}
iterations=$ITERATIONS
num_envs=$NUM_ENVS
rollouts=$NUM_ROLLOUTS
horizon_s=$HORIZON_S
s3_rule=median cell meets frozen Phase-0 rule; no cell below -5pp completion floor
l1_rule=all cells >= 0.70 absolute completion
c3_rule=screening, descriptive only
caveat=base seed 0 is the E35 continuation run; base seeds 1,2 are from scratch like all arms
predictions=pre-registered: S3 replication plausible (E36 CIs excluded 0); L1 floor-fragile (72% is 2pp above floor); c3 tests whether S3's gain needs the latent
EOF
fi

export PYTHONPATH="$MAIN"
touch "$OUT/training_map.tsv" "$OUT/manifest.tsv"
cd "$HOLOSOMA"

# --- training ---
for JOB in "${TRAIN_JOBS[@]}"; do
  IFS='|' read -r ARM SEED TRAIN_NAME ACTOR_FUNC CRITIC_FUNC MOTION_FILE <<< "$JOB"
  KEY="${ARM}_s${SEED}"
  TRAIN_ROW=$(awk -F $'\t' -v key="$KEY" '$1 == key { print }' \
    "$OUT/training_map.tsv")
  if [[ -n "$TRAIN_ROW" ]]; then
    IFS=$'\t' read -r _ ROW_NAME RUN_DIR CHECKPOINT CHECKPOINT_SHA <<< "$TRAIN_ROW"
    test "$ROW_NAME" = "$TRAIN_NAME"
    test -f "$CHECKPOINT"
    test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA"
    continue
  fi
  RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
    -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
    -exec test -f '{}/model_07999.pt' ';' -print | sort | tail -n 1)
  if [[ -z "$RUN_DIR" ]]; then
    echo "=== $TRAIN_NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    "$PYTHON" src/holosoma/holosoma/train_agent.py \
      exp:g1-29dof-wbt \
      simulator:mjwarp \
      logger:disabled \
      --training.num-envs "$NUM_ENVS" \
      --training.seed "$SEED" \
      --algo.config.num-learning-iterations "$ITERATIONS" \
      --algo.config.save-interval "$SAVE_INTERVAL" \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-file \
        "$MOTION_FILE" \
      --observation.groups.actor-obs.terms.motion-command.func "$ACTOR_FUNC" \
      --observation.groups.critic-obs.terms.motion-command.func "$CRITIC_FUNC" \
      --training.name "$TRAIN_NAME" \
      >> "$OUT/$TRAIN_NAME.train.log" 2>&1
    RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
      -name "*-${TRAIN_NAME}-locomotion" -newer "$OUT/protocol.sh" \
      -exec test -f '{}/model_07999.pt' ';' -print | sort | tail -n 1)
    test -n "$RUN_DIR"
    echo "=== $TRAIN_NAME complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  fi
  RUN_DIR=$(realpath "$RUN_DIR")
  CHECKPOINT="$RUN_DIR/model_07999.pt"
  CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$KEY" "$TRAIN_NAME" "$RUN_DIR" "$CHECKPOINT" "$CHECKPOINT_SHA" \
    >> "$OUT/training_map.tsv"
done

# --- assemble the full checkpoint set (trained + reused) ---
ALL_CKPTS=()
for JOB in "${TRAIN_JOBS[@]}"; do
  IFS='|' read -r ARM SEED TRAIN_NAME _ _ MOTION_FILE <<< "$JOB"
  KEY="${ARM}_s${SEED}"
  CHECKPOINT=$(awk -F $'\t' -v key="$KEY" '$1 == key { print $4 }' \
    "$OUT/training_map.tsv")
  test -n "$CHECKPOINT"
  ALL_CKPTS+=("$ARM|$SEED|$TRAIN_NAME|$CHECKPOINT|$MOTION_FILE")
done
ALL_CKPTS+=("${REUSED_CKPTS[@]}")

# --- evaluation ---
for ENTRY in "${ALL_CKPTS[@]}"; do
  IFS='|' read -r ARM SEED TRAIN_NAME CHECKPOINT MOTION_FILE <<< "$ENTRY"
  for EVAL_SEED in "${EVAL_SEEDS[@]}"; do
    # Reuse existing seed-404 reports for the three seed-0 checkpoints.
    if [[ "$SEED" = 0 && "$EVAL_SEED" = 404 ]]; then
      case "$ARM" in
        base) REPORT="$BASE0_REPORT404" ;;
        s3)   REPORT="$S30_REPORT404" ;;
        l1)   REPORT="$L10_REPORT404" ;;
        *)    REPORT="$OUT/reports/${TRAIN_NAME}_eval${EVAL_SEED}.json" ;;
      esac
    else
      REPORT="$OUT/reports/${TRAIN_NAME}_eval${EVAL_SEED}.json"
    fi
    ROW_KEY="${ARM}	${SEED}	${EVAL_SEED}"
    if awk -F $'\t' -v a="$ARM" -v s="$SEED" -v e="$EVAL_SEED" \
      '$1 == a && $2 == s && $3 == e { found = 1 } END { exit !found }' \
      "$OUT/manifest.tsv"; then
      test -f "$REPORT"
      continue
    fi
    if [[ ! -f "$REPORT" ]]; then
      EVAL_NAME="${TRAIN_NAME}_eval${EVAL_SEED}"
      echo "=== $EVAL_NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
      "$PYTHON" src/holosoma/holosoma/eval_agent.py \
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
        >> "$OUT/$EVAL_NAME.eval.log" 2>&1
      test -f "$REPORT"
      echo "=== $EVAL_NAME complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$ARM" "$SEED" "$EVAL_SEED" "$TRAIN_NAME" "$CHECKPOINT" "$REPORT" \
      >> "$OUT/manifest.tsv"
  done
done

# --- analysis ---
cd "$MAIN/scripts"
"$PYTHON" "$OUT/analyze_wbt_latent_phase2.py" \
  --manifest "$OUT/manifest.tsv" \
  --arm-rules "$OUT/arm_rules.json" \
  --motion-files "$OUT/motion_files.json" \
  --output "$OUT/analysis.json" > "$OUT/analysis.txt"

VERDICT=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' \
  "$OUT/analysis.json")
printf 'COMPLETE_%s\n' "$VERDICT" | tr '[:lower:],:' '[:upper:]__' \
  > "$OUT/ANALYSIS_STATUS"
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT latent phase2 complete: $(cat "$OUT/ANALYSIS_STATUS")" \
  | tee -a "$OUT/driver.log"
