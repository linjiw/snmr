#!/usr/bin/env bash
set -euo pipefail

# E50 Stage A — laundering premise check (docs/E50_PHYSICS_REPAIRED_TEACHER_PROTOCOL.md §2).
# Rolls out the best B1-confirmatory policy per source on its own walk1 reference with the
# repair recorder (all-env state capture), then exports segments + Stage-A metrics.
# GPU: one job at a time; do NOT launch while E49/WBT jobs run.

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
PY_SNMR=/home/ec2-user/work/retarget/.venv-snmr/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
OUT="$MAIN/runs/e50_stage_a"
EVALUATION_SEED=404
NUM_ROLLOUTS=100
HORIZON_STEPS=500
HORIZON_S=10.0

export PYTHONPATH="$MAIN"

# Best B1-confirmatory policies (runs/wbt_reference_confirmatory/analysis.json per_training_seed):
# GMR seed0 = 0.90 completion; SNMR seed2 = 0.91.
declare -A POLICY_DIR=(
  [gmr]="$HOLOSOMA/logs/WholeBodyTracking/20260721_230736-reference_confirm_gmr_walk1_seed0_to8000-locomotion"
  [snmr]="$HOLOSOMA/logs/WholeBodyTracking/20260722_141231-reference_confirm_snmr_walk1_seed2_to8000-locomotion"
)
declare -A REFERENCE=(
  [gmr]="$MAIN/runs/wbt_validation/gmr/walk1_subject5_mj.npz"
  [snmr]="$MAIN/runs/wbt_validation/snmr/walk1_subject5_mj.npz"
)

test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
mkdir -p "$OUT"
git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
git -C "$HOLOSOMA" rev-parse HEAD > "$OUT/holosoma_revision.txt"
cp "$0" "$OUT/protocol.sh"

for SOURCE in gmr snmr; do
  RUN_DIR="${POLICY_DIR[$SOURCE]}"
  CHECKPOINT="$RUN_DIR/model_07999.pt"
  test -f "$CHECKPOINT"
  test -f "${REFERENCE[$SOURCE]}"
  RECORDING="$OUT/${SOURCE}_walk1_recording.npz"
  REPORT="$OUT/${SOURCE}_walk1_rollouts.json"

  echo "=== E50-A $SOURCE eval start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  cd "$HOLOSOMA"
  nice -n 15 "$PY" "$MAIN/scripts/eval_agent_repair.py" \
    --checkpoint "$CHECKPOINT" \
    --wbt-metrics.config.enabled \
    --wbt-metrics.config.output-path "$REPORT" \
    --wbt-metrics.config.horizon-s "$HORIZON_S" \
    --recording.config.enabled \
    --recording.config.output-path "$RECORDING" \
    --training.headless True \
    --training.num-envs "$NUM_ROLLOUTS" \
    --training.seed "$EVALUATION_SEED" \
    --training.max-eval-steps "$HORIZON_STEPS" \
    --training.export-onnx False \
    --simulator.config.sim.max-episode-length-s 100000.0 \
    >> "$OUT/${SOURCE}_eval.log" 2>&1
  test -f "$RECORDING"
  test -f "$REPORT"
  sha256sum "$CHECKPOINT" "$RECORDING" "$REPORT" >> "$OUT/input_sha256.txt"

  echo "=== E50-A $SOURCE export start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  "$PY_SNMR" "$MAIN/scripts/export_e50_repaired_pairs.py" \
    --recording "$RECORDING" \
    --reference "${REFERENCE[$SOURCE]}" \
    --report "$REPORT" \
    --out "$OUT/$SOURCE" \
    | tee "$OUT/${SOURCE}_stage_a_summary.txt"
  echo "=== E50-A $SOURCE done $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
done

date -u +%FT%TZ > "$OUT/COMPLETE"
echo "E50 Stage A complete; gates in $OUT/{gmr,snmr}/stage_a_metrics.json" | tee -a "$OUT/driver.log"
