#!/usr/bin/env bash
set -euo pipefail

# E54 Stage 0 — T1 explicit-command teacher. This supporting portability run must
# never compete with E67 or write a success marker after a failed command.
MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOLOSOMA="${SNMR_HOLOSOMA_ROOT:-$MAIN/../holosoma}"
PY="${WBT_PYTHON:-$HOLOSOMA/.venv/hsmujoco/bin/python}"
REF="${E54_REFERENCE:-$MAIN/runs/wbt_validation/t1_gmr/walk1_subject5_mj.npz}"
OUT="${E54_OUT:-/data/robotixx/snmr-research/e54_t1}"
LOG_ROOT="${E54_LOG_ROOT:-/data/robotixx/snmr-research/e54_t1/holosoma_logs}"
MIN_FREE_MB="${E54_MIN_FREE_MB:-20000}"
export PYTHONPATH="$MAIN"
export PYTHONNOUSERSITE=1

test -x "$PY"
test -f "$REF"
test -d "$HOLOSOMA"
mkdir -p "$OUT" "$LOG_ROOT"
if [[ -e "$OUT/protocol.sh" ]]; then
  cmp "$0" "$OUT/protocol.sh"
else
  cp "$0" "$OUT/protocol.sh"
fi

# Fail before allocating a simulator if the local Holosoma clone has lost the custom
# T1 WBT registration (the current upstream checkout may contain T1 locomotion only).
T1_HELP=$("$PY" "$MAIN/scripts/train_agent_joint_reward.py" --help 2>&1)
if [[ "$T1_HELP" != *"exp:t1-29dof-wbt"* ]]; then
  echo "E54 blocked: exp:t1-29dof-wbt is not registered in $HOLOSOMA" >&2
  exit 2
fi

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
  | awk 'NR == 1 { print $1 }')
if [[ ! "$FREE_MB" =~ ^[0-9]+$ ]] || (( FREE_MB < MIN_FREE_MB )); then
  echo "E54 refused: GPU has ${FREE_MB:-unknown} MiB free; require $MIN_FREE_MB MiB" >&2
  exit 3
fi

cd "$HOLOSOMA"
if [[ ! -s "$OUT/teacher_ckpt.txt" ]]; then
  echo "=== E54 T1 teacher start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 nice -n 15 "$PY" \
    "$MAIN/scripts/train_agent_joint_reward.py" \
    exp:t1-29dof-wbt simulator:mjwarp logger:disabled \
    --logger.base-dir "$LOG_ROOT" \
    --training.num-envs 512 --training.seed 0 \
    --algo.config.num-learning-iterations 8000 --algo.config.save-interval 2000 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    --training.name e54_t1_teacher --training.headless True \
    >> "$OUT/teacher.train.log" 2>&1

  CHECKPOINT=$(find "$LOG_ROOT/WholeBodyTracking" -mindepth 2 -maxdepth 2 \
    -type f -path '*-e54_t1_teacher-locomotion/model_07999.pt' \
    -printf '%T@\t%p\n' | sort -nr | awk -F '\t' 'NR == 1 { print $2 }')
  test -n "$CHECKPOINT"
  test -s "$CHECKPOINT"
  printf '%s\n' "$CHECKPOINT" > "$OUT/teacher_ckpt.txt.tmp"
  mv "$OUT/teacher_ckpt.txt.tmp" "$OUT/teacher_ckpt.txt"

  "$PY" "$MAIN/scripts/eval_agent_repair.py" \
    --checkpoint "$CHECKPOINT" \
    --logger.base-dir "$LOG_ROOT" \
    --wbt-metrics.config.enabled \
    --wbt-metrics.config.output-path "$OUT/teacher_eval.json" \
    --wbt-metrics.config.horizon-s 10.0 \
    --training.headless True --training.num-envs 1024 --training.seed 404 \
    --training.max-eval-steps 500 --training.export-onnx False \
    --simulator.config.sim.max-episode-length-s 100000.0 \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    >> "$OUT/teacher.eval.log" 2>&1
fi

"$PY" - "$OUT/teacher_eval.json" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
completion = float(report["completion_rate"])
survival = float(report["mean_survival_s"])
if not math.isfinite(completion) or not math.isfinite(survival):
    raise SystemExit("nonfinite T1 teacher evaluation")
if completion < 0.80 or survival < 9.0:
    raise SystemExit(
        f"T1 teacher gate failed: completion={completion:.3f}, survival={survival:.2f}s"
    )
print(f"T1 teacher gate passed: completion={completion:.3f}, survival={survival:.2f}s")
PY

date -u +%FT%TZ > "$OUT/TEACHER_DONE.tmp"
mv "$OUT/TEACHER_DONE.tmp" "$OUT/TEACHER_DONE"
