#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETARGET_ROOT="$(cd "$ROOT/.." && pwd)"
HOLOSOMA="${HOLOSOMA_ROOT:-$RETARGET_ROOT/holosoma}"
PY="${WBT_PYTHON:-$RETARGET_ROOT/.venv-wbt/bin/python}"
MOTION_ROOT="$ROOT/runs/wbt_validation"
LOG_ROOT="$MOTION_ROOT/gate2a_smoke"

if [[ ! -x "$PY" ]]; then
  printf 'WBT Python is not executable: %s\n' "$PY" >&2
  exit 1
fi

for src in snmr gmr; do
  motion="$MOTION_ROOT/$src/walk1_subject5_mj.npz"
  if [[ ! -f "$motion" ]]; then
    printf 'Missing motion file: %s\n' "$motion" >&2
    exit 1
  fi
done

mkdir -p "$LOG_ROOT"
cd "$HOLOSOMA"

"$PY" -c 'import torch, warp, mujoco_warp; assert torch.cuda.is_available()'

for src in snmr gmr; do
  log="$LOG_ROOT/${src}_seed0.log"
  if [[ -e "$log" ]]; then
    printf 'Refusing to overwrite existing smoke log: %s\n' "$log" >&2
    exit 1
  fi
  motion="$MOTION_ROOT/$src/walk1_subject5_mj.npz"
  "$PY" src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    simulator:mjwarp \
    logger:disabled \
    --training.seed 0 \
    --training.num-envs 64 \
    --training.name "gate2a_${src}_walk_seed0" \
    --training.export-onnx False \
    --algo.config.num-learning-iterations 100 \
    --algo.config.save-interval 100 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$motion" \
    2>&1 | tee "$log"
done
