#!/usr/bin/env bash
set -euo pipefail

REV=4929924d6147b131f601ab5f6ac238a0fa45bae9
ROOT=/home/ec2-user/work/retarget/snmr-gate1
MAIN=/home/ec2-user/work/retarget/snmr
PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python
PAIRS=/home/ec2-user/work/retarget/data/pairs/unitree_g1
OUT_ROOT="$MAIN/runs/gate1_g1/diagnostics"
ANALYSIS="$OUT_ROOT/analysis.json"

test "$(git -C "$ROOT" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$ROOT" status --porcelain)"
test "$(cat "$OUT_ROOT/ANALYSIS_STATUS")" = RECALIBRATION_REQUIRED
test "$(jq -r '.arms[] | select(.arm == "c1_bce_seed0") | .terms.contact_bce.suggested_recalibrated_weight' "$ANALYSIS")" = 0.25
test "$(jq -r '.arms[] | select(.arm == "c2_edge_seed0") | .terms.contact_bce.suggested_recalibrated_weight' "$ANALYSIS")" = 0.25
test "$(jq -r '.arms[] | select(.arm == "c2_edge_seed0") | .terms.edge_velocity.suggested_recalibrated_weight' "$ANALYSIS")" = 0.021268901529295597

COMMON=(
  --robot unitree_g1
  --pairs_dir "$PAIRS"
  --window 64
  --lr 0.0003
  --min_lr 0.00001
  --latent_dim 128
  --enc_hidden 256
  --dec_hidden 256
  --contact_mask teacher_height
  --contact_weight 0
  --foot_vel_weight 0
  --device cuda
  --steps 5000
  --eval_every 1000
  --ckpt_every 5000
  --diag_every 500
  --seed 0
)

run_arm() {
  local name=$1
  shift
  local out="$OUT_ROOT/$name"
  test ! -e "$out"
  mkdir -p "$out"
  echo "=== $name start $(date -u +%FT%TZ) ===" | tee "$out/driver.log"
  (
    cd "$ROOT"
    "$PY" scripts/train_phase1.py "${COMMON[@]}" "$@" --out "$out"
  ) > "$out/train_stdout.log" 2>&1
  echo "=== $name complete $(date -u +%FT%TZ) ===" | tee -a "$out/driver.log"
}

run_arm c1_bce_seed0-r1 --contact_bce_weight 0.25
run_arm c2_edge_seed0-r1 \
  --contact_bce_weight 0.25 \
  --edge_velocity_weight 0.021268901529295597

if ! "$PY" "$OUT_ROOT/analyze_gate1_diagnostics.py" "$OUT_ROOT" \
  --output "$OUT_ROOT/analysis.json" > "$OUT_ROOT/analysis.txt"; then
  printf 'RECALIBRATION_REQUIRED\n' > "$OUT_ROOT/ANALYSIS_STATUS"
elif test "$(jq -r .passed "$OUT_ROOT/analysis.json")" = true; then
  printf 'PASS\n' > "$OUT_ROOT/ANALYSIS_STATUS"
else
  printf 'COMPLETE_WITH_DROPPED_ARMS\n' > "$OUT_ROOT/ANALYSIS_STATUS"
fi
