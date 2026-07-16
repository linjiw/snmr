#!/usr/bin/env bash
set -euo pipefail

MAIN=/home/ec2-user/work/retarget/snmr
PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python
PAIRS=/home/ec2-user/work/retarget/data/pairs
GMR=/home/ec2-user/work/retarget/GMR
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
OUT="$MAIN/runs/sharing_cost_screen"
STAGE=/tmp/snmr_sharing_cost_screen
CODE=/tmp/snmr_sharing_cost_code
ANALYZER="$MAIN/scripts/analyze_sharing_cost_screen.py"
ROBOTS=(
  unitree_g1
  booster_t1_29dof
  fourier_n1
  engineai_pm01
  stanford_toddy
)
MODE=${1:-launch}

ensure_code_worktree() {
  local revision
  revision=$(cat "$STAGE/launch_revision.txt")
  if [[ -e "$CODE" ]]; then
    test "$(git -C "$CODE" rev-parse HEAD)" = "$revision"
    test -z "$(git -C "$CODE" status --porcelain --untracked-files=no)"
  else
    git -C "$MAIN" worktree add --detach "$CODE" "$revision"
  fi
}

if [[ "$MODE" = launch ]]; then
  cd "$MAIN"
  test ! -e "$OUT"
  test -z "$(git status --porcelain --untracked-files=no)"
  if [[ ! -e "$STAGE" ]]; then
    mkdir -p "$STAGE"
    cp "$0" "$STAGE/protocol.sh"
    cp "$ANALYZER" "$STAGE/analyze_sharing_cost_screen.py"
    git rev-parse HEAD > "$STAGE/launch_revision.txt"
    sha256sum "$STAGE/protocol.sh" "$STAGE/analyze_sharing_cost_screen.py" \
      > "$STAGE/code_sha256.txt"
  fi
  ensure_code_worktree
  exec bash "$STAGE/protocol.sh" worker
fi

test "$MODE" = worker
test ! -e "$OUT"
test -d "$STAGE"
ensure_code_worktree
test "$(git -C "$CODE" rev-parse HEAD)" = "$(cat "$STAGE/launch_revision.txt")"
test -z "$(git -C "$CODE" status --porcelain --untracked-files=no)"

COMMON=(
  --pairs_root "$PAIRS"
  --window 64
  --robot_sampling balanced_combinations
  --lr 0.0003
  --min_lr 0.00001
  --latent_weight 1.0
  --contact_weight 0
  --contact_bce_weight 0
  --edge_velocity_weight 0
  --stance_velocity_weight 0
  --penetration_weight 0
  --teacher_velocity_weight 0
  --zr_decode_prob 0
  --latent_dim 128
  --diag_every 0
  --seed 0
  --device cuda
)

run_arm() {
  local arm=$1
  shift
  local expected_steps=$1
  shift
  local arm_out="$STAGE/$arm"
  local resume=()
  if [[ -f "$arm_out/manifest.json" ]] \
    && jq -e '.status == "completed" and .progress.completion_state == "completed"' \
      "$arm_out/manifest.json" >/dev/null \
    && [[ -f "$arm_out/final_eval.json" ]]; then
    return
  fi
  if [[ -f "$arm_out/ckpt.pt" ]]; then
    resume=(--resume)
  fi
  mkdir -p "$arm_out"
  echo "=== $arm start $(date -u +%FT%TZ) ===" | tee -a "$STAGE/driver.log"
  (
    cd "$CODE"
    SNMR_DATA_ROOT=/home/ec2-user/work/retarget/data \
    SNMR_GMR_ROOT="$GMR" \
    SNMR_HOLOSOMA_ROOT="$HOLOSOMA" \
      "$PY" scripts/train_phase2.py \
      "${COMMON[@]}" \
      "$@" \
      --out "$arm_out" \
      "${resume[@]}"
  ) >> "$arm_out/training.log" 2>&1
  jq -e --argjson steps "$expected_steps" '
    .status == "completed"
    and .progress.completion_state == "completed"
    and .progress.step == $steps
  ' "$arm_out/manifest.json" >/dev/null
  test -f "$arm_out/final_eval.json"
  echo "=== $arm complete $(date -u +%FT%TZ) ===" | tee -a "$STAGE/driver.log"
}

for ROBOT in "${ROBOTS[@]}"; do
  run_arm "specialist_${ROBOT}_seed0" \
    50000 \
    --steps 50000 \
    --robots "$ROBOT" \
    --robots_per_step 1 \
    --enc_hidden 256 \
    --dec_hidden 256 \
    --decoder_adapter_rank 0 \
    --eval_every 5000 \
    --ckpt_every 25000
done

run_arm shared_base_seed0 \
  125000 \
  --steps 125000 \
  --robots "${ROBOTS[@]}" \
  --robots_per_step 2 \
  --enc_hidden 256 \
  --dec_hidden 256 \
  --decoder_adapter_rank 0 \
  --eval_every 12500 \
  --ckpt_every 25000

run_arm shared_wide_seed0 \
  125000 \
  --steps 125000 \
  --robots "${ROBOTS[@]}" \
  --robots_per_step 2 \
  --enc_hidden 384 \
  --dec_hidden 384 \
  --decoder_adapter_rank 0 \
  --eval_every 12500 \
  --ckpt_every 25000

run_arm shared_adapter_seed0 \
  125000 \
  --steps 125000 \
  --robots "${ROBOTS[@]}" \
  --robots_per_step 2 \
  --enc_hidden 256 \
  --dec_hidden 256 \
  --decoder_adapter_rank 8 \
  --eval_every 12500 \
  --ckpt_every 25000

PYTHONPATH="$CODE" "$PY" "$STAGE/analyze_sharing_cost_screen.py" \
  --root "$STAGE" | tee "$STAGE/analysis.txt"
date -u +%FT%TZ > "$STAGE/COMPLETE"
git -C "$MAIN" worktree remove "$CODE"
mv "$STAGE" "$OUT"
