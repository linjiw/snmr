#!/usr/bin/env bash
set -euo pipefail

TRAIN_REV=4929924d6147b131f601ab5f6ac238a0fa45bae9
SCREEN_REV=e915bf9
EVALUATOR_SHA=76f736449b67bc8df4150bb8cfb2da44191a673abbd0d93c3af886cd46128cdd
ROOT=/home/ec2-user/work/retarget/snmr-gate1
MAIN=/home/ec2-user/work/retarget/snmr
PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python
PAIRS=/home/ec2-user/work/retarget/data/pairs/unitree_g1
SCREEN="$MAIN/runs/gate1_g1/screen"
OUT_ROOT="$MAIN/runs/gate1_g1/replication"

test "$(git -C "$ROOT" rev-parse HEAD)" = "$TRAIN_REV"
test -z "$(git -C "$ROOT" status --porcelain)"
git -C "$MAIN" merge-base --is-ancestor "$SCREEN_REV" HEAD
test -z "$(git -C "$MAIN" status --porcelain --untracked-files=no)"
test "$(cat "$SCREEN/ANALYSIS_STATUS")" = COMPLETE_PROMOTED_C4_C3
test "$(sha256sum "$MAIN/scripts/benchmark.py" | cut -d' ' -f1)" = "$EVALUATOR_SHA"
jq -e '
  .passed == true
  and .evaluator_sha256 == "76f736449b67bc8df4150bb8cfb2da44191a673abbd0d93c3af886cd46128cdd"
  and .promoted_arms == [
    "c4_teacher_velocity_seed0",
    "c3_stance_seed0"
  ]
' "$SCREEN/analysis.json" >/dev/null

test ! -e "$OUT_ROOT"
mkdir -p "$OUT_ROOT"
cp "$0" "$OUT_ROOT/protocol.sh"
cp "$MAIN/scripts/analyze_gate1_replication.py" "$OUT_ROOT/analyze_gate1_replication.py"
printf '%s\n' "$TRAIN_REV" > "$OUT_ROOT/trainer_revision.txt"
printf '%s\n' "$SCREEN_REV" > "$OUT_ROOT/screen_revision.txt"
printf '%s\n' "$EVALUATOR_SHA" > "$OUT_ROOT/evaluator_sha256.txt"
git -C "$MAIN" rev-parse HEAD > "$OUT_ROOT/launch_revision.txt"
sha256sum "$SCREEN/analysis.json" > "$OUT_ROOT/screen_analysis.sha256"
sha256sum "$OUT_ROOT/analyze_gate1_replication.py" > "$OUT_ROOT/analysis_analyzer.sha256"

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
  --steps 50000
  --eval_every 5000
  --ckpt_every 5000
  --diag_every 5000
)

run_arm() {
  local name=$1
  local seed=$2
  shift 2
  local out="$OUT_ROOT/$name"
  test ! -e "$out"
  mkdir -p "$out"
  echo "=== $name start $(date -u +%FT%TZ) ===" | tee -a "$OUT_ROOT/driver.log"
  (
    cd "$ROOT"
    "$PY" scripts/train_phase1.py \
      "${COMMON[@]}" \
      --seed "$seed" \
      "$@" \
      --out "$out"
  ) > "$out/train_stdout.log" 2>&1
  test "$(jq -r .status "$out/manifest.json")" = completed
  echo "=== $name train complete $(date -u +%FT%TZ) ===" | tee -a "$OUT_ROOT/driver.log"

  (
    cd "$MAIN"
    "$PY" scripts/benchmark.py \
      --ckpt "$out/ckpt.pt" \
      --robots unitree_g1 \
      --pairs_root /home/ec2-user/work/retarget/data/pairs \
      --window 192 \
      --windows_per_clip 6 \
      --timing_warmup 10 \
      --timing_repeats 30 \
      --bootstrap_samples 2000 \
      --bootstrap_seed 0 \
      --out "$out/benchmark" \
      --device cuda
  ) > "$out/benchmark_stdout.log" 2>&1
  test -f "$out/benchmark.json"
  echo "=== $name benchmark complete $(date -u +%FT%TZ) ===" | tee -a "$OUT_ROOT/driver.log"
}

for seed in 1 2; do
  run_arm "c0_seed${seed}" "$seed"
  run_arm "c3_stance_seed${seed}" "$seed" \
    --stance_velocity_weight 0.03
  run_arm "c4_teacher_velocity_seed${seed}" "$seed" \
    --teacher_velocity_weight 0.05 \
    --phase_balanced_velocity
done

date -u +%FT%TZ > "$OUT_ROOT/COMPLETE"
