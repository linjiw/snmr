#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/work/retarget/snmr
PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python
ANALYZER="$ROOT/scripts/analyze_gate1b_m3.py"
PAIRS=/home/ec2-user/work/retarget/data/pairs/unitree_g1
OUT="$ROOT/runs/gate1b/m3_teacher_height_head_seed0"
WORK=/tmp/snmr_gate1b_m3_teacher_height_head_seed0
INIT="$ROOT/runs/gate1_g1/screen/c4_teacher_velocity_seed0/ckpt.pt"
BASE="$ROOT/runs/phase1_g1_large/ckpt_100k_final.pt"
M1B="$ROOT/runs/gate1b/windowed_c6_m1b_decoded_clip_height_full.json"
SUPPORT="$ROOT/runs/gate1b/m1b_support_audit.json"
INIT_SHA=e791cc98bf33a7e25331cfb96317b511097d626f1b17c8857e2b67b8cfb84cfa

cd "$ROOT"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$OUT"
test ! -e "$WORK"
test "$(sha256sum "$INIT" | cut -d' ' -f1)" = "$INIT_SHA"
jq -e '
  .summary.windowed.teacher_height_stance_speed_ms > 0.08
  and .summary.windowed.source_contact_stance_speed_ms > 0.10
  and .decisions.windowed.all_relative_guards_pass == false
' "$M1B" >/dev/null
jq -e '
  .aggregate.windows == 42
  and .aggregate.support_bearing_windows == 8
  and .aggregate.zero_support_windows == 34
' "$SUPPORT" >/dev/null

mkdir -p "$WORK"
cp "$0" "$WORK/protocol.sh"
cp "$ANALYZER" "$WORK/analyze_gate1b_m3.py"
git rev-parse HEAD > "$WORK/launch_revision.txt"
sha256sum "$INIT" > "$WORK/initial_checkpoint.sha256"
sha256sum "$M1B" > "$WORK/m1b_result.sha256"
sha256sum "$SUPPORT" > "$WORK/m1b_support.sha256"
sha256sum "$WORK/protocol.sh" "$WORK/analyze_gate1b_m3.py" \
  > "$WORK/code_sha256.txt"

"$PY" scripts/train_phase1.py \
  --robot unitree_g1 \
  --pairs_dir "$PAIRS" \
  --window 64 \
  --lr 0.0003 \
  --min_lr 0.00001 \
  --latent_dim 128 \
  --enc_hidden 256 \
  --dec_hidden 256 \
  --contact_mask teacher_height \
  --contact_weight 0 \
  --foot_vel_weight 0 \
  --contact_bce_weight 0.25 \
  --init_ckpt "$INIT" \
  --train_contact_head_only \
  --device cuda \
  --steps 50000 \
  --eval_every 5000 \
  --ckpt_every 5000 \
  --diag_every 5000 \
  --seed 0 \
  --out "$WORK" \
  > "$WORK/train_stdout.log" 2>&1

jq -e '
  .status == "completed"
  and .progress.completion_state == "completed"
  and .progress.step == 50000
  and .config.train_contact_head_only == true
  and .config.contact_bce_weight == 0.25
  and .training.frozen_backbone_eval_mode == true
' "$WORK/manifest.json" >/dev/null

"$PY" scripts/audit_contact_masks.py \
  --ckpt "$BASE" \
  --mask_ckpt "$WORK/ckpt.pt" \
  --window 192 \
  --windows_per_clip 6 \
  --device cpu \
  --out "$WORK/mask_audit.json" \
  > "$WORK/mask_audit_stdout.log" 2>&1

"$PY" scripts/eval_footlock.py \
  --ckpt "$BASE" \
  --mask_ckpt "$WORK/ckpt.pt" \
  --method windowed \
  --lock_mask predicted_contact \
  --contact_probability_threshold 0.5 \
  --window 192 \
  --windows_per_clip 6 \
  --bootstrap_samples 2000 \
  --bootstrap_seed 0 \
  --device cpu \
  --out "$WORK/windowed_projection.json" \
  > "$WORK/projection_stdout.log" 2>&1

"$PY" "$WORK/analyze_gate1b_m3.py" --root "$WORK"
date -u +%FT%TZ > "$WORK/COMPLETE"
mv "$WORK" "$OUT"
