#!/usr/bin/env bash
set -uo pipefail
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REF="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/e65_hold"
mkdir -p "$OUT"
ln -sf "$MAIN/runs/e52_v4/c_prior_explicit_student.pt" "$OUT/"
export PYTHONPATH="$MAIN"
cd "$HOLOSOMA"

run_cell () {  # SRC_DIR ARM K TAG EXTRA_ENV
  local SRC=$1 ARM=$2 K=$3 TAG=$4 DET=${5:-}
  [ -f "$OUT/${TAG}.json" ] && return 0
  local CDIR="$OUT/work_$TAG"; mkdir -p "$CDIR"
  ln -sf "$SRC/${ARM}_student.pt" "$CDIR/"
  local DETVAR=""
  [ -n "$DET" ] && DETVAR=1
  E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="$CDIR" E52_EVAL_ONLY=1 \
    E52_EVAL_HOLD_Z="$K" E52_DET="$DETVAR" \
    nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 404 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    --training.name "e65_${TAG}" --training.headless True > "$OUT/${TAG}.log" 2>&1
  [ -f "$CDIR/${ARM}_eval.json" ] && mv "$CDIR/${ARM}_eval.json" "$OUT/${TAG}.json"
  echo "$TAG: $(grep -o '"completion_rate": [0-9.]*' "$OUT/${TAG}.json" 2>/dev/null)" | tee -a "$OUT/driver.log"
}

for K in 1 2 5 10; do
  run_cell "$MAIN/runs/e52_v4" c_prior_explicit "$K" "cvae_k${K}"
  run_cell "$MAIN/runs/e62_deterministic" c_prior_explicit "$K" "det_k${K}" det
done
date -u +%FT%TZ > "$OUT/COMPLETE"
