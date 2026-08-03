#!/usr/bin/env bash
set -uo pipefail
# E61-v4 seed extension: same noise cells on the seed-1/2 v4 checkpoints -> 3-seed gate
# verdict for the redundancy trend (pre-specified: D-C >= +5pp at some sigma with
# non-overlapping seed ranges).
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REF="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
export PYTHONPATH="$MAIN"
cd "$HOLOSOMA"

for SEED in 1 2; do
  OUT="$MAIN/runs/e61v4_noise/seed$SEED"
  mkdir -p "$OUT"
  ln -sf "$MAIN/runs/e52_v4_seeds/seed$SEED/c_prior_explicit_student.pt" "$OUT/"
  ln -sf "$MAIN/runs/e52_v4_seeds/seed$SEED/d_prior_explicit_snmr_student.pt" "$OUT/"
  ln -sf "$MAIN/runs/e52_v4_seeds/seed$SEED/a_prior_snmr_student.pt" "$OUT/"
  for CELL in "c_prior_explicit 0.25 0" "d_prior_explicit_snmr 0.25 0" \
              "c_prior_explicit 0.5 0" "d_prior_explicit_snmr 0.5 0" \
              "c_prior_explicit 1.0 0" "d_prior_explicit_snmr 1.0 0" \
              "a_prior_snmr 0 1.0"; do
    set -- $CELL; ARM=$1; NC=$2; NZ=$3
    TAG="${ARM}_nc${NC}_nz${NZ}"
    [ -f "$OUT/${TAG}.json" ] && continue
    E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="$OUT" E52_EVAL_ONLY=1 \
      E52_EVAL_NOISE_CMD="$NC" E52_EVAL_NOISE_ZRET="$NZ" \
      nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
      exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
      --training.num-envs 1024 --training.seed 404 \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
      --training.name "e61v4s_${TAG}" --training.headless True > "$OUT/${TAG}.log" 2>&1
    if [ -f "$OUT/${ARM}_eval.json" ]; then
      mv "$OUT/${ARM}_eval.json" "$OUT/${TAG}.json"
      echo "seed$SEED $TAG: $(grep -o '"completion_rate": [0-9.]*' "$OUT/${TAG}.json")" | tee -a "$MAIN/runs/e61v4_noise/driver.log"
    else
      echo "seed$SEED $TAG: FAILED" | tee -a "$MAIN/runs/e61v4_noise/driver.log"
    fi
  done
done
date -u +%FT%TZ > "$MAIN/runs/e61v4_noise/SEEDS_COMPLETE"
