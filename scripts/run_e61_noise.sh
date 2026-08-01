#!/usr/bin/env bash
set -euo pipefail
# E61: noise-redundancy sweep. Eval-only on existing C/D checkpoints (3 seeds each).
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REF="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/e61_noise"
export PYTHONPATH="$MAIN"
mkdir -p "$OUT"
cp "$0" "$OUT/protocol.sh"
declare -A CKPT_DIR=( [0]="$MAIN/runs/e52_v3" [1]="$MAIN/runs/e52_v3_seeds/seed1" [2]="$MAIN/runs/e52_v3_seeds/seed2" )
for SEED in 0 1 2; do
for ARM in c_prior_explicit d_prior_explicit_snmr; do
for SIG in 0 0.1 0.25 0.5 1.0; do
  TAG="${ARM}_seed${SEED}_sig${SIG}"
  E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="${CKPT_DIR[$SEED]}" \
  E52_EVAL_ONLY=1 E52_EVAL_NOISE_CMD="$SIG" \
  "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 100 --training.seed 404 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    --training.name "e61_$TAG" --training.headless True >> "$OUT/$TAG.log" 2>&1
  mv "${CKPT_DIR[$SEED]}/${ARM}_eval.json" "$OUT/${TAG}.json"
  echo "$TAG $(jq -r .completion_rate "$OUT/${TAG}.json")" | tee -a "$OUT/driver.log"
done; done; done
# symmetric control: corrupt z_ret only, arm D, sigma 1.0, seed 0
for SIG in 0.5 1.0 2.0; do
  TAG="d_zretnoise_seed0_sig${SIG}"
  E52_ARM=d_prior_explicit_snmr E52_TEACHER_CKPT="$TEACHER" E52_OUT="${CKPT_DIR[0]}" \
  E52_EVAL_ONLY=1 E52_EVAL_NOISE_ZRET="$SIG" \
  "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 100 --training.seed 404 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    --training.name "e61_$TAG" --training.headless True >> "$OUT/$TAG.log" 2>&1
  mv "${CKPT_DIR[0]}/d_prior_explicit_snmr_eval.json" "$OUT/${TAG}.json"
  echo "$TAG $(jq -r .completion_rate "$OUT/${TAG}.json")" | tee -a "$OUT/driver.log"
done
echo E61_DONE >> "$OUT/driver.log"
