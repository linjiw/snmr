#!/usr/bin/env bash
set -uo pipefail
# E61-v4 — clean noise-redundancy sweep on the post-DEFECT-2 v4 checkpoints (seed 0).
# The leak-era E61/E61b applied noise to full[:, :58] (wrong dims) while the decoder
# read the true reference — they tested nothing. Under fixed slicing, noise_cmd hits the
# real 64-d goal slice and the decoder has no leaked copy, so the redundancy hypothesis
# (z_ret = human-side channel independent of robot-space corruption) gets its first real
# test. Cells: sigma x {C, D} + z_ret-noise control on D. Eval-only (~4 min/cell).
# SAFETY: evals run from a scratch dir with symlinked checkpoints; runs/e52_v4 untouched.
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REF="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/e61v4_noise"
mkdir -p "$OUT"
ln -sf "$MAIN/runs/e52_v4/c_prior_explicit_student.pt" "$OUT/"
ln -sf "$MAIN/runs/e52_v4/d_prior_explicit_snmr_student.pt" "$OUT/"
ln -sf "$MAIN/runs/e52_v4/a_prior_snmr_student.pt" "$OUT/"
export PYTHONPATH="$MAIN"
cd "$HOLOSOMA"

run_cell () {  # ARM NOISE_CMD NOISE_ZRET
  local ARM=$1 NC=$2 NZ=$3
  local TAG="${ARM}_nc${NC}_nz${NZ}"
  [ -f "$OUT/${TAG}.json" ] && { echo "skip $TAG"; return 0; }
  E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="$OUT" E52_EVAL_ONLY=1 \
    E52_EVAL_NOISE_CMD="$NC" E52_EVAL_NOISE_ZRET="$NZ" \
    nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 404 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    --training.name "e61v4_${TAG}" --training.headless True > "$OUT/${TAG}.log" 2>&1
  if [ -f "$OUT/${ARM}_eval.json" ]; then
    mv "$OUT/${ARM}_eval.json" "$OUT/${TAG}.json"
    echo "$TAG: $(grep -o '"completion_rate": [0-9.]*' "$OUT/${TAG}.json")" | tee -a "$OUT/driver.log"
  else
    echo "$TAG: FAILED (no eval json)" | tee -a "$OUT/driver.log"
  fi
}

# Clean-baseline cells (noise 0 = should match v4 driver evals; also a slicing sanity check)
for SIG in 0 0.25 0.5 1.0; do
  run_cell c_prior_explicit "$SIG" 0
  run_cell d_prior_explicit_snmr "$SIG" 0
done
# Controls: corrupt z_ret channel on D (redundancy direction) and on A (dependence check)
run_cell d_prior_explicit_snmr 0 2.0
run_cell a_prior_snmr 0 1.0
date -u +%FT%TZ > "$OUT/COMPLETE"
