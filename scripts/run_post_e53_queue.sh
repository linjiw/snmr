#!/usr/bin/env bash
set -uo pipefail
# Sequential post-E53-2048 GPU queue: E62 (deterministic-encoder baseline, reviewer ask)
# then E57-B (harder-clips trackability). Waits for the E53 teacher checkpoint marker.
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REFERENCE_Z="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
export PYTHONPATH="$MAIN"

while [ ! -f "$MAIN/runs/e53_multiclip/teacher2048_ckpt.txt" ]; do sleep 600; done

# --- E62: deterministic 64-d goal encoder under identical DAgger recipe (walk1, seed 0)
OUT="$MAIN/runs/e62_deterministic"
mkdir -p "$OUT"
if [ ! -f "$OUT/c_prior_explicit_eval.json" ]; then
  cd "$HOLOSOMA"
  E52_ARM=c_prior_explicit E52_TEACHER_CKPT="$TEACHER" E52_OUT="$OUT" E52_ROUNDS=2000 \
    E52_DET=1 \
    nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 0 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE_Z" \
    --training.name e62_det --training.headless True > "$OUT/train.log" 2>&1
  echo "E62 done: $(tr -d '\n' < "$OUT/c_prior_explicit_eval.json" 2>/dev/null | head -c 200)"
fi

# --- E57-B
bash "$MAIN/scripts/run_e57b_harder_clips.sh"
