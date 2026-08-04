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

# --- E63: phase-only clock control (RL-reviewer's decisive ask). Same fetch path/dims
# as arm A but latents = fixed random projection of frame-index sinusoids. If ~0.6, arm
# A's 0.656 is an oracle clock; if <<0.6, z_ret content is established. 3 seeds.
for SEED in 0 1 2; do
  OUT="$MAIN/runs/e63_phase_only/seed$SEED"
  mkdir -p "$OUT"
  if [ ! -f "$OUT/a_prior_snmr_eval.json" ]; then
    cd "$HOLOSOMA"
    E52_ARM=a_prior_snmr E52_TEACHER_CKPT="$TEACHER" E52_OUT="$OUT" E52_ROUNDS=2000 \
      E52_PHASE_ONLY=1 \
      nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
      exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
      --training.num-envs 1024 --training.seed "$SEED" \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE_Z" \
      --training.name "e63_phase_s${SEED}" --training.headless True > "$OUT/train.log" 2>&1
    echo "E63 seed$SEED: $(tr -d '\n' < "$OUT/a_prior_snmr_eval.json" 2>/dev/null | head -c 160)"
  fi
done

# --- Teacher-bound precision (stats reviewer Q2): re-eval the SAME teacher checkpoint
# at 1024 rollouts so the "2.5pp below teacher" gap has a comparable-precision bound.
TOUT="$MAIN/runs/e51_teacher_1024eval"
mkdir -p "$TOUT"
if [ ! -f "$TOUT/eval404_1024.json" ]; then
  cd "$HOLOSOMA"
  nice -n 15 "$PY" "$MAIN/scripts/eval_agent_repair.py" \
    --checkpoint "$TEACHER" \
    --wbt-metrics.config.enabled \
    --wbt-metrics.config.output-path "$TOUT/eval404_1024.json" \
    --wbt-metrics.config.horizon-s 10.0 \
    --training.headless True --training.num-envs 1024 --training.seed 404 \
    --training.max-eval-steps 500 --training.export-onnx False \
    --simulator.config.sim.max-episode-length-s 100000.0 \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE_Z" \
    > "$TOUT/eval.log" 2>&1
  echo "teacher-1024: $(tr -d '\n' < "$TOUT/eval404_1024.json" 2>/dev/null | head -c 160)"
fi

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
