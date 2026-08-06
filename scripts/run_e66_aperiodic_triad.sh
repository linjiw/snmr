#!/usr/bin/env bash
set -uo pipefail
# E66 — THE DECISIVE TRIAD ON AN APERIODIC CLIP (cold-read reviewer's single change).
# push1 (box-push, aperiodic): teacher (E51 arm-A recipe) -> arms C (explicit), CLOCK
# (phase-only), A (z_ret-only) under the identical v4 DAgger recipe, seed 0 first.
# The question the clock null left open: can ANYTHING learned beat the clock where
# phase cannot suffice? Pre-specified readouts: completion(C) vs completion(clock) vs
# completion(A). If A > clock on aperiodic motion -> z_ret carries content beyond
# phase after all (single-clip-cyclic was the wrong instrument); if A ~ clock ->
# the latent-content negative generalizes; if clock collapses but A doesn't -> strongest
# H2 form. Teacher gate: >=0.5 completion on push1 (multi-clip teacher got 0.61 across
# starts; single-clip should exceed it).
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
REF="$MAIN/runs/wbt_latent_gmr_multi8/push1_subject2_mj_z.npz"
OUT="$MAIN/runs/e66_aperiodic"
export PYTHONPATH="$MAIN"
mkdir -p "$OUT"
cp "$0" "$OUT/protocol.sh"
cd "$HOLOSOMA"

# Stage 0: push1 explicit teacher
if [ ! -s "$OUT/teacher_ckpt.txt" ]; then
  echo "=== E66 teacher start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 nice -n 15 "$PY" \
    "$MAIN/scripts/train_agent_joint_reward.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 512 --training.seed 0 \
    --algo.config.num-learning-iterations 8000 --algo.config.save-interval 2000 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    --training.name e66_push1_teacher --training.headless True \
    >> "$OUT/teacher.train.log" 2>&1
  ls -1t "$HOLOSOMA"/logs/WholeBodyTracking/*-e66_push1_teacher-locomotion/model_07999.pt | head -1 > "$OUT/teacher_ckpt.txt"
  # teacher gate eval
  "$PY" "$MAIN/scripts/eval_agent_repair.py" \
    --checkpoint "$(cat "$OUT/teacher_ckpt.txt")" \
    --wbt-metrics.config.enabled \
    --wbt-metrics.config.output-path "$OUT/teacher_eval.json" \
    --wbt-metrics.config.horizon-s 10.0 \
    --training.headless True --training.num-envs 1024 --training.seed 404 \
    --training.max-eval-steps 500 --training.export-onnx False \
    --simulator.config.sim.max-episode-length-s 100000.0 \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    >> "$OUT/teacher.eval.log" 2>&1
  echo "teacher: $(grep -o '"completion_rate": [0-9.]*' "$OUT/teacher_eval.json")" | tee -a "$OUT/driver.log"
fi
TEACHER=$(cat "$OUT/teacher_ckpt.txt")
COMP=$(grep -o '"completion_rate": [0-9.]*' "$OUT/teacher_eval.json" | grep -o '[0-9.]*')
GATE_OK=$("$PY" -c "print(1 if $COMP >= 0.5 else 0)")
[ "$GATE_OK" = "0" ] && { echo "TEACHER GATE FAILED ($COMP < 0.5)" | tee -a "$OUT/driver.log"; exit 0; }

# Stage 1: the triad (identical v4 recipe, seed 0)
run_arm () {  # ARM PHASE_FLAG TAG
  local ARM=$1 PH=$2 TAG=$3
  [ -f "$OUT/$TAG/${ARM}_eval.json" ] && return 0
  mkdir -p "$OUT/$TAG"
  echo "=== E66 $TAG start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="$OUT/$TAG" E52_ROUNDS=2000 \
    E52_PHASE_ONLY="$PH" \
    nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 0 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    --training.name "e66_${TAG}" --training.headless True > "$OUT/$TAG/train.log" 2>&1
  echo "=== E66 $TAG done: $(tr -d '\n' < "$OUT/$TAG/${ARM}_eval.json" 2>/dev/null | head -c 160) ===" | tee -a "$OUT/driver.log"
}
run_arm c_prior_explicit "" explicit
run_arm a_prior_snmr 1 clock
run_arm a_prior_snmr "" zret
date -u +%FT%TZ > "$OUT/COMPLETE"
