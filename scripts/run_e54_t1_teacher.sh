#!/usr/bin/env bash
set -uo pipefail
# E54 Stage 0 — T1 explicit-command teacher on walk1 (GMR-qpos reference), using the
# freshly ported t1-29dof-wbt preset. Waits for E66 to finish (shares the GPU).
# Gate: >=0.5 completion (first-ever T1 WBT run; the G1 recipe may need tuning).
# If the teacher lands, E54 Stage 1 = the triad {explicit, clock, z_ret} on T1 with
# z_ret from the SHARED phase-2 all-5 checkpoint — the cross-embodiment command test:
# the SAME z_ret stream (from human motion) that commanded G1 now commands T1.
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
REF="$MAIN/runs/wbt_validation/t1_gmr/walk1_subject5_mj.npz"
OUT="$MAIN/runs/e54_t1"
export PYTHONPATH="$MAIN"
mkdir -p "$OUT"
cp "$0" "$OUT/protocol.sh"

while [ ! -f "$MAIN/runs/e66_aperiodic/COMPLETE" ]; do sleep 600; done
cd "$HOLOSOMA"
if [ ! -s "$OUT/teacher_ckpt.txt" ]; then
  echo "=== E54 T1 teacher start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 nice -n 15 "$PY" \
    "$MAIN/scripts/train_agent_joint_reward.py" \
    exp:t1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 512 --training.seed 0 \
    --algo.config.num-learning-iterations 8000 --algo.config.save-interval 2000 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REF" \
    --training.name e54_t1_teacher --training.headless True \
    >> "$OUT/teacher.train.log" 2>&1
  ls -1t "$HOLOSOMA"/logs/WholeBodyTracking/*-e54_t1_teacher-locomotion/model_07999.pt | head -1 > "$OUT/teacher_ckpt.txt"
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
  echo "T1 teacher: $(grep -o '"completion_rate": [0-9.]*' "$OUT/teacher_eval.json")" | tee -a "$OUT/driver.log"
fi
date -u +%FT%TZ > "$OUT/TEACHER_DONE"
