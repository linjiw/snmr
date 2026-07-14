#!/usr/bin/env bash
set -euo pipefail

# C6 Stage-B pilot: paired SNMR-vs-GMR WBT training, identical config, MuJoCo/Warp backend.
# One seed per pair for the first pass (audit Gate 2B; extend with seeds after effect-size look).
source /home/ec2-user/work/retarget/.venv-wbt/bin/activate
cd /home/ec2-user/work/retarget/holosoma
VAL=/home/ec2-user/work/retarget/snmr/runs/wbt_validation
OUT=/home/ec2-user/work/retarget/snmr/runs/wbt_pilot
ANALYZER=/home/ec2-user/work/retarget/snmr/scripts/analyze_wbt_pilot.py
test ! -e "$OUT"
mkdir -p "$OUT"
cp "$0" "$OUT/protocol.sh"
cp "$ANALYZER" "$OUT/analyze_wbt_pilot.py"
printf 'seed=0\nenvs=1024\niterations=1000\n' > "$OUT/protocol.txt"
git rev-parse HEAD > "$OUT/holosoma_revision.txt"
git status --porcelain > "$OUT/holosoma_status.txt"

for CLIP in walk1_subject5 dance2_subject4 fight1_subject3; do
  for SRC in gmr snmr; do
    NAME=pilot_${SRC}_${CLIP%%_*}_seed0
    echo "=== $NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
      --training.num-envs 1024 \
      --training.seed 0 \
      --algo.config.num-learning-iterations 1000 \
      --algo.config.save-interval 500 \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$VAL/$SRC/${CLIP}_mj.npz" \
      --training.name "$NAME" \
      > "$OUT/$NAME.log" 2>&1
    RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
      -name "*-${NAME}-locomotion" -newer "$OUT/protocol.sh" | sort | tail -n 1)
    test -n "$RUN_DIR"
    test -f "$RUN_DIR/holosoma_config.yaml"
    test -f "$RUN_DIR/events.out.tfevents"* 
    printf '%s\t%s\n' "$NAME" "$(realpath "$RUN_DIR")" >> "$OUT/run_dirs.tsv"
    echo "=== $NAME complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  done
done
python "$OUT/analyze_wbt_pilot.py" "$OUT/run_dirs.tsv" \
  --output "$OUT/analysis.json" > "$OUT/analysis.txt"
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT pilot complete" | tee -a "$OUT/driver.log"
