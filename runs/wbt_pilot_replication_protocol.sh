#!/usr/bin/env bash
set -euo pipefail

MAIN_REV=f9761eb
HOLOSOMA_REV=38009aad61851d59277fa4ebaf4f54c44ec483f7
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
VAL="$MAIN/runs/wbt_validation"
SEED0="$MAIN/runs/wbt_pilot"
OUT="$MAIN/runs/wbt_pilot_replication"
ANALYZER="$MAIN/scripts/analyze_wbt_replication.py"
BASE_ANALYZER="$MAIN/scripts/analyze_wbt_pilot.py"

git -C "$MAIN" merge-base --is-ancestor "$MAIN_REV" HEAD
test -z "$(git -C "$MAIN" status --porcelain --untracked-files=no)"
test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
diff -u "$SEED0/holosoma_status.txt" <(git -C "$HOLOSOMA" status --porcelain)
test -f "$SEED0/COMPLETE"
jq -e '
  .passed == true
  and .interpretation == "single-seed descriptive pilot; no inferential tracking claim"
  and (.runs | length == 6)
' "$SEED0/analysis.json" >/dev/null
test "$(cat "$SEED0/holosoma_revision.txt")" = "$HOLOSOMA_REV"

test ! -e "$OUT"
mkdir -p "$OUT"
cp "$0" "$OUT/protocol.sh"
cp "$ANALYZER" "$OUT/analyze_wbt_replication.py"
cp "$BASE_ANALYZER" "$OUT/analyze_wbt_pilot.py"
cp "$SEED0/run_dirs.tsv" "$OUT/combined_run_dirs.tsv"
printf 'seeds=1,2\nenvs=1024\niterations=1000\n' > "$OUT/protocol.txt"
printf '%s\n' "$HOLOSOMA_REV" > "$OUT/holosoma_revision.txt"
git -C "$HOLOSOMA" status --porcelain > "$OUT/holosoma_status.txt"
git -C "$MAIN" rev-parse HEAD > "$OUT/launch_revision.txt"
sha256sum "$SEED0/analysis.json" > "$OUT/seed0_analysis.sha256"
sha256sum "$OUT/analyze_wbt_replication.py" > "$OUT/analysis_analyzer.sha256"

source /home/ec2-user/work/retarget/.venv-wbt/bin/activate
cd "$HOLOSOMA"

for SEED in 1 2; do
  for CLIP in walk1_subject5 dance2_subject4 fight1_subject3; do
    for SRC in gmr snmr; do
      NAME=pilot_${SRC}_${CLIP%%_*}_seed${SEED}
      echo "=== $NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
      python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt \
        simulator:mjwarp \
        logger:disabled \
        --training.num-envs 1024 \
        --training.seed "$SEED" \
        --algo.config.num-learning-iterations 1000 \
        --algo.config.save-interval 500 \
        --randomization.ignore-unsupported True \
        --command.setup-terms.motion-command.params.motion-config.motion-file \
          "$VAL/$SRC/${CLIP}_mj.npz" \
        --training.name "$NAME" \
        > "$OUT/$NAME.log" 2>&1
      RUN_DIR=$(find logs/WholeBodyTracking -mindepth 1 -maxdepth 1 -type d \
        -name "*-${NAME}-locomotion" -newer "$OUT/protocol.sh" | sort | tail -n 1)
      test -n "$RUN_DIR"
      test -f "$RUN_DIR/holosoma_config.yaml"
      test -f "$RUN_DIR/events.out.tfevents"*
      test -f "$RUN_DIR/model_00999.pt"
      printf '%s\t%s\n' "$NAME" "$(realpath "$RUN_DIR")" \
        | tee -a "$OUT/run_dirs.tsv" >> "$OUT/combined_run_dirs.tsv"
      echo "=== $NAME complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    done
  done
done

python "$OUT/analyze_wbt_replication.py" "$OUT/combined_run_dirs.tsv" \
  --output "$OUT/analysis.json" > "$OUT/analysis.txt"
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "WBT pilot replication complete" | tee -a "$OUT/driver.log"
