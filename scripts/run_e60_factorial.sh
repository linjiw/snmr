#!/usr/bin/env bash
set -euo pipefail
# E60 — v2->v3 factorial (review blocker B6): un-confound {goal-conditioned prior} from
# {prior-path samples in the action loss}. Existing cells: (goal=explicit, mix=0) = v3 C
# 0.944; (goal=none, mix=0.5) = v2 B 0.021. This script runs the two missing cells:
#   cell GM: goal=explicit, mix=0.5  (was the mix alone the poison?)
#   cell NG: goal=none,     mix=0    (was goal-conditioning alone the cure?)
# 1 seed each, identical recipe/budget to v3 (2000 rounds @1024 envs, teacher E51-v2 A).

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REFERENCE_Z="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/e60_factorial"

export PYTHONPATH="$MAIN"
mkdir -p "$OUT"
git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
cp "$0" "$OUT/protocol.sh"

run_cell () {  # $1=cellname $2=arm $3=mix
  local CELL="$1" ARM="$2" MIX="$3"
  local CDIR="$OUT/$CELL"
  [ -f "$CDIR/${ARM}_eval.json" ] && { echo "cell $CELL already done"; return; }
  mkdir -p "$CDIR"
  echo "=== E60 cell $CELL (arm=$ARM mix=$MIX) start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  cd "$HOLOSOMA"
  E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="$CDIR" E52_ROUNDS=2000 \
    E52_PRIOR_MIX="$MIX" \
    nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 0 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE_Z" \
    --training.name "e60_${CELL}" --training.headless True \
    >> "$CDIR/train.log" 2>&1
  test -f "$CDIR/${ARM}_eval.json"
  echo "=== E60 cell $CELL done $(date -u +%FT%TZ): $(tr -d '\n' < "$CDIR/${ARM}_eval.json") ===" \
    | tee -a "$OUT/driver.log"
}

run_cell goal_mix   c_prior_explicit 0.5
run_cell nogoal_nomix b_prior_proprio 0
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "E60 complete" | tee -a "$OUT/driver.log"
