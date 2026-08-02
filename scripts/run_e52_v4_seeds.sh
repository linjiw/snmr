#!/usr/bin/env bash
set -euo pipefail
# E52 v4 seed replication (seeds 1,2; seed 0 = runs/e52_v4). Arms prioritized by
# decision value: C (headline bottleneck), A (interface-replacement) first; D (additive)
# next; B (control) last — B's seed-0 verdict gates whether more B seeds even matter.
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REFERENCE_Z="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/e52_v4_seeds"

export PYTHONPATH="$MAIN"
mkdir -p "$OUT"
git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
cp "$0" "$OUT/protocol.sh"

for ARM in c_prior_explicit a_prior_snmr d_prior_explicit_snmr b_prior_proprio; do
  for SEED in 1 2; do
    CDIR="$OUT/seed$SEED"
    [ -f "$CDIR/${ARM}_eval.json" ] && { echo "$ARM seed$SEED done" | tee -a "$OUT/driver.log"; continue; }
    mkdir -p "$CDIR"
    echo "=== E52v4 $ARM seed$SEED start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
    cd "$HOLOSOMA"
    E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="$CDIR" E52_ROUNDS=2000 \
      nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
      exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
      --training.num-envs 1024 --training.seed "$SEED" \
      --randomization.ignore-unsupported True \
      --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE_Z" \
      --training.name "e52v4_${ARM}_s${SEED}" --training.headless True \
      >> "$CDIR/${ARM}.train.log" 2>&1
    test -f "$CDIR/${ARM}_eval.json"
    echo "=== E52v4 $ARM seed$SEED done $(date -u +%FT%TZ): $(tr -d '\n' < "$CDIR/${ARM}_eval.json") ===" \
      | tee -a "$OUT/driver.log"
  done
done
date -u +%FT%TZ > "$OUT/COMPLETE"
