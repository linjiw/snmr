#!/usr/bin/env bash
set -euo pipefail
# E52 v4 — post-DEFECT-2 rerun (fixed obs slicing: proprio=[0:90), goal=[90:154)).
# All four arms, seed 0, identical v3 recipe/budget. Order = decision value:
#   c: explicit-goal prior (headline bottleneck result — does it survive the fix?)
#   a: z_ret-only prior    (TRUE interface-replacement test, never actually run before)
#   b: proprio-only prior  (no-goal control)
#   d: explicit+z_ret      (additive test)
MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PY=/home/ec2-user/work/retarget/.venv-wbt/bin/python
TEACHER="$HOLOSOMA/logs/WholeBodyTracking/20260727_123641-e51v2_bodyfix_jointrew_seed0-locomotion/model_07999.pt"
REFERENCE_Z="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/e52_v4"

export PYTHONPATH="$MAIN"
mkdir -p "$OUT"
git -C "$MAIN" rev-parse HEAD > "$OUT/snmr_revision.txt"
cp "$0" "$OUT/protocol.sh"
sha256sum "$TEACHER" "$REFERENCE_Z" > "$OUT/input_sha256.txt"

for ARM in c_prior_explicit a_prior_snmr b_prior_proprio d_prior_explicit_snmr; do
  [ -f "$OUT/${ARM}_eval.json" ] && { echo "$ARM already done" | tee -a "$OUT/driver.log"; continue; }
  echo "=== E52v4 $ARM start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  cd "$HOLOSOMA"
  E52_ARM="$ARM" E52_TEACHER_CKPT="$TEACHER" E52_OUT="$OUT" E52_ROUNDS=2000 \
    nice -n 15 "$PY" "$MAIN/scripts/train_e52_dagger.py" \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 --training.seed 0 \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$REFERENCE_Z" \
    --training.name "e52v4_${ARM}" --training.headless True \
    >> "$OUT/${ARM}.train.log" 2>&1
  test -f "$OUT/${ARM}_eval.json"
  echo "=== E52v4 $ARM done $(date -u +%FT%TZ): $(tr -d '\n' < "$OUT/${ARM}_eval.json") ===" \
    | tee -a "$OUT/driver.log"
done
date -u +%FT%TZ > "$OUT/COMPLETE"
echo "E52 v4 complete" | tee -a "$OUT/driver.log"
