#!/usr/bin/env bash
# E48 100k resolution: matched plain-distill control (A) vs contact-BCE-only (D)
# at full budget, current code, seed 0. Resolves whether E48's 30k fidelity-guard
# failure (+2-4 cm in aux arms) is slow convergence or a real fidelity price, and
# reports full-budget z-linear contact F1 (30k D-arm value was 0.257).
# Registered follow-up per the E48 verdict. Sequential (one GPU job at a time).
set -euo pipefail
MAIN=/home/ec2-user/work/retarget/snmr
PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python
OUT="$MAIN/runs/e48_ssl_100k"
cd "$MAIN"
mkdir -p "$OUT"

echo "=== A control 100k start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
nice -n 15 "$PY" scripts/train_phase1.py \
  --steps 100000 --lr 3e-4 --latent_dim 128 --enc_hidden 256 --dec_hidden 256 \
  --seed 0 --out "$OUT/a_control_100k" --device cuda \
  >> "$OUT/a_control_100k.train.log" 2>&1
echo "=== A control 100k complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"

echo "=== D bce_only 100k start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
nice -n 15 "$PY" scripts/train_phase1.py \
  --steps 100000 --lr 3e-4 --latent_dim 128 --enc_hidden 256 --dec_hidden 256 \
  --contact_bce_weight 0.5 --contact_mask teacher_height \
  --seed 0 --out "$OUT/d_bce_only_100k" --device cuda \
  >> "$OUT/d_bce_only_100k.train.log" 2>&1
echo "=== D bce_only 100k complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
echo "=== ALL DONE $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
