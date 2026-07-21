#!/usr/bin/env bash
# E48 100k post-training analysis: probe both arms' 100k checkpoints for z-linear contact F1,
# and pull final val MPJPE from each train log. Run after driver.sh completes both arms.
set -euo pipefail
MAIN=/home/ec2-user/work/retarget/snmr
PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python
OUT="$MAIN/runs/e48_ssl_100k"
cd "$MAIN"

for arm in a_control_100k d_bce_only_100k; do
  CK=$(ls -1 "$OUT/$arm"/ckpt*.pt 2>/dev/null | tail -1)
  echo "=== probing $arm ($CK) ==="
  "$PY" scripts/probe_latent_contact.py --ckpt "$CK" --device cpu \
    --out "$OUT/probe_${arm}.json"
done

echo ""
echo "=== E48 100k SUMMARY ==="
"$PY" - <<'PY'
import json, glob, os
OUT="/home/ec2-user/work/retarget/snmr/runs/e48_ssl_100k"
def final_mpjpe(arm):
    f=f"{OUT}/{arm}/log.jsonl"
    if not os.path.exists(f): return None
    last=None
    for line in open(f):
        try: last=json.loads(line)
        except: pass
    return last.get("val_mpjpe_m") if last else None
def probe_f1(arm):
    f=f"{OUT}/probe_{arm}.json"
    if not os.path.exists(f): return None
    d=json.load(open(f))
    return {k:d.get(k) for k in ("z_linear","z_context_linear","source_height_mask")}
for arm,label in (("a_control_100k","A control (distill only)"),
                  ("d_bce_only_100k","D contact-BCE 0.5 only")):
    m=final_mpjpe(arm); p=probe_f1(arm)
    print(f"{label}: final val MPJPE = {round(m*100,3) if m else '?'} cm ; probe F1 = {p}")
print("\nYardsticks: E48 30k -> A 0.023 / D 0.257 z-linear; source-height mask 0.127; E38 phase-2 z 0.044")
print("Fidelity question: does D's 100k MPJPE come within +0.3cm of A's? (convergence lag vs real price)")
PY
