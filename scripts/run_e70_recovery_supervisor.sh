#!/usr/bin/env bash
set -euo pipefail

# Wait for an uncontended GPU, then resume the frozen E70 launcher.  The launcher
# hash-checks and skips completed cells; it does not resume partial training state.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIN_FREE_MB="${E70_MIN_FREE_MB:-26000}"
POLL_SECONDS="${E70_POLL_SECONDS:-30}"
RECOVERY_LOG="${E70_RECOVERY_LOG:-/tmp/snmr-e70-recovery-supervisor.log}"

[[ "$MIN_FREE_MB" =~ ^[0-9]+$ ]]
[[ "$POLL_SECONDS" =~ ^[0-9]+$ ]]
exec > >(tee -a "$RECOVERY_LOG") 2>&1

printf '%s E70 recovery supervisor started; gate=%s MiB\n' \
    "$(date -u +%FT%TZ)" "$MIN_FREE_MB"
while true; do
    free_mb="$(
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            2>/dev/null | head -n 1 | tr -d ' ' || true
    )"
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
        printf '%s capacity gate observed %s MiB; launching frozen queue\n' \
            "$(date -u +%FT%TZ)" "$free_mb"
        break
    fi
    printf '%s waiting; free=%s MiB\n' \
        "$(date -u +%FT%TZ)" "${free_mb:-unavailable}"
    sleep "$POLL_SECONDS"
done

export E70_FULL_SEEDS=1
export E70_MIN_FREE_MB="$MIN_FREE_MB"
exec bash "$SNMR_ROOT/scripts/run_e70_multitraj.sh"
