#!/usr/bin/env bash
set -euo pipefail

# Pause one owned compute process and resume it only when no other CUDA compute
# process is present.  This never signals the external processes it observes.

if (( $# < 1 || $# > 3 )); then
    printf 'usage: %s TARGET_PID [MIN_FREE_MB] [POLL_SECONDS]\n' "$0" >&2
    exit 64
fi
target_pid="$1"
min_free_mb="${2:-28000}"
poll_seconds="${3:-30}"
[[ "$target_pid" =~ ^[0-9]+$ ]]
[[ "$min_free_mb" =~ ^[0-9]+$ ]]
[[ "$poll_seconds" =~ ^[0-9]+$ ]]
kill -0 "$target_pid"
kill -STOP "$target_pid"
printf '%s paused pid %s; waiting for exclusive GPU access\n' \
    "$(date -u +%FT%TZ)" "$target_pid"

while kill -0 "$target_pid" 2>/dev/null; do
    mapfile -t compute_pids < <(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | tr -d ' ' | sed '/^$/d' | sort -u
    )
    others=0
    for pid in "${compute_pids[@]}"; do
        if [[ "$pid" != "$target_pid" ]]; then
            others=$((others + 1))
        fi
    done
    free_mb="$(
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            | head -n 1 | tr -d ' '
    )"
    if (( others == 0 && free_mb >= min_free_mb )); then
        kill -CONT "$target_pid"
        printf '%s resumed pid %s with %s MiB free\n' \
            "$(date -u +%FT%TZ)" "$target_pid" "$free_mb"
        exit 0
    fi
    printf '%s waiting: %s other compute process(es), %s MiB free\n' \
        "$(date -u +%FT%TZ)" "$others" "$free_mb"
    sleep "$poll_seconds"
done

printf '%s target pid %s exited while waiting\n' "$(date -u +%FT%TZ)" "$target_pid" >&2
exit 1

