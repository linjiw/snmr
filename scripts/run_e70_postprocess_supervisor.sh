#!/usr/bin/env bash
set -euo pipefail

# Wait for the hash-frozen three-seed assay, then bind its analyzer output into
# the paper and produce the provenance-checked simulation video.  This never
# trains, tunes, publishes, or deploys a policy.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
ANALYSIS="$E70_ROOT/analysis_seed0-1-2.json"
COMPLETE="$E70_ROOT/SEED0-1-2_COMPLETE"
MIN_FREE_MB="${E70_MIN_FREE_MB:-26000}"
POLL_SECONDS="${E70_POST_POLL_SECONDS:-30}"
POST_LOG="${E70_POST_LOG:-/tmp/snmr-e70-postprocess-supervisor.log}"
VIDEO_ROOT="${E70_VIDEO_ROOT:-$SNMR_ROOT/exports/e70_video}"

[[ "$MIN_FREE_MB" =~ ^[0-9]+$ ]]
[[ "$POLL_SECONDS" =~ ^[0-9]+$ ]]
exec > >(tee -a "$POST_LOG") 2>&1
cd "$SNMR_ROOT"

printf '%s E70 postprocess supervisor started\n' "$(date -u +%FT%TZ)"
while [[ ! -f "$COMPLETE" || ! -f "$ANALYSIS" ]]; do
    printf '%s waiting for frozen three-seed analyzer\n' "$(date -u +%FT%TZ)"
    sleep "$POLL_SECONDS"
done

source "$SNMR_ROOT/scripts/activate_snmr.sh"
python "$SNMR_ROOT/scripts/check_e70_video_code_hashes.py" \
    --manifest "$SNMR_ROOT/autoresearch/iterate-260809-0351/e70_video_code_hashes.json" \
    --root "$SNMR_ROOT"
python "$SNMR_ROOT/scripts/render_e70_paper_values.py" \
    --analysis "$ANALYSIS" --out "$SNMR_ROOT/paper/e70_results.tex"

PAPER_BUILD="${E70_PAPER_BUILD:-/data/robotixx/snmr-research/paper-build-final}"
mkdir -p "$PAPER_BUILD"
XDG_CACHE_HOME="$SNMR_CACHE_ROOT/tectonic" \
    tectonic --keep-logs --outdir "$PAPER_BUILD" "$SNMR_ROOT/paper/main.tex"
pdfinfo "$PAPER_BUILD/main.pdf" | grep -Eq '^Pages:[[:space:]]+[1-8]$'
pdfinfo "$PAPER_BUILD/main.pdf" | grep -Eq '^Page size:[[:space:]]+612 x 792 pts \(letter\)$'
pdffonts "$PAPER_BUILD/main.pdf" | awk '
    NR > 2 { fonts += 1 }
    NR > 2 && ($(NF - 4) != "yes" || $(NF - 3) != "yes") { bad = 1 }
    END { exit (bad || fonts == 0) }
'

while true; do
    free_mb="$(
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            2>/dev/null | head -n 1 | tr -d ' ' || true
    )"
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
        printf '%s video capacity gate observed %s MiB\n' \
            "$(date -u +%FT%TZ)" "$free_mb"
        break
    fi
    printf '%s waiting for video capacity; free=%s MiB\n' \
        "$(date -u +%FT%TZ)" "${free_mb:-unavailable}"
    sleep "$POLL_SECONDS"
done

bash "$SNMR_ROOT/scripts/run_e70_video.sh"
python "$SNMR_ROOT/scripts/compose_e70_video.py" \
    --manifest "$SNMR_ROOT/autoresearch/iterate-260809-0351/e70_video_manifest.json" \
    --analysis "$ANALYSIS" \
    --capture-index "$VIDEO_ROOT/raw_capture_index.json" \
    --raw-dir "$VIDEO_ROOT/raw" \
    --out "$VIDEO_ROOT/snmr_e70_icra.mp4"

date -u +%FT%TZ > "$E70_ROOT/POSTPROCESS_COMPLETE"
printf '%s E70 paper/video postprocess complete\n' "$(date -u +%FT%TZ)"
