#!/usr/bin/env bash
set -euo pipefail

# Render the six policy-independently selected E70 simulation illustrations.
# Scientific aggregation must finish first; this script never trains or selects a
# rollout from policy outcomes.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
MANIFEST="${E70_VIDEO_MANIFEST:-$SNMR_ROOT/autoresearch/iterate-260809-0351/e70_video_manifest.json}"
ANALYSIS="$E70_ROOT/analysis_seed0-1-2.json"
COMPLETE="$E70_ROOT/SEED0-1-2_COMPLETE"
VIDEO_ROOT="${E70_VIDEO_ROOT:-$SNMR_ROOT/exports/e70_video}"
RAW_DIR="$VIDEO_ROOT/raw"
ATTEMPT_ROOT="$VIDEO_ROOT/capture_attempts"
MOTION_ROOT="$E70_ROOT/motions"
TEACHER_MANIFEST="$E70_ROOT/teacher_manifest.json"
VIDEO_CODE_HASHES="$SNMR_ROOT/autoresearch/iterate-260809-0351/e70_video_code_hashes.json"

test -f "$COMPLETE"
test -f "$ANALYSIS"
test -f "$MANIFEST"
test -f "$TEACHER_MANIFEST"
"$SNMR_ROOT/.venv/bin/python" "$SNMR_ROOT/scripts/check_e70_video_code_hashes.py" \
    --manifest "$VIDEO_CODE_HASHES" --root "$SNMR_ROOT"
mkdir -p "$RAW_DIR" "$ATTEMPT_ROOT"

mapfile -t capture_rows < <(
    "$SNMR_ROOT/.venv/bin/python" - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
if manifest.get("protocol") != "E70 paper-video rollout selection v1":
    raise SystemExit("unexpected E70 video manifest protocol")
if manifest.get("selection_uses_policy_outcomes") is not False:
    raise SystemExit("video manifest is not policy-independent")
for item in manifest["captures"]:
    values = (
        item["name"], item["arm"], int(item["phase_only"]),
        int(item["shuffle_latent"]), item["destroy_zcmd"],
        item["checkpoint"], item["checkpoint_sha256"], int(item["start_step"]),
    )
    print("\t".join(map(str, values)))
PY
)

for row in "${capture_rows[@]}"; do
    IFS=$'\t' read -r name arm phase shuffle destroy checkpoint expected_sha start_step <<<"$row"
    stable="$RAW_DIR/$name.mp4"
    stable_report="$RAW_DIR/$name.report.json"
    if [[ -s "$stable" && -s "$stable_report" ]]; then
        printf 'E70 video capture already exists: %s\n' "$stable"
        continue
    fi
    if [[ -e "$stable" || -e "$stable_report" ]]; then
        printf 'incomplete stable capture pair for %s; preserve and inspect it before retrying\n' \
            "$name" >&2
        exit 4
    fi
    test -f "$checkpoint"
    observed_sha="$(sha256sum "$checkpoint" | cut -d' ' -f1)"
    if [[ "$observed_sha" != "$expected_sha" ]]; then
        printf 'checkpoint hash mismatch for %s\n' "$name" >&2
        exit 2
    fi

    free_mb="$(
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            | head -n 1 | tr -d ' '
    )"
    if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < 26000 )); then
        printf 'video capture requires 26000 MiB free GPU memory; observed %s MiB\n' "$free_mb" >&2
        exit 5
    fi

    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    attempt="$ATTEMPT_ROOT/$name/$stamp"
    mkdir -p "$attempt"
    out="$(dirname "$checkpoint")"
    capture_envs=1
    if [[ "$destroy" == "marginal_random" ]]; then
        # The registered intervention samples each command dimension from the
        # contemporaneous batch marginal.  A one-environment capture has no
        # valid marginal (torch.std would be undefined), so env 0 is rendered
        # against the frozen 1,024-environment companion phase grid.
        capture_envs=1024
    fi
    printf 'Rendering %s from exact global start %s\n' "$name" "$start_step"
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        env \
            E52_ARM="$arm" E52_TEACHER_MANIFEST="$TEACHER_MANIFEST" E52_OUT="$out" \
            E52_PHASE_ONLY="$phase" E52_SHUFFLE_LATENT="$shuffle" \
            E52_EVAL_DESTROY_ZCMD="$destroy" E52_EVAL_EXACT_START="$start_step" \
            E70_VIDEO_CAPTURE_NAME="$name" \
            E70_VIDEO_REPORT_PATH="$attempt/report.json" PYTHONPATH="$SNMR_ROOT" \
            "$WBT_PYTHON" "$SNMR_ROOT/scripts/eval_e70_video.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs "$capture_envs" --training.seed 404 \
            --training.headless True --training.name "e70_video_${name}" \
            --logger.base-dir "$E70_ROOT/student_holosoma_logs" \
            --logger.video.enabled True --logger.video.interval 1 \
            --logger.video.width 1920 --logger.video.height 1080 \
            --logger.video.playback_rate 1.0 --logger.video.output_format mp4 \
            --logger.video.save_dir "$attempt" --logger.video.upload_to_wandb False \
            --logger.video.show_command_overlay False \
            --logger.video.record_env_id 0 --logger.video.use_recording_thread False \
            --logger.video.vertical_fov 45.0 \
            --randomization.ignore-unsupported True \
            --simulator.config.sim.max-episode-length-s 100000.0 \
            --command.setup-terms.motion-command.params.motion-config.motion-file "" \
            --command.setup-terms.motion-command.params.motion-config.motion-dir \
            "$MOTION_ROOT" logger.video.camera:cartesian-camera-config \
            --logger.video.camera.offset '[2.0,2.0,1.0]' \
            --logger.video.camera.target_offset '[0.0,0.0,0.3]' \
            --logger.video.camera.smoothing 0.95 \
            --logger.video.camera.tracking_body_name pelvis
    ) 2>&1 | tee "$attempt/capture.log"

    selected="$(
        "$SNMR_ROOT/.venv/bin/python" - "$attempt" <<'PY'
import pathlib
import sys

videos = list(pathlib.Path(sys.argv[1]).glob("*.mp4"))
if not videos:
    raise SystemExit("capture emitted no MP4")
print(max(videos, key=lambda path: (path.stat().st_size, path.name)))
PY
    )"
    cp --reflink=auto "$selected" "$stable"
    cp --reflink=auto "$attempt/report.json" "$stable_report"
    test -s "$stable" && test -s "$stable_report"
done

"$SNMR_ROOT/.venv/bin/python" "$SNMR_ROOT/scripts/index_e70_video_captures.py" \
    --manifest "$MANIFEST" --raw-dir "$RAW_DIR" \
    --student-root "$E70_ROOT/students" --out "$VIDEO_ROOT/raw_capture_index.json"

printf 'E70 raw captures complete: %s\n' "$RAW_DIR"
