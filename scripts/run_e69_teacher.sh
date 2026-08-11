#!/usr/bin/env bash
set -euo pipefail

# E69: train and gate only the reference-selected walk1_subject1 specialist.
# This launcher never creates a student or representation comparison.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E69_ROOT="${E69_ROOT:-/data/robotixx/snmr-research/e69}"
REFERENCE="$E69_ROOT/gmr_references/walk1_subject1_mj.npz"
LOG_ROOT="$E69_ROOT/teacher_holosoma_logs"
REPORT_ROOT="$E69_ROOT/teacher_reports"
REPORT="$REPORT_ROOT/walk1_subject1_eval404.json"
NAME="e69_walk1_subject1_teacher_seed0"
MIN_FREE_MB="${E69_MIN_FREE_MB:-16000}"

mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
test -f "$REFERENCE"

report_passes() {
    [[ -f "$REPORT" ]] && "$SNMR_ROOT/.venv/bin/python" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passes_gate") is True else 1)' \
        "$REPORT"
}

if report_passes; then
    printf 'E69 specialist report already passes; nothing to run.\n'
    exit 0
fi
if [[ -f "$REPORT" ]]; then
    printf 'E69 frozen report already exists and fails; it will not be overwritten.\n' >&2
    exit 2
fi

shopt -s nullglob
matches=("$LOG_ROOT"/WholeBodyTracking/*-"$NAME"-locomotion/model_07999.pt)
shopt -u nullglob
if (( ${#matches[@]} == 0 )); then
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
        printf 'E69 requires %s MiB free GPU memory; observed %s MiB.\n' \
            "$MIN_FREE_MB" "$free_mb" >&2
        exit 5
    fi
    printf '=== E69 specialist training start %s (free GPU %s MiB) ===\n' \
        "$(date -u +%FT%TZ)" "$free_mb"
    set +e
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 \
        PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
            "$SNMR_ROOT/scripts/train_agent_joint_reward.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs 512 --training.seed 0 --training.headless True \
            --training.name "$NAME" --logger.base-dir "$LOG_ROOT" \
            --algo.config.num-learning-iterations 8000 \
            --algo.config.save-interval 2000 \
            --randomization.ignore-unsupported True \
            --command.setup-terms.motion-command.params.motion-config.motion-file \
            "$REFERENCE"
    ) 2>&1 | tee "$REPORT_ROOT/walk1_subject1_train.log"
    train_status="${PIPESTATUS[0]}"
    set -e
    if (( train_status != 0 )); then
        printf 'E69 specialist training failed with status %s.\n' "$train_status" >&2
        exit "$train_status"
    fi
    shopt -s nullglob
    matches=("$LOG_ROOT"/WholeBodyTracking/*-"$NAME"-locomotion/model_07999.pt)
    shopt -u nullglob
fi
if (( ${#matches[@]} != 1 )); then
    printf 'Expected exactly one E69 model_07999.pt, found %s.\n' "${#matches[@]}" >&2
    exit 4
fi
FINAL_CKPT="${matches[0]}"

printf '=== E69 frozen specialist gate start %s ===\n' "$(date -u +%FT%TZ)"
set +e
(
    cd "$SNMR_HOLOSOMA_ROOT"
    E67_TEACHER_CKPT="$FINAL_CKPT" E67_TEACHER_REPORT="$REPORT" \
    PYTHONPATH="$SNMR_ROOT" "$WBT_PYTHON" \
        "$SNMR_ROOT/scripts/eval_e67_teacher.py" \
        exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
        --training.num-envs 1024 --training.seed 404 \
        --training.headless True --logger.base-dir "$LOG_ROOT" \
        --randomization.ignore-unsupported True \
        --simulator.config.sim.max-episode-length-s 100000.0 \
        --command.setup-terms.motion-command.params.motion-config.motion-file \
        "$REFERENCE"
) 2>&1 | tee "$REPORT_ROOT/walk1_subject1_eval404.log"
eval_status="${PIPESTATUS[0]}"
set -e
if [[ ! -f "$REPORT" ]]; then
    printf 'E69 evaluator exited %s without a persisted report.\n' "$eval_status" >&2
    exit "$eval_status"
fi
if ! report_passes; then
    printf 'E69 endpoint failed its frozen specialist gate.\n' >&2
    exit 3
fi
printf 'E69 endpoint passed its frozen specialist gate.\n'

