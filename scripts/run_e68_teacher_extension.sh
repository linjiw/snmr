#!/usr/bin/env bash
set -euo pipefail

# E68: one preregistered continuation of the failed walk3 specialist, followed by
# one final teacher-gate evaluation.  This script never launches a student.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E68_ROOT="${E68_ROOT:-/data/robotixx/snmr-research/e68}"
REFERENCE="${E68_REFERENCE:-/data/robotixx/snmr-research/e67/gmr_references/walk3_subject1_mj.npz}"
SOURCE_CKPT="${E68_SOURCE_CKPT:-/data/robotixx/snmr-research/e67/teacher_holosoma_logs/WholeBodyTracking/20260808_093956-e67_walk3_subject1_teacher_seed0-locomotion/model_08000.pt}"
SOURCE_SHA256="12f3e92b2d58a748dea768fcf9e442329470348d54c06af63041df5d2a6db32d"
LOG_ROOT="$E68_ROOT/teacher_holosoma_logs"
REPORT_ROOT="$E68_ROOT/teacher_reports"
REPORT="$REPORT_ROOT/walk3_subject1_eval404.json"
NAME="e68_walk3_subject1_teacher_seed0_extend8k"

mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
test -f "$REFERENCE"
test -f "$SOURCE_CKPT"
printf '%s  %s\n' "$SOURCE_SHA256" "$SOURCE_CKPT" | sha256sum --check --status

report_passes() {
    [[ -f "$REPORT" ]] && "$SNMR_ROOT/.venv/bin/python" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passes_gate") is True else 1)' \
        "$REPORT"
}

if report_passes; then
    printf 'E68 final teacher report already passes; nothing to resume.\n'
    exit 0
fi
if [[ -f "$REPORT" ]]; then
    printf 'E68 final report already exists and fails; frozen endpoint will not be overwritten.\n' >&2
    exit 2
fi

shopt -s nullglob
final_matches=("$LOG_ROOT"/WholeBodyTracking/*-"$NAME"-locomotion/model_15998.pt)
shopt -u nullglob
if (( ${#final_matches[@]} == 0 )); then
    printf '=== E68 extension training start %s ===\n' "$(date -u +%FT%TZ)"
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 \
        PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
            "$SNMR_ROOT/scripts/train_agent_joint_reward.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs 512 --training.seed 0 --training.headless True \
            --training.name "$NAME" --training.checkpoint "$SOURCE_CKPT" \
            --logger.base-dir "$LOG_ROOT" \
            --algo.config.num-learning-iterations 8000 \
            --algo.config.save-interval 2000 \
            --randomization.ignore-unsupported True \
            --command.setup-terms.motion-command.params.motion-config.motion-file \
            "$REFERENCE"
    ) 2>&1 | tee "$REPORT_ROOT/walk3_subject1_train.log"
    shopt -s nullglob
    final_matches=("$LOG_ROOT"/WholeBodyTracking/*-"$NAME"-locomotion/model_15998.pt)
    shopt -u nullglob
fi
if (( ${#final_matches[@]} != 1 )); then
    printf 'Expected exactly one E68 model_15998.pt, found %s.\n' "${#final_matches[@]}" >&2
    exit 4
fi
FINAL_CKPT="${final_matches[0]}"

printf '=== E68 frozen final gate start %s ===\n' "$(date -u +%FT%TZ)"
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
) 2>&1 | tee "$REPORT_ROOT/walk3_subject1_eval404.log"

if ! report_passes; then
    printf 'E68 endpoint failed its frozen gate.\n' >&2
    exit 3
fi
printf 'E68 endpoint passed its frozen gate.\n'

