#!/usr/bin/env bash
set -euo pipefail

# Reproducible E67 queue: two gated specialists, then deterministic student controls.
# Generated checkpoints/reports live on /data; existing completed cells are resumed, not
# overwritten.  Set E67_FULL_SEEDS=1 only after inspecting the seed-0 five-arm result.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E67_ROOT="${E67_ROOT:-/data/robotixx/snmr-research/e67}"
REFERENCE_ROOT="${E67_REFERENCE_ROOT:-$E67_ROOT/gmr_references}"
MOTION_ROOT="${E67_MOTION_ROOT:-$E67_ROOT/motions}"
PRECHECK="${E67_PRECHECK:-$SNMR_ROOT/autoresearch/iterate-260808-0338/e67_ambiguity_precheck.json}"
TEACHER_LOG_ROOT="$E67_ROOT/teacher_holosoma_logs"
TEACHER_REPORT_ROOT="$E67_ROOT/teacher_reports"
STUDENT_ROOT="$E67_ROOT/students"
MANIFEST="$E67_ROOT/teacher_manifest.json"
NUM_ENVS="${E67_NUM_ENVS:-1024}"
ROUNDS="${E67_ROUNDS:-2000}"

CLIPS=(walk1_subject5 walk3_subject1)
mkdir -p "$TEACHER_LOG_ROOT" "$TEACHER_REPORT_ROOT" "$STUDENT_ROOT"
test -f "$PRECHECK"
for clip in "${CLIPS[@]}"; do
    test -f "$REFERENCE_ROOT/${clip}_mj.npz"
    test -f "$MOTION_ROOT/${clip}_mj_z.npz"
done

latest_teacher_checkpoint() {
    local name="$1"
    local matches=()
    shopt -s nullglob
    matches=("$TEACHER_LOG_ROOT"/WholeBodyTracking/*-"$name"-locomotion/model_08000.pt)
    shopt -u nullglob
    if (( ${#matches[@]} > 0 )); then
        printf '%s\n' "${matches[${#matches[@]} - 1]}"
    fi
}

train_and_gate_teacher() {
    local clip="$1"
    local name="e67_${clip}_teacher_seed0"
    local reference="$REFERENCE_ROOT/${clip}_mj.npz"
    local report="$TEACHER_REPORT_ROOT/${clip}_eval404.json"
    local checkpoint
    checkpoint="$(latest_teacher_checkpoint "$name")"

    if [[ -z "$checkpoint" ]]; then
        printf '=== %s teacher training start %s ===\n' "$clip" "$(date -u +%FT%TZ)"
        (
            cd "$SNMR_HOLOSOMA_ROOT"
            E51_JOINT_POS_WEIGHT=1.0 E51_JOINT_POS_SIGMA=0.5 \
            PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
                "$SNMR_ROOT/scripts/train_agent_joint_reward.py" \
                exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
                --training.num-envs 512 --training.seed 0 --training.headless True \
                --training.name "$name" --logger.base-dir "$TEACHER_LOG_ROOT" \
                --algo.config.num-learning-iterations 8000 \
                --algo.config.save-interval 2000 \
                --randomization.ignore-unsupported True \
                --command.setup-terms.motion-command.params.motion-config.motion-file \
                "$reference"
        ) 2>&1 | tee "$TEACHER_REPORT_ROOT/${clip}_train.log"
        checkpoint="$(latest_teacher_checkpoint "$name")"
        test -f "$checkpoint"
    fi

    if [[ ! -f "$report" ]]; then
        printf '=== %s teacher gate start %s ===\n' "$clip" "$(date -u +%FT%TZ)"
        (
            cd "$SNMR_HOLOSOMA_ROOT"
            E67_TEACHER_CKPT="$checkpoint" E67_TEACHER_REPORT="$report" \
            PYTHONPATH="$SNMR_ROOT" "$WBT_PYTHON" \
                "$SNMR_ROOT/scripts/eval_e67_teacher.py" \
                exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
                --training.num-envs "$NUM_ENVS" --training.seed 404 \
                --training.headless True --logger.base-dir "$TEACHER_LOG_ROOT" \
                --randomization.ignore-unsupported True \
                --simulator.config.sim.max-episode-length-s 100000.0 \
                --command.setup-terms.motion-command.params.motion-config.motion-file \
                "$reference"
        ) 2>&1 | tee "$TEACHER_REPORT_ROOT/${clip}_eval404.log"
    fi

    # Do not rely on `set -e` through a function/pipeline/command-substitution
    # boundary: Bash can suppress that failure in subtle ways.  Re-read the
    # persisted report and return nonzero explicitly before exposing a
    # checkpoint to any student stage.
    if ! "$SNMR_ROOT/.venv/bin/python" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passes_gate") is True else 1)' \
        "$report"; then
        printf 'E67 %s specialist failed its persisted teacher gate; stopping before students.\n' \
            "$clip" >&2
        return 2
    fi
    printf '%s\n' "$checkpoint" > "$TEACHER_REPORT_ROOT/${clip}_passed_checkpoint.txt"
}

train_and_gate_teacher "${CLIPS[0]}"
train_and_gate_teacher "${CLIPS[1]}"
CKPT_FIRST="$(<"$TEACHER_REPORT_ROOT/${CLIPS[0]}_passed_checkpoint.txt")"
CKPT_SECOND="$(<"$TEACHER_REPORT_ROOT/${CLIPS[1]}_passed_checkpoint.txt")"
test -f "$CKPT_FIRST"
test -f "$CKPT_SECOND"

"$SNMR_ROOT/.venv/bin/python" - "$MANIFEST" \
    "${CLIPS[0]}" "$CKPT_FIRST" "${CLIPS[1]}" "$CKPT_SECOND" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "protocol": "E67 specialist manifest v1",
    "motions": [
        {"clip": sys.argv[2], "checkpoint": str(pathlib.Path(sys.argv[3]).resolve())},
        {"clip": sys.argv[4], "checkpoint": str(pathlib.Path(sys.argv[5]).resolve())},
    ],
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(path)
PY

student_command() {
    local seed="$1" arm="$2" tag="$3" phase="$4" shuffle="$5" eval_mode="$6"
    local out="$STUDENT_ROOT/seed${seed}_${tag}"
    local -a mode_env=()
    mkdir -p "$out"
    if [[ "$eval_mode" == "general" ]]; then
        mode_env+=(E52_EVAL_ONLY=1)
    elif [[ "$eval_mode" == "ambiguity" ]]; then
        mode_env+=(E52_EVAL_ONLY=1 E52_EVAL_STARTS_JSON="$PRECHECK")
    else
        mode_env+=(E52_SKIP_INLINE_EVAL=1)
    fi
    local run_seed="$seed"
    if [[ "$eval_mode" != "train" ]]; then
        run_seed=404
    fi
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        env \
            E52_ARM="$arm" E52_TEACHER_MANIFEST="$MANIFEST" E52_OUT="$out" \
            E52_ROUNDS="$ROUNDS" E52_DET=1 E52_PHASE_ONLY="$phase" \
            E52_SHUFFLE_LATENT="$shuffle" "${mode_env[@]}" \
            PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
            "$SNMR_ROOT/scripts/train_e52_dagger.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs "$NUM_ENVS" --training.seed "$run_seed" \
            --training.headless True --training.name "e67_seed${seed}_${tag}" \
            --logger.base-dir "$E67_ROOT/student_holosoma_logs" \
            --randomization.ignore-unsupported True \
            --simulator.config.sim.max-episode-length-s 100000.0 \
            --command.setup-terms.motion-command.params.motion-config.motion-file "" \
            --command.setup-terms.motion-command.params.motion-config.motion-dir \
            "$MOTION_ROOT"
    )
}

run_student() {
    local seed="$1" arm="$2" tag="$3" phase="$4" shuffle="$5"
    local out="$STUDENT_ROOT/seed${seed}_${tag}"
    local general="$out/${arm}_eval.json"
    local ambiguity="$out/${arm}_eval_ambiguity.json"
    # Create the directory before starting `tee`; creating it inside the
    # piped producer races tee's attempt to open its log file.
    mkdir -p "$out"
    if [[ ! -f "$general" ]]; then
        if [[ ! -f "$out/${arm}_student.pt" ]]; then
            student_command "$seed" "$arm" "$tag" "$phase" "$shuffle" train \
                2>&1 | tee "$out/train.log"
        fi
        # A fresh train runs with E52_SKIP_INLINE_EVAL=1.  Always produce the
        # general report before any ambiguity evaluation or validity gate.
        if [[ ! -f "$general" ]]; then
            student_command "$seed" "$arm" "$tag" "$phase" "$shuffle" general \
                2>&1 | tee "$out/general_eval.log"
        fi
    fi
    if [[ ! -f "$ambiguity" ]]; then
        student_command "$seed" "$arm" "$tag" "$phase" "$shuffle" ambiguity \
            2>&1 | tee "$out/ambiguity_eval.log"
    fi
}

passes_explicit_gate() {
    "$SNMR_ROOT/.venv/bin/python" - \
        "$STUDENT_ROOT/seed0_explicit/c_prior_explicit_eval.json" \
        "$TEACHER_REPORT_ROOT/${CLIPS[0]}_eval404.json" \
        "$TEACHER_REPORT_ROOT/${CLIPS[1]}_eval404.json" <<'PY'
import json
import sys

student = json.load(open(sys.argv[1]))["completion_rate"]
teachers = sum(json.load(open(path))["completion_rate"] for path in sys.argv[2:]) / 2
print(f"explicit={student:.6f}; teacher_macro={teachers:.6f}")
raise SystemExit(0 if student >= 0.80 or student >= teachers - 0.05 else 1)
PY
}

# Seed 0 is the fail-fast validity sequence fixed in the preregistration.
run_student 0 c_prior_explicit explicit 0 0
if ! passes_explicit_gate; then
    printf 'E67 explicit-control gate failed; stopping before representation arms.\n' >&2
    exit 3
fi
run_student 0 a_prior_snmr snmr 0 0
run_student 0 a_prior_snmr time 1 0
run_student 0 b_prior_proprio proprio 0 0
run_student 0 a_prior_snmr shuffled 0 1

if [[ "${E67_FULL_SEEDS:-0}" == "1" ]]; then
    for seed in 1 2; do
        run_student "$seed" c_prior_explicit explicit 0 0
        run_student "$seed" a_prior_snmr snmr 0 0
        run_student "$seed" a_prior_snmr time 1 0
        run_student "$seed" b_prior_proprio proprio 0 0
        run_student "$seed" a_prior_snmr shuffled 0 1
    done
fi

date -u +%FT%TZ > "$E67_ROOT/SEED0_COMPLETE"
printf 'E67 requested queue complete.\n'
