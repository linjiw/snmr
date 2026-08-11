#!/usr/bin/env bash
set -euo pipefail

# E70: fresh, loader-order-safe two-walk student experiment.  This consumes only
# frozen E67/E69 teachers and motions; it never loads an E67 student artifact.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
MOTION_ROOT="$E70_ROOT/motions"
STUDENT_ROOT="$E70_ROOT/students"
MANIFEST="$E70_ROOT/teacher_manifest.json"
PRECHECK="${E70_PRECHECK:-$SNMR_ROOT/autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json}"
NUM_ENVS="${E70_NUM_ENVS:-1024}"
ROUNDS="${E70_ROUNDS:-2000}"
MIN_FREE_MB="${E70_MIN_FREE_MB:-26000}"

CLIPS=(walk1_subject1 walk1_subject5)
SOURCE_MOTIONS=(
    /data/robotixx/snmr-research/e69/motions/walk1_subject1_mj_z.npz
    /data/robotixx/snmr-research/e67/motions/walk1_subject5_mj_z.npz
)
MOTION_HASHES=(
    b78f294395a5c74f37edc1c09dd6de0909d966a9b7fd948291c8dd803f7106aa
    d8de93425c14e90dce2930450d722d3eb2b6fcbb09e9c4ff3d59725025424f51
)
TEACHER_REPORTS=(
    /data/robotixx/snmr-research/e69/teacher_reports/walk1_subject1_eval404.json
    /data/robotixx/snmr-research/e67/teacher_reports/walk1_subject5_eval404.json
)
REPORT_HASHES=(
    60a151c007f1fa5f806120a684110dac5c3e991ed42e6d6b68abe9a78cca8f86
    8e768491128af8edbf32e106fd74a2edde7b542148a3628acdd22d8a48c856a7
)
PRECHECK_HASH=3c03b89e2bda939e9a8c5a6dd58caeb771944bb9e5755b14eb37ab75ed9c502e

hash_is() {
    local path="$1" expected="$2" actual
    actual="$(sha256sum "$path" | cut -d' ' -f1)"
    if [[ "$actual" != "$expected" ]]; then
        printf 'SHA-256 mismatch for %s: expected %s, got %s\n' \
            "$path" "$expected" "$actual" >&2
        return 2
    fi
}

hash_is "$PRECHECK" "$PRECHECK_HASH"
free_mb="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' '
)"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
    printf 'E70 requires %s MiB free GPU memory; observed %s MiB. No output was changed.\n' \
        "$MIN_FREE_MB" "$free_mb" >&2
    exit 5
fi
printf 'E70 capacity gate passed with %s MiB free GPU memory.\n' "$free_mb"
mkdir -p "$MOTION_ROOT" "$STUDENT_ROOT" "$E70_ROOT/student_holosoma_logs"

for index in 0 1; do
    source_path="${SOURCE_MOTIONS[$index]}"
    destination="$MOTION_ROOT/${CLIPS[$index]}_mj_z.npz"
    test -f "$source_path"
    hash_is "$source_path" "${MOTION_HASHES[$index]}"
    if [[ -e "$destination" || -L "$destination" ]]; then
        if [[ "$(readlink -f "$destination")" != "$(readlink -f "$source_path")" ]]; then
            printf 'Refusing unexpected E70 motion path %s\n' "$destination" >&2
            exit 2
        fi
    else
        ln -s "$source_path" "$destination"
    fi
    test "$(basename "$destination")" = "${CLIPS[$index]}_mj_z.npz"
    hash_is "${TEACHER_REPORTS[$index]}" "${REPORT_HASHES[$index]}"
done

"$SNMR_ROOT/.venv/bin/python" - "$MANIFEST" \
    "${CLIPS[0]}" "${TEACHER_REPORTS[0]}" \
    "${CLIPS[1]}" "${TEACHER_REPORTS[1]}" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
motions = []
for clip, report_name in ((sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5])):
    report_path = pathlib.Path(report_name)
    report = json.loads(report_path.read_text())
    if report.get("passes_gate") is not True or report.get("num_rollouts") != 1024:
        raise SystemExit(f"teacher report for {clip} is not a frozen passing endpoint")
    checkpoint = pathlib.Path(report["checkpoint"])
    if not checkpoint.is_file():
        raise SystemExit(f"missing teacher checkpoint {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != report["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint hash mismatch for {clip}")
    motions.append(
        {
            "clip": clip,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "teacher_report": str(report_path.resolve()),
        }
    )
payload = {"protocol": "E70 specialist manifest v1", "motions": motions}
if output.exists() and json.loads(output.read_text()) != payload:
    raise SystemExit(f"refusing to overwrite different manifest {output}")
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(output)
PY

student_command() {
    local seed="$1" arm="$2" tag="$3" phase="$4" shuffle="$5" eval_mode="$6"
    local out="$STUDENT_ROOT/seed${seed}_${tag}"
    local run_seed="$seed"
    local -a mode_env=()
    mkdir -p "$out"
    case "$eval_mode" in
        train) mode_env+=(E52_SKIP_INLINE_EVAL=1) ;;
        general) mode_env+=(E52_EVAL_ONLY=1) ; run_seed=404 ;;
        ambiguity)
            mode_env+=(E52_EVAL_ONLY=1 E52_EVAL_STARTS_JSON="$PRECHECK")
            run_seed=404
            ;;
        destroy_zero)
            mode_env+=(E52_EVAL_ONLY=1 E52_EVAL_DESTROY_ZCMD=zero)
            run_seed=404
            ;;
        destroy_shuffle)
            mode_env+=(E52_EVAL_ONLY=1 E52_EVAL_DESTROY_ZCMD=shuffle)
            run_seed=404
            ;;
        destroy_marginal)
            mode_env+=(E52_EVAL_ONLY=1 E52_EVAL_DESTROY_ZCMD=marginal_random)
            run_seed=404
            ;;
        *) printf 'unknown evaluation mode %s\n' "$eval_mode" >&2; return 2 ;;
    esac
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        env \
            E52_ARM="$arm" E52_TEACHER_MANIFEST="$MANIFEST" E52_OUT="$out" \
            E52_ROUNDS="$ROUNDS" E52_DET=1 E52_PHASE_ONLY="$phase" \
            E52_SHUFFLE_LATENT="$shuffle" E52_REPLAY_ROUNDS=4 \
            E52_TEACHER_FLOOR=0.1 E52_TEACHER_ANNEAL_ROUNDS=200 \
            E52_CKPT_EVERY=50 E52_VAL_EVERY=50 E52_VAL_SAMPLES=4096 \
            E52_BEST_AFTER=50 "${mode_env[@]}" \
            PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
            "$SNMR_ROOT/scripts/train_e52_dagger.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs "$NUM_ENVS" --training.seed "$run_seed" \
            --training.headless True --training.name "e70_seed${seed}_${tag}" \
            --logger.base-dir "$E70_ROOT/student_holosoma_logs" \
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
    mkdir -p "$out"
    if [[ ! -f "$out/${arm}_student.pt" ]]; then
        student_command "$seed" "$arm" "$tag" "$phase" "$shuffle" train \
            2>&1 | tee "$out/train.log"
    fi
    if [[ ! -f "$general" ]]; then
        student_command "$seed" "$arm" "$tag" "$phase" "$shuffle" general \
            2>&1 | tee "$out/general_eval.log"
    fi
    if [[ ! -f "$ambiguity" ]]; then
        student_command "$seed" "$arm" "$tag" "$phase" "$shuffle" ambiguity \
            2>&1 | tee "$out/ambiguity_eval.log"
    fi
}

passes_explicit_gate() {
    local seed="$1"
    "$SNMR_ROOT/.venv/bin/python" - \
        "$STUDENT_ROOT/seed${seed}_explicit/c_prior_explicit_eval.json" \
        "${TEACHER_REPORTS[0]}" "${TEACHER_REPORTS[1]}" <<'PY'
import json
import sys
student = json.load(open(sys.argv[1]))["completion_rate"]
teachers = sum(json.load(open(path))["completion_rate"] for path in sys.argv[2:]) / 2
print(f"explicit={student:.6f}; teacher_macro={teachers:.6f}")
raise SystemExit(0 if student >= 0.80 or student >= teachers - 0.05 else 1)
PY
}

run_explicit_destruction() {
    local seed="$1"
    local out="$STUDENT_ROOT/seed${seed}_explicit"
    if [[ ! -f "$out/c_prior_explicit_eval_destroy_zero.json" ]]; then
        student_command "$seed" c_prior_explicit explicit 0 0 destroy_zero \
            2>&1 | tee "$out/destroy_zero_eval.log"
    fi
    if [[ ! -f "$out/c_prior_explicit_eval_destroy_shuffle.json" ]]; then
        student_command "$seed" c_prior_explicit explicit 0 0 destroy_shuffle \
            2>&1 | tee "$out/destroy_shuffle_eval.log"
    fi
    if [[ ! -f "$out/c_prior_explicit_eval_destroy_marginal_random.json" ]]; then
        student_command "$seed" c_prior_explicit explicit 0 0 destroy_marginal \
            2>&1 | tee "$out/destroy_marginal_eval.log"
    fi
}

run_seed() {
    local seed="$1"
    run_student "$seed" c_prior_explicit explicit 0 0
    if ! passes_explicit_gate "$seed"; then
        printf 'E70 seed %s explicit-control gate failed; stopping before representation arms.\n' \
            "$seed" >&2
        return 3
    fi
    run_explicit_destruction "$seed"
    run_student "$seed" a_prior_snmr snmr 0 0
    run_student "$seed" a_prior_snmr time 1 0
    run_student "$seed" b_prior_proprio proprio 0 0
    run_student "$seed" a_prior_snmr shuffled 0 1
}

run_seed 0
SEEDS=(0)
if [[ "${E70_FULL_SEEDS:-0}" == "1" ]]; then
    run_seed 1
    run_seed 2
    SEEDS=(0 1 2)
fi

"$SNMR_ROOT/.venv/bin/python" "$SNMR_ROOT/scripts/analyze_e67_results.py" \
    --students_root "$STUDENT_ROOT" \
    --teacher_reports "${TEACHER_REPORTS[@]}" \
    --seeds "${SEEDS[@]}" \
    --protocol "E70 preregistered analysis v1" \
    --out "$E70_ROOT/analysis_seed$(IFS=-; printf '%s' "${SEEDS[*]}").json"

date -u +%FT%TZ > "$E70_ROOT/SEED$(IFS=-; printf '%s' "${SEEDS[*]}")_COMPLETE"
printf 'E70 requested queue complete.\n'
