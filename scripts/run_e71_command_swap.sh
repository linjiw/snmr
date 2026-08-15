#!/usr/bin/env bash
set -euo pipefail

# E71 state machine.  The explicit arm must finish and pass the manifest-bound
# feasibility preflight before any SNMR report can be launched.

if (( $# != 1 )); then
    printf 'usage: %s {status|smoke|confirmatory}\n' "$0" >&2
    exit 64
fi

mode="$1"
SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E70_ROOT=/data/robotixx/snmr-research/e70
E71_ROOT=/data/robotixx/snmr-research/e71
E71_DRAFT_MANIFEST="$SNMR_ROOT/autoresearch/iterate-260813-2350/e71_freeze_draft.json"
E71_MANIFEST="$SNMR_ROOT/autoresearch/iterate-260813-2350/e71_freeze_manifest.json"
PRECHECK="$SNMR_ROOT/autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json"
MIN_FREE_MB=26000
POSTPROCESS_GATE="$E70_ROOT/POSTPROCESS_COMPLETE"

[[ "$MIN_FREE_MB" =~ ^[0-9]+$ ]]
if (( MIN_FREE_MB < 26000 )); then
    printf 'E71_MIN_FREE_MB cannot lower the frozen 26000-MiB gate.\n' >&2
    exit 64
fi

gpu_free_mb() {
    local observed
    if [[ -n "${WBT_PYTHON:-}" && -x "${WBT_PYTHON:-}" ]]; then
        observed="$(CUDA_VISIBLE_DEVICES=0 "$WBT_PYTHON" -c \
            'import torch; print(torch.cuda.mem_get_info(0)[0] // (1024 * 1024))' \
            2>/dev/null | tail -n 1 | tr -d ' ' || true)"
    else
        observed="$(nvidia-smi --id=0 --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
    fi
    if [[ "$observed" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$observed"
    fi
}

manifest_status() {
    "$SNMR_ROOT/.venv/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$1"
}

manifest_smoke_pair() {
    "$SNMR_ROOT/.venv/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["smoke_pair_id"])' "$1"
}

show_status() {
    local free_mb draft_status final_status
    free_mb="$(gpu_free_mb)"
    draft_status="missing"
    final_status="missing"
    if [[ -f "$E71_DRAFT_MANIFEST" ]]; then
        draft_status="$(manifest_status "$E71_DRAFT_MANIFEST")"
    fi
    if [[ -f "$E71_MANIFEST" ]]; then
        final_status="$(manifest_status "$E71_MANIFEST")"
    fi
    printf 'E70 postprocess gate: %s\n' "$([[ -f "$POSTPROCESS_GATE" ]] && printf pass || printf blocked)"
    printf 'E71 draft manifest: %s (%s)\n' "$E71_DRAFT_MANIFEST" "$draft_status"
    printf 'E71 final manifest: %s (%s)\n' "$E71_MANIFEST" "$final_status"
    printf 'GPU free MiB: %s (required %s)\n' "${free_mb:-unavailable}" "$MIN_FREE_MB"
}

if [[ "$mode" == "status" ]]; then
    show_status
    exit 0
fi
if [[ "$mode" != "smoke" && "$mode" != "confirmatory" ]]; then
    printf 'unknown E71 mode %s\n' "$mode" >&2
    exit 64
fi
if [[ ! -f "$POSTPROCESS_GATE" ]]; then
    printf 'E71 blocked: missing cross-project gate %s; no output was changed.\n' \
        "$POSTPROCESS_GATE" >&2
    exit 6
fi
if [[ "$mode" == "smoke" ]]; then
    ACTIVE_MANIFEST="$E71_DRAFT_MANIFEST"
    required_status="DRAFT"
else
    ACTIVE_MANIFEST="$E71_MANIFEST"
    required_status="PREREGISTERED"
fi
if [[ ! -f "$ACTIVE_MANIFEST" ]]; then
    printf 'E71 blocked: missing %s freeze manifest %s; no output was changed.\n' \
        "$required_status" "$ACTIVE_MANIFEST" >&2
    exit 6
fi
if [[ "$(manifest_status "$ACTIVE_MANIFEST")" != "$required_status" ]]; then
    printf 'E71 %s run requires a %s manifest; no output was changed.\n' \
        "$mode" "$required_status" >&2
    exit 6
fi
if ! PYTHONPATH="$SNMR_ROOT" "$SNMR_ROOT/.venv/bin/python" -c \
    'import json,pathlib,sys; from scripts.audit_e71_bundle import validate_manifest; p=pathlib.Path(sys.argv[1]); m=json.loads(p.read_text()); s=validate_manifest(m, base_dir=p.parent, require_preregistered=sys.argv[2]=="PREREGISTERED"); raise SystemExit(0 if s["runtime_ready"] else 9)' \
    "$ACTIVE_MANIFEST" "$required_status"; then
    printf 'E71 %s manifest failed integrity/runtime-ready validation; no output was changed.\n' \
        "$required_status" >&2
    exit 6
fi

require_gpu() {
    local free_mb
    free_mb="$(gpu_free_mb)"
    if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
        printf 'E71 requires %s MiB free GPU memory; observed %s. No cell was started.\n' \
            "$MIN_FREE_MB" "${free_mb:-unavailable}" >&2
        return 5
    fi
    printf 'E71 capacity gate passed with %s MiB free GPU memory.\n' "$free_mb"
}

source "$SNMR_ROOT/scripts/activate_snmr.sh"
export CUDA_VISIBLE_DEVICES=0
require_gpu

MANIFEST="$E70_ROOT/teacher_manifest.json"
MOTION_ROOT="$E70_ROOT/motions"
mkdir -p "$E71_ROOT/reports" "$E71_ROOT/logs"
exec 9>"$E71_ROOT/.e71.lock"
if ! flock -n 9; then
    printf 'another E71 state-machine process owns %s/.e71.lock\n' "$E71_ROOT" >&2
    exit 7
fi

run_report() {
    local seed="$1" tag="$2" num_envs="$3" report="$4" smoke_pair="${5:-}"
    local arm out log
    case "$tag" in
        explicit) arm=c_prior_explicit ;;
        snmr) arm=a_prior_snmr ;;
        *) printf 'unsupported E71 arm %s\n' "$tag" >&2; return 64 ;;
    esac
    out="$E70_ROOT/students/seed${seed}_${tag}"
    log="$E71_ROOT/logs/$(basename "${report%.json}").log"
    test -f "$out/${arm}_student.pt"
    if [[ -e "$report" || -e "$log" ]]; then
        printf 'refusing existing E71 cell output: %s or %s\n' "$report" "$log" >&2
        return 2
    fi
    require_gpu
    local -a smoke_env=()
    if [[ -n "$smoke_pair" ]]; then
        smoke_env+=(E71_SMOKE_PAIR_ID="$smoke_pair")
    fi
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        env \
            E52_ARM="$arm" \
            E52_OUT="$out" \
            E52_TEACHER_MANIFEST="$MANIFEST" \
            E71_STARTS_JSON="$PRECHECK" \
            E71_FREEZE_MANIFEST="$ACTIVE_MANIFEST" \
            E71_REPORT_PATH="$report" \
            E71_TRAINING_SEED="$seed" \
            "${smoke_env[@]}" \
            PYTHONPATH="$SNMR_ROOT" \
            nice -n 10 "$WBT_PYTHON" "$SNMR_ROOT/scripts/eval_e71_command_swap.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs "$num_envs" \
            --training.seed 404 \
            --training.headless True \
            --training.name "e71_${tag}_seed${seed}" \
            --logger.base-dir "$E71_ROOT/holosoma_logs" \
            --simulator.config.sim.max-episode-length-s 20.0 \
            --command.setup-terms.motion-command.params.motion-config.motion-file "" \
            --command.setup-terms.motion-command.params.motion-config.motion-dir "$MOTION_ROOT"
    ) 2>&1 | tee "$log"
}

if [[ "$mode" == "smoke" ]]; then
    smoke_pair_id="$(manifest_smoke_pair "$ACTIVE_MANIFEST")"
    [[ "$smoke_pair_id" =~ ^[0-9]+$ ]]
    smoke_report="$E71_ROOT/reports/smoke_explicit_seed0_pair${smoke_pair_id}.json"
    smoke_audit="$E71_ROOT/smoke_audit.json"
    run_report 0 explicit 4 \
        "$smoke_report" \
        "$smoke_pair_id"
    "$SNMR_ROOT/.venv/bin/python" "$SNMR_ROOT/scripts/audit_e71_bundle.py" \
        --manifest "$ACTIVE_MANIFEST" \
        --smoke-report "$smoke_report" \
        --smoke-out "$smoke_audit"
    exit 0
fi

explicit_reports=()
for seed in 0 1 2; do
    report="$E71_ROOT/reports/explicit_seed${seed}.json"
    run_report "$seed" explicit 276 "$report"
    explicit_reports+=("$report")
done

"$SNMR_ROOT/.venv/bin/python" "$SNMR_ROOT/scripts/audit_e71_bundle.py" \
    --manifest "$E71_MANIFEST" \
    --explicit-reports "${explicit_reports[@]}" \
    --explicit-preflight

gate="$E71_ROOT/explicit_gate.json"
"$SNMR_ROOT/.venv/bin/python" "$SNMR_ROOT/scripts/analyze_e71_command_swap.py" \
    --explicit-reports "${explicit_reports[@]}" \
    --gate-out "$gate"
"$SNMR_ROOT/.venv/bin/python" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["valid_explicit_gate"] else 3)' \
    "$gate"

snmr_reports=()
for seed in 0 1 2; do
    report="$E71_ROOT/reports/snmr_seed${seed}.json"
    run_report "$seed" snmr 276 "$report"
    snmr_reports+=("$report")
done

analysis="$E71_ROOT/analysis.json"
"$SNMR_ROOT/.venv/bin/python" "$SNMR_ROOT/scripts/analyze_e71_command_swap.py" \
    --explicit-reports "${explicit_reports[@]}" \
    --snmr-reports "${snmr_reports[@]}" \
    --out "$analysis"
"$SNMR_ROOT/.venv/bin/python" "$SNMR_ROOT/scripts/audit_e71_bundle.py" \
    --manifest "$E71_MANIFEST" \
    --explicit-reports "${explicit_reports[@]}" \
    --snmr-reports "${snmr_reports[@]}" \
    --gate "$gate" \
    --analysis "$analysis" \
    --out "$E71_ROOT/integrity_bundle.json"
