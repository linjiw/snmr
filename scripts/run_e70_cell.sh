#!/usr/bin/env bash
set -euo pipefail

# Run exactly one frozen E70 train/evaluation cell.  This exists so an externally
# interrupted long queue can restart one cell from round 0 without re-entering or
# modifying completed cells.  It deliberately does not implement partial checkpoint
# resume: the E70 checkpoints do not contain simulator/RNG/replay state.

if (( $# != 3 )); then
    printf 'usage: %s SEED {explicit|snmr|time|proprio|shuffled} {train|general|ambiguity}\n' "$0" >&2
    exit 64
fi

seed="$1"
tag="$2"
mode="$3"
[[ "$seed" =~ ^[0-9]+$ ]]

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
MANIFEST="$E70_ROOT/teacher_manifest.json"
MOTION_ROOT="$E70_ROOT/motions"
PRECHECK="$SNMR_ROOT/autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json"
PRECHECK_HASH=3c03b89e2bda939e9a8c5a6dd58caeb771944bb9e5755b14eb37ab75ed9c502e

case "$tag" in
    explicit) arm=c_prior_explicit; phase=0; shuffle=0 ;;
    snmr) arm=a_prior_snmr; phase=0; shuffle=0 ;;
    time) arm=a_prior_snmr; phase=1; shuffle=0 ;;
    proprio) arm=b_prior_proprio; phase=0; shuffle=0 ;;
    shuffled) arm=a_prior_snmr; phase=0; shuffle=1 ;;
    *) printf 'unknown E70 tag %s\n' "$tag" >&2; exit 64 ;;
esac

case "$mode" in
    train) run_seed="$seed"; mode_env=(E52_SKIP_INLINE_EVAL=1) ;;
    general) run_seed=404; mode_env=(E52_EVAL_ONLY=1) ;;
    ambiguity)
        run_seed=404
        mode_env=(E52_EVAL_ONLY=1 E52_EVAL_STARTS_JSON="$PRECHECK")
        ;;
    *) printf 'unknown E70 cell mode %s\n' "$mode" >&2; exit 64 ;;
esac

test -f "$MANIFEST"
test -d "$MOTION_ROOT"
test "$(sha256sum "$PRECHECK" | cut -d' ' -f1)" = "$PRECHECK_HASH"

free_mb="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' '
)"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < 26000 )); then
    printf 'E70 cell requires 26000 MiB free GPU memory; observed %s MiB.\n' "$free_mb" >&2
    exit 5
fi

out="$E70_ROOT/students/seed${seed}_${tag}"
mkdir -p "$out" "$E70_ROOT/student_holosoma_logs"
case "$mode" in
    train)
        if [[ -e "$out/${arm}_student.pt" || -e "$out/train.log" ]]; then
            printf 'refusing nonempty training cell %s; quarantine it before a round-0 restart\n' "$out" >&2
            exit 2
        fi
        log="$out/train.log"
        ;;
    general)
        test -f "$out/${arm}_student.pt"
        test ! -e "$out/${arm}_eval.json"
        log="$out/general_eval.log"
        ;;
    ambiguity)
        test -f "$out/${arm}_student.pt"
        test ! -e "$out/${arm}_eval_ambiguity.json"
        log="$out/ambiguity_eval.log"
        ;;
esac

printf 'E70 exact cell: seed=%s tag=%s mode=%s free=%s MiB\n' \
    "$seed" "$tag" "$mode" "$free_mb"
(
    cd "$SNMR_HOLOSOMA_ROOT"
    env \
        E52_ARM="$arm" E52_TEACHER_MANIFEST="$MANIFEST" E52_OUT="$out" \
        E52_ROUNDS=2000 E52_DET=1 E52_PHASE_ONLY="$phase" \
        E52_SHUFFLE_LATENT="$shuffle" E52_REPLAY_ROUNDS=4 \
        E52_TEACHER_FLOOR=0.1 E52_TEACHER_ANNEAL_ROUNDS=200 \
        E52_CKPT_EVERY=50 E52_VAL_EVERY=50 E52_VAL_SAMPLES=4096 \
        E52_BEST_AFTER=50 "${mode_env[@]}" \
        PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
        "$SNMR_ROOT/scripts/train_e52_dagger.py" \
        exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
        --training.num-envs 1024 --training.seed "$run_seed" \
        --training.headless True --training.name "e70_seed${seed}_${tag}" \
        --logger.base-dir "$E70_ROOT/student_holosoma_logs" \
        --randomization.ignore-unsupported True \
        --simulator.config.sim.max-episode-length-s 100000.0 \
        --command.setup-terms.motion-command.params.motion-config.motion-file "" \
        --command.setup-terms.motion-command.params.motion-config.motion-dir \
        "$MOTION_ROOT"
) 2>&1 | tee "$log"
