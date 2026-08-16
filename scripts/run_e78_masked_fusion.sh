#!/usr/bin/env bash
set -euo pipefail

# E78 — masked-fusion prototype (Track A of docs/LATENT_BENEFIT_PROGRAM_2026-08-15.md).
#
# STATUS: PROTOTYPE HARNESS.  Gates and arms are preregistered in the program plan (§E4);
# this launcher does not decide anything, it only produces the arrays the plan's analysis
# consumes.  Nothing here touches the frozen E70 tree: frozen students are reached by
# symlink and evaluated read-only; everything is written under $E78_ROOT.
#
# usage:
#   scripts/run_e78_masked_fusion.sh train  <seed> <arm-tag>
#   scripts/run_e78_masked_fusion.sh sweep  <seed> <arm-tag>      # clean + dropout severities
#   scripts/run_e78_masked_fusion.sh frozen <seed> {explicit|snmr} # frozen E70 student, same sweep
#
# arm tags (all trained WITH reference dropout, scope=$E78_MASK_SCOPE, fraction $E78_MASK_FRAC):
#   mE   explicit-only                          (c_prior_explicit)
#   mS   snmr-only                              (a_prior_snmr)
#   mZc  explicit+snmr, concat fusion           (d_prior_explicit_snmr, E78_FUSION=concat)
#   mZf  explicit+snmr, FiLM fusion             (E78_FUSION=film)
#   mZg  explicit+snmr, gated-residual fusion   (E78_FUSION=gated)
#   mTf  explicit+TIME-CODE, FiLM  (control: latent replaced by the E70 time code)
#   mShf explicit+SHUFFLED z, FiLM (control: other clip's latent at matched phase)
#
# Severity sweep (physical units: masked-tick fraction x segment length in ticks at 50 Hz),
# seeded identically for every arm (E78_EVAL_MASK_SEED=404) so contrasts are paired.

if (( $# != 3 )); then
    printf 'usage: %s {train|sweep|frozen} SEED TAG\n' "$0" >&2; exit 64
fi
mode="$1"; seed="$2"; tag="$3"
[[ "$seed" =~ ^[0-9]+$ ]]

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
E78_ROOT="${E78_ROOT:-/data/robotixx/snmr-research/e78_masked_fusion}"
MOTION_ROOT="$E70_ROOT/motions"
MANIFEST="$E70_ROOT/teacher_manifest.json"
NUM_ENVS="${E78_NUM_ENVS:-1024}"
MIN_FREE_MB="${E78_MIN_FREE_MB:-26000}"
MASK_FRAC="${E78_MASK_FRAC:-0.3}"
MASK_SCOPE="${E78_MASK_SCOPE:-all}"
SWEEP_FRACS="${E78_SWEEP_FRACS:-0.1 0.3 0.5}"
SWEEP_SEGS="${E78_SWEEP_SEGS:-5-25 25-50}"

test -f "$MANIFEST"; test -d "$MOTION_ROOT"

case "$tag" in
    mE)   arm=c_prior_explicit;      fusion=concat; phase=0; shuffle=0 ;;
    mS)   arm=a_prior_snmr;          fusion=concat; phase=0; shuffle=0 ;;
    mZc)  arm=d_prior_explicit_snmr; fusion=concat; phase=0; shuffle=0 ;;
    mZf)  arm=d_prior_explicit_snmr; fusion=film;   phase=0; shuffle=0 ;;
    mZg)  arm=d_prior_explicit_snmr; fusion=gated;  phase=0; shuffle=0 ;;
    mTf)  arm=d_prior_explicit_snmr; fusion=film;   phase=1; shuffle=0 ;;
    mShf) arm=d_prior_explicit_snmr; fusion=film;   phase=0; shuffle=1 ;;
    explicit) arm=c_prior_explicit;  fusion=concat; phase=0; shuffle=0 ;;   # frozen only
    snmr)     arm=a_prior_snmr;      fusion=concat; phase=0; shuffle=0 ;;   # frozen only
    *) printf 'unknown E78 tag %s\n' "$tag" >&2; exit 64 ;;
esac

free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
    printf 'E78 requires %s MiB free GPU memory; observed %s MiB.\n' "$MIN_FREE_MB" "$free_mb" >&2
    exit 5
fi

if [[ "$mode" == frozen ]]; then
    out="$E78_ROOT/frozen_seed${seed}_${tag}"
    flag_dim=0
    mkdir -p "$out"
    ln -sfn "$E70_ROOT/students/seed${seed}_${tag}/${arm}_student.pt" "$out/${arm}_student.pt"
else
    out="$E78_ROOT/seed${seed}_${tag}"
    flag_dim=2
    mkdir -p "$out"
fi
mkdir -p "$E78_ROOT/logs" "$E78_ROOT/holosoma_logs"

run_cell() {   # run_cell <run_seed> <log> <extra env...>
    local run_seed="$1" log="$2"; shift 2
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        env \
            E52_ARM="$arm" E52_TEACHER_MANIFEST="$MANIFEST" E52_OUT="$out" \
            E52_ROUNDS="${E78_ROUNDS:-2000}" E52_DET=1 E52_PHASE_ONLY="$phase" \
            E52_SHUFFLE_LATENT="$shuffle" E52_REPLAY_ROUNDS=4 \
            E52_TEACHER_FLOOR=0.1 E52_TEACHER_ANNEAL_ROUNDS=200 \
            E52_CKPT_EVERY=50 E52_VAL_EVERY=50 E52_VAL_SAMPLES=4096 E52_BEST_AFTER=50 \
            E78_FUSION="$fusion" E78_FLAG_DIM="$flag_dim" \
            E78_MASK_FRAC="$MASK_FRAC" E78_MASK_SCOPE="$MASK_SCOPE" \
            E78_MASK_SEG_MIN=5 E78_MASK_SEG_MAX=25 E78_MASK_RAMP_ROUNDS=300 \
            "$@" \
            PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
            "$SNMR_ROOT/scripts/train_e78_masked_fusion.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs "$NUM_ENVS" --training.seed "$run_seed" \
            --training.headless True --training.name "e78_$(basename "$out")" \
            --logger.base-dir "$E78_ROOT/holosoma_logs" \
            --randomization.ignore-unsupported True \
            --simulator.config.sim.max-episode-length-s 100000.0 \
            --command.setup-terms.motion-command.params.motion-config.motion-file "" \
            --command.setup-terms.motion-command.params.motion-config.motion-dir "$MOTION_ROOT"
    ) > "$log" 2>&1
}

case "$mode" in
    train)
        if [[ -e "$out/${arm}_student.pt" ]]; then
            printf 'refusing nonempty training cell %s\n' "$out" >&2; exit 2
        fi
        run_cell "$seed" "$out/train.log" E52_SKIP_INLINE_EVAL=1
        ;;
    sweep|frozen)
        test -f "$out/${arm}_student.pt"
        # clean condition first (mandatory: reproduces the frozen numbers for `frozen`)
        [[ -e "$out/${arm}_eval.json" ]] || run_cell 404 "$out/eval_clean.log" E52_EVAL_ONLY=1
        for f in $SWEEP_FRACS; do for seg in $SWEEP_SEGS; do
            lo="${seg%-*}"; hi="${seg#*-}"
            report="$out/${arm}_eval_mask${MASK_SCOPE}_hold_f${f}_s${lo}-${hi}.json"
            [[ -e "$report" ]] && continue
            run_cell 404 "$out/eval_mask_f${f}_s${seg}.log" E52_EVAL_ONLY=1 \
                E78_EVAL_MASK_FRAC="$f" E78_EVAL_MASK_SEG_MIN="$lo" E78_EVAL_MASK_SEG_MAX="$hi" \
                E78_EVAL_MASK_SCOPE="$MASK_SCOPE" E78_EVAL_MASK_SEED=404
        done; done
        ;;
    *) printf 'unknown mode %s\n' "$mode" >&2; exit 64 ;;
esac
printf 'E78 %s %s seed %s done -> %s\n' "$mode" "$tag" "$seed" "$out"
