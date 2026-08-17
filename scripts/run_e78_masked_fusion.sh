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
#   scripts/run_e78_masked_fusion.sh frozen <seed> {explicit|snmr|time|shuffled} # frozen E70 student, same sweep
#
# arm tags (all trained WITH reference dropout, scope=$E78_MASK_SCOPE, fraction $E78_MASK_FRAC,
# and identical flag bits — including mE, so no arm's edge is "knowing when it is blind"):
#   mE   explicit-only                          (c_prior_explicit)
#   mB   goal-blind floor                        (b_prior_proprio) -- denominator for R
#   mS   snmr-only                              (a_prior_snmr)
#   mZc  explicit+snmr, concat fusion           (d_prior_explicit_snmr, E78_FUSION=concat)
#   mZf  explicit+snmr, FiLM fusion             (E78_FUSION=film)             <- treatment
#   mZg  explicit+snmr, gated-residual fusion   (E78_FUSION=gated)
#   mGf  explicit + EXPLICIT FUTURE WINDOW [g_t, g_t+0.1s], FiLM (E78_GOAL_WINDOW=1)
#          window-matched control; pre-committed reading: mGf >= mZf is the likely outcome
#          (single embodiment: explicit content is a superset) and is a HANDOFF-style
#          "expose a future window" finding, not a failure.
#   mTf  explicit + E70 time code, FiLM, code FROZEN with the reference during dropout
#          (matched-masking control)
#   mTl  explicit + E70 time code, FiLM, code LIVE from the tick counter (E78_TIME_LIVE=1)
#          (a deployed system never loses its clock; isolates content beyond known time)
#   mShf explicit + SHUFFLED z (other clip at matched phase), FiLM (identity-vs-content control)
#   cfut UNMASKED, flag-free C-future arm for the paper: explicit window [g_t, g_t+0.1s]
#          through the frozen A-arm projection path (a_prior_snmr + E78_GOAL_WINDOW=1,
#          E78_MASK_FRAC=0, E78_FLAG_DIM=0).  Post-hoc, non-registered addition to E70.
#
# Registered GPU order (advisor guidance 2026-08-15): frozen sanity -> mE, mZf + sweep (kill
# check) -> mGf, mTf, mTl, mShf, mZc, mZg, mS.  cfut in any gap.
#
# Severity sweep (physical units: masked-tick fraction x segment length in ticks at 50 Hz),
# seeded identically for every arm (E78_EVAL_MASK_SEED=404) so contrasts are paired.
# `sweep` also evaluates the frozen 69-pair AMBIGUITY start grid (E70 precheck, hash-bound)
# clean and at the registered severities — the co-secondary endpoint where content should
# separate hardest (an early outage leaves frozen z saying which clip; a clock cannot).

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
EVAL_MASK_MODE="${E78_EVAL_MASK_MODE:-hold}"   # hold | zero | extrapolate (fill during an outage)
SWEEP_FRACS="${E78_SWEEP_FRACS:-0.1 0.3 0.5}"
SWEEP_SEGS="${E78_SWEEP_SEGS:-5-25 25-50}"
AMB_SEVERITIES="${E78_AMB_SEVERITIES:-0.3:5-25 0.5:5-25 0.3:25-50}"   # registered secondaries
PRECHECK="$SNMR_ROOT/autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json"
PRECHECK_HASH=3c03b89e2bda939e9a8c5a6dd58caeb771944bb9e5755b14eb37ab75ed9c502e

test -f "$MANIFEST"; test -d "$MOTION_ROOT"; test -f "$PRECHECK"
test "$(sha256sum "$PRECHECK" | cut -d' ' -f1)" = "$PRECHECK_HASH"

gwin=0; tlive=0; arm_mask_frac="$MASK_FRAC"; arm_flag_dim=2
case "$tag" in
    mE)   arm=c_prior_explicit;      fusion=concat; phase=0; shuffle=0 ;;
    mB)   arm=b_prior_proprio;      fusion=concat; phase=0; shuffle=0 ;;   # goal-blind floor,
    # trained under the same masking recipe and carrying the same flag bits.  Dropout is a
    # structural no-op for it, so its curve is the denominator of floor-relative retention
    # (docs/COMMAND_INTERFACE_SYNTHESIS_2026-08-16.md II.1) for the masked family.
    mS)   arm=a_prior_snmr;          fusion=concat; phase=0; shuffle=0 ;;
    mZc)  arm=d_prior_explicit_snmr; fusion=concat; phase=0; shuffle=0 ;;
    mZf)  arm=d_prior_explicit_snmr; fusion=film;   phase=0; shuffle=0 ;;
    mZg)  arm=d_prior_explicit_snmr; fusion=gated;  phase=0; shuffle=0 ;;
    mGf)  arm=d_prior_explicit_snmr; fusion=film;   phase=0; shuffle=0; gwin=1 ;;
    mTf)  arm=d_prior_explicit_snmr; fusion=film;   phase=1; shuffle=0 ;;
    mTl)  arm=d_prior_explicit_snmr; fusion=film;   phase=1; shuffle=0; tlive=1 ;;
    mShf) arm=d_prior_explicit_snmr; fusion=film;   phase=0; shuffle=1 ;;
    cfut) arm=a_prior_snmr;          fusion=concat; phase=0; shuffle=0; gwin=1
          arm_mask_frac=0; arm_flag_dim=0 ;;
    explicit) arm=c_prior_explicit;  fusion=concat; phase=0; shuffle=0 ;;   # frozen only
    snmr)     arm=a_prior_snmr;      fusion=concat; phase=0; shuffle=0 ;;   # frozen only
    time)     arm=a_prior_snmr;      fusion=concat; phase=1; shuffle=0 ;;   # frozen only (E70 T)
    shuffled) arm=a_prior_snmr;      fusion=concat; phase=0; shuffle=1 ;;   # frozen only (E70 S)
    proprio)  arm=b_prior_proprio;   fusion=concat; phase=0; shuffle=0 ;;   # frozen only (E70 B):
    # goal-blind floor.  Dropout is a structural no-op for this arm (its encoder reads no
    # reference at all), so its curve is the flat line every other arm must be read against.
    *) printf 'unknown E78 tag %s\n' "$tag" >&2; exit 64 ;;
esac
if [[ "$mode" != frozen && ( "$tag" == explicit || "$tag" == snmr || "$tag" == time || "$tag" == shuffled || "$tag" == proprio ) ]]; then
    printf 'tags explicit/snmr are frozen-only\n' >&2; exit 64
fi

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
    flag_dim="$arm_flag_dim"
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
            E78_MASK_FRAC="$arm_mask_frac" E78_MASK_SCOPE="$MASK_SCOPE" \
            E78_GOAL_WINDOW="$gwin" E78_TIME_LIVE="$tlive" \
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
        # clean condition first (mandatory sanity: `frozen` must reproduce the seed-exact,
        # hash-bound E70 report values within the E76 evaluation-noise tolerance)
        [[ -e "$out/${arm}_eval.json" ]] || run_cell 404 "$out/eval_clean.log" E52_EVAL_ONLY=1
        [[ -e "$out/${arm}_eval_ambiguity.json" ]] || run_cell 404 "$out/eval_amb_clean.log" \
            E52_EVAL_ONLY=1 E52_EVAL_STARTS_JSON="$PRECHECK"
        if [[ "$mode" == frozen ]]; then
            python "$SNMR_ROOT/scripts/check_e78_frozen_sanity.py" \
                --frozen-dir "$E70_ROOT/students/seed${seed}_${tag}" --arm "$arm" \
                --replay-dir "$out" --tolerance "${E78_SANITY_TOL:-0.02}"
        fi
        for f in $SWEEP_FRACS; do for seg in $SWEEP_SEGS; do
            lo="${seg%-*}"; hi="${seg#*-}"
            report="$out/${arm}_eval_mask${MASK_SCOPE}_${EVAL_MASK_MODE}_f${f}_s${lo}-${hi}.json"
            [[ -e "$report" ]] && continue
            run_cell 404 "$out/eval_${EVAL_MASK_MODE}_f${f}_s${seg}.log" E52_EVAL_ONLY=1 \
                E78_EVAL_MASK_FRAC="$f" E78_EVAL_MASK_SEG_MIN="$lo" E78_EVAL_MASK_SEG_MAX="$hi" \
                E78_EVAL_MASK_SCOPE="$MASK_SCOPE" E78_EVAL_MASK_SEED=404 \
                E78_EVAL_MASK_MODE="$EVAL_MASK_MODE"
        done; done
        for fs in $AMB_SEVERITIES; do
            f="${fs%%:*}"; seg="${fs#*:}"; lo="${seg%-*}"; hi="${seg#*-}"
            report="$out/${arm}_eval_ambiguity_mask${MASK_SCOPE}_${EVAL_MASK_MODE}_f${f}_s${lo}-${hi}.json"
            [[ -e "$report" ]] && continue
            run_cell 404 "$out/eval_amb_${EVAL_MASK_MODE}_f${f}_s${seg}.log" E52_EVAL_ONLY=1 \
                E52_EVAL_STARTS_JSON="$PRECHECK" \
                E78_EVAL_MASK_FRAC="$f" E78_EVAL_MASK_SEG_MIN="$lo" E78_EVAL_MASK_SEG_MAX="$hi" \
                E78_EVAL_MASK_SCOPE="$MASK_SCOPE" E78_EVAL_MASK_SEED=404 \
                E78_EVAL_MASK_MODE="$EVAL_MASK_MODE"
        done
        ;;
    *) printf 'unknown mode %s\n' "$mode" >&2; exit 64 ;;
esac
printf 'E78 %s %s seed %s done -> %s\n' "$mode" "$tag" "$seed" "$out"
