#!/usr/bin/env bash
set -euo pipefail

# E77 PILOT: locate the interpretable severity window for degraded-command experiments.
#
# THIS IS A DESCRIPTIVE PILOT, NOT A CONFIRMATORY EXPERIMENT.  It has no gate, no preregistration,
# and no seed replication.  Its only purpose is to answer one prerequisite question before any
# degraded-command study is designed:
#
#     Is there a corruption severity at which BOTH arms are neither saturated nor broken?
#
# E61-v4 failed precisely here -- its only positive sigma was one where "both arms are broken
# (~0.1 completion)".  Any confirmatory design must first prove the window exists.  Nothing
# measured here may be promoted to a confirmatory result without its own preregistration.
#
# Three severity axes, all using existing eval-only knobs on the frozen trainer (no code edited):
#   hold   -- zero-order hold on z_cmd, the SHARED bottleneck.  Matched across arms by construction:
#             identical units (control ticks), identical tensor, identical decoder.
#   noise_z   -- Gaussian sigma on the standardized SNMR latent window.  SNMR arm only.
#   noise_cmd -- Gaussian sigma on the 64-D explicit motion-command observation.  Explicit arm only.
#             (For the SNMR arm the explicit cmd tensor is structurally ignored, so noise_cmd is a
#             no-op there -- it is run on the explicit arm only, deliberately.)
#
# Evaluation-only on frozen checkpoints; writes only under $E77_ROOT; general start grid, eval seed
# 404, 1024 rollouts, matching the frozen E75/E65 protocol.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
E77_ROOT="${E77_ROOT:-/data/robotixx/snmr-research/e77_degradation_pilot}"
MOTION_ROOT="$E70_ROOT/motions"
MANIFEST="$E70_ROOT/teacher_manifest.json"
FROZEN_STUDENTS="$E70_ROOT/students"
NUM_ENVS="${E77_NUM_ENVS:-1024}"
SEED="${E77_SEED:-0}"
MIN_FREE_MB="${E77_MIN_FREE_MB:-26000}"

free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
    printf 'E77 requires %s MiB free GPU memory; observed %s MiB.\n' "$MIN_FREE_MB" "$free_mb" >&2
    exit 5
fi
printf 'E77 pilot: capacity gate passed with %s MiB free.\n' "$free_mb"

mkdir -p "$E77_ROOT/logs" "$E77_ROOT/holosoma_logs"

# cell <tag> <arm> <phase_only> <label> <extra env assignments...>
cell() {
    local tag="$1" arm="$2" phase="$3" label="$4"; shift 4
    local out="$E77_ROOT/${tag}/${label}"
    local report="$out/${arm}_eval.json"
    [[ -f "$report" ]] && { printf 'E77 skip (done): %s/%s\n' "$tag" "$label"; return 0; }
    mkdir -p "$out"
    ln -sfn "$FROZEN_STUDENTS/seed${SEED}_${tag}/${arm}_student.pt" "$out/${arm}_student.pt"
    printf 'E77 running %s %s\n' "$tag" "$label"
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        env \
            E52_ARM="$arm" E52_TEACHER_MANIFEST="$MANIFEST" E52_OUT="$out" \
            E52_ROUNDS=2000 E52_DET=1 E52_PHASE_ONLY="$phase" \
            E52_SHUFFLE_LATENT=0 E52_REPLAY_ROUNDS=4 \
            E52_TEACHER_FLOOR=0.1 E52_TEACHER_ANNEAL_ROUNDS=200 \
            E52_CKPT_EVERY=50 E52_VAL_EVERY=50 E52_VAL_SAMPLES=4096 \
            E52_BEST_AFTER=50 E52_EVAL_ONLY=1 \
            "$@" \
            PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
            "$SNMR_ROOT/scripts/train_e52_dagger.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs "$NUM_ENVS" --training.seed 404 \
            --training.headless True --training.name "e77_${tag}_${label}" \
            --logger.base-dir "$E77_ROOT/holosoma_logs" \
            --randomization.ignore-unsupported True \
            --simulator.config.sim.max-episode-length-s 100000.0 \
            --command.setup-terms.motion-command.params.motion-config.motion-file "" \
            --command.setup-terms.motion-command.params.motion-config.motion-dir "$MOTION_ROOT"
    ) > "$E77_ROOT/logs/${tag}_${label}.log" 2>&1
}

# --- axis 1: zero-order hold on the shared z_cmd bottleneck (matched across arms) -------------
for k in 1 2 5 10 20; do
    cell snmr     a_prior_snmr     0 "hold_k${k}" E52_EVAL_HOLD_Z="$k"
    cell explicit c_prior_explicit 0 "hold_k${k}" E52_EVAL_HOLD_Z="$k"
done

# --- axis 2: Gaussian noise on each arm's OWN upstream channel --------------------------------
for s in 0.1 0.25 0.5 1.0 2.0; do
    cell snmr     a_prior_snmr     0 "noisez_${s}"   E52_EVAL_NOISE_ZRET="$s"
    cell explicit c_prior_explicit 0 "noisecmd_${s}" E52_EVAL_NOISE_CMD="$s"
done

printf 'E77 pilot complete under %s\n' "$E77_ROOT"
