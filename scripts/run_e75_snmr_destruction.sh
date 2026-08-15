#!/usr/bin/env bash
set -euo pipefail

# E75: command destruction on the frozen SNMR students.
#
# This is a REPLICATION of the frozen explicit-arm destruction in
# scripts/run_e70_multitraj.sh:202-216 (`run_explicit_destruction`), changed only in the arm.
# Every other registered field -- start grid (general), evaluation seed (404), rollouts (1024),
# determinism, rounds, phase/shuffle flags -- is copied verbatim from that launcher's
# `student_command`, so the two arms are directly comparable.
#
# Preregistration: docs/E75_SNMR_DESTRUCTION_PREREG.md (written before any run).
#
# Isolation: writes ONLY under $E75_ROOT. The frozen E70 checkpoints are reached through
# symlinks and hash-verified before use. Nothing under /data/robotixx/snmr-research/e70/ is
# written, moved, or deleted.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
E75_ROOT="${E75_ROOT:-/data/robotixx/snmr-research/e75_snmr_destruction}"
MOTION_ROOT="$E70_ROOT/motions"
MANIFEST="$E70_ROOT/teacher_manifest.json"
FROZEN_STUDENTS="$E70_ROOT/students"
NUM_ENVS="${E75_NUM_ENVS:-1024}"
ROUNDS="${E75_ROUNDS:-2000}"
MIN_FREE_MB="${E75_MIN_FREE_MB:-26000}"

ARM=a_prior_snmr
TAG=snmr
SEEDS=(0 1 2)
MODES=(zero shuffle marginal_random)

# Registered in docs/E75_SNMR_DESTRUCTION_PREREG.md section 3 and, independently, in
# docs/E71_COMMAND_SWAP_PROTOCOL.md section 3.
CKPT_HASHES=(
    f88984971c3435e3c377f038ed2ef5abef788aa1a0f68a80ad7011b23bb9b93a
    185ac3991cb6bcdd451d719c72d5d72273d4351a45bd93e7f532ebeaec730d38
    6d23363133df4ba30f6ddb12887aa22b932f6255a9e27031bdf57daf683e42c1
)

hash_is() {
    local path="$1" expected="$2" actual
    actual="$(sha256sum "$path" | cut -d' ' -f1)"
    if [[ "$actual" != "$expected" ]]; then
        printf 'SHA-256 mismatch for %s: expected %s, got %s\n' \
            "$path" "$expected" "$actual" >&2
        return 2
    fi
}

free_mb="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' '
)"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
    printf 'E75 requires %s MiB free GPU memory; observed %s MiB. No output was changed.\n' \
        "$MIN_FREE_MB" "$free_mb" >&2
    exit 5
fi
printf 'E75 capacity gate passed with %s MiB free GPU memory.\n' "$free_mb"

# Fail closed before any GPU work if a frozen checkpoint has drifted.
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    hash_is "$FROZEN_STUDENTS/seed${seed}_${TAG}/${ARM}_student.pt" "${CKPT_HASHES[$index]}"
done
printf 'E75 verified 3 frozen SNMR checkpoints against the registered hashes.\n'

mkdir -p "$E75_ROOT/students" "$E75_ROOT/logs" "$E75_ROOT/holosoma_logs"

destroy_cell() {
    local seed="$1" mode="$2"
    local out="$E75_ROOT/students/seed${seed}_${TAG}"
    local report="$out/${ARM}_eval_destroy_${mode}.json"
    local log="$E75_ROOT/logs/seed${seed}_destroy_${mode}.log"

    if [[ -f "$report" ]]; then
        printf 'E75 cell already complete, skipping: %s\n' "$report"
        return 0
    fi

    mkdir -p "$out"
    # eval-only reads exactly one file out of $E52_OUT: "${arm}_student.pt".  A symlink keeps the
    # frozen checkpoint canonical while every write lands under $E75_ROOT.
    ln -sfn "$FROZEN_STUDENTS/seed${seed}_${TAG}/${ARM}_student.pt" "$out/${ARM}_student.pt"

    printf 'E75 running seed %s mode %s\n' "$seed" "$mode"
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        env \
            E52_ARM="$ARM" E52_TEACHER_MANIFEST="$MANIFEST" E52_OUT="$out" \
            E52_ROUNDS="$ROUNDS" E52_DET=1 E52_PHASE_ONLY=0 \
            E52_SHUFFLE_LATENT=0 E52_REPLAY_ROUNDS=4 \
            E52_TEACHER_FLOOR=0.1 E52_TEACHER_ANNEAL_ROUNDS=200 \
            E52_CKPT_EVERY=50 E52_VAL_EVERY=50 E52_VAL_SAMPLES=4096 \
            E52_BEST_AFTER=50 \
            E52_EVAL_ONLY=1 E52_EVAL_DESTROY_ZCMD="$mode" \
            PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
            "$SNMR_ROOT/scripts/train_e52_dagger.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs "$NUM_ENVS" --training.seed 404 \
            --training.headless True --training.name "e75_seed${seed}_${TAG}_destroy_${mode}" \
            --logger.base-dir "$E75_ROOT/holosoma_logs" \
            --randomization.ignore-unsupported True \
            --simulator.config.sim.max-episode-length-s 100000.0 \
            --command.setup-terms.motion-command.params.motion-config.motion-file "" \
            --command.setup-terms.motion-command.params.motion-config.motion-dir \
            "$MOTION_ROOT"
    ) 2>&1 | tee "$log"

    if [[ ! -f "$report" ]]; then
        printf 'E75 cell produced no report: %s\n' "$report" >&2
        return 4
    fi
}

for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        destroy_cell "$seed" "$mode"
    done
done

printf 'E75 complete: 9 cells under %s\n' "$E75_ROOT/students"
