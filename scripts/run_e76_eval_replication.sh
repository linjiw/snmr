#!/usr/bin/env bash
set -euo pipefail

# E76: how reproducible is one E70 ambiguity evaluation?
#
# Discovered 2026-08-15 while running the E72 control gate: re-running the frozen SNMR seed-0
# ambiguity evaluation, with the frozen checkpoint, the frozen motion directory, the frozen
# precheck, evaluation seed 404 and E52_DET=1, does NOT reproduce the recorded completion.  About
# 18% of the 1,024 rollout outcomes flip between runs.  Nothing about the inputs differs; the
# harness itself is not bit-reproducible, and 10-second closed-loop rollouts amplify any last-bit
# difference until near-threshold rollouts change outcome.
#
# This script measures the resulting run-to-run spread so the paper can state it, and so E72's
# intervention effects can be judged against a measured noise floor instead of an assumed zero.
#
# Evaluation-only. Reads frozen artifacts, writes only under $E76_ROOT.

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
E76_ROOT="${E76_ROOT:-/data/robotixx/snmr-research/e76_eval_replication}"
MOTION_ROOT="$E70_ROOT/motions"
MANIFEST="$E70_ROOT/teacher_manifest.json"
FROZEN_STUDENTS="$E70_ROOT/students"
PRECHECK="${E76_PRECHECK:-$SNMR_ROOT/autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json}"
NUM_ENVS="${E76_NUM_ENVS:-1024}"
REPEATS="${E76_REPEATS:-8}"
MIN_FREE_MB="${E76_MIN_FREE_MB:-26000}"

# The two arms whose contrast is the paper's headline (A-T), at one seed.  Same start grid, same
# evaluation seed; only the arm differs -- exactly as in the frozen experiment.
#   arm_spec = "<tag>:<arm>:<phase_only>"
ARM_SPECS=("snmr:a_prior_snmr:0" "time:a_prior_snmr:1")
SEED="${E76_SEED:-0}"

free_mb="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' '
)"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
    printf 'E76 requires %s MiB free GPU memory; observed %s MiB.\n' "$MIN_FREE_MB" "$free_mb" >&2
    exit 5
fi
printf 'E76 capacity gate passed with %s MiB free GPU memory.\n' "$free_mb"

mkdir -p "$E76_ROOT/logs" "$E76_ROOT/holosoma_logs"

for spec in "${ARM_SPECS[@]}"; do
    IFS=':' read -r tag arm phase <<<"$spec"
    for repeat in $(seq 1 "$REPEATS"); do
        out="$E76_ROOT/${tag}/seed${SEED}/repeat${repeat}"
        report="$out/${arm}_eval_ambiguity.json"
        if [[ -f "$report" ]]; then
            printf 'E76 repeat already complete, skipping: %s\n' "$report"
            continue
        fi
        mkdir -p "$out"
        ln -sfn "$FROZEN_STUDENTS/seed${SEED}_${tag}/${arm}_student.pt" "$out/${arm}_student.pt"
        printf 'E76 running %s seed %s repeat %s/%s\n' "$tag" "$SEED" "$repeat" "$REPEATS"
        (
            cd "$SNMR_HOLOSOMA_ROOT"
            env \
                E52_ARM="$arm" E52_TEACHER_MANIFEST="$MANIFEST" E52_OUT="$out" \
                E52_ROUNDS=2000 E52_DET=1 E52_PHASE_ONLY="$phase" \
                E52_SHUFFLE_LATENT=0 E52_REPLAY_ROUNDS=4 \
                E52_TEACHER_FLOOR=0.1 E52_TEACHER_ANNEAL_ROUNDS=200 \
                E52_CKPT_EVERY=50 E52_VAL_EVERY=50 E52_VAL_SAMPLES=4096 \
                E52_BEST_AFTER=50 \
                E52_EVAL_ONLY=1 E52_EVAL_STARTS_JSON="$PRECHECK" \
                PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
                "$SNMR_ROOT/scripts/train_e52_dagger.py" \
                exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
                --training.num-envs "$NUM_ENVS" --training.seed 404 \
                --training.headless True --training.name "e76_${tag}_seed${SEED}_r${repeat}" \
                --logger.base-dir "$E76_ROOT/holosoma_logs" \
                --randomization.ignore-unsupported True \
                --simulator.config.sim.max-episode-length-s 100000.0 \
                --command.setup-terms.motion-command.params.motion-config.motion-file "" \
                --command.setup-terms.motion-command.params.motion-config.motion-dir \
                "$MOTION_ROOT"
        ) > "$E76_ROOT/logs/${tag}_seed${SEED}_r${repeat}.log" 2>&1
    done
done

printf 'E76 complete: %s arms x %s repeats under %s\n' "${#ARM_SPECS[@]}" "$REPEATS" "$E76_ROOT"
