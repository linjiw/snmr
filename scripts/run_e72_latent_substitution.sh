#!/usr/bin/env bash
set -euo pipefail

# E72: source intervention on the SNMR latent, delivered as substituted-latent motion NPZs.
#
# Derived from scripts/run_e70_multitraj.sh (`student_command`, eval_mode=ambiguity, arm
# a_prior_snmr, run_seed 404) with exactly three deviations, all recorded in
# docs/E72_LATENT_SUBSTITUTION_PROTOCOL.md section 8:
#   1. motion-dir points at the arm's substituted motion directory;
#   2. E52_OUT and --logger.base-dir point under the new E72 root;
#   3. the frozen student checkpoint is reached by symlink so nothing is written under
#      /data/robotixx/snmr-research/e70/.
# Every other registered field -- ambiguity start grid, evaluation seed 404, 1024 rollouts,
# determinism, phase/shuffle flags -- is copied verbatim.
#
# The `control` arm runs FIRST and is a mandatory validity check, not a scientific arm: its
# motions are byte-identical to the frozen E70 motions, so its replication distribution must
# contain the frozen per-seed ambiguity completion. If it does not, no other arm may be
# interpreted. (Amended 2026-08-15 from exact equality -- see E76.)

SNMR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNMR_ROOT/scripts/activate_snmr.sh"

E70_ROOT="${E70_ROOT:-/data/robotixx/snmr-research/e70}"
E72_ROOT="${E72_ROOT:-/data/robotixx/snmr-research/e72_latent_sub}"
MOTION_ARMS_ROOT="$E72_ROOT/motions"
MANIFEST="$E70_ROOT/teacher_manifest.json"
FROZEN_STUDENTS="$E70_ROOT/students"
PRECHECK="${E72_PRECHECK:-$SNMR_ROOT/autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json}"
PRECHECK_HASH=3c03b89e2bda939e9a8c5a6dd58caeb771944bb9e5755b14eb37ab75ed9c502e
NUM_ENVS="${E72_NUM_ENVS:-1024}"
# Amendment 1 (2026-08-15): the harness is not bit-reproducible (E76, per-arm sd 0.0083), so
# every cell is replicated and the unit of analysis is the mean over repeats.
REPEATS="${E72_REPEATS:-3}"
CONTROL_SD="${E72_CONTROL_SD:-0.008310}"
MIN_FREE_MB="${E72_MIN_FREE_MB:-26000}"

ARM=a_prior_snmr
TAG=snmr
SEEDS=(0 1 2)
# control FIRST -- registered mandatory determinism check.
LATENT_ARMS=(control shift_m0250 shift_p0250 shift_p0500 first_frame clip_mean)

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

hash_is "$PRECHECK" "$PRECHECK_HASH"

free_mb="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' '
)"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
    printf 'E72 requires %s MiB free GPU memory; observed %s MiB. No output was changed.\n' \
        "$MIN_FREE_MB" "$free_mb" >&2
    exit 5
fi
printf 'E72 capacity gate passed with %s MiB free GPU memory.\n' "$free_mb"

for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    hash_is "$FROZEN_STUDENTS/seed${seed}_${TAG}/${ARM}_student.pt" "${CKPT_HASHES[$index]}"
done
for latent_arm in "${LATENT_ARMS[@]}"; do
    for clip in walk1_subject1 walk1_subject5; do
        if [[ ! -f "$MOTION_ARMS_ROOT/$latent_arm/${clip}_mj_z.npz" ]]; then
            printf 'E72 missing substituted motion: %s/%s\n' "$latent_arm" "$clip" >&2
            exit 3
        fi
    done
done
printf 'E72 verified 3 frozen checkpoints and 6 substituted motion arms.\n'

mkdir -p "$E72_ROOT/students" "$E72_ROOT/logs" "$E72_ROOT/holosoma_logs"

eval_cell() {
    local latent_arm="$1" seed="$2" repeat="$3"
    local out="$E72_ROOT/students/${latent_arm}/seed${seed}_${TAG}/repeat${repeat}"
    local report="$out/${ARM}_eval_ambiguity.json"
    local log="$E72_ROOT/logs/${latent_arm}_seed${seed}_r${repeat}.log"

    if [[ -f "$report" ]]; then
        printf 'E72 cell already complete, skipping: %s\n' "$report"
        return 0
    fi

    mkdir -p "$out"
    ln -sfn "$FROZEN_STUDENTS/seed${seed}_${TAG}/${ARM}_student.pt" "$out/${ARM}_student.pt"

    printf 'E72 running arm %s seed %s repeat %s/%s\n' "$latent_arm" "$seed" "$repeat" "$REPEATS"
    (
        cd "$SNMR_HOLOSOMA_ROOT"
        env \
            E52_ARM="$ARM" E52_TEACHER_MANIFEST="$MANIFEST" E52_OUT="$out" \
            E52_ROUNDS=2000 E52_DET=1 E52_PHASE_ONLY=0 \
            E52_SHUFFLE_LATENT=0 E52_REPLAY_ROUNDS=4 \
            E52_TEACHER_FLOOR=0.1 E52_TEACHER_ANNEAL_ROUNDS=200 \
            E52_CKPT_EVERY=50 E52_VAL_EVERY=50 E52_VAL_SAMPLES=4096 \
            E52_BEST_AFTER=50 \
            E52_EVAL_ONLY=1 E52_EVAL_STARTS_JSON="$PRECHECK" \
            PYTHONPATH="$SNMR_ROOT" nice -n 10 "$WBT_PYTHON" \
            "$SNMR_ROOT/scripts/train_e52_dagger.py" \
            exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
            --training.num-envs "$NUM_ENVS" --training.seed 404 \
            --training.headless True --training.name "e72_${latent_arm}_seed${seed}_r${repeat}" \
            --logger.base-dir "$E72_ROOT/holosoma_logs" \
            --randomization.ignore-unsupported True \
            --simulator.config.sim.max-episode-length-s 100000.0 \
            --command.setup-terms.motion-command.params.motion-config.motion-file "" \
            --command.setup-terms.motion-command.params.motion-config.motion-dir \
            "$MOTION_ARMS_ROOT/$latent_arm"
    ) 2>&1 | tee "$log"

    if [[ ! -f "$report" ]]; then
        printf 'E72 cell produced no report: %s\n' "$report" >&2
        return 4
    fi
}

# Registered gate, as amended 2026-08-15 (docs/E72_LATENT_SUBSTITUTION_PROTOCOL.md, Amendment 1):
# the frozen per-seed ambiguity completion must lie within 3 sd of the control arm's own
# replication mean.  Exact equality was the original requirement and is not achievable -- the
# harness is not bit-reproducible (E76, per-arm sd 0.0083), and a re-run of the FROZEN motion
# directory does not reproduce the frozen number either.
check_control() {
    "$SNMR_ROOT/.venv/bin/python" - "$E72_ROOT" "$FROZEN_STUDENTS" "$CONTROL_SD" "$REPEATS" <<'PY'
import json
import pathlib
import statistics as st
import sys

e72, frozen = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sd, repeats = float(sys.argv[3]), int(sys.argv[4])
TOLERANCE_SD = 3.0
bad = []
for seed in (0, 1, 2):
    runs = [
        json.loads(
            (
                e72 / "students" / "control" / f"seed{seed}_snmr" / f"repeat{r}"
                / "a_prior_snmr_eval_ambiguity.json"
            ).read_text()
        )["completion_rate"]
        for r in range(1, repeats + 1)
    ]
    mean = st.mean(runs)
    want = json.loads(
        (frozen / f"seed{seed}_snmr" / "a_prior_snmr_eval_ambiguity.json").read_text()
    )["completion_rate"]
    z = (want - mean) / sd
    ok = abs(z) <= TOLERANCE_SD
    print(
        f"control seed{seed}: replication mean {mean:.6f} over {len(runs)} runs "
        f"{[round(v, 4) for v in runs]}; frozen {want:.6f}; z={z:+.2f} -> "
        f"{'ok' if ok else 'OUTSIDE 3 sd'}"
    )
    if not ok:
        bad.append(seed)
raise SystemExit(1 if bad else 0)
PY
}

for seed in "${SEEDS[@]}"; do
    for repeat in $(seq 1 "$REPEATS"); do
        eval_cell control "$seed" "$repeat"
    done
done

if ! check_control; then
    printf '\nE72 STOP: the control arm did not reproduce the frozen ambiguity completions.\n' >&2
    printf 'Per the preregistration, no intervention arm may be interpreted. Not running them.\n' >&2
    exit 6
fi
printf '\nE72 control gate PASSED: the substituted-motion path reproduces the frozen computation\nto within the harness reproducibility measured in E76.\n\n'

for latent_arm in "${LATENT_ARMS[@]}"; do
    [[ "$latent_arm" == "control" ]] && continue
    for seed in "${SEEDS[@]}"; do
        for repeat in $(seq 1 "$REPEATS"); do
            eval_cell "$latent_arm" "$seed" "$repeat"
        done
    done
done

printf 'E72 complete: %s arms x %s seeds under %s\n' \
    "${#LATENT_ARMS[@]}" "${#SEEDS[@]}" "$E72_ROOT/students"
