#!/usr/bin/env bash
set -euo pipefail

# E49 act-through-latent WBT co-training (docs/E49_ACT_THROUGH_LATENT_PROTOCOL.md).
# Config-only extension of the Phase-1 L1 arm. Controls (NOT retrained here):
#   baseline explicit-command = 0.88 completion (runs/wbt_horizon_calibration GMR seed0/eval404)
#   l1_frozen tangent+MLP      = 0.72 completion (runs/wbt_latent_phase1, seed0/eval404)
# E49 arms (seed0, eval404, 8k iters, matched to Phase-1):
#   e49_window_mlp  raw latent window [z, z+.1, z+.2, z+.5s] -> plain MLP  (zero config risk; L1 path)
#   e49_window_enc  same window -> MLPEncoder (enc 256->64)               (Stage-0 gated; may drop)
# Reward + termination stay on the explicit GMR robot-space reference (unchanged); only actor obs changes.
# GPU: one job at a time. Stage 0 (window_enc smoke) and Stage 2 (full arms) are separate invocations
# controlled by E49_STAGE. Never launch while another GPU job runs.

MAIN=/home/ec2-user/work/retarget/snmr
HOLOSOMA=/home/ec2-user/work/retarget/holosoma
PYTHON=/home/ec2-user/work/retarget/.venv-wbt/bin/python
HOLOSOMA_REV=9fb2b57470e3863dadb9d98719504a7a5d67a9d7
REFERENCE="$MAIN/runs/wbt_validation/gmr/walk1_subject5_mj.npz"
AUGMENTED="$MAIN/runs/wbt_latent_gmr/walk1_subject5_mj_z.npz"
OUT="$MAIN/runs/e49_act_through_latent"
TRAINING_SEED=0
EVALUATION_SEED=404
NUM_ENVS="${E49_NUM_ENVS:-1024}"
ITERATIONS="${E49_ITERS:-8000}"
SAVE_INTERVAL="${E49_SAVE:-2000}"
NUM_ROLLOUTS=100
HORIZON_STEPS=500
HORIZON_S=10.0
STAGE="${E49_STAGE:-smoke_enc}"   # smoke_enc | window_mlp | window_enc

WINDOW_FUNC=snmr.integration.wbt_latent:snmr_latent_window
PREVIEW_FUNC=snmr.integration.wbt_latent:motion_command_with_latent_preview  # critic (privileged), as in L1

export PYTHONPATH="$MAIN"

test "$(git -C "$HOLOSOMA" rev-parse HEAD)" = "$HOLOSOMA_REV"
test -z "$(git -C "$HOLOSOMA" status --porcelain --untracked-files=no)"
test -f "$REFERENCE"
test -f "$AUGMENTED"
mkdir -p "$OUT/reports"

run_train() {
  local TRAIN_NAME="$1"; shift
  echo "=== $TRAIN_NAME (stage=$STAGE, envs=$NUM_ENVS, iters=$ITERATIONS) start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  cd "$HOLOSOMA"
  nice -n 15 "$PYTHON" src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    simulator:mjwarp \
    logger:disabled \
    --training.num-envs "$NUM_ENVS" \
    --training.seed "$TRAINING_SEED" \
    --algo.config.num-learning-iterations "$ITERATIONS" \
    --algo.config.save-interval "$SAVE_INTERVAL" \
    --randomization.ignore-unsupported True \
    --command.setup-terms.motion-command.params.motion-config.motion-file "$AUGMENTED" \
    --training.name "$TRAIN_NAME" \
    "$@" \
    >> "$OUT/$TRAIN_NAME.train.log" 2>&1
  echo "=== $TRAIN_NAME train exit $? $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
}

case "$STAGE" in
  # Stage 0: prove MLPEncoder + new-obs-group wiring constructs & steps. Tiny + short.
  smoke_enc)
    E49_NUM_ENVS=64; NUM_ENVS=64; ITERATIONS=20; SAVE_INTERVAL=20
    run_train e49_smoke_enc \
      --observation.groups.actor-obs.terms.motion-command.func "$WINDOW_FUNC" \
      --algo.config.module-dict.actor.type MLPEncoder \
      --algo.config.module-dict.actor.layer-config.encoder-input-name actor_obs \
      --algo.config.module-dict.actor.layer-config.encoder-hidden-dims '[256]' \
      --algo.config.module-dict.actor.layer-config.encoder-output-dim 64
    echo "smoke_enc finished; inspect $OUT/e49_smoke_enc.train.log for an 'Actor Module' with an encoder submodule and finite loss." | tee -a "$OUT/driver.log"
    ;;

  # Stage 2a: primary config-only arm — raw window -> plain MLP (exact L1 mechanism, zero config risk).
  window_mlp)
    run_train e49_window_mlp_walk1_seed0_to8000 \
      --observation.groups.actor-obs.terms.motion-command.func "$WINDOW_FUNC" \
      --observation.groups.critic-obs.terms.motion-command.func "$PREVIEW_FUNC"
    ;;

  # Stage 2b: treatment — raw window -> MLPEncoder bottleneck (only if smoke_enc passed).
  window_enc)
    run_train e49_window_enc_walk1_seed0_to8000 \
      --observation.groups.actor-obs.terms.motion-command.func "$WINDOW_FUNC" \
      --observation.groups.critic-obs.terms.motion-command.func "$PREVIEW_FUNC" \
      --algo.config.module-dict.actor.type MLPEncoder \
      --algo.config.module-dict.actor.layer-config.encoder-input-name actor_obs \
      --algo.config.module-dict.actor.layer-config.encoder-hidden-dims '[256]' \
      --algo.config.module-dict.actor.layer-config.encoder-output-dim 64
    ;;
  *)
    echo "unknown E49_STAGE=$STAGE (use smoke_enc | window_mlp | window_enc)" >&2; exit 2 ;;
esac

# --- eval (full arms only): 100 phase-stratified 10-s rollouts, matched to Phase-1/B1 ---
if [[ "$STAGE" == window_mlp || "$STAGE" == window_enc ]]; then
  TRAIN_NAME="e49_${STAGE}_walk1_seed0_to8000"
  RUN_DIR=$(ls -1dt "$HOLOSOMA"/logs/WholeBodyTracking/*-"$TRAIN_NAME"-locomotion 2>/dev/null | head -1)
  test -n "$RUN_DIR"
  CHECKPOINT=$(ls -1 "$RUN_DIR"/model_*.pt | sort -V | tail -1)
  test -f "$CHECKPOINT"
  EVAL_NAME="${TRAIN_NAME}_eval${EVALUATION_SEED}"
  REPORT="$OUT/reports/$EVAL_NAME.json"
  echo "=== $EVAL_NAME start $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
  cd "$HOLOSOMA"
  "$PYTHON" src/holosoma/holosoma/eval_agent.py \
    --checkpoint "$CHECKPOINT" \
    --wbt-metrics.config.enabled \
    --wbt-metrics.config.output-path "$REPORT" \
    --wbt-metrics.config.horizon-s "$HORIZON_S" \
    --training.headless True \
    --training.num-envs "$NUM_ROLLOUTS" \
    --training.seed "$EVALUATION_SEED" \
    --training.max-eval-steps "$HORIZON_STEPS" \
    --training.export-onnx False \
    --simulator.config.sim.max-episode-length-s 100000.0 \
    >> "$OUT/$EVAL_NAME.eval.log" 2>&1
  test -f "$REPORT"
  sha256sum "$CHECKPOINT" "$REPORT" >> "$OUT/input_sha256.txt"
  echo "=== $EVAL_NAME complete $(date -u +%FT%TZ) ===" | tee -a "$OUT/driver.log"
fi
