# MorphoRetarget foundation freeze

This directory is the clean-checkout evidence bundle for source commit
`078d3116aa233ca09c5c68134d9fd5dd5eb55944` on
`feat/morpho-retarget-foundation`.

The source workers ran from detached clean SNMR and Newton worktrees. Every generated simulator
report records these exact revisions with `dirty=false`:

- SNMR: `078d3116aa233ca09c5c68134d9fd5dd5eb55944`
- Newton: `7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a`
- Isaac Lab: `3c6e67bb5c7ada942a6d1884ab69338f57596f77`

## Outcome

- The six local torque-twin intervention checks reproduce exactly. This remains instrumentation,
  not a learned-retargeting result.
- Independently re-hashed MuJoCo and Newton inputs share the same motion, referenced MJCF bundle,
  controller, RobotSpec, and rollout-contract identifiers.
- MuJoCo and Newton again agree that both open-loop endpoint rollouts fail. The proxy remains
  rejected as a ranker, reward, or preference label.
- The deterministic 10,000 x 29 MuJoCo FK reference reproduced bit-for-bit under the clean source
  commit.
- The clean live PhysX worker again stalled after articulation initialization and returned no body
  poses before the 240 s cap. Therefore `g0_evaluated=false`, `g0_pass=false`; there is no FK
  parity claim.
- The complete repository suite passes: 553 passed, 5 skipped, 27 pre-existing warnings.

`foundation_freeze_manifest.json` binds the artifact hashes, commands, source states, test result,
and interpretation boundaries.

## Reproduction

From a clean checkout of the bound SNMR revision, use fresh output paths:

```bash
SNMR_FREEZE_OUT="$(mktemp -d)"

SNMR_HOLOSOMA_ROOT=/home/robotixx/holosoma \
  .venv/bin/python scripts/experiment_dynamics_twins.py \
  --newton-root /path/to/clean/newton \
  --isaac-lab-root /path/to/clean/IsaacLab \
  --out "$SNMR_FREEZE_OUT/dynamics_twins_mujoco.json"

/path/to/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json "$SNMR_FREEZE_OUT/dynamics_twins_mujoco.json" \
  --scale 0.5 --isaac-lab-root /path/to/clean/IsaacLab \
  --out "$SNMR_FREEZE_OUT/newton_torque_0.5.json"

.venv/bin/python scripts/g0_fk_parity_mujoco.py \
  --num-samples 10000 --seed 0 \
  --newton-root /path/to/clean/newton \
  --isaac-lab-root /path/to/clean/IsaacLab \
  --out-npz "$SNMR_FREEZE_OUT/g0_reference.npz" \
  --out-json "$SNMR_FREEZE_OUT/g0_reference.json"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q
```

The Newton worker must import the clean checkout being recorded. If its environment is editable
against a different tree, put the intended clean checkout first on `PYTHONPATH`.

The PhysX worker must run in the Isaac/Holosoma simulator environment. A timeout, missing pose
tensor, mapping error, or asset mismatch is a failed/unevaluated gate, never a skipped pass.

## Interpretation boundary

This bundle freezes contracts and diagnostic infrastructure. It does not establish G0, a learned
RobotSpec-conditioned retargeter, unseen-T1 generalization, dynamics-aware motion adaptation,
strong-verifier qualification, physics preferences, repair, downstream utility, or hardware
readiness.
