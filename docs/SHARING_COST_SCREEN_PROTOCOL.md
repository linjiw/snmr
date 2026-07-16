# Sharing-Cost Screen Protocol

Frozen before training. This screen tests the two leading explanations for the measured shared
model fidelity cost: insufficient shared capacity and insufficient robot-specific conditional
capacity. It does not test gradient surgery because E28 found mostly positive mean gradient
cosines and only moderate norm imbalance.

## Question

Can a wider shared trunk or small per-robot decoder residuals close at least half of the
shared-versus-specialist pose gap under exactly matched robot exposure and LR progress?

## Controls and sampling

- Robots: G1, T1, N1, PM01, and Toddy.
- Seed: 0 for the screen.
- Trainer and objective: `scripts/train_phase2.py` for every arm, including specialists, so the
  distillation, smoothness, limits, and latent-consistency objectives are identical.
- Data: the existing 70/7 train/validation clip split; final evaluation uses the trainer's dense
  16-window-per-clip evaluation.
- Exposure: exactly 50,000 sampled windows per robot.
- Sampling: `balanced_combinations` enumerates every size-K robot subset once per cycle. For five
  robots and K=2, a ten-step cycle contains every pair and exposes each robot four times. The
  cycle uses a private seeded RNG, so it neither perturbs nor depends on clip/window sampling.
- LR progress: specialists use 50k optimizer steps; shared arms use 125k steps with two robots
  per step. Both schedules therefore move from `3e-4` to `1e-5` over 50k exposures per robot.
- Architecture held fixed unless named by the arm: temporal encoder, latent 128, four encoder and
  decoder layers, four attention heads, no contact objective, no robot-source decode augmentation.

## Arms

| Arm | Optimizer steps | Width | Robot-specific parameters | Purpose |
|---|---:|---:|---:|---|
| Five specialists | 50k each | 256 | full separate model | Per-robot lower-bound controls |
| Shared base | 125k | 256 | 0 | Re-establish the sharing cost at matched exposure |
| Shared wider | 125k | 384 | 0 | Capacity intervention |
| Shared + adapters | 125k | 256 | 20,480 total (rank 8) | Conditional-capacity intervention |

Parameter contracts from the implementation are 1,540,810 for base/specialists, 2,751,178 for
width 384 (`1.79x` base), and 1,561,290 for adapters, of which 20,480 (`1.31%`) are
robot-specific. Each adapter is a node-shared `256 -> 8 -> 256` residual immediately before the
shared output heads. Its up-projection starts at zero, preserving the base decoder function at
initialization.

## Decision

For each robot, define the paired sharing gap as shared MPJPE minus that robot's specialist
MPJPE. The assay is valid only if the shared base has positive mean and worst-robot paired gaps.
A candidate promotes only if all conditions hold:

1. It closes at least 50% of the base mean paired gap.
2. It closes at least 50% of the base worst-robot paired gap.
3. Total parameters are no more than `2x` the base.
4. Robot-specific parameters are no more than 20% of the candidate.

If both candidates pass, promote the lower mean-gap arm; break an exact tie by fewer parameters,
then arm name. Run seeds 1 and 2 only for the winner. If neither passes, retain the matched shared
base as the paper result and investigate morphology-complete conditioning rather than PCGrad.

## Reproducibility

`scripts/run_sharing_cost_screen.sh` creates a detached code worktree at the launch revision and
runs from `/tmp`, so later edits to the main checkout cannot alter an active long-running shell or
trainer. The driver is resumable from retained 25k checkpoints. Raw checkpoints and stdout logs
remain ignored; manifests, exposure counts, checkpoint hashes, learning curves, final metrics,
and the frozen analyzer are the reviewable record.
