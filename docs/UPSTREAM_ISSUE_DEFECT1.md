# DRAFT upstream issue — holosoma (github.com/amazon-far/holosoma)

Status: DRAFT, not yet filed. Filing a public issue is an outward-facing action —
awaiting maintainer-contact decision (public issue vs private report). Content below is
self-contained and verified against pinned rev `9fb2b57`.

---

**Title:** MuJoCo backend: world-body off-by-one silently zeroes body-position tracking
rewards in WBT (MuJoCo/MuJoCo-Warp only; Isaac backends unaffected)

## Summary

On the MuJoCo(-Warp) simulator path, rigid-body state tensors are built from raw
`mjw_data.xpos`/`xquat`/... whose index 0 is the **world body** (width = `model.nbody`),
but `_body_list` **excludes** the world body. Body indices resolved by name against
`_body_list` are then applied to the raw-width tensors, shifting every simulation-side
body read one body rootward. In WholeBodyTracking, tracked slot 0 (pelvis) reads the
world body — always at the origin — so `error_body_pos` sits at the robot's distance
from the origin (meters) and the exponential body-position reward kernel underflows to
**exactly 0.0**. The primary tracking reward carries no gradient for the entire run.

Policies still learn to walk (orientation and velocity terms survive, joint-space terms
if configured), which is why this is easy to miss: training "works", TensorBoard shows a
healthy total reward, and only the raw per-term log (`raw_rew_..._position_error_exp ≡
0.0` for 8k iterations) reveals it.

## Affected / unaffected

- Affected: `simulator/mujoco/mujoco.py` (`num_bodies = model.nbody` with a world-less
  `_body_list`), consumed by `managers/command/terms/wbt.py` (`tracked_body_indexes` /
  `ref_body_index` built by name against `_body_list`, applied to raw-width sim tensors).
- Unaffected: IsaacGym / IsaacSim backends (both lists are robot-only), and the
  motion-npz side of WBT (name-mapped independently, correct).
- Related, same root cause (documented, lower stakes): on the Warp path
  `UndesiredContacts` reads `contact_forces` created from `cfrc_ext[..., :3]`, which is
  the **torque** half (force is `[..., 3:6]`), and is also world-indexed; `cfrc_ext` is
  additionally not populated unless `rne_postconstraint` runs.

## Reproduction (pinned rev 9fb2b57)

1. Train any G1 WBT motion on `simulator:mjwarp` (e.g.
   `exp:g1-29dof-wbt simulator:mjwarp --training.num-envs 512 ...`).
2. Inspect the raw term log for the body-position reward: it is exactly 0.0 at every
   iteration; `Env/motion/error_body_pos` is O(robot-to-origin distance), not O(cm).
3. One-line check without training: after env construction, compare
   `sim._rigid_body_pos.shape[1]` (33 for G1) with `len(sim._body_list)` (32).

## Fix

Offset sim-side body indices by `tensor_width - len(_body_list)` (1 on MuJoCo paths, 0 on
Isaac) wherever `_body_list`-derived indices touch raw-width tensors — or, more robustly,
strip the world entry when building the state tensors so both conventions match.

Monkeypatch we train with (verifies the fix end-to-end):
`https://github.com/linjiw/snmr/blob/main/snmr/integration/wbt_bodyfix.py`

## Measured impact (single-clip G1 walking, LAFAN1 walk1, 8k iters @512 envs)

| condition | completion | joint RMSE (rad) |
|---|---|---|
| defective indexing | 0.90 | 0.263 |
| fixed indexing, same config | 0.94 | 0.142 (−46%) |
| fixed + standard joint-space reward term | 0.98 | 0.122 |

The fixed stack matches an external MuJoCo-Warp WBT calibration band (mjlab 187-run
nightly, 97.9% completion), which the defective stack does not.
