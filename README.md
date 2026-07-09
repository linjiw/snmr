# SNMR — Shared Neural Motion Retargeting

Implementation of the neural retargeter (contribution **C1**) from
[`NEURAL_RETARGETING_DESIGN.md`](NEURAL_RETARGETING_DESIGN.md): a SAME-style skeleton-agnostic
graph autoencoder that maps a motion into a **shared latent space** and decodes it onto **any robot
embodiment** as MuJoCo `qpos`, with joint-limit satisfaction by construction and differentiable
forward kinematics in the loss loop.

This is a from-scratch, dependency-light (torch + numpy + scipy; mujoco to load robots) research
codebase, built and validated against the real Unitree G1 model and a real holosoma whole-body-tracking
motion that ship in this repo.

## What is implemented and validated

| Module | Role | Validated by |
|---|---|---|
| `snmr/rotation.py` | wxyz quaternion ops, 6D rotation rep (Zhou et al.), geodesic metrics | `tests/test_rotation.py` — cross-checked against scipy |
| `snmr/robot_model.py` | MJCF → embodiment graph + **differentiable batched FK** | `tests/test_fk.py` — **matches `mujoco.mj_forward` to <1e-4 m / <1e-4 rad** |
| `snmr/skeleton.py` | shared skeleton graph (human + robot), SMPL-X body-22 topology | `tests/test_data.py` |
| `snmr/data.py` | canonical motion, real-NPZ loader, heading-invariant graph pose features | `tests/test_data.py` — incl. translation/yaw invariance |
| `snmr/model.py` | GAT encoder → shared latent → embodiment-conditioned (AdaLN) decoder → qpos | `tests/test_model.py` — shapes, unit quats, in-limit dof, grad flow, variable topology |
| `snmr/losses.py` | distill, task-FK, joint-limit, smoothness, foot-contact, latent-consistency | `tests/test_model.py`, `tests/test_overfit.py` |
| `snmr/train.py` | single-motion fitting loop (basis of the Phase-1 trainer) | `tests/test_overfit.py` |
| `snmr/human.py` | LAFAN1 24-body skeleton, human pose/static features, contact flags, pair loader | `tests/test_human.py`, `tests/test_human_to_robot.py` — incl. **end-to-end human→latent→robot training convergence vs real GMR teacher output** |
| `scripts/make_pairs_lafan1.py` | data engine: LAFAN1 BVH + GMR teacher → paired NPZs per robot | generated the full dataset: 77 clips × 5 robots = **2.48M teacher frames (~23 h), 1.6 GB** in `data/pairs/` |
| `scripts/train_phase1.py` | Phase-1 trainer: human→z→robot, clip-split train/val, held-out MPJPE eval, resumable | smoke-validated; root predicted in the **scaled-human-heading frame** (see below) |
| `scripts/export_wbt_npz.py` | SNMR output → holosoma WBT training NPZ (50 fps resample + MuJoCo FK replay) | schema-validated against the real holosoma sample NPZ |

### Root-pose parametrisation (hard-won lesson)

The decoder predicts the robot root **relative to the scaled human root in its heading frame**, not
in world coordinates: (1) world targets are unlearnable because encoder features are deliberately
heading/translation invariant (measured: ~3.3 m val MPJPE of pure root drift); (2) even human-relative
targets are ill-posed because GMR scales the trajectory (robot_xy ≈ 0.875·human_xy for LAFAN1→G1) so
the offset grows with distance from the origin (measured: 59 of 61 cm val MPJPE was root error).
The per-robot scale is least-squares fitted on the training set, stored in the checkpoint, and the
world pose is recomposed from the known human trajectory at inference
(`world_root_to_local`/`local_root_to_world`).

**End-to-end proof (`tests/test_overfit.py`, `scripts/overfit_batch.py`):** fitting the full
encode→latent→decode pipeline to a real 60-frame G1 clip drives loss 1.56→~0.05 and whole-body
**MPJPE to ~5–10 cm**, with **zero joint-limit violations** throughout. This establishes that the
architecture has the capacity and gradients to represent genuine motion — the prerequisite for Phase-1
dataset training.

```bash
# create env (torch CPU + mujoco), then:
python -m pytest -q                       # 28 tests, ~4min on CPU
python scripts/overfit_batch.py --steps 800   # end-to-end demonstration
```

### Robot model choice (matters for correctness)

FK and the joint-limit head load the **holosoma_retargeting** G1 MJCF
(`.../models/g1/g1_29dof.xml`), not GMR's `g1_mocap_29dof.xml`. The GMR mocap model narrows hip-pitch
to `[-1.57, 1.57]`, but the training NPZ (produced against the true hardware model) spans down to
~`-2.27` rad — so ~38% of frames would be **unreachable** by the tanh limit-head, injecting a silent
reconstruction floor. `tests/test_fk.py::test_joint_limits_cover_training_data` now guards this
contract. (This was one of two correctness bugs surfaced by an adversarial multi-agent review of the
implementation; the other — silently dropping slide/ball joints instead of raising — is fixed in
`robot_model.py` and guarded by `test_slide_joint_raises`.)

## Design → code mapping

- **Shared latent** (`MotionEncoder`): node-shared graph attention + global max-pool → a single
  per-frame latent, so any skeleton/joint-count is accepted; a temporal transformer over the frame
  latents gives velocity-consistent output.
- **Embodiment conditioning** (`EmbodimentEncoder` + `AdaLN` in `MotionDecoder`): a target robot's
  static graph features (rest offsets, joint axes, dof/EE flags) are pooled into an embodiment code
  that conditions every decoder layer — enabling zero-shot decode onto a new robot from its MJCF alone.
- **Joint limits by construction**: per-node angle heads emit `tanh`-scaled values mapped into each
  hinge's `[lo, hi]`; the `limits` loss is a redundant safety term (observed ≈0).
- **Differentiable FK** (`RobotKinematics.forward_kinematics`): task-space and contact losses are
  computed on FK'd bodies, so supervision can live in Cartesian space (GBC-style), not just qpos.

## What is intentionally *not* built here (needs data/GPU absent in this environment)

Faithful to the design's phasing, the following are scoped as documented extensions rather than
stubbed code:

- **AMASS/SMPL-X ingestion.** SMPL-X body models require registration and are not present; the
  human→robot path is fully implemented and tested on **LAFAN1** (`snmr/human.py`,
  `SNMR.retarget_human_to_robot`), and the SMPL-X body-22 topology is already in
  `smplx_body_skeleton` for when AMASS is provisioned. The skeleton-agnostic encoder needs no
  changes to accept it.
- **Multi-robot L_z consistency training (Phase 2)** and **holosoma WBT integration (Phase 3–4)**:
  `losses.latent_consistency_loss` and the `convert_data_format_mj.py` contract (`RobotMotion.qpos()`
  emits exactly `[root_pos, root_quat wxyz, dof]`) are in place; the multi-embodiment training loop and
  IsaacSim runs are the next milestones.

## Conventions (fixed package-wide)

Quaternions are **wxyz** internally (matching MuJoCo `qpos` and the GMR intermediate dict); the numpy
I/O boundary converts to/from xyzw where needed. FK assumes single-hinge, origin-anchored joints
(true for the humanoids in scope) and raises otherwise rather than emitting silently-wrong kinematics.
