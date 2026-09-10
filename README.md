# SNMR — Shared Neural Motion Retargeting

Implementation of the neural retargeter (contribution **C1**) from
[`NEURAL_RETARGETING_DESIGN.md`](NEURAL_RETARGETING_DESIGN.md): a SAME-style skeleton-agnostic
graph autoencoder that maps a motion into a **shared latent space** and decodes it onto a **target robot
embodiment** as MuJoCo `qpos`, with joint-limit satisfaction by construction and differentiable
forward kinematics in the loss loop.

For a diagram-first explanation of the implemented GMR -> SNMR -> Holosoma WBT workflow, its
current experiment status, and the proposed latent-command/shared-policy extensions, see
[`docs/CURRENT_ARCHITECTURE.md`](docs/CURRENT_ARCHITECTURE.md).

This is a from-scratch, dependency-light (torch + numpy + scipy; mujoco to load robots) research
codebase, built and validated against the real Unitree G1 model and a real holosoma whole-body-tracking
motion that ship in this repo.

## Environment setup

The core retargeter/GMR data environment and the Holosoma WBT environment should remain separate.
For a local Python 3.10 environment:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e ".[dev,robot,data,video]"
uv pip install --python .venv/bin/python -e ../GMR --no-deps
source scripts/activate_snmr.sh
python -m pytest -q
```

The optional paper-video pipeline additionally requires the `ffmpeg` and `ffprobe`
system executables; `scripts/compose_e70_video.py` checks for both before encoding.

`scripts/activate_snmr.sh` sets the external/data roots, headless MuJoCo rendering, and isolates
pytest from host-level ROS plugins. The GMR editable install should point at the commit recorded in
`THIRD_PARTY.md`; `scripts/fetch_externals.sh` provisions that layout on a fresh machine. GMR is
needed only to regenerate teacher pairs, not for ordinary SNMR training or inference.

The next-stage controller uses Holosoma's separate MuJoCo/MJWarp environment (created by
`holosoma/scripts/setup_mujoco_via_uv.sh` or `setup_mujoco.sh`). Isaac Lab and SONIC are not SNMR
dependencies and should not be installed into `.venv`.

The activation script also exposes the portable Tectonic installation on the data disk. Build the
current six-page draft without adding host packages using:

```bash
source scripts/activate_snmr.sh
mkdir -p /data/robotixx/snmr-research/paper-build
XDG_CACHE_HOME="$SNMR_CACHE_ROOT/tectonic" \
  tectonic --outdir /data/robotixx/snmr-research/paper-build paper/main.tex
```

The original multi-trajectory E67 pair and its bounded E68 extension are closed because the
`walk3_subject1` specialist missed the frozen quality gate. The reference-only E69 screen then
selected `walk1_subject1`; its new specialist passed with 0.987 completion and 9.94 s survival.
The fresh, loader-order-safe two-walk comparison is preregistered in
`docs/E70_MULTITRAJ_PROTOCOL.md` and launched with `bash scripts/run_e70_multitraj.sh`. Seed 0
passes every registered content gate; seeds 1--2 are the frozen confirmation queue. The video
and hardware boundaries are recorded in `docs/E70_VIDEO_PROTOCOL.md` and
`docs/REAL_WORLD_DEPLOYMENT_PLAN.md`. All large artifacts remain under
`/data/robotixx/snmr-research/`.

## What is implemented and validated

| Module | Role | Validated by |
|---|---|---|
| `snmr/rotation.py` | wxyz quaternion ops, 6D rotation rep (Zhou et al.), geodesic metrics | `tests/test_rotation.py` — cross-checked against scipy |
| `snmr/robot_model.py` | MJCF → embodiment graph + **differentiable batched FK** | `tests/test_fk.py` — **matches `mujoco.mj_forward` to <1e-4 m / <1e-4 rad** |
| `snmr/robot_spec.py` | versioned, identity-free kinematics/dynamics/actuation/control contract + coherent dynamics twins | `tests/test_robot_spec.py` — parse/round-trip, hashes, masks, dimensionless features, twin invariants |
| `snmr/motion_spec.py` | canonical 50 Hz human-motion schema with provenance, SLERP/cubic resampling, scale descriptors, contacts, and normalized-buffer hashes | `tests/test_motion_spec.py` — strict validation, derivation, immutable buffers, JSON/hash round-trip |
| `snmr/robot_tokens.py` | padded variable-node RobotSpec tensors with tree-distance bias and per-node joint-limit contract | `tests/test_robot_tokens.py` — variable DoF, renaming and permutation equivariance, identity exclusion |
| `snmr/provenance.py` | worker-side immutable input snapshots, referenced MJCF-bundle hashing, and exact Git states | `tests/test_provenance.py` — mutation detection, materialization, clean/dirty/unavailable revisions |
| `snmr/skeleton.py` | shared skeleton graph (human + robot), SMPL-X body-22 topology | `tests/test_data.py` |
| `snmr/data.py` | canonical motion, real-NPZ loader, heading-invariant graph pose features | `tests/test_data.py` — incl. translation/yaw invariance |
| `snmr/model.py` | GAT encoder → shared latent → embodiment-conditioned (AdaLN) decoder → qpos | `tests/test_model.py` — shapes, unit quats, in-limit dof, grad flow, variable topology |
| `snmr/losses.py` | distill, task-FK, joint-limit, smoothness, foot-contact, latent-consistency | `tests/test_model.py`, `tests/test_overfit.py` |
| `snmr/train.py` | single-motion fitting loop (basis of the Phase-1 trainer) | `tests/test_overfit.py` |
| `snmr/human.py` | LAFAN1 24-body skeleton, human pose/static features, contact flags, pair loader | `tests/test_human.py`, `tests/test_human_to_robot.py` — incl. **end-to-end human→latent→robot training convergence vs real GMR teacher output** |
| `scripts/make_pairs_lafan1.py` | data engine: LAFAN1 BVH + GMR teacher → paired NPZs per robot | generated the full dataset: 77 clips × 5 robots = **2.48M teacher frames (~23 h), 1.6 GB** in `data/pairs/` |
| `scripts/train_phase1.py` | Phase-1 trainer: human→z→robot, clip-split train/val, held-out MPJPE eval, resumable | smoke-validated; root predicted in the **scaled-human-heading frame** (see below) |
| `scripts/export_wbt_npz.py` | SNMR output → holosoma WBT training NPZ (50 fps resample + MuJoCo FK replay) | schema-validated against the real holosoma sample NPZ |
| `snmr/verification.py` | backend-neutral physics reports with provenance and localized failures | `tests/test_verification.py`; matched MuJoCo CPU/Newton-MJWarp pilot |

### RobotSpec-conditioned physics pilot

The MorphoRetarget foundation is implemented without changing legacy checkpoint inputs. A controlled
G1 torque-twin experiment passes all six of that pilot's RobotSpec *contract checks* (these are not the
seven G0--G6 program gates; see the status note below), but also falsifies open-loop PD replay as a
candidate-ranking/RL reward: MuJoCo CPU and Newton/MJWarp agree that the clip fails, while rollout
saturation is not monotonic with motor strength. The retained design uses a frozen closed-loop
tracker in PhysX as the primary strong verifier and Newton/MJWarp as an independent second solver.
See [`docs/MORPHORETARGET_FOUNDATION_2026-08-30.md`](docs/MORPHORETARGET_FOUNDATION_2026-08-30.md)
for the original pilot and
[`docs/RESEARCH_STATUS_2026-09-01_MORPHORETARGET.md`](docs/RESEARCH_STATUS_2026-09-01_MORPHORETARGET.md)
for the frozen status, PM01 diagnosis, clean provenance bundle, and explicit G0 failure boundary.

> **Program status (iteration 2, 2026-09-01): FAILED — `STOP_A1_AND_A2`.** All seven program gates
> G0--G6 are **not met**. The fixed-G1 amortization screen stopped after five successive corrective
> runs, and the held-out-morphology experiment (A2) was blocked before generating any data, so
> **there is no learned zero-shot retargeting result of any kind** — G2 is not met and not partial.
> The cross-backend controller audit stands at 42 pass / 14 fail / 8 missing, with PhysX adopted as
> the sole primary verifier by an explicit scope decision rather than a parity result. The Booster T1
> tracker **completed** training on 2026-09-02 but is **not qualified**: no rollout has been run
> against it and no qualification floor was ever registered. Full record:
> [`docs/RESEARCH_STATUS_2026-09-01_MORPHORETARGET_ITERATION2.md`](docs/RESEARCH_STATUS_2026-09-01_MORPHORETARGET_ITERATION2.md).
>
> One measured finding did come out of that failure: the registered stop threshold is
> **mis-specified**. Scored on the same 13 validation clips with the same ruler, the GMR teacher
> accumulates 35 fidelity violations across 11/13 clips and reaches amplitude ratio 1.598 — above the
> 1.50 bound that stopped its own student at 1.510, which recorded 6 violations across 4/13 clips.

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
python -m pytest -q                       # 875 tests across 106 files
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
  that conditions every decoder layer. The interface accepts a new MJCF, but current zero-shot
  decoding fails at the available five-robot training scale.
- **Joint limits by construction**: per-node angle heads emit `tanh`-scaled values mapped into each
  hinge's `[lo, hi]`; the `limits` loss is a redundant safety term (observed ≈0).
- **Differentiable FK** (`RobotKinematics.forward_kinematics`): task-space and contact losses are
  computed on FK'd bodies, so supervision can live in Cartesian space (GBC-style), not just qpos.

## Current scope boundaries and deferred work

- **AMASS/SMPL-X ingestion.** SMPL-X body models require registration and are not present; the
  human→robot path is fully implemented and tested on **LAFAN1** (`snmr/human.py`,
  `SNMR.retarget_human_to_robot`), and the SMPL-X body-22 topology is already in
  `smplx_body_skeleton` for when AMASS is provisioned. The skeleton-agnostic encoder needs no
  changes to accept it.
- **Multi-robot SNMR is implemented and trained.** Phase 2 covers five trained robots with `L_z`
  consistency. Zero-shot decoding to a held-out robot currently fails, so embodiment augmentation
  and robot-source decode training remain deferred experiments.
- **The Holosoma latent-command instrument is implemented and validated on MuJoCo/Warp.** On the
  single cyclic clip, the explicit 64-d interface matches its evaluated teacher, while an absolute
  time-index control outperforms the frozen SNMR latent; see `paper/main.tex`. In the frozen E70
  two-walk assay, the frozen three-seed result is SNMR over time by +0.191 completion
  (69-cluster 95% CI [0.124, 0.274]) and over matched-phase shuffled SNMR by +0.199
  ([0.127, 0.279]), positive on each clip and at each training seed. Shared multi-robot control
  and RL-to-retargeter feedback remain proposed extensions.

## Conventions (fixed package-wide)

Quaternions are **wxyz** internally (matching MuJoCo `qpos` and the GMR intermediate dict); the numpy
I/O boundary converts to/from xyzw where needed. FK assumes single-hinge, origin-anchored joints
(true for the humanoids in scope) and raises otherwise rather than emitting silently-wrong kinematics.
