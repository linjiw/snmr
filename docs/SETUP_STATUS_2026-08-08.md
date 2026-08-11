# Verified local research environment — 2026-08-08

This file records the working environment used for the current paper experiments.  It is a
machine-local status snapshot, not a claim that generated datasets or checkpoints are
redistributable.

## Environment boundaries

- SNMR, data generation, analysis, and paper utilities: `/home/robotixx/snmr/.venv`
  (Python 3.10.12, PyTorch 2.13.0+cu130, NumPy 2.2.6, SciPy 1.15.3, MuJoCo 3.11.0).
- GMR teacher: editable `general_motion_retargeting` import from
  `/data/robotixx/snmr-externals/GMR` at
  `bb1bbe40774794fceb2a7c579a3464a28e68c844`.
- WBT control: separate `/home/robotixx/holosoma/.venv/hsmujoco` environment and Holosoma
  source at `20699ffa20f494b9563aa68601940c53397bf088`.  The active simulator is
  MuJoCo-Warp/Warp 1.16.0 on an NVIDIA RTX 5090 (32,607 MiB, driver 590.48.01).
- Paper: portable Tectonic 0.16.9 at `/data/robotixx/snmr-tools/bin/tectonic`.
- Large caches and new experiment outputs live on `/data`; the root filesystem has only about
  14 GB free while `/data` has about 700 GB free.

`source scripts/activate_snmr.sh` selects these roots, keeps the two Python environments
separate, redirects caches to `/data`, enables headless MuJoCo, and disables unrelated host ROS
pytest plugins.  The core environment was provisioned with `uv`, so `uv pip --python ...` is the
authoritative installer; an in-environment `python -m pip` module is not required.

Isaac Lab and SONIC are intentionally not installed.  Neither is imported by SNMR or by the
current Holosoma MuJoCo/MJWarp experiment; adding them would not unlock a currently registered
paper result.

## Data and verified artifacts

- Full LAFAN1/GMR paired corpus: 385 NPZ files (77 clips x 5 robots), 2,483,360 frames.
- Five-robot Phase-2 endpoint:
  `/data/robotixx/snmr-research/e67/phase2_all5_seed0/ckpt.pt`.
- E70's two frozen latent WBT motions are sourced from the E67/E69 motion directories; the
  loader-order precheck is
  `autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json`.
- Current manuscript PDF:
  `/data/robotixx/snmr-research/paper-build/main.pdf`.
- Full repository verification before the E67 launch: 204 passed, 4 skipped.  Focused E67/E68
  regression suite after the fail-closed repair: 12 passed.

## Current research state

E67 stopped at its preregistered specialist gate because `walk3_subject1` failed.  E68's separate
8k-to-16k extension also failed, so that pair is closed.  E69 then selected
`walk1_subject1` using references only, and its one frozen 8k specialist passed (0.9873
completion, 9.9376 s survival, 0.1815 rad joint RMSE).  The next registered command is:

```bash
bash scripts/run_e70_multitraj.sh
```

E70 retrains every student in `/data/robotixx/snmr-research/e70`; it cannot reuse the invalid
post-gate E67 artifact.  At the 2026-08-09 snapshot the NVIDIA kernel modules and RTX 5090 are
visible under `/proc`, but the container has no `/dev/nvidia*` nodes, so `nvidia-smi` and E70 are
temporarily blocked pending host/container device restoration.  Do not treat that infrastructure
condition as an experiment result.

## 2026-08-09 live update

The NVIDIA devices were restored and E70 launched without changing the frozen recipe.  Seed 0
completed all five arms and passed every preregistered content gate; the requested seed-1/2
confirmation queue is live under `/data/robotixx/snmr-research/e70/`.  The queue is resumable and
must finish before video rendering or a full test run competes for the GPU.

The repository now also has a policy-independent rollout-selection manifest for the paper video
(`docs/E70_VIDEO_PROTOCOL.md`) and a Holosoma-compatible offline ONNX exporter with checker and
ONNX Runtime parity validation (`scripts/export_e70_policy_onnx.py`).  These additions establish a
deployment toolchain; `docs/REAL_WORLD_DEPLOYMENT_PLAN.md` records why hardware remains gated.

## Quick verification

```bash
source scripts/activate_snmr.sh
python -c 'import snmr, general_motion_retargeting, torch, mujoco'
python -m pytest -q
bash -n scripts/run_e67_multitraj.sh scripts/run_e68_teacher_extension.sh \
  scripts/run_e69_teacher.sh scripts/run_e70_multitraj.sh
tectonic --version
```
