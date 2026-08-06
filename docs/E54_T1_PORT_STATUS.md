# E54 groundwork: Booster T1 29-DoF WBT config port — status

Date: 2026-08-06. Unblocks E54 (cross-embodiment latent command). All work verified at the
pinned holosoma clone rev 9fb2b57 (clone stays untracked/dirty by design — the config files
below live in the clone, NOT in this repo; this doc + the exporter are the committed record).

## What was created

In the holosoma clone (`holosoma/src/holosoma/holosoma/config_values/`):

| File | Content |
| --- | --- |
| `wbt/t1/__init__.py` | package marker |
| `wbt/t1/command.py` | `t1_29dof_wbt_command` — MotionCommand w/ T1 tracked bodies, ref body `Trunk`, default motion_file = walk1 GMR export (absolute path; override per run) |
| `wbt/t1/observation.py` | `t1_29dof_wbt_observation` — term-identical to G1 (all terms name-agnostic) |
| `wbt/t1/reward.py` | `t1_29dof_wbt_reward` — same terms/weights/sigmas as G1; UndesiredContacts allowlist regex swapped to T1 feet/hands |
| `wbt/t1/termination.py` | `t1_29dof_wbt_termination` — BadTrackingZOnly w/ T1 body names |
| `wbt/t1/curriculum.py` | `t1_29dof_wbt_curriculum` — copy of G1 (name-agnostic) |
| `wbt/t1/randomization.py` | `t1_29dof_wbt_randomization` — copy of G1 robot-only preset (name-agnostic); object DR omitted |
| `wbt/t1/experiment.py` | `t1_29dof_wbt` — PPO variant mirroring `g1_29dof_wbt`; robot = `robot.t1_29dof_waist_wrist` + `action_scale=0.25` + `action_scales_by_effort_limit_over_p_gain=True` + self-collisions on + init z=0.68 (T1 loco default; G1 uses 0.76); action = `action.t1_29dof_joint_pos` |
| `experiment.py`, `command.py`, `reward.py`, `termination.py`, `observation.py`, `randomization.py`, `curriculum.py` (top-level aggregators) | registered the `t1_29dof_wbt` presets; `exp:t1-29dof-wbt` now a tyro subcommand |

In this repo:

* `scripts/export_wbt_t1_from_pair.py` — exports a holosoma-WBT T1 reference NPZ from a
  `data/pairs/booster_t1_29dof/*.npz` pair. `--source gmr` (default) uses the pair's
  GMR-teacher qpos; `--source snmr --ckpt <phase-1/2 ckpt>` runs the human motion through the
  SNMR model (T1 kinematics from GMR `t1_mocap.xml`, per-robot `xy_scales` read from the ckpt).
  Both remap GMR-layout qpos (27 hinges) into holosoma-layout (29 hinges, head joints
  zero-filled) and FK-replay through holosoma's `t1/t1_29dof.xml`.
* `scripts/export_wbt_npz.py` — `mujoco_replay()` gained a `root_body` parameter
  (default `"pelvis"` = old behavior; T1 passes `"Trunk"`). Existing tests pass.

Exported references (regenerable, not committed):

* `runs/wbt_validation/t1_gmr/walk1_subject5_mj.npz` — 13066 frames @ 50 fps, schema OK
* `runs/wbt_validation/t1_snmr/walk1_subject5_mj.npz` — same clip through the phase-2 all-5
  checkpoint `runs/phase2_all5/ckpt_100k_final.pt`, schema OK

## Body-name mapping (G1 link -> T1 link, 14 tracked bodies)

T1 link names read from `holosoma/data/robots/t1/t1_29dof.xml`; T1 is 29-DoF like G1 but
spends 2 DoF on the head (no waist roll/pitch) and has 7-DoF arms with 4-DoF legs' hip cluster
ordered pitch->roll->yaw (G1: pitch->roll->yaw too, but link naming is `Hip_Roll_Left` style).

| G1 (tracked) | T1 | Note |
| --- | --- | --- |
| `pelvis` | `Waist` | hips attach here; T1's floating base is `Trunk` |
| `left_hip_roll_link` | `Hip_Roll_Left` | driven by `Left_Hip_Roll` |
| `left_knee_link` | `Shank_Left` | driven by `Left_Knee_Pitch` |
| `left_ankle_roll_link` | `left_foot_link` | driven by `Left_Ankle_Roll` (foot body) |
| `right_hip_roll_link` | `Hip_Roll_Right` | |
| `right_knee_link` | `Shank_Right` | |
| `right_ankle_roll_link` | `right_foot_link` | |
| `torso_link` | `Trunk` | also the WBT reference body (`body_name_ref`) |
| `left_shoulder_roll_link` | `AL2` | driven by `Left_Shoulder_Roll` |
| `left_elbow_link` | `AL3` | driven by `Left_Elbow_Pitch` |
| `left_wrist_yaw_link` | `left_hand_link` | arm end-effector (G1 tracks its last wrist link; T1's hand body is the equivalent distal link) |
| `right_shoulder_roll_link` | `AR2` | |
| `right_elbow_link` | `AR3` | |
| `right_wrist_yaw_link` | `right_hand_link` | |

End-effector set for `bad_motion_body_pos_body_names` (termination): `left_foot_link`,
`right_foot_link`, `left_hand_link`, `right_hand_link` (G1: ankle_roll + wrist_yaw links).

## Smoke-test results (all PASS)

Environment: `.venv-wbt`, `PYTHONPATH=<snmr repo>`, mjwarp backend, 4 envs, headless,
A10G with ~14.5 GB held by an external job (fits fine).

1. **Config parse**: `exp:t1-29dof-wbt simulator:mjwarp logger:disabled --training.num-envs 4
   --training.headless True` parses via `tyro.cli(AnnotatedExperimentConfig)`;
   `robot_type=t1_29dof`, tracked bodies as in the table above.
2. **Env construction**: `WholeBodyTrackingManager` constructs on MuJoCo-Warp with the walk1
   GMR T1 reference (13066 frames). Requires `--randomization.ignore-unsupported True`
   (same as all local mjwarp G1 runs — the material/COM startup DR terms are IsaacSim-only).
   `snmr.integration.wbt_bodyfix.patch()` applied and reports offset 1 (DEFECT-1 fix works
   unchanged for T1: 33-wide tensors, 32-name list).
3. **Reset + steps**: `reset_all()` OK — actor_obs (4, 154), critic_obs (4, 286); identical
   widths to G1 (29 DoF + 14 bodies => same term dims). 10 zero-action steps OK; per-env
   rewards finite (0.003–0.067 at zero action).

## Gotchas found (worth knowing before E54)

* **`FAKE_BODY_NAME_ALIASES` is G1-hardcoded upstream** (`managers/command/terms/wbt.py:515`):
  the fake bodies `left/right_foot_contact_point` are unconditionally aliased to
  `left/right_ankle_roll_link` when indexing motion npz data — for ANY robot. Handled on the
  data side: the T1 exporter renames the two contact-point rows in `body_names` to the G1
  alias names (the row data is the T1 contact point itself, which is what the alias stands in
  for). Clone untouched.
* **GMR vs holosoma T1 MJCF differ**: GMR `t1_mocap.xml` has 27 hinges (no `AAHead_yaw` /
  `Head_pitch`), narrower elbow-yaw ranges, and `left/right_toe_link` instead of
  `*_foot_contact_point`. The exporter remaps by joint name and zero-fills the head joints.
* **The default motion_file in `wbt/t1/command.py` is an absolute path** into this repo's
  `runs/wbt_validation/t1_gmr/` (no T1 motion ships inside the holosoma package). Override
  with `--command.setup_terms.motion_command.params.motion_config.motion_file=...` per run.
* Randomization on mjwarp needs `--randomization.ignore-unsupported True` (pre-existing,
  robot-independent).

## Blockers

None for E54 groundwork. Open items:

* Head joints (`AAHead_yaw`, `Head_pitch`) are zero-filled in references — fine for tracking
  (no head body is in the tracked-body set; the stock WBT reward has no joint-space term, and
  the E51 joint-reward add-on would simply pull them toward the zeros in the npz).
* The clone is dirty by design; if it is ever re-cloned, the `wbt/t1/` package and the 7
  aggregator registrations must be re-applied (file list above is complete).
* Batch export of all 77 T1 clips not yet run (one command per clip with
  `export_wbt_t1_from_pair.py`; walk1_subject5 verified end-to-end).

## Repro commands

```bash
# export a T1 reference (GMR ground-truth path)
cd snmr && PYTHONPATH=. ../.venv-wbt/bin/python scripts/export_wbt_t1_from_pair.py \
  --pair ../data/pairs/booster_t1_29dof/walk1_subject5.npz \
  --out runs/wbt_validation/t1_gmr/walk1_subject5_mj.npz

# ... or through the phase-2 all-5 SNMR checkpoint
PYTHONPATH=. ../.venv-wbt/bin/python scripts/export_wbt_t1_from_pair.py \
  --pair ../data/pairs/booster_t1_29dof/walk1_subject5.npz \
  --source snmr --ckpt runs/phase2_all5/ckpt_100k_final.pt \
  --out runs/wbt_validation/t1_snmr/walk1_subject5_mj.npz

# train (local mjwarp; add wbt_bodyfix like train_agent_joint_reward.py does)
cd ../holosoma && PYTHONPATH=$PWD/../snmr ../.venv-wbt/bin/python \
  ../snmr/scripts/train_agent_joint_reward.py exp:t1-29dof-wbt simulator:mjwarp \
  --randomization.ignore-unsupported True \
  --command.setup_terms.motion_command.params.motion_config.motion_file=<t1_ref.npz>
```
