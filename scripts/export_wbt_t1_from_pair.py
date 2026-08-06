#!/usr/bin/env python
"""Export a Booster T1 29-DoF holosoma WBT reference NPZ from a LAFAN1 pair NPZ (E54 groundwork).

Two sources:
  * ``--source gmr``  (default): the pair's GMR-teacher qpos is the motion (ground truth path).
  * ``--source snmr``: run the pair's human motion through a phase-1/2 SNMR checkpoint
    (``--ckpt``) with the GMR ``t1_mocap.xml`` kinematics (the embodiment graph the model was
    trained on), then convert the predicted qpos.

Either way the qpos is in GMR ``t1_mocap.xml`` layout (7 root + 27 hinges, no head joints) and
must be remapped into holosoma's ``t1/t1_29dof.xml`` layout (7 root + 29 hinges, head joints
first) before FK replay. Missing joints (AAHead_yaw, Head_pitch) are zero-filled.

Gotcha handled here: holosoma's ``MotionCommand`` unconditionally aliases the fake bodies
``left/right_foot_contact_point`` to the *G1* names ``left/right_ankle_roll_link`` when indexing
motion data (``managers/command/terms/wbt.py: FAKE_BODY_NAME_ALIASES``). A T1 FK replay contains
the contact points but not those G1 names, so we rename the two contact-point rows in
``body_names`` accordingly — the row data is the contact point itself, which is exactly the body
the alias stands in for.

    PYTHONPATH=. .venv-wbt/bin/python scripts/export_wbt_t1_from_pair.py \
        --pair ../data/pairs/booster_t1_29dof/walk1_subject5.npz \
        --out runs/wbt_validation/t1_gmr/walk1_subject5_mj.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.paths import holosoma_root, robot_mjcf  # noqa: E402

from export_wbt_npz import REQUIRED_KEYS, mujoco_replay, resample_qpos  # noqa: E402

# mirrors holosoma managers/command/terms/wbt.py FAKE_BODY_NAME_ALIASES (G1-named on purpose)
CONTACT_POINT_RENAMES = {
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def holosoma_t1_mjcf() -> pathlib.Path:
    return holosoma_root() / "src/holosoma/holosoma/data/robots/t1/t1_29dof.xml"


def hinge_names(model: "mujoco.MjModel") -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]


def remap_qpos_gmr_to_holosoma(qpos_gmr: np.ndarray, gmr_joints: list[str], holo_joints: list[str]) -> np.ndarray:
    """(T, 7+len(gmr_joints)) -> (T, 7+len(holo_joints)), matched by joint name, zeros elsewhere."""
    T = qpos_gmr.shape[0]
    assert qpos_gmr.shape[1] == 7 + len(gmr_joints), (
        f"qpos has {qpos_gmr.shape[1]} cols, expected 7+{len(gmr_joints)}"
    )
    out = np.zeros((T, 7 + len(holo_joints)), dtype=np.float64)
    out[:, :7] = qpos_gmr[:, :7]
    gmr_index = {name: i for i, name in enumerate(gmr_joints)}
    missing = []
    for k, name in enumerate(holo_joints):
        if name in gmr_index:
            out[:, 7 + k] = qpos_gmr[:, 7 + gmr_index[name]]
        else:
            missing.append(name)
    assert set(missing) <= {"AAHead_yaw", "Head_pitch"}, f"unexpected unmapped joints: {missing}"
    return out


def snmr_qpos_t1(ckpt: str, pair_path: str, device: str) -> np.ndarray:
    """Predict T1 qpos (GMR t1_mocap layout) from the pair's human motion via an SNMR checkpoint."""
    import torch

    from snmr.data import local_root_to_world
    from snmr.human import human_static_features, lafan1_skeleton, load_pair_npz
    from snmr.model import SNMR, SNMRConfig
    from snmr.robot_model import RobotKinematics

    state = torch.load(ckpt, map_location=device, weights_only=False)
    tc = state.get("config", {})
    model = SNMR(
        SNMRConfig(
            latent_dim=tc.get("latent_dim", 64),
            enc_hidden=tc.get("enc_hidden", 128),
            dec_hidden=tc.get("dec_hidden", 128),
        )
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    # phase-1 ckpts store scalar xy_scale; phase-2 a per-robot dict
    xy_scale = float(state.get("xy_scale", state.get("xy_scales", {}).get("booster_t1_29dof", 1.0)))

    pair = load_pair_npz(pair_path)
    rk = RobotKinematics(str(robot_mjcf("booster_t1_29dof")), device=device)
    skel = lafan1_skeleton(device=device)
    static = human_static_features(skel, body_pos_sample=pair["human_pos"].to(device))

    hp = pair["human_pos"].to(device)
    hq = pair["human_quat"].to(device)
    anchor = hp[:, 0, :].clone()
    anchor[:, :2] *= xy_scale
    with torch.no_grad():
        pred = model.retarget_human_to_robot(hp, hq, skel, static, rk)
        wp, wq = local_root_to_world(anchor, hq[:, 0, :], pred["root_pos"], pred["root_quat"])
    return torch.cat([wp, wq, pred["dof_pos"]], dim=-1).cpu().numpy().astype(np.float64)


def validate_t1(out: dict, holo_model: "mujoco.MjModel") -> list[str]:
    """Self-consistency + holosoma-sim-compatibility checks (no G1 reference applies here)."""
    problems = []
    for k in REQUIRED_KEYS:
        if k not in out:
            problems.append(f"missing key {k}")
    if problems:
        return problems
    T = out["joint_pos"].shape[0]
    n_joints = len(out["joint_names"])
    n_bodies = len(out["body_names"])
    if out["joint_pos"].shape != (T, 7 + n_joints):
        problems.append(f"joint_pos {out['joint_pos'].shape} != (T, 7+{n_joints})")
    if out["joint_vel"].shape != (T, 6 + n_joints):
        problems.append(f"joint_vel {out['joint_vel'].shape} != (T, 6+{n_joints})")
    for k in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        if out[k].shape != (T, n_bodies, 3):
            problems.append(f"{k} {out[k].shape} != (T, {n_bodies}, 3)")
    if out["body_quat_w"].shape != (T, n_bodies, 4):
        problems.append(f"body_quat_w {out['body_quat_w'].shape} != (T, {n_bodies}, 4)")
    qn = np.linalg.norm(out["joint_pos"][:, 3:7], axis=1)
    if not np.allclose(qn, 1, atol=1e-3):
        problems.append("root quat not unit")
    # every sim-side name holosoma will look up must exist in the npz
    holo_joints = hinge_names(holo_model)
    sim_bodies = [
        mujoco.mj_id2name(holo_model, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(holo_model.nbody)
        if mujoco.mj_id2name(holo_model, mujoco.mjtObj.mjOBJ_BODY, b) != "world"
    ]
    lookup_bodies = [CONTACT_POINT_RENAMES.get(b, b) for b in sim_bodies]
    npz_bodies = set(out["body_names"].tolist())
    npz_joints = set(out["joint_names"].tolist())
    for b in lookup_bodies:
        if b not in npz_bodies:
            problems.append(f"sim body lookup '{b}' missing from npz body_names")
    for j in holo_joints:
        if j not in npz_joints:
            problems.append(f"sim joint '{j}' missing from npz joint_names")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="LAFAN1 pair NPZ (booster_t1_29dof)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["gmr", "snmr"], default="gmr")
    ap.add_argument("--ckpt", default=None, help="SNMR checkpoint (required for --source snmr)")
    ap.add_argument("--output_fps", type=float, default=50.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    pair = np.load(args.pair, allow_pickle=True)
    robot = str(np.asarray(pair["robot"]).reshape(-1)[0])
    assert robot == "booster_t1_29dof", f"pair is for '{robot}', expected booster_t1_29dof"
    src_fps = float(np.asarray(pair["fps"]).reshape(-1)[0])

    gmr_model = mujoco.MjModel.from_xml_path(str(robot_mjcf("booster_t1_29dof")))
    holo_path = str(holosoma_t1_mjcf())
    holo_model = mujoco.MjModel.from_xml_path(holo_path)
    gmr_joints = hinge_names(gmr_model)
    holo_joints = hinge_names(holo_model)

    if args.source == "gmr":
        qpos_gmr = np.asarray(pair["qpos"], dtype=np.float64)
    else:
        assert args.ckpt, "--source snmr requires --ckpt"
        qpos_gmr = snmr_qpos_t1(args.ckpt, args.pair, args.device)

    qpos = remap_qpos_gmr_to_holosoma(qpos_gmr, gmr_joints, holo_joints)
    qpos50 = resample_qpos(qpos, src_fps=src_fps, dst_fps=args.output_fps)
    out = mujoco_replay(holo_path, qpos50, args.output_fps, root_body="Trunk")

    # satisfy holosoma's FAKE_BODY_NAME_ALIASES lookup (see module docstring)
    out["body_names"] = np.array(
        [CONTACT_POINT_RENAMES.get(b, b) for b in out["body_names"].tolist()]
    )

    problems = validate_t1(out, holo_model)
    outpath = pathlib.Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outpath, **out)
    print(f"wrote {outpath}  ({qpos50.shape[0]} frames @ {args.output_fps} fps, source={args.source})")
    if problems:
        print("SCHEMA PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("T1 WBT schema validation: OK")


if __name__ == "__main__":
    main()
