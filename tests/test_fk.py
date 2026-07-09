"""Validate the differentiable torch FK against MuJoCo's own forward kinematics.

This is the load-bearing test for the whole package: if FK indexing/ordering is wrong, every
downstream loss is silently wrong. We assert agreement to <1e-4 m and <1e-4 rad on random,
in-limit configurations of the real Unitree G1 model.
"""

import numpy as np
import torch

from snmr import rotation as rot
from snmr.robot_model import RobotKinematics


def test_fk_matches_mujoco(g1_mjcf):
    import mujoco

    rk = RobotKinematics(g1_mjcf, device="cpu")
    model = mujoco.MjModel.from_xml_path(g1_mjcf)
    data = mujoco.MjData(model)

    lo, hi = rk.dof_limits()
    lo_np, hi_np = lo.numpy(), hi.numpy()
    rng = np.random.default_rng(0)

    # Map our body ordering to MuJoCo body ids once.
    mj_body_id = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in rk.body_names
    }

    max_pos_err = 0.0
    max_ang_err = 0.0
    for _ in range(16):
        # random in-limit dof, random root pose
        dof = rng.uniform(lo_np, hi_np)
        root_pos = rng.uniform(-1, 1, size=3)
        root_quat_xyzw = _rand_quat(rng)
        root_quat_wxyz = np.r_[root_quat_xyzw[3], root_quat_xyzw[:3]]

        # ---- MuJoCo reference ----
        data.qpos[:3] = root_pos
        data.qpos[3:7] = root_quat_wxyz
        data.qpos[7:] = dof
        mujoco.mj_forward(model, data)
        ref_pos = np.array([data.xpos[mj_body_id[n]].copy() for n in rk.body_names])
        ref_quat = np.array([data.xquat[mj_body_id[n]].copy() for n in rk.body_names])  # wxyz

        # ---- torch FK ----
        bp, bq = rk.forward_kinematics(
            torch.tensor(root_pos, dtype=torch.float64),
            torch.tensor(root_quat_wxyz, dtype=torch.float64),
            torch.tensor(dof, dtype=torch.float64),
        )
        pos_err = np.abs(bp.numpy() - ref_pos).max()
        ang_err = (
            rot.quat_geodesic_angle(bq, torch.tensor(ref_quat, dtype=torch.float64)).max().item()
        )
        max_pos_err = max(max_pos_err, pos_err)
        max_ang_err = max(max_ang_err, ang_err)

    assert max_pos_err < 1e-4, f"position error too high: {max_pos_err}"
    assert max_ang_err < 1e-4, f"orientation error too high: {max_ang_err}"


def test_fk_is_batched_and_differentiable(g1_mjcf):
    rk = RobotKinematics(g1_mjcf, device="cpu")
    T = 5
    dof = torch.zeros(T, rk.num_dof, requires_grad=True)
    root_pos = torch.zeros(T, 3, requires_grad=True)
    root_quat = torch.tensor([1.0, 0, 0, 0]).expand(T, 4).clone().requires_grad_(True)
    bp, bq = rk.forward_kinematics(root_pos, root_quat, dof)
    assert bp.shape == (T, rk.num_bodies, 3)
    assert bq.shape == (T, rk.num_bodies, 4)
    loss = bp.pow(2).sum()
    loss.backward()
    assert dof.grad is not None and torch.isfinite(dof.grad).all()


def test_fk_matches_mujoco_multiple_robots():
    """FK must be correct across embodiments, not just G1 — guards against G1-specific coincidences
    (e.g. all-axis-aligned joints). Skips any robot whose MJCF is absent."""
    import pathlib

    import mujoco

    from snmr import paths

    try:
        assets = paths.gmr_root() / "assets"
    except FileNotFoundError:
        import pytest

        pytest.skip("GMR clone not found (run scripts/fetch_externals.sh)")
    candidates = {
        "unitree_h1": assets / "unitree_h1" / "h1.xml",
        "unitree_h1_2": assets / "unitree_h1_2" / "h1_2_handless.xml",
        "fourier_n1": assets / "fourier_n1" / "n1_mocap.xml",
        "booster_t1_29dof": assets / "booster_t1_29dof" / "t1_mocap.xml",
        "stanford_toddy": assets / "stanford_toddy" / "toddy_mocap.xml",
    }
    tested = 0
    for name, path in candidates.items():
        if not path.exists():
            continue
        tested += 1
        rk = RobotKinematics(str(path))
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        lo, hi = (t.numpy() for t in rk.dof_limits())
        rng = np.random.default_rng(1)
        mj_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in rk.body_names}
        max_pos = max_ang = 0.0
        for _ in range(8):
            dof = rng.uniform(lo, hi)
            rp = rng.uniform(-1, 1, 3)
            rq = _rand_quat(rng)
            rq_w = np.r_[rq[3], rq[:3]]
            data.qpos[:3] = rp
            data.qpos[3:7] = rq_w
            data.qpos[7 : 7 + rk.num_dof] = dof
            mujoco.mj_forward(model, data)
            refp = np.array([data.xpos[mj_id[n]] for n in rk.body_names])
            refq = np.array([data.xquat[mj_id[n]] for n in rk.body_names])
            bp, bq = rk.forward_kinematics(
                torch.tensor(rp), torch.tensor(rq_w), torch.tensor(dof)
            )
            max_pos = max(max_pos, np.abs(bp.numpy() - refp).max())
            max_ang = max(
                max_ang, rot.quat_geodesic_angle(bq, torch.tensor(refq)).max().item()
            )
        assert max_pos < 1e-4, f"{name}: pos err {max_pos}"
        assert max_ang < 1e-4, f"{name}: ang err {max_ang}"
    if tested == 0:
        import pytest

        pytest.skip("no additional robot MJCFs found")


def test_dof_and_body_counts(g1_mjcf):
    rk = RobotKinematics(g1_mjcf, device="cpu")
    assert rk.num_dof == 29
    # 'pelvis' root present, hand/toe/head cosmetic bodies included; must contain the tracked links.
    for name in ["pelvis", "left_knee_link", "torso_link", "left_wrist_yaw_link", "right_elbow_link"]:
        assert name in rk.body_names


def test_joint_limits_cover_training_data(g1_mjcf, g1_train_npz):
    """The model whose joint limits the tanh head enforces MUST be able to represent the training
    data. Guards against the hip-pitch range mismatch an adversarial review caught: GMR's mocap G1
    narrows hip-pitch to [-1.57, 1.57] while the NPZ spans down to ~-2.27 rad."""
    import numpy as np

    rk = RobotKinematics(g1_mjcf, device="cpu")
    data = np.load(g1_train_npz, allow_pickle=True)
    npz_joint_names = [str(x) for x in data["joint_names"]]
    dof = np.asarray(data["joint_pos"])[:, 7:]  # (T, D)

    lo, hi = (t.numpy() for t in rk.dof_limits())
    # map our dof order (dof_body_index -> body -> its single joint) to npz columns by joint name
    import mujoco

    model = mujoco.MjModel.from_xml_path(g1_mjcf)
    body_to_jointname = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j]))
            body_to_jointname[b] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)

    violations = 0
    for d in range(rk.num_dof):
        body = rk.body_names[int(rk.graph.dof_body_index[d])]
        jname = body_to_jointname[body]
        col = dof[:, npz_joint_names.index(jname)]
        # allow a small tolerance for float noise
        if col.min() < lo[d] - 1e-3 or col.max() > hi[d] + 1e-3:
            violations += 1
    assert violations == 0, f"{violations} dof have training data outside the model's joint limits"


def test_slide_joint_raises(tmp_path):
    """A model with an actuated slide joint must raise, not silently drop the dof (fail-loud
    contract). This is the latent bug an adversarial review found: dropping a slide/ball would
    mis-align every subsequent hinge's dof index."""
    import pytest

    mjcf = tmp_path / "slide.xml"
    mjcf.write_text(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="root" pos="0 0 1">
              <freejoint name="root"/>
              <geom type="sphere" size="0.1"/>
              <body name="l1" pos="0.1 0 0">
                <joint name="slide1" type="slide" axis="0 0 1" range="-1 1"/>
                <geom type="sphere" size="0.05"/>
                <body name="l2" pos="0.1 0 0">
                  <joint name="hinge1" type="hinge" axis="0 1 0" range="-1 1"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    with pytest.raises(NotImplementedError, match="slide/ball"):
        RobotKinematics(str(mjcf), device="cpu")


def _rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)
