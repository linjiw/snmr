"""Tests for the embodiment-variant generator (scripts/make_embodiment_variants.py).

Verifies:
  - Variant XMLs load via RobotKinematics without error.
  - torch-FK matches mujoco mj_forward (<1e-4) on random configs.
  - Scale factor is measurably applied (limb-length ratio ~= factor).
  - DOF order is preserved (same count + same body ordering).
  - Base file is unmodified (hash before == hash after).

Skips gracefully if holosoma/GMR assets are absent (follows conftest patterns).
"""

import hashlib
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snmr import paths  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402
from snmr import rotation as rot  # noqa: E402


def _file_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def base_mjcf_path() -> pathlib.Path:
    """The base G1 MJCF; skip if holosoma not present."""
    try:
        p = paths.g1_mjcf()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if not p.exists():
        pytest.skip(f"G1 MJCF not found at {p}")
    return p


@pytest.fixture(scope="module")
def variant_artifacts(base_mjcf_path: pathlib.Path) -> dict:
    """Generate all variants and return info needed by tests.

    Returns dict with keys: base_hash, variant_paths, manifest, base_mjcf_path.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_embodiment_variants import generate_variants  # noqa: E402

    base_hash = _file_hash(base_mjcf_path)
    variant_paths, manifest = generate_variants("unitree_g1")
    return {
        "base_hash": base_hash,
        "variant_paths": variant_paths,
        "manifest": manifest,
        "base_mjcf_path": base_mjcf_path,
    }


class TestVariantGeneration:
    """Core tests for generated variant MJCFs."""

    def test_base_file_unmodified(self, variant_artifacts):
        """The original MJCF must not be altered by variant generation."""
        after_hash = _file_hash(variant_artifacts["base_mjcf_path"])
        assert after_hash == variant_artifacts["base_hash"], (
            "Base MJCF file was modified during variant generation!"
        )

    def test_all_variants_load_via_robot_kinematics(self, variant_artifacts):
        """Every variant XML must load cleanly through RobotKinematics."""
        for vpath in variant_artifacts["variant_paths"]:
            rk = RobotKinematics(str(vpath), device="cpu")
            assert rk.num_dof > 0

    def test_dof_count_preserved(self, variant_artifacts):
        """All variants must have the same DOF count as the base."""
        base_rk = RobotKinematics(str(variant_artifacts["base_mjcf_path"]), device="cpu")
        for vpath in variant_artifacts["variant_paths"]:
            rk = RobotKinematics(str(vpath), device="cpu")
            assert rk.num_dof == base_rk.num_dof, (
                f"{vpath.name}: dof {rk.num_dof} != base {base_rk.num_dof}"
            )

    def test_dof_order_preserved(self, variant_artifacts):
        """Body names (which encode DOF order) must be identical to base."""
        base_rk = RobotKinematics(str(variant_artifacts["base_mjcf_path"]), device="cpu")
        for vpath in variant_artifacts["variant_paths"]:
            rk = RobotKinematics(str(vpath), device="cpu")
            assert rk.body_names == base_rk.body_names, (
                f"{vpath.name}: body name order differs from base"
            )

    def test_fk_matches_mujoco(self, variant_artifacts):
        """Torch FK must agree with mujoco mj_forward to <1e-4 on each variant."""
        import mujoco

        for vpath in variant_artifacts["variant_paths"]:
            rk = RobotKinematics(str(vpath), device="cpu")
            model = mujoco.MjModel.from_xml_path(str(vpath))
            data = mujoco.MjData(model)
            lo, hi = rk.dof_limits()
            lo_np, hi_np = lo.numpy(), hi.numpy()
            rng = np.random.default_rng(123)

            mj_body_id = {
                name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in rk.body_names
            }

            max_pos_err = 0.0
            for _ in range(4):
                dof = rng.uniform(lo_np, hi_np)
                root_pos = np.array([0.0, 0.0, 0.75])
                root_quat = np.array([1.0, 0.0, 0.0, 0.0])

                data.qpos[:3] = root_pos
                data.qpos[3:7] = root_quat
                data.qpos[7:] = dof
                mujoco.mj_forward(model, data)

                bp, _ = rk.forward_kinematics(
                    torch.tensor(root_pos, dtype=torch.float64),
                    torch.tensor(root_quat, dtype=torch.float64),
                    torch.tensor(dof, dtype=torch.float64),
                )
                ref_pos = np.array([data.xpos[mj_body_id[n]].copy() for n in rk.body_names])
                pos_err = np.abs(bp.numpy() - ref_pos).max()
                max_pos_err = max(max_pos_err, pos_err)

            assert max_pos_err < 1e-4, (
                f"{vpath.name}: FK error {max_pos_err:.2e} >= 1e-4"
            )

    def test_leg_factor_applied(self, variant_artifacts):
        """For leg variants, pelvis-to-foot distance must scale approximately by the factor."""
        base_rk = RobotKinematics(str(variant_artifacts["base_mjcf_path"]), device="cpu")
        zero_dof = torch.zeros(base_rk.num_dof, dtype=torch.float64)
        root_p = torch.tensor([0.0, 0.0, 0.75], dtype=torch.float64)
        root_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        bp_base, _ = base_rk.forward_kinematics(root_p, root_q, zero_dof)
        pelvis_idx = base_rk.body_names.index("pelvis")
        foot_idx = base_rk.body_names.index("left_ankle_roll_link")
        base_leg_len = (bp_base[foot_idx] - bp_base[pelvis_idx]).norm().item()

        for info in variant_artifacts["manifest"]["variants"]:
            if info["region"] != "legs":
                continue
            rk = RobotKinematics(info["path"], device="cpu")
            bp_var, _ = rk.forward_kinematics(root_p, root_q, zero_dof)
            var_leg_len = (bp_var[foot_idx] - bp_var[pelvis_idx]).norm().item()
            ratio = var_leg_len / base_leg_len
            expected = info["factor"]
            # Allow 5% tolerance (non-linear chain, pos offsets include lateral offsets)
            assert abs(ratio - expected) < 0.05, (
                f"legs variant factor={expected}: measured ratio={ratio:.4f}, "
                f"tolerance exceeded"
            )

    def test_arm_factor_applied(self, variant_artifacts):
        """For arm variants, shoulder-to-hand distance must scale approximately by the factor."""
        base_rk = RobotKinematics(str(variant_artifacts["base_mjcf_path"]), device="cpu")
        zero_dof = torch.zeros(base_rk.num_dof, dtype=torch.float64)
        root_p = torch.tensor([0.0, 0.0, 0.75], dtype=torch.float64)
        root_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        bp_base, _ = base_rk.forward_kinematics(root_p, root_q, zero_dof)
        shoulder_idx = base_rk.body_names.index("left_shoulder_pitch_link")
        hand_idx = base_rk.body_names.index("left_rubber_hand_link")
        base_arm_len = (bp_base[hand_idx] - bp_base[shoulder_idx]).norm().item()

        for info in variant_artifacts["manifest"]["variants"]:
            if info["region"] != "arms":
                continue
            rk = RobotKinematics(info["path"], device="cpu")
            bp_var, _ = rk.forward_kinematics(root_p, root_q, zero_dof)
            var_arm_len = (bp_var[hand_idx] - bp_var[shoulder_idx]).norm().item()
            ratio = var_arm_len / base_arm_len
            expected = info["factor"]
            assert abs(ratio - expected) < 0.05, (
                f"arms variant factor={expected}: measured ratio={ratio:.4f}, "
                f"tolerance exceeded"
            )

    def test_uniform_factor_applied(self, variant_artifacts):
        """For uniform variants, overall limb lengths should scale by ~factor."""
        base_rk = RobotKinematics(str(variant_artifacts["base_mjcf_path"]), device="cpu")
        zero_dof = torch.zeros(base_rk.num_dof, dtype=torch.float64)
        root_p = torch.tensor([0.0, 0.0, 0.75], dtype=torch.float64)
        root_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        bp_base, _ = base_rk.forward_kinematics(root_p, root_q, zero_dof)
        pelvis_idx = base_rk.body_names.index("pelvis")
        foot_idx = base_rk.body_names.index("left_ankle_roll_link")
        base_len = (bp_base[foot_idx] - bp_base[pelvis_idx]).norm().item()

        for info in variant_artifacts["manifest"]["variants"]:
            if info["region"] != "uniform":
                continue
            rk = RobotKinematics(info["path"], device="cpu")
            bp_var, _ = rk.forward_kinematics(root_p, root_q, zero_dof)
            var_len = (bp_var[foot_idx] - bp_var[pelvis_idx]).norm().item()
            ratio = var_len / base_len
            expected = info["factor"]
            assert abs(ratio - expected) < 0.05, (
                f"uniform variant factor={expected}: measured ratio={ratio:.4f}, "
                f"tolerance exceeded"
            )


class TestRetargeting:
    """Tests for teacher-on-variant retargeting (skip if GMR/LAFAN1 absent)."""

    @pytest.fixture(scope="class")
    @classmethod
    def retarget_result(cls, variant_artifacts):
        """Run retarget on one clip for legs_0.85 variant."""
        try:
            bvh_dir = paths.data_root() / "lafan1_bvh"
        except FileNotFoundError as e:
            pytest.skip(str(e))

        clip = bvh_dir / "walk1_subject1.bvh"
        if not clip.exists():
            pytest.skip(f"LAFAN1 clip not found: {clip}")

        try:
            import general_motion_retargeting  # noqa: F401
        except ImportError:
            pytest.skip("GMR not installed")

        sys.path.insert(0, str(ROOT / "scripts"))
        from make_embodiment_variants import retarget_variant

        # Use the legs_0.85 variant
        legs085_path = None
        for vpath in variant_artifacts["variant_paths"]:
            if "legs_0.85" in vpath.name:
                legs085_path = vpath
                break
        if legs085_path is None:
            pytest.skip("legs_0.85 variant not found")

        results = retarget_variant(legs085_path, "legs_0.85", [clip], "unitree_g1")
        return results[0] if results else None

    def test_retarget_runs(self, retarget_result):
        """Teacher retargeting must complete without error."""
        assert retarget_result is not None

    def test_retarget_finite(self, retarget_result):
        """All qpos values must be finite."""
        assert retarget_result["qpos_finite"]

    def test_retarget_pelvis_direction(self, retarget_result, variant_artifacts):
        """For legs_0.85, the retargeted pelvis height should be LOWER than base (if available)."""
        try:
            base_pair_dir = paths.data_root() / "pairs" / "unitree_g1"
        except FileNotFoundError:
            pytest.skip("data root not found")

        base_npz = base_pair_dir / "walk1_subject1.npz"
        if not base_npz.exists():
            pytest.skip("Base pair NPZ not available for direction comparison")

        base_z = np.load(base_npz)["qpos"][:, 2].mean()
        var_z = retarget_result["mean_pelvis_z"]
        assert var_z < base_z, (
            f"legs_0.85 pelvis_z ({var_z:.4f}) should be lower than base ({base_z:.4f})"
        )
