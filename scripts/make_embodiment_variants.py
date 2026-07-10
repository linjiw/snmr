#!/usr/bin/env python
"""Generate kinematically scaled MJCF embodiment variants for SNMR augmentation.

Motivation (SAME, SIGGRAPH Asia 2023): zero-shot decoding to an unseen robot fails without
synthetic embodiment diversity in training. This script produces 6 scaled variants of a base
robot (default: Unitree G1) by region-specific scaling of body-frame position offsets.

Variants produced:
    legs_0.85, legs_1.15   -- subtree below each hip joint scaled
    arms_0.85, arms_1.15   -- subtree below each shoulder scaled
    uniform_0.9, uniform_1.1 -- all body pos offsets scaled

Method (robust to includes/meshes):
    1. Load MJCF via mujoco.MjModel.from_xml_path (resolves includes/defaults).
    2. mj_saveLastXML into the SOURCE model's directory (so relative meshdir stays valid).
    3. ElementTree-edit that resolved file: scale <body pos="..."> by region.
    4. Validate: load via RobotKinematics + raw mujoco; dof count/order identical to base;
       torch-FK matches mj_forward (<1e-4) on 4 random in-limit configs.

Geoms are NOT scaled (FK-only use for distillation targets; visual/collision fidelity
is not needed). Joint ranges and axes are preserved verbatim.

Usage:
    python scripts/make_embodiment_variants.py [--robot unitree_g1] [--retarget --clips walk1_subject1.bvh]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.paths import robot_mjcf, data_root  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402
from snmr import rotation as rot  # noqa: E402


# -------------------------------------------------------------------------------------
# Region definitions for G1 (generalizable: we match body names by pattern)
# -------------------------------------------------------------------------------------
# Hip roots: the first body in each leg subtree (direct child of pelvis with 'hip' in name)
HIP_ROOTS = ["left_hip_pitch_link", "right_hip_pitch_link"]
# Shoulder roots: the first body in each arm subtree
SHOULDER_ROOTS = ["left_shoulder_pitch_link", "right_shoulder_pitch_link"]

VARIANT_SPECS = [
    ("legs_0.85", "legs", 0.85),
    ("legs_1.15", "legs", 1.15),
    ("arms_0.85", "arms", 0.85),
    ("arms_1.15", "arms", 1.15),
    ("uniform_0.9", "uniform", 0.9),
    ("uniform_1.1", "uniform", 1.1),
]


def _subtree_bodies(tree_root: ET.Element, root_body_names: list[str]) -> set[str]:
    """Return names of all <body> elements in the subtree(s) rooted at root_body_names (inclusive)."""
    names = set()

    def _find_body_by_name(elem: ET.Element, name: str) -> ET.Element | None:
        if elem.tag == "body" and elem.get("name") == name:
            return elem
        for child in elem:
            result = _find_body_by_name(child, name)
            if result is not None:
                return result
        return None

    def _collect(elem: ET.Element) -> None:
        if elem.tag == "body":
            bname = elem.get("name")
            if bname:
                names.add(bname)
        for child in elem:
            _collect(child)

    for rname in root_body_names:
        body_el = _find_body_by_name(tree_root, rname)
        if body_el is not None:
            _collect(body_el)

    return names


def _bodies_to_scale(tree_root: ET.Element, region: str) -> set[str]:
    """Return the set of body names whose 'pos' attribute should be scaled."""
    if region == "legs":
        return _subtree_bodies(tree_root, HIP_ROOTS)
    elif region == "arms":
        return _subtree_bodies(tree_root, SHOULDER_ROOTS)
    elif region == "uniform":
        # All bodies except worldbody itself
        names = set()

        def _collect_all(elem: ET.Element) -> None:
            if elem.tag == "body":
                bname = elem.get("name")
                if bname:
                    names.add(bname)
            for child in elem:
                _collect_all(child)

        _collect_all(tree_root)
        return names
    else:
        raise ValueError(f"Unknown region: {region}")


def _scale_body_positions(xml_path: pathlib.Path, region: str, factor: float, out_path: pathlib.Path) -> None:
    """Read an MJCF XML, scale body pos offsets for bodies in the given region, write result."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Get worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> in {xml_path}")

    bodies_to_edit = _bodies_to_scale(worldbody, region)

    def _scale_pos_recursive(elem: ET.Element) -> None:
        for child in elem:
            if child.tag == "body":
                bname = child.get("name")
                if bname and bname in bodies_to_edit:
                    pos_str = child.get("pos")
                    if pos_str:
                        pos = [float(x) for x in pos_str.split()]
                        scaled = [x * factor for x in pos]
                        child.set("pos", " ".join(f"{v:.8g}" for v in scaled))
            _scale_pos_recursive(child)

    _scale_pos_recursive(worldbody)
    tree.write(str(out_path), xml_declaration=False)


def _verify_variant(
    base_mjcf: str, variant_mjcf: str, tag: str, region: str, factor: float
) -> dict:
    """Verify a variant: loads, dof match, FK matches mujoco, limb-length change measurable."""
    import mujoco

    # Load both
    rk_base = RobotKinematics(base_mjcf, device="cpu")
    rk_var = RobotKinematics(variant_mjcf, device="cpu")

    # DOF count and order must be identical
    assert rk_var.num_dof == rk_base.num_dof, (
        f"Variant {tag}: dof mismatch {rk_var.num_dof} vs {rk_base.num_dof}"
    )
    assert rk_var.body_names == rk_base.body_names, (
        f"Variant {tag}: body name mismatch"
    )

    # FK vs mujoco on 4 random configs
    model_var = mujoco.MjModel.from_xml_path(variant_mjcf)
    data_var = mujoco.MjData(model_var)
    lo, hi = rk_var.dof_limits()
    lo_np, hi_np = lo.numpy(), hi.numpy()
    rng = np.random.default_rng(42)

    mj_body_id = {
        name: mujoco.mj_name2id(model_var, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in rk_var.body_names
    }

    max_pos_err = 0.0
    for _ in range(4):
        dof = rng.uniform(lo_np, hi_np)
        root_pos = np.array([0.0, 0.0, 0.75])
        root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz

        data_var.qpos[:3] = root_pos
        data_var.qpos[3:7] = root_quat
        data_var.qpos[7:] = dof
        mujoco.mj_forward(model_var, data_var)

        bp, bq = rk_var.forward_kinematics(
            torch.tensor(root_pos, dtype=torch.float64),
            torch.tensor(root_quat, dtype=torch.float64),
            torch.tensor(dof, dtype=torch.float64),
        )
        ref_pos = np.array([data_var.xpos[mj_body_id[n]].copy() for n in rk_var.body_names])
        pos_err = np.abs(bp.numpy() - ref_pos).max()
        max_pos_err = max(max_pos_err, pos_err)

    assert max_pos_err < 1e-4, f"Variant {tag}: FK error {max_pos_err:.2e}"

    # Measure limb-length change at zero pose
    zero_dof = torch.zeros(rk_base.num_dof, dtype=torch.float64)
    root_p = torch.tensor([0.0, 0.0, 0.75], dtype=torch.float64)
    root_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    bp_base, _ = rk_base.forward_kinematics(root_p, root_q, zero_dof)
    bp_var, _ = rk_var.forward_kinematics(root_p, root_q, zero_dof)

    pelvis_idx = rk_base.body_names.index("pelvis")

    # Compute ratios for relevant endpoints
    ratios = {}
    if region == "legs":
        for foot_name in ["left_ankle_roll_link", "right_ankle_roll_link"]:
            if foot_name in rk_base.body_names:
                foot_idx = rk_base.body_names.index(foot_name)
                base_dist = (bp_base[foot_idx] - bp_base[pelvis_idx]).norm().item()
                var_dist = (bp_var[foot_idx] - bp_var[pelvis_idx]).norm().item()
                ratios[f"pelvis_to_{foot_name}"] = var_dist / base_dist if base_dist > 0 else 1.0
    elif region == "arms":
        for hand_name in ["left_rubber_hand_link", "right_rubber_hand_link"]:
            if hand_name in rk_base.body_names:
                hand_idx = rk_base.body_names.index(hand_name)
                # measure from shoulder to hand
                shoulder_name = "left_shoulder_pitch_link" if "left" in hand_name else "right_shoulder_pitch_link"
                shoulder_idx = rk_base.body_names.index(shoulder_name)
                base_dist = (bp_base[hand_idx] - bp_base[shoulder_idx]).norm().item()
                var_dist = (bp_var[hand_idx] - bp_var[shoulder_idx]).norm().item()
                ratios[f"shoulder_to_{hand_name}"] = var_dist / base_dist if base_dist > 0 else 1.0
    elif region == "uniform":
        for foot_name in ["left_ankle_roll_link", "right_ankle_roll_link"]:
            if foot_name in rk_base.body_names:
                foot_idx = rk_base.body_names.index(foot_name)
                base_dist = (bp_base[foot_idx] - bp_base[pelvis_idx]).norm().item()
                var_dist = (bp_var[foot_idx] - bp_var[pelvis_idx]).norm().item()
                ratios[f"pelvis_to_{foot_name}"] = var_dist / base_dist if base_dist > 0 else 1.0

    return {
        "tag": tag,
        "region": region,
        "factor": factor,
        "num_dof": rk_var.num_dof,
        "max_fk_error": float(max_pos_err),
        "limb_ratios": ratios,
    }


def generate_variants(base_robot: str = "unitree_g1") -> tuple[list[pathlib.Path], dict]:
    """Generate all 6 embodiment variants. Returns (variant_paths, manifest_dict)."""
    import mujoco

    base_path = robot_mjcf(base_robot)
    base_dir = base_path.parent
    base_stem = base_path.stem

    # Step 1: resolve the base XML into the source directory
    model = mujoco.MjModel.from_xml_path(str(base_path))
    resolved_path = base_dir / f"{base_stem}_resolved.xml"
    mujoco.mj_saveLastXML(str(resolved_path), model)

    manifest = {
        "base_robot": base_robot,
        "base_mjcf": str(base_path),
        "variants": [],
    }
    variant_paths = []

    for tag, region, factor in VARIANT_SPECS:
        out_name = f"{base_stem}_var_{tag}.xml"
        out_path = base_dir / out_name

        _scale_body_positions(resolved_path, region, factor, out_path)

        # Verify
        info = _verify_variant(str(base_path), str(out_path), tag, region, factor)
        info["path"] = str(out_path)
        manifest["variants"].append(info)
        variant_paths.append(out_path)

        print(f"  [OK] {tag}: dof={info['num_dof']}, FK_err={info['max_fk_error']:.2e}, "
              f"ratios={info['limb_ratios']}")

    # Write manifest
    manifest_path = base_dir / f"{base_stem}_variants_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest_path"] = str(manifest_path)

    # Clean up the resolved intermediate
    resolved_path.unlink(missing_ok=True)

    print(f"\nManifest written to {manifest_path}")
    return variant_paths, manifest


# -------------------------------------------------------------------------------------
# Retargeting support: run GMR teacher on each variant
# -------------------------------------------------------------------------------------
def retarget_variant(
    variant_path: pathlib.Path,
    variant_tag: str,
    clips: list[pathlib.Path],
    base_robot: str = "unitree_g1",
) -> list[dict]:
    """Run the GMR teacher on a variant, producing pair NPZs with the standard schema."""
    import general_motion_retargeting.params as gmr_params
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    # Patch ROBOT_XML_DICT so GMR loads the variant model
    original_xml = gmr_params.ROBOT_XML_DICT[base_robot]
    gmr_params.ROBOT_XML_DICT[base_robot] = variant_path

    LAFAN1_BODIES = [
        "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
        "Spine", "Spine1", "Spine2", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftFootMod", "RightFootMod",
    ]

    out_dir = data_root() / "pairs_variants" / variant_tag
    results = []

    try:
        for bvh_path in clips:
            frames, human_height = load_bvh_file(str(bvh_path))
            T = len(frames)
            J = len(LAFAN1_BODIES)

            human_pos = np.zeros((T, J, 3), dtype=np.float32)
            human_quat = np.zeros((T, J, 4), dtype=np.float32)
            for t, frame in enumerate(frames):
                for j, name in enumerate(LAFAN1_BODIES):
                    p, q = frame[name]
                    human_pos[t, j] = p
                    human_quat[t, j] = q

            retargeter = GMR(
                src_human="bvh_lafan1", tgt_robot=base_robot,
                actual_human_height=human_height, verbose=False
            )
            qpos = np.stack([retargeter.retarget(frame).copy() for frame in frames]).astype(np.float32)

            out_file = out_dir / f"{bvh_path.stem}.npz"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_file,
                human_pos=human_pos,
                human_quat=human_quat,
                human_names=np.array(LAFAN1_BODIES),
                qpos=qpos,
                fps=np.array(30.0),
                robot=np.array(base_robot),
                human_height=np.array(human_height),
                variant_tag=np.array(variant_tag),
            )
            results.append({
                "clip": bvh_path.stem,
                "variant": variant_tag,
                "frames": T,
                "out": str(out_file),
                "qpos_finite": bool(np.isfinite(qpos).all()),
                "mean_pelvis_z": float(qpos[:, 2].mean()),
            })
            print(f"    {bvh_path.stem}: {T} frames, finite={results[-1]['qpos_finite']}, "
                  f"pelvis_z={results[-1]['mean_pelvis_z']:.4f}")
    finally:
        # Restore original
        gmr_params.ROBOT_XML_DICT[base_robot] = original_xml

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="unitree_g1", help="Base robot name")
    ap.add_argument("--retarget", action="store_true",
                    help="Also run GMR teacher on variants (requires GMR + LAFAN1 data)")
    ap.add_argument("--clips", nargs="+", default=["walk1_subject1.bvh"],
                    help="BVH clip filenames (relative to lafan1_bvh dir) for retargeting")
    args = ap.parse_args()

    print(f"Generating embodiment variants for {args.robot}...")
    variant_paths, manifest = generate_variants(args.robot)

    if args.retarget:
        bvh_dir = data_root() / "lafan1_bvh"
        clip_paths = [bvh_dir / c for c in args.clips]
        missing = [c for c in clip_paths if not c.exists()]
        if missing:
            print(f"ERROR: BVH clips not found: {missing}")
            sys.exit(1)

        print("\nRunning teacher retargeting on variants...")
        all_results = []
        for vpath, (tag, region, factor) in zip(variant_paths, VARIANT_SPECS):
            print(f"  Variant: {tag}")
            results = retarget_variant(vpath, tag, clip_paths, args.robot)
            all_results.extend(results)

        # Direction validation for legs_0.85: pelvis height should be LOWER than base
        print("\nDirection validation (legs_0.85 vs base)...")
        base_pair_dir = data_root() / "pairs" / args.robot
        legs085_dir = data_root() / "pairs_variants" / "legs_0.85"
        for clip_name in args.clips:
            stem = pathlib.Path(clip_name).stem
            base_npz = base_pair_dir / f"{stem}.npz"
            var_npz = legs085_dir / f"{stem}.npz"
            if base_npz.exists() and var_npz.exists():
                base_z = np.load(base_npz)["qpos"][:, 2].mean()
                var_z = np.load(var_npz)["qpos"][:, 2].mean()
                print(f"  {stem}: base_pelvis_z={base_z:.4f}, legs085_pelvis_z={var_z:.4f}, "
                      f"lower={'YES' if var_z < base_z else 'NO'}")
            elif var_npz.exists():
                var_data = np.load(var_npz)
                print(f"  {stem}: legs085_pelvis_z={var_data['qpos'][:, 2].mean():.4f} "
                      f"(base pair not available for comparison)")


if __name__ == "__main__":
    main()
