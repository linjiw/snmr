#!/usr/bin/env python3
"""Write one validated coherent G1 MJCF variant from immutable captured bytes."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snmr.mjcf_variants import (
    GeneratorProvenance,
    build_semantic_limb_variant_plan,
    materialize_mjcf_variant,
    write_materialized_mjcf_variant,
    zero_perturbation_plan,
)
from snmr.paths import g1_mjcf
from snmr.provenance import MjcfBundleSnapshot, source_revision_manifest
from snmr.robot_spec import SemanticManifest


def g1_semantics() -> SemanticManifest:
    return SemanticManifest(
        root_link="pelvis",
        torso_link="torso_link",
        left_hand_link="left_rubber_hand_link",
        right_hand_link="right_rubber_hand_link",
        left_foot_link="left_ankle_roll_link",
        right_foot_link="right_ankle_roll_link",
        allowed_contact_links=("left_ankle_roll_link", "right_ankle_roll_link"),
        symmetry_pairs=(
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            ("left_rubber_hand_link", "right_rubber_hand_link"),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help="G1 MJCF entrypoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--link-fraction", type=float, default=0.15)
    parser.add_argument("--newton-root", type=Path, default=None)
    parser.add_argument("--isaac-lab-root", type=Path, default=None)
    parser.add_argument(
        "--joint-limit-shift-deg",
        type=float,
        default=0.0,
        help="must remain 0 until mirrored joint-coordinate signs are qualified",
    )
    parser.add_argument(
        "--zero",
        action="store_true",
        help="publish an exact byte-for-byte zero-perturbation copy",
    )
    return parser.parse_args()


def main() -> None:
    import mujoco

    args = parse_args()
    source = args.source if args.source is not None else g1_mjcf()
    snapshot = MjcfBundleSnapshot.capture(source)
    semantics = g1_semantics()
    if args.zero:
        plan = zero_perturbation_plan(snapshot, semantics)
    else:
        plan = build_semantic_limb_variant_plan(
            snapshot,
            semantics,
            seed=args.seed,
            link_length_fraction=args.link_fraction,
            joint_limit_shift_rad=math.radians(args.joint_limit_shift_deg),
        )
    revisions = source_revision_manifest(
        snmr_path=ROOT,
        newton_path=args.newton_root,
        isaac_lab_path=args.isaac_lab_root,
    )
    provenance = GeneratorProvenance(
        command=(sys.executable, *sys.argv),
        python_version=platform.python_version(),
        mujoco_version=str(mujoco.__version__),
        revisions=tuple(sorted(revisions.items())),
    )
    variant = materialize_mjcf_variant(
        snapshot,
        plan,
        semantics,
        generator_provenance=provenance,
    )
    entrypoint = write_materialized_mjcf_variant(variant, args.output_dir)
    print(
        json.dumps(
            {**variant.manifest(), "output_entrypoint": str(entrypoint)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
