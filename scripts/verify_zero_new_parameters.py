#!/usr/bin/env python3
"""Does an unseen robot cost any new parameters? Answer it against real MJCFs.

This checks the load-bearing positioning claim of the MorphoRetarget line: one
parameter set serves every humanoid, so a robot absent at training time needs no
new weights.  Per-robot-adapter methods (a learned prompt bank plus an
embodiment-specific output head) and per-pair optimizers cannot say this.

The durable unit-test form of the property lives in
``tests/test_morpho_model.py`` and uses synthetic chains, so it does not depend
on external assets.  This script is the real-robot confirmation.

Usage:
    python scripts/verify_zero_new_parameters.py [--assets DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import torch

from snmr.morpho_model import KinematicMorphoRetargeter, MorphoRetargetConfig
from snmr.robot_spec import RobotSpec, SemanticManifest
from snmr.robot_tokens import RobotGraphTokenizer

DEFAULT_ASSETS = Path("/home/robotixx/GMR/assets")
ROBOTS = (
    "unitree_g1",
    "booster_t1_29dof",
    "fourier_n1",
    "stanford_toddy",
    "unitree_h1",
    "engineai_pm01",
    "kuavo_s45",
    "booster_k1",
)
HUMAN_TOKEN_DIM = 128


def _candidate_mjcfs(assets: Path, robot: str) -> list[Path]:
    return [p for p in sorted(assets.glob(f"{robot}/*.xml")) if "scene" not in p.name.lower()]


def _guess_semantics(path: Path) -> SemanticManifest:
    """Best-effort semantic links, for a shape check only.

    A real experiment uses a hand-declared manifest per robot; name matching is
    adequate here because nothing downstream depends on the anchors being right.
    """
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [
        name
        for i in range(model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
    ]

    def pick(*keys: str) -> str | None:
        for key in keys:
            for name in names:
                if key in name.lower():
                    return name
        return None

    root = pick("pelvis", "base_link", "torso", "trunk") or names[min(1, len(names) - 1)]
    return SemanticManifest(
        root_link=root,
        torso_link=pick("torso", "waist", "chest"),
        head_link=pick("head"),
        left_hand_link=pick("left_hand", "left_wrist", "l_hand"),
        right_hand_link=pick("right_hand", "right_wrist", "r_hand"),
        left_foot_link=pick("left_ankle_roll", "left_foot", "left_ankle", "l_foot"),
        right_foot_link=pick("right_ankle_roll", "right_foot", "right_ankle", "r_foot"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()

    if not args.assets.is_dir():
        print(f"assets directory not found: {args.assets}")
        return 2

    specs: list[tuple[str, RobotSpec]] = []
    for robot in ROBOTS:
        for path in _candidate_mjcfs(args.assets, robot):
            try:
                specs.append((robot, RobotSpec.from_mjcf(path, _guess_semantics(path))))
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"  skip {robot}/{path.name}: {type(error).__name__}: {error}")
                continue
            break

    if not specs:
        print("no robot specs could be built")
        return 2

    tokenizer = RobotGraphTokenizer(feature_set="kinematic")
    model = KinematicMorphoRetargeter(
        MorphoRetargetConfig(human_token_dim=HUMAN_TOKEN_DIM)
    ).eval()
    reference = sum(p.numel() for p in model.parameters())
    print(f"\none model, {reference:,} parameters\n")
    print(f"{'robot':<20}{'joints':>7}{'nodes':>7}{'output':>16}{'parameters':>13}")

    counts, joint_counts = set(), set()
    for robot, spec in specs:
        batch = tokenizer([spec])
        with torch.no_grad():
            output = model(torch.randn(1, 4, HUMAN_TOKEN_DIM), batch)
        total = sum(p.numel() for p in model.parameters())
        counts.add(total)
        joint_counts.add(len(spec.joints))
        print(
            f"{robot:<20}{len(spec.joints):>7}{batch.node_features.shape[1]:>7}"
            f"{str(tuple(output.joint_positions.shape)):>16}{total:>13,}"
        )

    print()
    if len(counts) != 1:
        print(f"FAIL: parameter count varied across robots: {sorted(counts)}")
        return 1
    print(
        f"PASS: {len(specs)} robots spanning DoF {sorted(joint_counts)} share one "
        f"parameter set.\n      An unseen robot adds ZERO parameters; only the output "
        f"width tracks the robot."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
