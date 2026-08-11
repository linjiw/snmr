#!/usr/bin/env python
"""Generate SNMR training pairs: LAFAN1 human motion + GMR teacher qpos, per robot.

For each BVH clip and each target robot, saves one NPZ with:
    human_pos   (T, J, 3)  world positions of the LAFAN1 human bodies (fixed 24-body order)
    human_quat  (T, J, 4)  world orientations, wxyz
    human_names (J,)       body names (constant across clips)
    qpos        (T, 7+D)   GMR teacher output [root_pos, root_quat wxyz, dof]
    fps                     scalar (30 by default, GMR-aligned)
    robot                   robot name string
    human_height            scalar from the BVH loader

This is the Phase-1 data engine (design doc §3.3): the human dict is the encoder input, the teacher
qpos is the distillation target, and FK task-space losses come from the same NPZ.

Usage:
    python scripts/make_pairs_lafan1.py --bvh_dir ../data/lafan1_bvh --out_dir ../data/pairs \
        --robots unitree_g1 booster_t1_29dof --max_clips 3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# GMR is installed in the environment (pip install -e GMR --no-deps + its used deps).
from snmr.paths import data_root  # noqa: E402
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

LAFAN1_BODIES = [
    "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftFootMod", "RightFootMod",
]


def process_clip(bvh_path: pathlib.Path, robot: str, out_dir: pathlib.Path) -> dict:
    frames, human_height = load_bvh_file(str(bvh_path))
    T = len(frames)
    J = len(LAFAN1_BODIES)

    human_pos = np.zeros((T, J, 3), dtype=np.float32)
    human_quat = np.zeros((T, J, 4), dtype=np.float32)
    for t, frame in enumerate(frames):
        for j, name in enumerate(LAFAN1_BODIES):
            p, q = frame[name]
            human_pos[t, j] = p
            human_quat[t, j] = q  # wxyz (GMR intermediate convention)

    retargeter = GMR(src_human="bvh_lafan1", tgt_robot=robot, actual_human_height=human_height)
    qpos = np.stack([retargeter.retarget(frame).copy() for frame in frames]).astype(np.float32)

    out = out_dir / robot / (bvh_path.stem + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically so an interrupted multi-hour generation run never leaves a
    # corrupt file that looks complete to a resume pass.
    tmp = out.with_name(f"{out.name}.tmp.npz")
    np.savez_compressed(
        tmp,
        human_pos=human_pos,
        human_quat=human_quat,
        human_names=np.array(LAFAN1_BODIES),
        qpos=qpos,
        fps=np.array(30.0),
        robot=np.array(robot),
        human_height=np.array(human_height),
    )
    tmp.replace(out)
    return {"clip": bvh_path.stem, "robot": robot, "frames": T, "out": str(out)}


def _process_task(task: tuple[pathlib.Path, str, pathlib.Path]) -> dict:
    return process_clip(*task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_dir", default=str(data_root() / "lafan1_bvh"))
    ap.add_argument("--out_dir", default=str(data_root() / "pairs"))
    ap.add_argument("--robots", nargs="+", default=["unitree_g1"])
    ap.add_argument("--max_clips", type=int, default=0, help="0 = all clips")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel clip/robot workers; 1 preserves sequential behavior")
    ap.add_argument("--skip_existing", action="store_true",
                    help="resume without regenerating pair files that already exist")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")

    bvh_files = sorted(pathlib.Path(args.bvh_dir).glob("*.bvh"))
    if args.max_clips:
        bvh_files = bvh_files[: args.max_clips]
    out_dir = pathlib.Path(args.out_dir)
    tasks = []
    skipped = 0
    for robot in args.robots:
        for bvh in bvh_files:
            out = out_dir / robot / f"{bvh.stem}.npz"
            if args.skip_existing and out.exists():
                skipped += 1
                continue
            tasks.append((bvh, robot, out_dir))

    t0 = time.time()
    total_frames = 0
    if args.workers == 1:
        results = map(_process_task, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        results = executor.map(_process_task, tasks)

    try:
        for info in results:
            total_frames += info["frames"]
            elapsed = time.time() - t0
            print(
                f"[{elapsed:7.1f}s] {info['robot']:18s} {info['clip']:28s} "
                f"{info['frames']:6d} frames -> {info['out']}"
            )
    finally:
        if args.workers != 1:
            executor.shutdown()
    dt = time.time() - t0
    print(f"\nDone: {len(tasks)} generated, {skipped} skipped, "
          f"{total_frames} teacher frames in {dt:.0f}s ({total_frames/max(dt,1e-9):.0f} fps)")


if __name__ == "__main__":
    main()
