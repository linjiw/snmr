#!/usr/bin/env python
"""Render a sim2sim review video and tracking metrics from a state capture.

Consumes the ``.npz`` written by ``run_sim2sim_review_capture.py``: replays the
logged MuJoCo ``qpos`` through the saved model (or the stock G1 WBT plane
scene) with an offscreen EGL renderer, encodes H.264 via ffmpeg, and computes
joint-position tracking RMSE against the reference motion from the
``motion_started`` event onward.  Engineering artifact for the Phase E sim2sim
campaign; not preregistered, never enters the paper or the frozen paper-video
pipeline.  Run with the SNMR ``.venv`` Python (MUJOCO_GL=egl).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import mujoco
import numpy as np

FPS = 25
WIDTH = 1280
HEIGHT = 720
FALLBACK_SCENE = pathlib.Path(
    "/home/robotixx/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/"
    "scene_g1_29dof_wbt_plane.xml"
)


def load_model(model_mjb: pathlib.Path, nq: int) -> mujoco.MjModel:
    for candidate in (model_mjb, FALLBACK_SCENE):
        try:
            if candidate.suffix == ".mjb":
                model = mujoco.MjModel.from_binary_path(str(candidate))
            else:
                model = mujoco.MjModel.from_xml_path(str(candidate))
        except Exception as error:  # noqa: BLE001 - fall through to next source
            print(f"model load failed for {candidate}: {error}")
            continue
        if model.nq == nq:
            print(f"render model: {candidate} (nq={model.nq})")
            return model
        print(f"model {candidate} has nq={model.nq}, capture has {nq}; skipping")
    raise RuntimeError("no render model matches the captured qpos dimension")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=pathlib.Path, required=True)
    parser.add_argument("--capture-report", type=pathlib.Path, required=True)
    parser.add_argument("--reference-motion", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--metrics-out", type=pathlib.Path, required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    states = np.load(args.states, allow_pickle=False)
    report = json.loads(args.capture_report.read_text())
    qpos = states["qpos"]
    dof_pos = states["dof_pos"]
    times = states["sample_times"]
    dof_names = [str(n) for n in states["dof_names"]]
    if qpos.shape[0] == 0:
        raise RuntimeError("capture contains no samples")

    model = load_model(pathlib.Path(report["model_mjb"]), qpos.shape[1])
    data = mujoco.MjData(model)

    events = {e["event"]: e["time_s"] for e in report["events"]}
    motion_start = events.get("motion_started")
    release = events.get("gantry_released")
    handoff = events.get("safety_handoff_requested")

    # Tracking metric: reference frame k = floor((t - motion_start) * 50) at the
    # policy's 50 Hz motion clock; compared joints are matched by name.
    metrics: dict[str, object] = {
        "reference_motion": str(args.reference_motion),
        "motion_started_s": motion_start,
        "gantry_released_s": release,
        "safety_handoff_s": handoff,
        "note": (
            "engineering tracking metric from a wall-clock-aligned replay; "
            "not preregistered, not a paper value"
        ),
    }
    if motion_start is not None:
        ref = np.load(args.reference_motion, allow_pickle=False)
        ref_fps = float(ref["fps"][0])
        ref_names = [str(n) for n in ref["joint_names"]]
        ref_dofs = ref["joint_pos"][:, 7:]
        col = [ref_names.index(n) for n in dof_names]
        ref_dofs = ref_dofs[:, col]
        errors, err_times = [], []
        for t, q in zip(times, dof_pos):
            if t < motion_start:
                continue
            k = int((t - motion_start) * ref_fps)
            if k >= ref_dofs.shape[0]:
                break
            errors.append(q - ref_dofs[k])
            err_times.append(t)
        errors = np.asarray(errors)
        err_times = np.asarray(err_times)

        def window_rmse(lo: float | None, hi: float | None) -> float | None:
            mask = np.ones(len(err_times), dtype=bool)
            if lo is not None:
                mask &= err_times >= lo
            if hi is not None:
                mask &= err_times < hi
            if not mask.any():
                return None
            return float(np.sqrt(np.mean(errors[mask] ** 2)))

        metrics["joint_rmse_rad_full_motion"] = window_rmse(motion_start, handoff)
        metrics["joint_rmse_rad_assisted"] = window_rmse(motion_start, release)
        metrics["joint_rmse_rad_unassisted"] = window_rmse(release, handoff)
        metrics["mean_abs_error_rad_unassisted"] = (
            float(np.mean(np.abs(errors[(err_times >= release) & (err_times < handoff)])))
            if release is not None and handoff is not None
            else None
        )

    # Base pose telemetry over the same windows, from the logged root states.
    root = states["root_states"]
    up_axis = 1.0 - 2.0 * (root[:, 3] ** 2 + root[:, 4] ** 2)
    if release is not None and handoff is not None:
        mask = (times >= release) & (times < handoff)
        metrics["min_base_height_m_unassisted"] = float(root[mask, 2].min())
        metrics["min_up_axis_z_unassisted"] = float(up_axis[mask].min())
    mask_hold = times >= (handoff if handoff is not None else times[-1])
    if mask_hold.any():
        metrics["min_base_height_m_safety_hold"] = float(root[mask_hold, 2].min())
        metrics["min_up_axis_z_safety_hold"] = float(up_axis[mask_hold].min())

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")

    # Render: subsample 200 Hz samples to FPS, chase camera on the base link.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 3.2
    camera.azimuth = 140.0
    camera.elevation = -12.0

    stride = max(1, round(200 / FPS))
    frame_indices = range(0, qpos.shape[0], stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    label = args.title.replace(":", r"\:").replace("'", "")
    drawtext = (
        f"drawtext=text='{label}':x=24:y=24:fontsize=28:fontcolor=white:"
        "box=1:boxcolor=black@0.45:boxborderw=8"
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
            "-vf", drawtext if label else "null",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.out),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    for i in frame_indices:
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = [qpos[i][0], qpos[i][1], 0.8]
        renderer.update_scene(data, camera=camera)
        encoder.stdin.write(renderer.render().tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    print(f"video={args.out} frames={len(list(frame_indices))} metrics={args.metrics_out}")


if __name__ == "__main__":
    main()
