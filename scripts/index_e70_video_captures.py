#!/usr/bin/env python
"""Validate and index raw E70 video captures against the frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
from fractions import Fraction
from typing import Any


HORIZON_STEPS = 500
CONTROL_DT = 0.02


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_capture_report(item: dict[str, Any], report: dict[str, Any]) -> None:
    expected_start = int(item["start_step"])
    expected = {
        "protocol": "E70 exact simulation capture report v1",
        "capture_name": item["name"],
        "arm": item["arm"],
        "phase_only": bool(item["phase_only"]),
        "shuffle_latent": bool(item["shuffle_latent"]),
        "destroy_zcmd": item["destroy_zcmd"],
        "evaluation_seed": 404,
        "num_rollouts": 1,
        "simulator_num_envs": 1024 if item["destroy_zcmd"] == "marginal_random" else 1,
        "intervention_pool_size": 1024 if item["destroy_zcmd"] == "marginal_random" else 1,
        "exact_start": expected_start,
        "video_capture": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"capture {item['name']} report {key}={report.get(key)!r}, "
                f"expected {value!r}"
            )
    if report.get("start_steps") != [expected_start]:
        raise ValueError(f"capture {item['name']} did not realize its exact start")
    if report.get("motion_ids") != [int(item["side"])] or report.get("clip") != item["clip"]:
        raise ValueError(f"capture {item['name']} resolved to the wrong motion")
    if report.get("student_checkpoint_sha256") != item["checkpoint_sha256"]:
        raise ValueError(f"capture {item['name']} used a different student checkpoint")
    if len(report.get("completed", [])) != 1 or len(report.get("survival_s", [])) != 1:
        raise ValueError(f"capture {item['name']} has invalid outcome-vector lengths")
    try:
        steps_executed = int(report["steps_executed"])
        survival_s = float(report["survival_s"][0])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"capture {item['name']} lacks a valid executed-step count") from exc
    completed = bool(report["completed"][0])
    if not 1 <= steps_executed <= HORIZON_STEPS:
        raise ValueError(f"capture {item['name']} has invalid steps_executed={steps_executed}")
    if completed and steps_executed != HORIZON_STEPS:
        raise ValueError(
            f"capture {item['name']} completion disagrees with steps_executed={steps_executed}"
        )
    if abs(survival_s - steps_executed * CONTROL_DT) > CONTROL_DT / 2:
        raise ValueError(
            f"capture {item['name']} survival={survival_s} disagrees with "
            f"steps_executed={steps_executed}"
        )
    config = report.get("video_config")
    if not isinstance(config, dict):
        raise ValueError(f"capture {item['name']} lacks video configuration")
    for key, value in (
        ("enabled", True),
        ("width", 1920),
        ("height", 1080),
        ("playback_rate", 1.0),
        ("record_env_id", 0),
        ("vertical_fov", 45.0),
        ("use_recording_thread", False),
    ):
        if config.get(key) != value:
            raise ValueError(
                f"capture {item['name']} video {key}={config.get(key)!r}, expected {value!r}"
            )
    expected_camera = {
        "type": "cartesian",
        "offset": [2.0, 2.0, 1.0],
        "target_offset": [0.0, 0.0, 0.3],
        "smoothing": 0.95,
        "tracking_body_name": "pelvis",
    }
    if config.get("camera") != expected_camera:
        raise ValueError(
            f"capture {item['name']} camera={config.get('camera')!r}, "
            f"expected {expected_camera!r}"
        )

    runtime = pathlib.Path(str(report.get("runtime", "")))
    expected_runtime = pathlib.Path(__file__).with_name("eval_e70_video.py").resolve()
    if runtime.resolve() != expected_runtime or report.get("runtime_sha256") != sha256_file(expected_runtime):
        raise ValueError(f"capture {item['name']} evaluator provenance does not match")
    for paths_key, hashes_key in (
        ("teacher_ckpts", "teacher_checkpoint_sha256"),
        ("motion_files", "motion_sha256"),
    ):
        paths = report.get(paths_key, [])
        hashes = report.get(hashes_key, [])
        if len(paths) != 2 or len(hashes) != 2:
            raise ValueError(f"capture {item['name']} lacks {paths_key} provenance")
        if [sha256_file(pathlib.Path(path)) for path in paths] != hashes:
            raise ValueError(f"capture {item['name']} {paths_key} hashes do not match")


def probe(path: pathlib.Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames",
            "-show_entries", "format=duration,size", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    stream, fmt = payload["streams"][0], payload["format"]
    return {
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream["r_frame_rate"])),
        "frames": int(stream["nb_frames"]),
        "duration_seconds": float(fmt["duration"]),
        "bytes": int(fmt["size"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--student-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("protocol") != "E70 paper-video rollout selection v1":
        raise ValueError("unexpected E70 video manifest protocol")
    captures = []
    for item in manifest["captures"]:
        checkpoint = pathlib.Path(item["checkpoint"])
        if checkpoint.parent.parent != args.student_root.resolve():
            raise ValueError(f"capture checkpoint is outside the E70 student root: {checkpoint}")
        if sha256_file(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash changed for {item['name']}")
        report_path = args.raw_dir / f"{item['name']}.report.json"
        raw_path = args.raw_dir / f"{item['name']}.mp4"
        report = json.loads(report_path.read_text())
        validate_capture_report(item, report)
        media = probe(raw_path)
        if media["width"] != 1920 or media["height"] != 1080 or media["frames"] < 1:
            raise ValueError(f"capture {item['name']} has invalid media properties: {media}")
        if abs(media["frames"] - int(report["steps_executed"])) > 3:
            raise ValueError(
                f"capture {item['name']} encoded {media['frames']} frames for "
                f"{report['steps_executed']} executed steps"
            )
        captures.append(
            {
                "name": item["name"],
                "clip": item["clip"],
                "exact_start": item["start_step"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": item["checkpoint_sha256"],
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "raw_video": str(raw_path.resolve()),
                "raw_video_sha256": sha256_file(raw_path),
                "media": media,
                "completed": bool(report["completed"][0]),
                "survival_s": float(report["survival_s"][0]),
                "steps_executed": int(report["steps_executed"]),
                "video_config": report["video_config"],
            }
        )

    index = {
        "protocol": "E70 raw simulation capture index v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "selection_uses_policy_outcomes": False,
        "captures": captures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(index, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"validated {len(captures)} captures -> {args.out}")


if __name__ == "__main__":
    main()
