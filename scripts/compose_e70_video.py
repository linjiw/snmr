#!/usr/bin/env python
"""Compose the frozen E70 simulation captures into an ICRA-ready video.

Raw rollout selection is fixed by ``prepare_e70_video.py``.  This script only
lays out those captures, adds result cards populated from the frozen analyzer,
encodes the final MP4, and verifies the submission constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
from fractions import Fraction
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 30
# Keep a full decimal megabyte below the official 20 MB ceiling so upload-system
# unit conventions and small remuxing differences cannot turn a local pass into a
# submission rejection.
MAX_BYTES = 19_000_000
FONT = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
BG = "#07111f"
INK = "#edf4ff"
MUTED = "#9fb0c6"
BLUE = "#5ba7ff"
GREEN = "#55d69e"
ORANGE = "#ffb45e"
GRAY = "#8291a6"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Extract only the preregistered quantities displayed in the video."""
    if analysis.get("protocol") != "E70 preregistered analysis v1":
        raise ValueError("analysis does not use the frozen E70 protocol")
    seeds = [int(seed) for seed in analysis.get("seeds", [])]
    if seeds != [0, 1, 2]:
        raise ValueError("final paper video requires completed training seeds 0, 1, and 2")
    arms = analysis.get("arms", {})
    order = ("explicit", "snmr", "time", "proprio", "shuffled")
    completions = {}
    for arm in order:
        try:
            value = float(arms[arm]["ambiguity_completion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"analysis is missing {arm} ambiguity completion") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid {arm} completion: {value}")
        completions[arm] = value

    contrasts = {}
    for label, key in (("A-T", "snmr_minus_time"), ("A-S", "snmr_minus_shuffled")):
        item = analysis.get(key, {})
        try:
            contrasts[label] = {
                "difference": float(item["difference"]),
                "ci95_low": float(item["ci95_low"]),
                "ci95_high": float(item["ci95_high"]),
                "clusters": int(item["clusters"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"analysis is missing {label} contrast") from exc
        values = contrasts[label]
        if (
            not all(math.isfinite(value) for value in values.values())
            or values["clusters"] != 69
            or not -1.0 <= values["ci95_low"] <= values["difference"] <= values["ci95_high"] <= 1.0
        ):
            raise ValueError(f"analysis has an invalid {label} contrast: {values}")

    if not isinstance(analysis.get("explicit_general_gate"), bool):
        raise ValueError("analysis is missing the frozen explicit capability gate")
    if not isinstance(analysis.get("positive_content_gate"), bool):
        raise ValueError("analysis is missing the frozen positive-content gate")
    if analysis["positive_content_gate"] and not analysis["explicit_general_gate"]:
        raise ValueError("positive-content gate cannot pass when explicit capability fails")

    return {
        "seeds": seeds,
        "completions": completions,
        "contrasts": contrasts,
        "explicit_general_gate": analysis["explicit_general_gate"],
        "positive_content_gate": analysis["positive_content_gate"],
        "interpretation": str(analysis.get("interpretation", "")),
    }


def capture_status(capture: dict[str, Any]) -> tuple[str, str]:
    """Return an honest, compact rollout-status label and display color."""
    survival_s = float(capture["survival_s"])
    if not 0.0 < survival_s <= 10.01:
        raise ValueError(f"invalid capture survival {survival_s}")
    if bool(capture["completed"]):
        return f"completed · {survival_s:.2f} s", GREEN
    return f"terminated · {survival_s:.2f} s", ORANGE


def validated_capture_index(
    index: dict[str, Any],
    *,
    manifest_sha256: str,
    raw_paths: dict[str, pathlib.Path],
) -> dict[str, dict[str, Any]]:
    """Bind capture outcomes and media hashes to this manifest and raw directory."""
    if index.get("protocol") != "E70 raw simulation capture index v1":
        raise ValueError("unexpected raw-capture index protocol")
    if index.get("manifest_sha256") != manifest_sha256:
        raise ValueError("raw-capture index does not match the video manifest")
    if index.get("selection_uses_policy_outcomes") is not False:
        raise ValueError("raw-capture index does not preserve policy-independent selection")
    entries = {str(item["name"]): item for item in index.get("captures", [])}
    if set(entries) != set(raw_paths):
        raise ValueError("raw-capture index names do not match the manifest")
    for name, path in raw_paths.items():
        if entries[name].get("raw_video_sha256") != sha256_file(path):
            raise ValueError(f"raw-capture hash mismatch for {name}")
        capture_status(entries[name])
    return entries


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT
    return ImageFont.truetype(str(path), size=size)


def _centered(draw: ImageDraw.ImageDraw, text: str, y: int, font: ImageFont.FreeTypeFont, fill: str) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    draw.text(((WIDTH - (bounds[2] - bounds[0])) / 2, y), text, font=font, fill=fill)


def make_text_card(path: pathlib.Path, *, kicker: str, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((128, 116, WIDTH - 128, HEIGHT - 116), radius=28, fill="#0d1b2d", outline="#294363", width=3)
    _centered(draw, kicker.upper(), 190, _font(28, bold=True), BLUE)
    _centered(draw, title, 292, _font(64, bold=True), INK)
    for index, line in enumerate(lines):
        _centered(draw, line, 480 + 70 * index, _font(35), MUTED if index else INK)
    image.save(path)


def make_results_card(path: pathlib.Path, summary: dict[str, Any]) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    seed_label = ", ".join(str(seed) for seed in summary["seeds"])
    draw.text((120, 95), "PREREGISTERED AMBIGUITY-START COMPLETION", font=_font(28, bold=True), fill=BLUE)
    draw.text((120, 148), f"training seeds {seed_label} · 1,024 rollouts per arm/seed", font=_font(29), fill=MUTED)
    labels = {
        "explicit": "C  explicit goal",
        "snmr": "A  SNMR latent",
        "time": "T  absolute time",
        "proprio": "B  proprio only",
        "shuffled": "S  shuffled content",
    }
    colors = {"explicit": BLUE, "snmr": GREEN, "time": ORANGE, "proprio": GRAY, "shuffled": ORANGE}
    x0, bar_width = 600, 1050
    for row, (arm, value) in enumerate(summary["completions"].items()):
        y = 285 + row * 132
        draw.text((120, y + 12), labels[arm], font=_font(34, bold=arm in {"explicit", "snmr"}), fill=INK)
        draw.rounded_rectangle((x0, y, x0 + bar_width, y + 68), radius=12, fill="#17273b")
        draw.rounded_rectangle((x0, y, x0 + max(8, int(bar_width * value)), y + 68), radius=12, fill=colors[arm])
        draw.text((x0 + bar_width + 28, y + 7), f"{value:.3f}", font=_font(38, bold=True), fill=INK)
    image.save(path)


def make_contrast_card(path: pathlib.Path, summary: dict[str, Any]) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((120, 104), "PAIRED CONTENT CONTRASTS", font=_font(30, bold=True), fill=BLUE)
    draw.text((120, 164), "Hierarchy: training seed → ambiguity pair → rollout", font=_font(31), fill=MUTED)
    for row, (label, item) in enumerate(summary["contrasts"].items()):
        y = 335 + row * 230
        color = GREEN if item["ci95_low"] > 0 else ORANGE
        draw.text((150, y), label, font=_font(68, bold=True), fill=color)
        draw.text((420, y + 4), f"{item['difference']:+.3f}", font=_font(64, bold=True), fill=INK)
        draw.text(
            (790, y + 16),
            f"95% CI [{item['ci95_low']:+.3f}, {item['ci95_high']:+.3f}]",
            font=_font(42),
            fill=INK,
        )
        draw.text((420, y + 92), f"{item['clusters']} reference-only state-matched pairs", font=_font(30), fill=MUTED)
    gate_label, gate_color = outcome_gate_label(summary)
    _centered(draw, gate_label, 875, _font(38, bold=True), gate_color)
    image.save(path)


def outcome_gate_label(summary: dict[str, Any]) -> tuple[str, str]:
    """Return outcome-conditioned wording and color for the video result card."""
    if not summary["explicit_general_gate"]:
        return "Frozen assay validity: INVALID (explicit control failed)", ORANGE
    if summary["positive_content_gate"]:
        return "Frozen positive-content gate: PASS", GREEN
    return "Frozen positive-content gate: DOES NOT PASS", ORANGE


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")


def encode_card(image: pathlib.Path, output: pathlib.Path, duration: float) -> None:
    _run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(image), "-t", str(duration),
            "-vf", f"fps={FPS},format=yuv420p", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", str(output),
        ]
    )


def _drawtext(text: str) -> str:
    return text.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def encode_panel(
    left: pathlib.Path,
    right: pathlib.Path,
    output: pathlib.Path,
    *,
    title: str,
    left_label: str,
    right_label: str,
    left_status: tuple[str, str],
    right_status: tuple[str, str],
    footer: str,
    duration: float = 10.0,
) -> None:
    for raw in (left, right):
        if not raw.is_file():
            raise FileNotFoundError(raw)
    common = (
        f"fps={FPS},scale=960:720:force_original_aspect_ratio=decrease,"
        "pad=960:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},setpts=PTS-STARTPTS"
    )
    font = str(FONT_BOLD)
    graph = (
        f"[0:v]{common}[left];[1:v]{common}[right];"
        "[left][right]hstack=inputs=2[body];"
        f"[body]pad={WIDTH}:{HEIGHT}:0:205:color={BG},"
        f"drawtext=fontfile={font}:text='{_drawtext(title)}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=48,"
        f"drawtext=fontfile={font}:text='{_drawtext(left_label)}':fontcolor={GREEN}:fontsize=35:x=480-text_w/2:y=135,"
        f"drawtext=fontfile={font}:text='{_drawtext(right_label)}':fontcolor={ORANGE}:fontsize=35:x=1440-text_w/2:y=135,"
        f"drawtext=fontfile={FONT}:text='{_drawtext(left_status[0])}':fontcolor={left_status[1]}:fontsize=25:x=480-text_w/2:y=177,"
        f"drawtext=fontfile={FONT}:text='{_drawtext(right_status[0])}':fontcolor={right_status[1]}:fontsize=25:x=1440-text_w/2:y=177,"
        f"drawtext=fontfile={FONT}:text='{_drawtext(footer)}':fontcolor={MUTED}:fontsize=28:x=(w-text_w)/2:y=970,"
        "format=yuv420p[out]"
    )
    _run(
        [
            "ffmpeg", "-y", "-i", str(left), "-i", str(right),
            "-filter_complex", graph, "-map", "[out]", "-an", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(output),
        ]
    )


def probe_video(path: pathlib.Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,field_order",
            "-show_entries", "format=duration,size", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    fmt = payload["format"]
    frame_result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-read_intervals", "%+#1", "-show_entries", "frame=interlaced_frame",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    first_frame = json.loads(frame_result.stdout).get("frames", [{}])[0]
    fps = float(Fraction(stream["r_frame_rate"]))
    observed = {
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "field_order": stream.get("field_order", "unknown"),
        "first_frame_interlaced": int(first_frame.get("interlaced_frame", -1)),
        "duration_seconds": float(fmt["duration"]),
        "bytes": int(fmt["size"]),
        "sha256": sha256_file(path),
    }
    checks = {
        "h264": observed["codec"] == "h264",
        "yuv420p": observed["pixel_format"] == "yuv420p",
        "minimum_height_480": observed["height"] >= 480,
        "minimum_fps_20": observed["fps"] >= 20.0,
        "progressive": observed["field_order"] == "progressive"
        or observed["first_frame_interlaced"] == 0,
        "maximum_duration_180s": observed["duration_seconds"] <= 180.0,
        "maximum_size_19MB": observed["bytes"] <= MAX_BYTES,
    }
    return {"observed": observed, "checks": checks, "passes": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--analysis", type=pathlib.Path, required=True)
    parser.add_argument("--capture-index", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required")
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --force")

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("protocol") != "E70 paper-video rollout selection v1":
        raise ValueError("video manifest does not use the frozen selection protocol")
    analysis = json.loads(args.analysis.read_text())
    summary = result_summary(analysis)
    captures = {item["name"]: args.raw_dir / f"{item['name']}.mp4" for item in manifest["captures"]}
    missing = [str(path) for path in captures.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing raw captures:\n" + "\n".join(missing))
    capture_index = json.loads(args.capture_index.read_text())
    capture_records = validated_capture_index(
        capture_index,
        manifest_sha256=sha256_file(args.manifest),
        raw_paths=captures,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = args.out.parent / "composition_work"
    work.mkdir(parents=True, exist_ok=True)
    segments: list[pathlib.Path] = []

    card_specs = [
        (
            "intro", 6.0, "CAUSAL HUMANOID INTERFACES",
            "What must retargeting tell tracking?",
            ["A learned 64-d command is the action decoder's only goal channel.", "Simulation · Unitree G1 · deterministic deployment"],
        ),
        (
            "assay", 7.0, "THE TWO-WALK AMBIGUITY ASSAY",
            "Similar present. Different future.",
            ["69 pairs selected from reference motion only.", "Same starts, architecture, training budget, and evaluation seed across arms."],
        ),
    ]
    for name, duration, kicker, title, lines in card_specs:
        png = work / f"{name}.png"
        mp4 = work / f"{name}.mp4"
        make_text_card(png, kicker=kicker, title=title, lines=lines)
        encode_card(png, mp4, duration)
        segments.append(mp4)

    for clip in ("walk1_subject1", "walk1_subject5"):
        panel = work / f"panel_{clip}.mp4"
        encode_panel(
            captures[f"snmr_{clip}"], captures[f"time_{clip}"], panel,
            title=clip.replace("_", " "), left_label="A · SNMR latent", right_label="T · absolute time",
            left_status=capture_status(capture_records[f"snmr_{clip}"]),
            right_status=capture_status(capture_records[f"time_{clip}"]),
            footer="Same exact start and camera · seed-0 illustration · real-time simulation", duration=10.0,
        )
        segments.append(panel)

    explicit_panel = work / "panel_explicit.mp4"
    encode_panel(
        captures["explicit_clean_walk1_subject1"],
        captures["explicit_destroy_marginal_walk1_subject1"],
        explicit_panel,
        title="Causal use of the exclusive command",
        left_label="C · clean command",
        right_label="C · marginal-resampled command",
        left_status=capture_status(capture_records["explicit_clean_walk1_subject1"]),
        right_status=capture_status(capture_records["explicit_destroy_marginal_walk1_subject1"]),
        footer="Same policy, start, and camera · only the learned command is intervened on",
        duration=10.0,
    )
    segments.append(explicit_panel)

    results_png, results_mp4 = work / "results.png", work / "results.mp4"
    make_results_card(results_png, summary)
    encode_card(results_png, results_mp4, 10.0)
    segments.append(results_mp4)

    contrast_png, contrast_mp4 = work / "contrasts.png", work / "contrasts.mp4"
    make_contrast_card(contrast_png, summary)
    encode_card(contrast_png, contrast_mp4, 9.0)
    segments.append(contrast_mp4)

    boundary_png, boundary_mp4 = work / "boundary.png", work / "boundary.mp4"
    make_text_card(
        boundary_png,
        kicker="EVIDENCE BOUNDARY",
        title="Simulation result, not a hardware claim",
        lines=[
            "Production CPU loopback and WBT-to-safety handoff pass.",
            "Still unvalidated: robustness · HIL · tethered hardware · sim-to-real.",
        ],
    )
    encode_card(boundary_png, boundary_mp4, 8.0)
    segments.append(boundary_mp4)

    concat_file = work / "segments.txt"
    concat_file.write_text("".join(f"file '{path.resolve()}'\n" for path in segments))
    _run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-an", "-r", str(FPS), "-c:v", "libx264", "-preset", "slow",
            "-b:v", "1800k", "-maxrate", "2000k", "-bufsize", "4000k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
        ]
    )

    contact_sheet = args.out.with_name(args.out.stem + "_contact_sheet.png")
    _run(
        [
            "ffmpeg", "-y", "-i", str(args.out),
            "-vf", "fps=1/6,scale=480:270,tile=4x3", "-frames:v", "1", str(contact_sheet),
        ]
    )
    validation = probe_video(args.out)
    validation.update(
        {
            "protocol": "E70 ICRA video validation v1",
            "video": str(args.out.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "analysis_sha256": sha256_file(args.analysis),
            "capture_index_sha256": sha256_file(args.capture_index),
            "raw_captures": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in captures.items()},
            "result_summary": summary,
            "contact_sheet": str(contact_sheet.resolve()),
        }
    )
    validation_path = args.out.with_name(args.out.stem + "_validation.json")
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    if not validation["passes"]:
        raise RuntimeError(f"video failed submission checks; see {validation_path}")
    print(json.dumps(validation["observed"], indent=2))
    print(f"validation={validation_path}")


if __name__ == "__main__":
    main()
