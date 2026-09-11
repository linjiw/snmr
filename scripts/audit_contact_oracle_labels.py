#!/usr/bin/env python
"""Label-source audit for the teacher-height contact oracle (no checkpoint, CPU only).

The Gate 1b oracle is *teacher-height hysteresis*: per foot, the ankle-body origin height minus
that foot's clip-wide minimum, enter <= 0.03 m / exit < 0.05 m.  It is a proxy for stance, not a
force-verified label.  Near-zero stance prevalence on six of seven audited clips (frozen
``runs/gate1b/mask_audit.json``) is a reason to inspect the label construction, not a
conclusion that the oracle is wrong on those clips.  This script inspects, per clip and per
foot, every ingredient the reviewer asked about:

* ground plane: per-foot clip minimum (the oracle's choice) vs a shared clip minimum vs a robust
  lower-tail plateau (histogram mode of the lowest 5 cm) vs a low percentile;
* sole point vs ankle origin: the origin-to-plateau offset and the foot tilt at the minimum
  frame vs the clip median (a tilted or penetrating foot drags the origin below the stance
  plateau, which then hides ordinary stance under the 3 cm enter threshold);
* frame convention and world-axis orientation: pelvis height, mean up-axis, fps and frame counts
  for the human and robot streams;
* threshold hysteresis: stance prevalence under an enter/exit sweep and under each ground choice;
* stance-run structure: count and duration of oracle stance runs, and left/right alternation;
* the human-side deployable mask (``source_contact``, height+speed on the toe joints) and a
  human-side height-only mask for comparison — reported, not used to redefine the oracle.

Nothing here redefines the oracle or changes any frozen result.  It writes one JSON with
provenance so a reviewer can see *why* prevalence is low where it is low.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.experiment import git_state, runtime_state, sha256_file, utc_now  # noqa: E402
from snmr.human import LAFAN1_CONTACT_BODIES, load_pair_npz  # noqa: E402
from snmr.metrics import FOOT_BODIES, detect_contact, detect_contact_height_hysteresis  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402

ENTER, EXIT = 0.03, 0.05
SWEEP = ((0.02, 0.04), (0.03, 0.05), (0.05, 0.07), (0.08, 0.10))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a 1-D bool array as (start, end_exclusive)."""
    out = []
    start = None
    for i, v in enumerate(mask.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def _tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Angle between the body z-axis and world z, in degrees, for (T, 4) wxyz quaternions."""
    w, x, y, z = (quat_wxyz[:, i] for i in range(4))
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(r22, -1.0, 1.0)))


def _plateau(z: np.ndarray, bin_m: float = 0.005, span_m: float = 0.05) -> float:
    """Histogram mode of the lowest ``span_m`` of the height distribution (a stance plateau)."""
    lo = float(z.min())
    edges = np.arange(lo, lo + span_m + bin_m, bin_m)
    hist, _ = np.histogram(z, bins=edges)
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def _prevalence(height_rel: np.ndarray, enter: float, exit_: float) -> float:
    mask = detect_contact_height_hysteresis(
        torch.tensor(height_rel[:, None, None].repeat(3, axis=2)), enter_height=enter,
        exit_height=exit_, ground_z=0.0,
    )
    return float(mask.float().mean())


def audit_clip(ctx: RobotContext, pair_path: pathlib.Path, window: int, windows_per_clip: int) -> dict:
    pair = load_pair_npz(str(pair_path))
    hp, q = pair["human_pos"], pair["qpos"]
    fps = float(pair["fps"])
    T = int(q.shape[0])
    with torch.no_grad():
        body_pos, body_quat = ctx.kin.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])
    feet = FOOT_BODIES[ctx.name]
    foot_idx = [ctx.kin.body_index(n) for n in feet]
    fz = body_pos[:, foot_idx, 2].numpy()  # (T, 2)
    fq = body_quat[:, foot_idx, :].numpy()
    pelvis_z = body_pos[:, ctx.kin.body_index("pelvis"), 2].numpy() if "pelvis" in ctx.kin.graph.body_names else q[:, 2].numpy()
    oracle = detect_contact_height_hysteresis(body_pos[:, foot_idx, :], ENTER, EXIT).numpy()

    starts = np.linspace(0, max(T - window, 0), num=min(windows_per_clip, max(T // window, 1)), dtype=int)
    sampled = np.zeros(T, dtype=bool)
    for s in starts:
        sampled[int(s) : int(s) + window] = True

    source_idx = [pair["human_names"].index(n) for n in LAFAN1_CONTACT_BODIES]
    hz = hp[:, source_idx, 2].numpy()
    source_mask = detect_contact(hp[:, source_idx, :], fps, height_threshold=0.08, speed_threshold=0.24).numpy()
    human_height_only = detect_contact_height_hysteresis(hp[:, source_idx, :], ENTER, EXIT).numpy()
    hips_idx = pair["human_names"].index("Hips") if "Hips" in pair["human_names"] else 0

    shared_min = float(fz.min())
    per_foot = []
    for f, name in enumerate(feet):
        z = fz[:, f]
        zmin = float(z.min())
        imin = int(np.argmin(z))
        plateau = _plateau(z)
        rel_min = z - zmin
        rel_plateau = z - plateau
        rel_shared = z - shared_min
        rel_p2 = z - float(np.percentile(z, 2))
        tilt = _tilt_deg(fq[:, f, :])
        runs = _runs(oracle[:, f])
        run_len = np.array([e - s for s, e in runs], dtype=float)
        per_foot.append(
            {
                "body": name,
                "z_min_m": zmin,
                "z_min_frame": imin,
                "z_min_frame_in_sampled_windows": bool(sampled[imin]),
                "z_percentiles_m": {p: float(np.percentile(z, p)) for p in (1, 2, 5, 25, 50, 75)},
                "plateau_m": plateau,
                "plateau_minus_min_m": plateau - zmin,
                "frames_within_enter_of_min": int((rel_min <= ENTER).sum()),
                "frames_within_enter_of_plateau": int((np.abs(rel_plateau) <= ENTER).sum()),
                "tilt_deg_at_min_frame": float(tilt[imin]),
                "tilt_deg_median": float(np.median(tilt)),
                "tilt_deg_p95": float(np.percentile(tilt, 95)),
                "oracle_prevalence": float(oracle[:, f].mean()),
                "oracle_prevalence_in_sampled_windows": float(oracle[sampled, f].mean()) if sampled.any() else None,
                "oracle_runs": len(runs),
                "oracle_run_len_frames": {
                    "median": float(np.median(run_len)) if run_len.size else None,
                    "max": float(run_len.max()) if run_len.size else None,
                    "min": float(run_len.min()) if run_len.size else None,
                },
                "prevalence_by_ground": {
                    "per_foot_clip_min (oracle)": _prevalence(rel_min, ENTER, EXIT),
                    "shared_clip_min": _prevalence(rel_shared, ENTER, EXIT),
                    "lower_tail_plateau": _prevalence(np.maximum(rel_plateau, 0.0), ENTER, EXIT),
                    "percentile_2": _prevalence(np.maximum(rel_p2, 0.0), ENTER, EXIT),
                },
                "prevalence_by_threshold_per_foot_min": {
                    f"enter{e}_exit{x}": _prevalence(rel_min, e, x) for e, x in SWEEP
                },
                "human_toe": {
                    "body": LAFAN1_CONTACT_BODIES[f],
                    "z_min_m": float(hz[:, f].min()),
                    "z_p50_m": float(np.percentile(hz[:, f], 50)),
                    "source_contact_prevalence": float(source_mask[:, f].mean()),
                    "height_only_hysteresis_prevalence": float(human_height_only[:, f].mean()),
                },
            }
        )

    both = oracle[:, 0] & oracle[:, 1]
    either = oracle[:, 0] | oracle[:, 1]
    return {
        "clip": pair_path.stem,
        "input": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
        "frames": T,
        "fps": fps,
        "duration_s": T / fps,
        "human_frames": int(hp.shape[0]),
        "frame_convention": {
            "robot_pelvis_z_mean_m": float(pelvis_z.mean()),
            "robot_pelvis_z_min_m": float(pelvis_z.min()),
            "human_hips_z_mean_m": float(hp[:, hips_idx, 2].mean()),
            "human_toe_z_mean_m": float(hz.mean()),
            "robot_foot_origin_z_mean_m": float(fz.mean()),
            "up_axis": "z (positive up) assumed by detect_contact_height_hysteresis",
        },
        "sampled_windows": {"starts": [int(s) for s in starts], "window": window, "coverage_fraction": float(sampled.mean())},
        "oracle_clip_summary": {
            "prevalence_any_foot": float(either.mean()),
            "prevalence_both_feet": float(both.mean()),
            "prevalence_left": float(oracle[:, 0].mean()),
            "prevalence_right": float(oracle[:, 1].mean()),
            "double_support_share_of_stance": float(both.sum() / max(either.sum(), 1)),
        },
        "per_foot": per_foot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "runs/gate1b/oracle_label_audit_2026-09-10.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {outp}")

    ctx = RobotContext(args.robot, "cpu")
    pairs_root = data_root() / "pairs" / args.robot
    clips = [audit_clip(ctx, pairs_root / f"{c}.npz", args.window, args.windows_per_clip) for c in args.clips]
    out = {
        "created_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "git": git_state(ROOT),
        "runtime": runtime_state("cpu"),
        "auditor": {"path": __file__, "sha256": sha256_file(__file__)},
        "oracle_definition": {
            "signal": "ankle body origin height from differentiable FK of the GMR teacher qpos",
            "ground": "per-foot clip-wide minimum of that height",
            "enter_m": ENTER,
            "exit_m": EXIT,
            "status": "proxy for stance; not force-verified",
        },
        "clips": clips,
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    for c in clips:
        print(f"== {c['clip']}  T={c['frames']} fps={c['fps']:.0f}  oracle any-foot prevalence {c['oracle_clip_summary']['prevalence_any_foot']:.3f}")
        for f in c["per_foot"]:
            g = f["prevalence_by_ground"]
            print(
                f"   {f['body']:<22} zmin {f['z_min_m']:+.3f} @{f['z_min_frame']:>5} (sampled={f['z_min_frame_in_sampled_windows']})"
                f" plateau-min {f['plateau_minus_min_m']:.3f} m | tilt@min {f['tilt_deg_at_min_frame']:.1f} deg (median {f['tilt_deg_median']:.1f})"
                f" | prev oracle {g['per_foot_clip_min (oracle)']:.3f} plateau {g['lower_tail_plateau']:.3f} p2 {g['percentile_2']:.3f}"
                f" | runs {f['oracle_runs']} medlen {f['oracle_run_len_frames']['median']}"
                f" | human src {f['human_toe']['source_contact_prevalence']:.3f} hgt {f['human_toe']['height_only_hysteresis_prevalence']:.3f}"
            )
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
