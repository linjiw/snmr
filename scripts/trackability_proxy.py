#!/usr/bin/env python
"""Trackability proxy: open-loop PD-replay stability of WBT motion NPZs (RL-free, local).

WHY: the decisive SNMR-vs-GMR question is "does the retargeted data train an equally good
tracking policy?" — but WBT RL needs IsaacSim + GPU-days we don't have locally. This script
measures a cheap, deterministic PROXY: initialize the G1 at a clip's first state in MuJoCo,
then run physics forward with ONLY joint-space PD torques tracking the reference dof
trajectory (no root actuation, gravity on) and measure how long the robot survives before
toppling, plus tracking error while alive. Open-loop replay eventually falls for ANY clip;
the RELATIVE comparison on the same clip isolates data quality — jitter, contact
inconsistency, and dynamically infeasible poses topple PD replay sooner. This is the standard
"sim replay sanity check" used to vet retargeted datasets before RL.

Input NPZs are holosoma WBT format (see export_wbt_npz.py): 50 fps,
joint_pos = [root_pos(3), root_quat wxyz(4), dof(29)], joint_vel = [root lin(3) world,
root ang(3) WORLD (converted to body-local for MuJoCo's free joint), dof(29)].

    python scripts/trackability_proxy.py \
        --dir_a runs/wbt_validation/gmr --dir_b runs/wbt_validation/snmr \
        --out runs/trackability
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.paths import g1_mjcf  # noqa: E402

# ---------------------------------------------------------------------------------------
# holosoma g1_29dof deploy-matched PD gains + effort limits, keyed by joint-name substring.
# Stiffness/damping copied verbatim from
#   holosoma/src/holosoma/holosoma/config_values/robot.py:488-505 (stiffness) and
#   robot.py:506-523 (damping); per-joint effort limits from robot.py:212-242
#   (dof_effort_limit_list, in dof_names order robot.py:24-54 — left/right symmetric, so a
#   substring map is lossless). Using the deploy gains makes the proxy measure what the
#   downstream WBT stack would actually experience.
# ---------------------------------------------------------------------------------------
HOLOSOMA_G1_STIFFNESS = {
    "hip_yaw": 40.179238471,
    "hip_roll": 99.098427777,
    "hip_pitch": 40.179238471,
    "knee": 99.098427777,
    "ankle_pitch": 28.501246196,
    "ankle_roll": 28.501246196,
    "waist_yaw": 40.179238471,
    "waist_roll": 28.501246196,
    "waist_pitch": 28.501246196,
    "shoulder_pitch": 14.250623098,
    "shoulder_roll": 14.250623098,
    "shoulder_yaw": 14.250623098,
    "elbow": 14.250623098,
    "wrist_roll": 14.250623098,
    "wrist_pitch": 16.778327481,
    "wrist_yaw": 16.778327481,
}
HOLOSOMA_G1_DAMPING = {
    "hip_yaw": 2.557889765,
    "hip_roll": 6.308801854,
    "hip_pitch": 2.557889765,
    "knee": 6.308801854,
    "ankle_pitch": 1.814445687,
    "ankle_roll": 1.814445687,
    "waist_yaw": 2.557889765,
    "waist_roll": 1.814445687,
    "waist_pitch": 1.814445687,
    "shoulder_pitch": 0.907222843,
    "shoulder_roll": 0.907222843,
    "shoulder_yaw": 0.907222843,
    "elbow": 0.907222843,
    "wrist_roll": 0.907222843,
    "wrist_pitch": 1.068141502,
    "wrist_yaw": 1.068141502,
}
HOLOSOMA_G1_EFFORT = {  # Nm, robot.py:212-242
    "hip_pitch": 88.0,
    "hip_roll": 139.0,
    "hip_yaw": 88.0,
    "knee": 139.0,
    "ankle_pitch": 50.0,
    "ankle_roll": 50.0,
    "waist_yaw": 88.0,
    "waist_roll": 50.0,
    "waist_pitch": 50.0,
    "shoulder_pitch": 25.0,
    "shoulder_roll": 25.0,
    "shoulder_yaw": 25.0,
    "elbow": 25.0,
    "wrist_roll": 25.0,
    "wrist_pitch": 5.0,
    "wrist_yaw": 5.0,
}

# Divergence thresholds (a fallen/escaped robot, not a momentary wobble).
FALL_ROOT_Z_M = 0.35
ROOT_XY_DEV_M = 0.5
TILT_LIMIT_RAD = math.radians(60.0)


def _match_gain(joint_name: str, table: dict[str, float]) -> float:
    hits = [k for k in table if k in joint_name]
    if len(hits) != 1:
        raise ValueError(f"gain lookup for {joint_name!r} matched {hits} — table ambiguous")
    return table[hits[0]]


def _hinge_maps(m: mujoco.MjModel) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Names + qpos/dof addresses of the hinge joints, in model order (== NPZ dof order)."""
    names, qadr, vadr = [], [], []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
            qadr.append(m.jnt_qposadr[j])
            vadr.append(m.jnt_dofadr[j])
    return names, np.asarray(qadr), np.asarray(vadr)


def _tilt_rad(root_quat_wxyz: np.ndarray) -> float:
    """Angle between the body z-axis and world up = combined |pitch/roll| gravity tilt."""
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, root_quat_wxyz)
    return math.acos(min(max(R[8], -1.0), 1.0))


def replay(npz_path: str, mjcf: str, seconds_max: float = 10.0, control_hz: float = 50.0,
           start_frame: int = 0) -> dict:
    """Open-loop PD replay of one WBT NPZ; returns survival + while-alive tracking metrics.

    Torque tau = kp*(q_ref - q) - kd*qd on the 29 hinge dofs only (root free joint gets zero
    applied force), recomputed at every physics substep with the 50 Hz reference target held
    over the control tick (zero-order hold — what a deployed PD loop does under decimation),
    clamped to holosoma per-joint effort limits. Fully deterministic.
    """
    data = np.load(npz_path, allow_pickle=True)
    jp = np.asarray(data["joint_pos"], dtype=np.float64)   # (T, 7+D)
    jv = np.asarray(data["joint_vel"], dtype=np.float64)   # (T, 6+D)
    fps = float(np.asarray(data["fps"]).ravel()[0])
    if abs(fps - control_hz) > 1e-6:
        raise ValueError(f"{npz_path}: NPZ fps {fps} != control_hz {control_hz}")

    m = mujoco.MjModel.from_xml_path(mjcf)
    d = mujoco.MjData(m)
    names, qadr, vadr = _hinge_maps(m)
    if "joint_names" in data and list(data["joint_names"]) != names:
        raise ValueError(f"{npz_path}: joint_names do not match MJCF hinge order")
    if jp.shape[1] != m.nq or jv.shape[1] != m.nv:
        raise ValueError(f"{npz_path}: joint_pos/vel widths {jp.shape[1]}/{jv.shape[1]} "
                         f"!= model nq/nv {m.nq}/{m.nv}")

    kp = np.array([_match_gain(n, HOLOSOMA_G1_STIFFNESS) for n in names])
    kd = np.array([_match_gain(n, HOLOSOMA_G1_DAMPING) for n in names])
    tau_max = np.array([_match_gain(n, HOLOSOMA_G1_EFFORT) for n in names])

    n_sub = round(1.0 / (control_hz * m.opt.timestep))
    if n_sub < 1 or abs(n_sub * m.opt.timestep * control_hz - 1.0) > 1e-6:
        raise ValueError(f"model timestep {m.opt.timestep} does not divide 1/{control_hz}s")

    # Initial state straight from the NPZ. MuJoCo qpos layout matches joint_pos directly;
    # qvel root linear part is world-frame like the NPZ, but the free-joint ANGULAR part is
    # body-local while the NPZ stores world-frame — rotate it into the pelvis frame.
    T = jp.shape[0]
    start_frame = int(start_frame)
    if not 0 <= start_frame < T - 1:
        raise ValueError(f"start_frame {start_frame} out of range for {T} frames")
    d.qpos[:] = jp[start_frame]
    R0 = np.empty(9)
    mujoco.mju_quat2Mat(R0, jp[start_frame, 3:7])
    d.qvel[0:3] = jv[start_frame, 0:3]
    d.qvel[3:6] = R0.reshape(3, 3).T @ jv[start_frame, 3:6]
    d.qvel[6:] = 0.0
    d.qvel[vadr] = jv[start_frame, 6:]
    mujoco.mj_forward(m, d)

    ticks_max = min(int(round(seconds_max * control_hz)), T - 1 - start_frame)
    ticks_alive = 0
    diverge_reason = None
    dof_errs, z_errs = [], []
    for k in range(ticks_max):
        ref = jp[start_frame + k + 1]        # target = reference at the END of this tick
        q_ref = ref[7:]
        for _ in range(n_sub):
            tau = kp * (q_ref - d.qpos[qadr]) - kd * d.qvel[vadr]
            d.qfrc_applied[:] = 0.0          # root dofs stay unactuated
            d.qfrc_applied[vadr] = np.clip(tau, -tau_max, tau_max)
            mujoco.mj_step(m, d)

        z = float(d.qpos[2])
        xy_dev = float(np.linalg.norm(d.qpos[0:2] - ref[0:2]))
        tilt = _tilt_rad(d.qpos[3:7])
        if z < FALL_ROOT_Z_M:
            diverge_reason = f"root z {z:.2f} < {FALL_ROOT_Z_M}"
        elif xy_dev > ROOT_XY_DEV_M:
            diverge_reason = f"root xy deviation {xy_dev:.2f} > {ROOT_XY_DEV_M}"
        elif tilt > TILT_LIMIT_RAD:
            diverge_reason = f"tilt {math.degrees(tilt):.0f} deg > 60"
        if diverge_reason is not None:
            break
        ticks_alive += 1
        dof_errs.append(float(np.mean(np.abs(d.qpos[qadr] - q_ref))))
        z_errs.append(abs(z - float(ref[2])))

    seconds_eval = ticks_max / control_hz
    return {
        "npz": str(npz_path),
        "start_frame": start_frame,
        "seconds_evaluated": seconds_eval,
        "survival_time_s": ticks_alive / control_hz,
        "survived_fraction": (ticks_alive / control_hz) / seconds_eval if seconds_eval else 0.0,
        "diverged": diverge_reason is not None,
        "diverge_reason": diverge_reason,
        "mean_dof_err_rad": float(np.mean(dof_errs)) if dof_errs else float("nan"),
        "mean_root_height_err_m": float(np.mean(z_errs)) if z_errs else float("nan"),
    }


def replay_clip(npz_path: str, mjcf: str, seconds_max: float, control_hz: float,
                num_starts: int) -> dict:
    """Aggregate replay over evenly spaced deterministic start frames.

    A single 10 s window from t=0 samples almost none of a multi-minute clip; averaging a few
    evenly spaced windows makes the per-clip statistic representative while staying
    deterministic (no randomness anywhere).
    """
    T = int(np.load(npz_path, allow_pickle=True)["joint_pos"].shape[0])
    need = int(round(seconds_max * control_hz))
    last = max(T - 1 - need, 0)
    starts = sorted({int(s) for s in np.linspace(0, last, max(1, num_starts))})
    runs = [replay(npz_path, mjcf, seconds_max, control_hz, start_frame=s) for s in starts]
    return {
        "starts": runs,
        "survival_time_s": float(np.mean([r["survival_time_s"] for r in runs])),
        "survived_fraction": float(np.mean([r["survived_fraction"] for r in runs])),
        "mean_dof_err_rad": float(np.nanmean([r["mean_dof_err_rad"] for r in runs])),
        "mean_root_height_err_m": float(np.nanmean([r["mean_root_height_err_m"] for r in runs])),
    }


def _fmt(v: float, nd: int = 2) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"


def to_markdown(results: dict, label_a: str, label_b: str, seconds_max: float,
                num_starts: int) -> str:
    lines = [
        "# Trackability proxy — open-loop PD replay stability",
        "",
        f"Per clip: mean over {num_starts} evenly spaced start windows of {seconds_max:.0f} s "
        f"each. Survival = time until root z<{FALL_ROOT_Z_M} m, root xy off-reference by "
        f">{ROOT_XY_DEV_M} m, or tilt >60 deg. Errors are means while alive. "
        f"delta = {label_b} - {label_a} (survival: positive favors {label_b}; "
        "errors: negative favors it).",
        "",
        f"| clip | surv {label_a} (s) | surv {label_b} (s) | dsurv | "
        f"dof err {label_a} (rad) | dof err {label_b} (rad) | ddof | "
        f"z err {label_a} (m) | z err {label_b} (m) | dz |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for clip, row in results.items():
        a, b = row[label_a], row[label_b]
        lines.append(
            f"| {clip} | {_fmt(a['survival_time_s'])} | {_fmt(b['survival_time_s'])} "
            f"| {_fmt(b['survival_time_s'] - a['survival_time_s'], 2)} "
            f"| {_fmt(a['mean_dof_err_rad'], 3)} | {_fmt(b['mean_dof_err_rad'], 3)} "
            f"| {_fmt(b['mean_dof_err_rad'] - a['mean_dof_err_rad'], 3)} "
            f"| {_fmt(a['mean_root_height_err_m'], 3)} | {_fmt(b['mean_root_height_err_m'], 3)} "
            f"| {_fmt(b['mean_root_height_err_m'] - a['mean_root_height_err_m'], 3)} |"
        )
    ms = {lab: {k: float(np.mean([row[lab][k] for row in results.values()]))
                for k in ("survival_time_s", "mean_dof_err_rad", "mean_root_height_err_m")}
          for lab in (label_a, label_b)}
    a, b = ms[label_a], ms[label_b]
    lines.append(
        f"| **mean** | {_fmt(a['survival_time_s'])} | {_fmt(b['survival_time_s'])} "
        f"| {_fmt(b['survival_time_s'] - a['survival_time_s'], 2)} "
        f"| {_fmt(a['mean_dof_err_rad'], 3)} | {_fmt(b['mean_dof_err_rad'], 3)} "
        f"| {_fmt(b['mean_dof_err_rad'] - a['mean_dof_err_rad'], 3)} "
        f"| {_fmt(a['mean_root_height_err_m'], 3)} | {_fmt(b['mean_root_height_err_m'], 3)} "
        f"| {_fmt(b['mean_root_height_err_m'] - a['mean_root_height_err_m'], 3)} |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir_a", default=str(ROOT / "runs" / "wbt_validation" / "gmr"),
                    help="baseline NPZ dir (e.g. GMR teacher)")
    ap.add_argument("--dir_b", default=str(ROOT / "runs" / "wbt_validation" / "snmr"),
                    help="comparison NPZ dir (e.g. SNMR)")
    ap.add_argument("--mjcf", default=str(g1_mjcf()))
    ap.add_argument("--seconds_max", type=float, default=10.0)
    ap.add_argument("--control_hz", type=float, default=50.0)
    ap.add_argument("--num_starts", type=int, default=3,
                    help="evenly spaced start windows per clip (deterministic)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "trackability"))
    args = ap.parse_args()

    dir_a, dir_b = pathlib.Path(args.dir_a), pathlib.Path(args.dir_b)
    label_a, label_b = dir_a.name, dir_b.name
    clips = sorted({p.name for p in dir_a.glob("*.npz")} & {p.name for p in dir_b.glob("*.npz")})
    if not clips:
        sys.exit(f"no matching NPZs between {dir_a} and {dir_b}")

    results = {}
    for name in clips:
        clip = name.removesuffix(".npz").removesuffix("_mj")
        results[clip] = {}
        for label, path in ((label_a, dir_a / name), (label_b, dir_b / name)):
            r = replay_clip(str(path), args.mjcf, args.seconds_max, args.control_hz,
                            args.num_starts)
            results[clip][label] = r
            print(f"{clip:24s} {label:6s} survival {r['survival_time_s']:6.2f}s  "
                  f"dof err {r['mean_dof_err_rad']:.3f} rad  "
                  f"z err {r['mean_root_height_err_m']:.3f} m")

    md = to_markdown(results, label_a, label_b, args.seconds_max, args.num_starts)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text(md)
    with open(out / "comparison.json", "w") as fh:
        json.dump({"config": vars(args), "labels": [label_a, label_b], "results": results},
                  fh, indent=2)
    print("\n" + md)
    print(f"wrote {out}/comparison.md and comparison.json")


if __name__ == "__main__":
    main()
