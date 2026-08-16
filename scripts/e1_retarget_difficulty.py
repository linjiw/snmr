#!/usr/bin/env python
"""E1 — do retarget-derived features predict where a tracker fails? (Track B pilot)

Program plan: docs/LATENT_BENEFIT_PROGRAM_2026-08-15.md §E1.

Inputs
------
* WBT motion NPZs (the frozen E70 motion directory by default): ``joint_pos`` (7 root +
  29 joints), ``joint_vel``, ``body_pos_w``, ``body_lin_vel_w``, ``latent_z``, ``fps``.
* Evaluation reports written by the E52-family trainers (general start grid): per-rollout
  ``start_steps`` (global frame), ``motion_ids``, ``completed``, ``survival_s``.

Labels (per 1-s bin of each clip, exactly what HoloSoma's AdaptiveTimestepsSampler
estimates online)
------
* ``hazard``      failures whose (start + survival) lands in the bin, divided by the number
                  of active-rollout ticks that pass through the bin (exposure).
* ``start_fail``  1 - completion of rollouts that *start* in the bin.

Features (per bin)
------
* ``kin``    kinematics anyone has from the explicit reference: root speed/height, joint
             velocity/acceleration RMS, max joint speed, angular momentum proxy.
* ``ret``    retargeting byproducts the reference format does not carry: min joint-limit
             margin (MJCF limits), foot-skate magnitude, heuristic contact-switch density,
             foot clearance.
* ``z``      SNMR latent statistics: PCA(16) of the bin-mean latent (fit on training
             folds only), mean ||dz/dt||, distance to the clip-mean latent.

Model: ridge regression on standardised features, held out by leave-one-temporal-block-out
(blocks of ``--block-s`` seconds, both clips) and by leave-one-clip-out.  Reports R^2 for
each feature group and the incremental R^2 over ``kin`` alone — the E1 gate is
incremental R^2 >= +0.10 on held-out data (plan §E1).

This is a *pilot* on the two E70 walks (26 clips-seconds each, three explicit seeds pooled).
The confirmatory E1 needs the multi-clip pool; the code path is the same.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

DEFAULT_MOTIONS = "/data/robotixx/snmr-research/e70/motions"
DEFAULT_REPORTS = "/data/robotixx/snmr-research/e70/students/seed*_explicit/c_prior_explicit_eval.json"
DEFAULT_MJCF = "/data/robotixx/snmr-externals/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


# ----------------------------------------------------------------------------- features
def joint_limits(mjcf: str, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import mujoco
    except ImportError:  # pragma: no cover
        return None
    model = mujoco.MjModel.from_xml_path(mjcf)
    lo, hi = [], []
    for name in joint_names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if j < 0 or not model.jnt_limited[j]:
            lo.append(-np.pi); hi.append(np.pi)
        else:
            lo.append(model.jnt_range[j][0]); hi.append(model.jnt_range[j][1])
    return np.asarray(lo), np.asarray(hi)


def bin_features(npz: dict, *, bin_s: float, limits: tuple[np.ndarray, np.ndarray] | None) -> dict[str, np.ndarray]:
    fps = int(np.asarray(npz["fps"]).reshape(-1)[0])
    q = np.asarray(npz["joint_pos"], dtype=float)          # (T, 7+29)
    qd = np.asarray(npz["joint_vel"], dtype=float)         # (T, 6+29)
    body_pos = np.asarray(npz["body_pos_w"], dtype=float)  # (T, B, 3)
    body_vel = np.asarray(npz["body_lin_vel_w"], dtype=float)
    z = np.asarray(npz["latent_z"], dtype=float)           # (T, 128)
    names = [str(b) for b in npz["body_names"]]
    T = q.shape[0]
    L = int(round(bin_s * fps))
    n_bins = T // L
    dt = 1.0 / fps

    root_pos, root_vel = q[:, :3], qd[:, :3]
    root_ang = qd[:, 3:6]
    joints, joint_vel = q[:, 7:], qd[:, 6:]
    joint_acc = np.gradient(joint_vel, dt, axis=0)
    speed = np.linalg.norm(root_vel[:, :2], axis=1)
    feet = [names.index(b) for b in FOOT_BODIES]
    foot_h = body_pos[:, feet, 2]                                   # (T, 2)
    foot_hspeed = np.linalg.norm(body_vel[:, feet, :2], axis=2)     # (T, 2)
    ground = np.percentile(foot_h, 2)
    contact = (foot_h - ground < 0.03) & (foot_hspeed < 0.25)       # heuristic stance
    skate = np.where(foot_h - ground < 0.03, foot_hspeed, 0.0)      # horizontal slip near ground
    switches = np.abs(np.diff(contact.astype(int), axis=0)).sum(1)  # (T-1,)
    if limits is not None:
        lo, hi = limits
        margin = np.minimum(joints - lo, hi - joints).min(1)         # (T,)
    else:
        margin = np.full(T, np.nan)
    dz = np.linalg.norm(np.diff(z, axis=0), axis=1)
    z_clip_mean = z.mean(0)

    rows = {"kin": [], "ret": [], "z_raw": [], "z_scalar": []}
    for b in range(n_bins):
        s = slice(b * L, (b + 1) * L)
        rows["kin"].append([
            speed[s].mean(), speed[s].max(), np.abs(np.gradient(speed[s], dt)).mean(),
            np.linalg.norm(root_ang[s], axis=1).mean(),
            root_pos[s, 2].mean(), root_pos[s, 2].min(), root_pos[s, 2].std(),
            np.sqrt((joint_vel[s] ** 2).mean()), np.abs(joint_vel[s]).max(),
            np.sqrt((joint_acc[s] ** 2).mean()),
        ])
        rows["ret"].append([
            np.nanmin(margin[s]), np.nanmean(margin[s]),
            skate[s].mean(), skate[s].max(),
            switches[s.start:min(s.stop, T - 1)].sum() / bin_s,
            (foot_h[s] - ground).max(), contact[s].mean(),
        ])
        rows["z_raw"].append(z[s].mean(0))
        rows["z_scalar"].append([dz[s.start:min(s.stop, T - 1)].mean(),
                                 np.linalg.norm(z[s].mean(0) - z_clip_mean)])
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["n_bins"] = n_bins
    out["bin_len"] = L
    out["fps"] = fps
    out["T"] = T
    return out


# ------------------------------------------------------------------------------- labels
def bin_labels(reports: list[dict], clip_offsets: list[int], clip_frames: list[int],
               *, bin_len: int, fps: int) -> dict[str, np.ndarray]:
    """Per-clip, per-bin hazard and start-failure labels pooled across reports."""
    n_bins = [t // bin_len for t in clip_frames]
    failures = [np.zeros(n) for n in n_bins]
    exposure = [np.zeros(n) for n in n_bins]
    start_n = [np.zeros(n) for n in n_bins]
    start_fail = [np.zeros(n) for n in n_bins]
    for r in reports:
        starts = np.asarray(r["start_steps"]); mids = np.asarray(r["motion_ids"])
        done = np.asarray(r["completed"], dtype=bool); surv = np.asarray(r["survival_s"], dtype=float)
        for s, m, d, sv in zip(starts, mids, done, surv):
            local = int(s) - clip_offsets[m]
            b0 = local // bin_len
            if b0 < n_bins[m]:
                start_n[m][b0] += 1
                start_fail[m][b0] += (not d)
            ticks = int(round(sv * fps))
            end = local + ticks
            for b in range(b0, min(end // bin_len + 1, n_bins[m])):
                lo, hi = max(local, b * bin_len), min(end, (b + 1) * bin_len)
                if hi > lo:
                    exposure[m][b] += hi - lo
            if not d and end // bin_len < n_bins[m]:
                failures[m][end // bin_len] += 1
    return {
        "hazard": [np.divide(f, e, out=np.zeros_like(f), where=e > 0) for f, e in zip(failures, exposure)],
        "exposure": exposure,
        "start_fail": [np.divide(f, n, out=np.full_like(f, np.nan), where=n > 0) for f, n in zip(start_fail, start_n)],
        "start_n": start_n,
    }


# ------------------------------------------------------------------------------- model
def ridge_fit_predict(Xtr, ytr, Xte, alpha: float = 1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr1 = np.c_[Xtr, np.ones(len(Xtr))]; Xte1 = np.c_[Xte, np.ones(len(Xte))]
    reg = alpha * np.eye(Xtr1.shape[1]); reg[-1, -1] = 0.0
    w = np.linalg.solve(Xtr1.T @ Xtr1 + reg, Xtr1.T @ ytr)
    return Xte1 @ w


def pca_fit(X, k):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k].T


def r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def build_matrix(feats: dict, group: str, train_idx, pca_k: int = 16):
    """Assemble features for ``group`` in {kin, kin+ret, kin+z, all}; PCA fit on train rows."""
    parts = [feats["kin"]]
    if "ret" in group or group == "all":
        parts.append(feats["ret"])
    if "z" in group or group == "all":
        mu, P = pca_fit(feats["z_raw"][train_idx], pca_k)
        parts.append((feats["z_raw"] - mu) @ P)
        parts.append(feats["z_scalar"])
    return np.concatenate(parts, axis=1)


def cross_validate(feats, y, folds, groups=("kin", "kin+ret", "kin+z", "all"), alpha=1.0):
    results = {}
    for g in groups:
        preds = np.full_like(y, np.nan)
        for te in folds:
            tr = np.setdiff1d(np.arange(len(y)), te)
            X = build_matrix(feats, g, tr)
            preds[te] = ridge_fit_predict(X[tr], y[tr], X[te], alpha)
        results[g] = {"r2": r2(y, preds), "pred": preds}
    base = results["kin"]["r2"]
    for g in results:
        results[g]["incremental_r2_over_kin"] = results[g]["r2"] - base
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motions", default=DEFAULT_MOTIONS)
    ap.add_argument("--reports", default=DEFAULT_REPORTS, help="comma-separated globs of eval JSONs (pooled)")
    ap.add_argument("--mjcf", default=DEFAULT_MJCF)
    ap.add_argument("--bin-s", type=float, default=1.0)
    ap.add_argument("--block-s", type=float, default=20.0, help="temporal CV block length")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    files = sorted(glob.glob(str(pathlib.Path(args.motions) / "*.npz")))
    report_paths = sorted({p for pat in args.reports.split(",") for p in glob.glob(pat)})
    reports = [json.loads(pathlib.Path(p).read_text()) for p in report_paths]
    if not files or not reports:
        raise SystemExit("no motions or no reports found")
    npzs = [dict(np.load(f, allow_pickle=True)) for f in files]
    limits = joint_limits(args.mjcf, [str(j) for j in npzs[0]["joint_names"]])
    per_clip = [bin_features(n, bin_s=args.bin_s, limits=limits) for n in npzs]
    frames = [f["T"] for f in per_clip]
    offsets = np.concatenate([[0], np.cumsum(frames)[:-1]]).tolist()
    labels = bin_labels(reports, offsets, frames, bin_len=per_clip[0]["bin_len"], fps=per_clip[0]["fps"])

    # Stack clips into one design; keep clip id and bin index for folds.
    feats = {k: np.concatenate([f[k] for f in per_clip]) for k in ("kin", "ret", "z_raw", "z_scalar")}
    clip_id = np.concatenate([np.full(f["n_bins"], i) for i, f in enumerate(per_clip)])
    bin_id = np.concatenate([np.arange(f["n_bins"]) for f in per_clip])
    hazard = np.concatenate(labels["hazard"]) * per_clip[0]["fps"]  # failures per exposed second
    exposure = np.concatenate(labels["exposure"])
    start_fail = np.concatenate(labels["start_fail"])
    keep = exposure > 0
    log_hazard = np.log1p(hazard)

    block_len = int(round(args.block_s / args.bin_s))
    block = clip_id * 10_000 + bin_id // block_len
    temporal_folds = [np.where((block == b) & keep)[0] for b in np.unique(block)]
    temporal_folds = [f for f in temporal_folds if len(f)]
    clip_folds = [np.where((clip_id == c) & keep)[0] for c in np.unique(clip_id)]

    def run(y, mask, folds):
        idx = np.where(mask)[0]
        sub = {k: v[idx] for k, v in feats.items()}
        remap = {g: i for i, g in enumerate(idx)}
        f2 = [np.array([remap[i] for i in f if i in remap]) for f in folds]
        f2 = [f for f in f2 if len(f)]
        res = cross_validate(sub, y[idx], f2, alpha=args.alpha)
        return {g: {k: v for k, v in r.items() if k != "pred"} for g, r in res.items()}

    ok_sf = keep & np.isfinite(start_fail)
    summary = {
        "n_bins_total": int(len(hazard)), "n_bins_with_exposure": int(keep.sum()),
        "n_reports_pooled": len(reports), "clips": [pathlib.Path(f).stem for f in files],
        "joint_limits_source": args.mjcf if limits is not None else None,
        "label_summary": {
            "hazard_per_s_mean": float(hazard[keep].mean()), "hazard_per_s_max": float(hazard[keep].max()),
            "hazard_nonzero_bins": int((hazard[keep] > 0).sum()),
            "start_fail_mean": float(np.nanmean(start_fail)),
        },
        "temporal_block_cv": {
            "log_hazard": run(log_hazard, keep, temporal_folds),
            "start_fail": run(np.nan_to_num(start_fail), ok_sf, temporal_folds),
        },
        "leave_one_clip_out_cv": {
            "log_hazard": run(log_hazard, keep, clip_folds),
            "start_fail": run(np.nan_to_num(start_fail), ok_sf, clip_folds),
        },
        "gate": "incremental R^2 over kin >= +0.10 on held-out folds (plan §E1); pilot only",
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
