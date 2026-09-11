#!/usr/bin/env python
"""CPU half of the G1 locomotion pilot: a SMALL candidate bank, scored by the intent filter and
kinematic guards, written as a ledger with the closed-loop columns left explicitly unrun.

Per development clip (the oracle-valid walk clips of
``reproducibility/reports/locomotion_dev_set_2026-09-10.json``) and per 192-frame window on the
frozen 6-window grid, the bank is:

    noop            the GMR-G1 reference unchanged
    smooth_s1/s2    Gaussian low-pass of dof_pos (sigma 1 / 2 frames at 30 fps), root unchanged
    footlock_dls    snmr.footlock.foot_lock_masked with the deployable source_contact mask
    c6_projection   snmr.projection.windowed_contact_projection, frozen Gate-1b config
    spline_bounded  c6_projection's leg-joint dof delta least-squares projected onto the C2
                    clamped cubic B-spline family (8-frame knots) and box-clipped at 0.35 rad;
                    root kept at the proposal.  A bounded-correction variant, not a search.

Stage 0 scoring (all CPU, no simulator): joint limits, penetration vs the proposal's ground,
the revised intent filter (``snmr.repair_intent``, PROVISIONAL tolerances; events defined on
the proposal's anchors until the human-side adapter exists), and the Gate-1b kinematic guards
(stance foot speed under the source_contact and teacher-height masks, MPJPE-to-proposal, jerk
ratio).  Every (candidate, clip, window) is in the ledger with a status; nothing is dropped.

Stage 1 (closed-loop, GPU) is NOT run here.  The ledger carries ``closed_loop: null`` with the
command template from the pilot manifest.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.correction_basis import EndpointConstraint, clamped_cubic_basis  # noqa: E402
from snmr.experiment import git_state, runtime_state, sha256_file, utc_now  # noqa: E402
from snmr.footlock import _leg_dof_indices, foot_lock_masked  # noqa: E402
from snmr.g1_semantics import g1_semantic_manifest  # noqa: E402
from snmr.human import LAFAN1_CONTACT_BODIES, load_pair_npz  # noqa: E402
from snmr.metrics import FOOT_BODIES, detect_contact, detect_contact_height_hysteresis  # noqa: E402
from snmr.paths import data_root  # noqa: E402
from snmr.projection import WindowedProjectionConfig, windowed_contact_projection  # noqa: E402
from snmr.repair_intent import PROVISIONAL_TOLERANCES, evaluate_intent  # noqa: E402

from train_phase2 import RobotContext  # noqa: E402

KNOT = 8
JOINT_BOUND = 0.35
FPS = 30.0


def _anchor_bodies(ctx: RobotContext) -> tuple[list[str], list[int]]:
    manifest = g1_semantic_manifest()
    names = []
    for field in manifest.__dataclass_fields__:
        if field.endswith("_link"):
            value = getattr(manifest, field)
            if isinstance(value, str):
                names.append(value)
    return names, [ctx.kin.body_index(n) for n in names]


def _fk(ctx, root_pos, root_quat, dof):
    with torch.no_grad():
        body, _ = ctx.kin.forward_kinematics(root_pos, root_quat, dof)
    return body


def _stance_speed(feet_xyz: torch.Tensor, mask: torch.Tensor) -> float | None:
    """Mean xy speed (m/s) of feet over frames where ``mask`` is true (never inferred from speed)."""
    speed = torch.zeros(feet_xyz.shape[:2])
    speed[1:] = (feet_xyz[1:, :, :2] - feet_xyz[:-1, :, :2]).norm(dim=-1) * FPS
    speed[0] = speed[1]
    m = mask.bool()
    return float(speed[m].mean()) if m.any() else None


def _jerk(dof: torch.Tensor) -> float:
    d3 = dof[3:] - 3 * dof[2:-1] + 3 * dof[1:-2] - dof[:-3]
    return float((d3 ** 2).mean())


def _spline_project_delta(delta: np.ndarray, columns: list[int]) -> np.ndarray:
    """Least-squares projection of selected delta channels onto the C2 spline family, box-clipped."""
    L = delta.shape[0]
    basis = clamped_cubic_basis(L, KNOT)
    nullspace = EndpointConstraint(L, KNOT, 2).nullspace
    design = basis @ nullspace  # (L, F)
    out = np.zeros_like(delta)
    for c in columns:
        theta, *_ = np.linalg.lstsq(design, delta[:, c], rcond=None)
        coefficients = np.clip(nullspace @ theta, -JOINT_BOUND, JOINT_BOUND)
        out[:, c] = basis @ coefficients
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_set", default=str(ROOT / "reproducibility/reports/locomotion_dev_set_2026-09-10.json"))
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "reproducibility/reports/pilot_candidate_ledger_cpu_2026-09-10.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(outp)

    dev = json.load(open(args.dev_set))
    clips = args.clips or dev["valid_walk_clips"]
    ctx = RobotContext("unitree_g1", "cpu")
    feet = FOOT_BODIES["unitree_g1"]
    foot_idx = [ctx.kin.body_index(n) for n in feet]
    anchor_names, anchor_idx = _anchor_bodies(ctx)
    leg_cols = sorted({i for f in feet for i in _leg_dof_indices(ctx.kin, f)})
    lower, upper = ctx.kin.dof_limits()
    lower, upper = lower.cpu(), upper.cpu()
    pairs_root = data_root() / "pairs" / "unitree_g1"
    tol = PROVISIONAL_TOLERANCES

    ledger = []
    t_start = time.time()
    for clip in clips:
        pair_path = pairs_root / f"{clip}.npz"
        pair = load_pair_npz(str(pair_path))
        q_all, hp_all = pair["qpos"], pair["human_pos"]
        T = q_all.shape[0]
        root_pos_all, root_quat_all, dof_all = q_all[:, 0:3], q_all[:, 3:7], q_all[:, 7:]
        body_all = _fk(ctx, root_pos_all, root_quat_all, dof_all)
        feet_all = body_all[:, foot_idx, :]
        ground = feet_all[..., 2].min(dim=0).values  # the frozen oracle's per-foot clip minimum
        oracle_all = detect_contact_height_hysteresis(feet_all, 0.03, 0.05)
        source_idx = [pair["human_names"].index(n) for n in LAFAN1_CONTACT_BODIES]
        source_all = detect_contact(hp_all[:, source_idx, :], FPS, height_threshold=0.08, speed_threshold=0.24)
        starts = np.linspace(0, max(T - args.window, 0), num=min(args.windows_per_clip, max(T // args.window, 1)), dtype=int)

        for start in starts:
            s, e = int(start), int(start) + args.window
            rp, rq, dof = root_pos_all[s:e].clone(), root_quat_all[s:e].clone(), dof_all[s:e].clone()
            src_mask, orc_mask = source_all[s:e], oracle_all[s:e]
            timestamps = np.arange(e - s) / FPS
            prop_body = body_all[s:e]
            prop_anchors = prop_body[:, anchor_idx, :].numpy()
            prop_feet = prop_body[:, foot_idx, :]
            prop_stance_src = _stance_speed(prop_feet, src_mask)
            prop_stance_orc = _stance_speed(prop_feet, orc_mask)
            prop_jerk = _jerk(dof)
            prop_pen = float(((prop_feet[..., 2] - ground[None, :]) < -0.005).float().mean())

            candidates: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]] = {}
            candidates["noop"] = (rp, rq, dof, {})
            for sigma in (1.0, 2.0):
                sm = torch.tensor(gaussian_filter1d(dof.numpy(), sigma=sigma, axis=0, mode="nearest"), dtype=dof.dtype)
                candidates[f"smooth_s{int(sigma)}"] = (rp, rq, sm, {"sigma_frames": sigma})
            t0 = time.time()
            locked = foot_lock_masked(ctx.kin, rp, rq, dof, feet, src_mask, iters=12, lr=0.5, damping=1e-2, blend=2, merge_gap=6, extend=2)
            candidates["footlock_dls"] = (rp, rq, locked, {"seconds": time.time() - t0})
            t0 = time.time()
            proj = windowed_contact_projection(ctx.kin, rp, rq, dof, feet, src_mask, config=WindowedProjectionConfig())
            candidates["c6_projection"] = (proj.root_pos, proj.root_quat, proj.dof_pos, {"seconds": time.time() - t0})
            delta = (proj.dof_pos - dof).numpy()
            spl = _spline_project_delta(delta, leg_cols)
            candidates["spline_bounded"] = (rp, rq, dof + torch.tensor(spl, dtype=dof.dtype), {"leg_dof_columns": leg_cols, "max_abs_delta_rad": float(np.abs(spl).max())})

            for cid, (crp, crq, cdof, meta) in candidates.items():
                body = _fk(ctx, crp, crq, cdof)
                anchors = body[:, anchor_idx, :].numpy()
                cfeet = body[:, foot_idx, :]
                limit_viol = int(((cdof < lower[None, :] - 1e-6) | (cdof > upper[None, :] + 1e-6)).sum())
                pen = float(((cfeet[..., 2] - ground[None, :]) < -0.005).float().mean())
                # Candidate contact for the sequence check: the oracle definition with the PROPOSAL's
                # per-foot ground, applied identically to proposal and candidate feet.
                ground_offset = torch.cat([torch.zeros(len(feet), 2), ground[:, None]], dim=1)[None, :, :]
                cand_contact = detect_contact_height_hysteresis(cfeet - ground_offset, 0.03, 0.05, ground_z=0.0)
                ref_contact = detect_contact_height_hysteresis(prop_feet - ground_offset, 0.03, 0.05, ground_z=0.0)
                report = evaluate_intent(
                    prop_anchors, anchors, timestamps, tolerances=tol, anchor_labels=anchor_names,
                    reference_contact=ref_contact.numpy(), candidate_contact=cand_contact.numpy(),
                )
                mpjpe_cm = float((body - prop_body).norm(dim=-1).mean()) * 100.0
                jerk_ratio = _jerk(cdof) / max(prop_jerk, 1e-12)
                stance_src = _stance_speed(cfeet, src_mask)
                stance_orc = _stance_speed(cfeet, orc_mask)
                guards = {
                    "joint_limit_violations": limit_viol,
                    "penetration_fraction": pen,
                    "penetration_guard_pass": pen <= prop_pen + 0.02,
                    "mpjpe_to_proposal_cm": mpjpe_cm,
                    "mpjpe_guard_pass": mpjpe_cm <= 0.5,
                    "jerk_ratio": jerk_ratio,
                    "jerk_guard_pass": jerk_ratio <= 1.2,
                    "stance_speed_source_mask_m_s": stance_src,
                    "stance_speed_teacher_mask_m_s": stance_orc,
                    "stance_speed_source_endpoint_pass": (stance_src is not None and stance_src <= 0.10),
                    "stance_speed_teacher_endpoint_pass": (stance_orc is not None and stance_orc <= 0.08),
                }
                guard_pass = limit_viol == 0 and guards["penetration_guard_pass"] and guards["mpjpe_guard_pass"] and guards["jerk_guard_pass"]
                if cid == "noop":
                    status = "baseline"
                elif not report.accepted:
                    status = "rejected_intent"
                elif not guard_pass:
                    status = "rejected_guard"
                else:
                    status = "stage0_pass"
                ledger.append({
                    "clip": clip, "window_start": s, "window_end": e, "candidate": cid, "meta": meta,
                    "status": status,
                    "intent": report.to_dict(),
                    "guards": guards,
                    "proposal": {"stance_speed_source_mask_m_s": prop_stance_src, "stance_speed_teacher_mask_m_s": prop_stance_orc, "penetration_fraction": prop_pen},
                    "closed_loop": None,
                    "closed_loop_note": "NOT RUN: requires GPU approval; see pilot manifest closed_loop_execution.exact_command_template",
                })
            print(f"{clip} window {s:>5}: " + ", ".join(f"{r['candidate']}={r['status']}" for r in ledger if r['clip'] == clip and r['window_start'] == s) + f"  [{time.time() - t_start:.0f}s]")

    # Aggregate per candidate.
    summary = {}
    for cid in ("noop", "smooth_s1", "smooth_s2", "footlock_dls", "c6_projection", "spline_bounded"):
        rows = [r for r in ledger if r["candidate"] == cid]
        reasons: dict[str, int] = {}
        for r in rows:
            for reason in r["intent"]["rejection_reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        def _mean(key, sub):
            vals = [r[sub][key] for r in rows if r[sub][key] is not None]
            return float(np.mean(vals)) if vals else None
        summary[cid] = {
            "windows": len(rows),
            "status_counts": {st: sum(r["status"] == st for r in rows) for st in ("baseline", "stage0_pass", "rejected_intent", "rejected_guard")},
            "intent_rejection_reason_counts": reasons,
            "mean_stance_speed_source_mask_m_s": _mean("stance_speed_source_mask_m_s", "guards"),
            "mean_stance_speed_teacher_mask_m_s": _mean("stance_speed_teacher_mask_m_s", "guards"),
            "mean_mpjpe_to_proposal_cm": _mean("mpjpe_to_proposal_cm", "guards"),
            "mean_jerk_ratio": _mean("jerk_ratio", "guards"),
            "windows_with_limit_violations": sum(r["guards"]["joint_limit_violations"] > 0 for r in rows),
        }
    proposal_means = {
        "stance_speed_source_mask_m_s": float(np.mean([r["proposal"]["stance_speed_source_mask_m_s"] for r in ledger if r["candidate"] == "noop" and r["proposal"]["stance_speed_source_mask_m_s"] is not None])),
        "stance_speed_teacher_mask_m_s": float(np.mean([r["proposal"]["stance_speed_teacher_mask_m_s"] for r in ledger if r["candidate"] == "noop" and r["proposal"]["stance_speed_teacher_mask_m_s"] is not None])),
    }
    out = {
        "created_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "git": git_state(ROOT),
        "runtime": runtime_state("cpu"),
        "generator": {"path": __file__, "sha256": sha256_file(__file__)},
        "inputs": {"dev_set": {"path": args.dev_set, "sha256": sha256_file(args.dev_set)}, "pairs": {c: sha256_file(pairs_root / f"{c}.npz") for c in clips}},
        "protocol": {
            "clips": clips, "window": args.window, "windows_per_clip": args.windows_per_clip, "fps": FPS,
            "anchors": anchor_names,
            "intent_tolerances_sha256": tol.sha256, "intent_tolerances_status": tol.calibration_status,
            "event_source": "proposal (teacher FK anchors); the human-side HumanMotionSpec adapter is pending, so this is not yet the spec's source-defined event set",
            "contact_for_sequence_check": "teacher-height hysteresis (enter 0.03 / exit 0.05) on proposal and candidate feet with the PROPOSAL's per-foot clip-minimum ground",
            "guards": "Gate-1b: joint limits 0; penetration fraction <= proposal + 0.02; MPJPE-to-proposal <= 0.5 cm (all bodies, world frame); jerk ratio <= 1.2; stance-speed endpoints reported (<= 0.10 source mask, <= 0.08 teacher mask), not applied as guards",
            "closed_loop": "NOT RUN",
        },
        "summary": summary,
        "proposal_means": proposal_means,
        "ledger": ledger,
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("\nSUMMARY (CPU stage 0 only; closed loop not run)")
    print(f"proposal stance speed: source-mask {proposal_means['stance_speed_source_mask_m_s']:.3f} m/s, teacher-mask {proposal_means['stance_speed_teacher_mask_m_s']:.3f} m/s")
    for cid, sm in summary.items():
        print(f"  {cid:<15} {sm['status_counts']}  stance src {sm['mean_stance_speed_source_mask_m_s']:.3f} tea {sm['mean_stance_speed_teacher_mask_m_s']:.3f} | mpjpe {sm['mean_mpjpe_to_proposal_cm']:.2f} cm | jerk x{sm['mean_jerk_ratio']:.2f} | limits {sm['windows_with_limit_violations']} | intent rejections {sm['intent_rejection_reason_counts']}")
    print("wrote", outp)


if __name__ == "__main__":
    main()
