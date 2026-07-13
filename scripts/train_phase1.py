#!/usr/bin/env python
"""Phase-1 training: LAFAN1 human motion → shared latent → single robot (default G1).

Design-doc step N3. Trains `SNMR.retarget_human_to_robot` on the paired dataset produced by
`make_pairs_lafan1.py`, with a clip-level train/val split, window sampling, and periodic held-out
evaluation of the Gate-G1 metric: whole-body MPJPE between the prediction's FK and the GMR teacher's
FK on unseen clips.

    python scripts/train_phase1.py --robot unitree_g1 --steps 60000 --window 64 \
        --out runs/phase1_g1

Checkpoints (model + optimizer + step) are written every --ckpt_every steps and at the end;
`--resume` continues from the latest checkpoint in --out.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.data import local_root_to_world, world_root_to_local  # noqa: E402
from snmr.diagnostics import (  # noqa: E402
    binary_contact_metrics,
    loss_gradient_diagnostics,
    model_parameter_groups,
)
from snmr.experiment import (  # noqa: E402
    RunManifest,
    capture_rng_state,
    dataset_fingerprint,
    restore_rng_state,
)
from snmr.human import (  # noqa: E402
    LAFAN1_BODY_NAMES,
    LAFAN1_CONTACT_BODIES,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.losses import (  # noqa: E402
    DEFAULT_LOSS_WEIGHTS,
    collect_loss_terms,
    contact_prediction_loss,
    contact_self_consistency_loss,
    contact_self_consistency_velocity_loss,
    foot_contact_loss,
    foot_penetration_loss,
    foot_velocity_distill_loss,
    teacher_foot_velocity_loss,
    teacher_stance_velocity_loss,
    weighted_loss,
)
from snmr.metrics import (  # noqa: E402
    FOOT_BODIES,
    detect_contact,
    detect_contact_height_hysteresis,
)
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

# The decoder's root head predicts the robot root **in the scaled-human-root heading frame**:
#  (1) world-frame root targets are unlearnable — encoder features are heading/translation invariant
#      (first smoke run: ~3.3 m val MPJPE of pure root drift);
#  (2) anchoring at the *raw* human root still leaves an unlearnable residual, because GMR scales the
#      human trajectory before IK: robot_xy ≈ s·human_xy with s≈0.875 for LAFAN1→G1 (measured residual
#      ≤1.5 cm), so (robot − human) grows with distance from the world origin (measured: 59 cm root
#      error dominating a 61 cm val MPJPE while dof-only error was 12 cm);
#  (3) therefore the anchor is s·human_xy with per-robot constant s fitted on the training set by
#      least squares (stored in the checkpoint; a known retargeting constant at inference, no leakage).
# World pose is recovered at eval/inference by composing with the known (scaled) human trajectory.

from snmr.paths import data_root, robot_mjcf  # noqa: E402

# data-consistent models resolved centrally (snmr/paths.py); see THIRD_PARTY.md
ROBOT_MJCF_KEYS = ["unitree_g1", "booster_t1_29dof", "fourier_n1", "engineai_pm01", "stanford_toddy"]

# LAFAN1 clip prefixes; we hold out whole clips (all subjects of a held-out sequence stay together
# would be even stricter, but per-clip split matches the design doc and keeps val diverse).
VAL_CLIPS = [
    "walk1_subject5", "dance2_subject4", "fight1_subject3",
    "run2_subject1", "jumps1_subject2", "sprint1_subject4", "aiming2_subject3",
]


def load_dataset(pairs_dir: pathlib.Path, device: str):
    files = sorted(pairs_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no pair NPZs in {pairs_dir}")
    train, val = [], []
    for f in files:
        pair = load_pair_npz(str(f))
        item = {
            "name": f.stem,
            "human_pos": pair["human_pos"].to(device),
            "human_quat": pair["human_quat"].to(device),
            "qpos": pair["qpos"].to(device),
            "fps": pair["fps"],
        }
        (val if f.stem in VAL_CLIPS else train).append(item)
    return train, val


def fit_xy_scale(clips: list[dict]) -> float:
    """Least-squares fit of s in robot_xy ≈ s * human_root_xy over the training clips.

    This is the trajectory scale GMR applies (human_scale_table x height ratio); for LAFAN1→G1 it is
    ~0.875 with ≤1.5 cm residual. Fitted once, stored in the checkpoint, and treated as a constant of
    the retargeting config at inference.
    """
    num = 0.0
    den = 0.0
    for clip in clips:
        hx = clip["human_pos"][:, 0, :2]
        rx = clip["qpos"][:, 0:2]
        num += float((rx * hx).sum())
        den += float((hx * hx).sum())
    return num / max(den, 1e-9)


def scaled_anchor(clip: dict, s: int, e: int, xy_scale: float) -> tuple:
    anchor_pos = clip["human_pos"][s:e, 0, :].clone()  # Hips
    anchor_pos[:, :2] *= xy_scale
    anchor_quat = clip["human_quat"][s:e, 0, :]
    return anchor_pos, anchor_quat


def make_teacher(clip: dict, s: int, e: int, xy_scale: float) -> tuple:
    """Slice a window and build the teacher dict with the root pose in the scaled-human-heading frame."""
    q = clip["qpos"][s:e]
    anchor_pos, anchor_quat = scaled_anchor(clip, s, e, xy_scale)
    local_pos, local_quat = world_root_to_local(anchor_pos, anchor_quat, q[:, 0:3], q[:, 3:7])
    teacher = {"root_pos": local_pos, "root_quat": local_quat, "dof_pos": q[:, 7:]}
    return clip["human_pos"][s:e], clip["human_quat"][s:e], teacher, (anchor_pos, anchor_quat), q


def sample_window(clip: dict, window: int, xy_scale: float):
    T = clip["qpos"].shape[0]
    s = 0 if T <= window else random.randint(0, T - window)
    e = min(s + window, T)
    hp, hq, teacher, anchor, q = make_teacher(clip, s, e, xy_scale)
    return hp, hq, teacher, anchor, q


def split_pair_paths(pairs_dir: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    paths = {"train": [], "validation": []}
    for path in sorted(pairs_dir.glob("*.npz")):
        split = "validation" if path.stem in VAL_CLIPS else "train"
        paths[split].append(path)
    return paths


def factorized_contact_mask(
    kind: str,
    human_pos: torch.Tensor,
    teacher_feet: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    if kind == "teacher_legacy":
        return detect_contact(teacher_feet, fps)
    if kind == "teacher_height":
        return detect_contact_height_hysteresis(
            teacher_feet, enter_height=0.03, exit_height=0.05
        )
    human_idx = [LAFAN1_BODY_NAMES.index(name) for name in LAFAN1_CONTACT_BODIES]
    human_feet = human_pos[:, human_idx, :]
    if kind == "source_contact":
        return detect_contact(
            human_feet, fps, height_threshold=0.08, speed_threshold=0.24
        )
    return detect_contact_height_hysteresis(
        human_feet, enter_height=0.08, exit_height=0.10
    )


def objective_manifest(args: argparse.Namespace) -> dict:
    return {
        "distill": {
            "weight": 1.0,
            "semantics": "configuration-space root position, root rotation, and DOF MSE",
        },
        "limits": {"weight": 0.1, "semantics": "squared joint-limit excess"},
        "smooth": {"weight": 0.01, "semantics": "frame-difference acceleration and jerk"},
        "legacy_teacher_mask_contact": {
            "weight": args.contact_weight,
            "active": args.contact_weight > 0,
            "semantics": "legacy frame-displacement teacher-mask velocity plus penetration",
        },
        "legacy_edge_self_consistency": {
            "weight": args.contact_weight,
            "active": args.edge_contact and args.contact_weight > 0,
            "semantics": "legacy predicted-contact displacement plus penetration",
        },
        "legacy_contact_bce": {
            "weight": args.contact_weight,
            "active": args.edge_contact and args.contact_weight > 0,
            "semantics": "BCE against legacy height-and-speed teacher mask",
        },
        "legacy_teacher_foot_displacement": {
            "weight": args.foot_vel_weight,
            "active": args.foot_vel_weight > 0,
            "semantics": "E25 all-phase 3D teacher foot displacement matching",
        },
        "contact_bce": {
            "weight": args.contact_bce_weight,
            "active": args.contact_bce_weight > 0,
            "mask": args.contact_mask,
        },
        "edge_velocity": {
            "weight": args.edge_velocity_weight,
            "active": args.edge_velocity_weight > 0,
            "semantics": "mean squared XY velocity computed in m/s, normalized by contact mass",
        },
        "teacher_stance_velocity": {
            "weight": args.stance_velocity_weight,
            "active": args.stance_velocity_weight > 0,
            "mask": args.contact_mask,
            "semantics": "mean squared XY velocity computed in m/s over explicit stance samples",
        },
        "penetration": {
            "weight": args.penetration_weight,
            "active": args.penetration_weight > 0,
            "semantics": "mean squared absolute ground penetration",
        },
        "teacher_velocity": {
            "weight": args.teacher_velocity_weight,
            "active": args.teacher_velocity_weight > 0,
            "phase_balanced": args.phase_balanced_velocity,
            "semantics": "mean squared 3D teacher foot-velocity error computed in m/s",
        },
    }


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


@torch.no_grad()
def evaluate(model, rk, skel, static, clips, window: int, xy_scale: float, max_windows_per_clip: int = 4):
    """Held-out whole-body MPJPE (m) between predicted-FK and teacher-FK, plus dof/root errors."""
    model.eval()
    mpjpes, dof_errs = [], []
    for clip in clips:
        T = clip["qpos"].shape[0]
        starts = np.linspace(0, max(T - window, 0), num=min(max_windows_per_clip, max(T // window, 1)), dtype=int)
        for s in starts:
            e = s + window
            hp, hq, teacher, (anchor_pos, anchor_quat), q = make_teacher(clip, int(s), int(e), xy_scale)
            pred = model.retarget_human_to_robot(hp, hq, skel, static, rk)
            # recover the predicted world root from the local prediction, then FK both in world frame
            wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
            bp_p, _ = rk.forward_kinematics(wp, wq, pred["dof_pos"])
            bp_t, _ = rk.forward_kinematics(q[:, 0:3], q[:, 3:7], teacher["dof_pos"])
            mpjpes.append((bp_p - bp_t).norm(dim=-1).mean().item())
            dof_errs.append((pred["dof_pos"] - teacher["dof_pos"]).abs().mean().item())
    model.train()
    return float(np.mean(mpjpes)), float(np.mean(dof_errs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="unitree_g1", choices=ROBOT_MJCF_KEYS)
    ap.add_argument("--pairs_dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "phase1_g1"))
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--enc_hidden", type=int, default=128)
    ap.add_argument("--dec_hidden", type=int, default=128)
    ap.add_argument("--contact_weight", type=float, default=0.0,
                    help="weight of the foot-contact loss (0 = off; contact mask from teacher feet)")
    ap.add_argument("--edge_contact", action="store_true",
                    help="add the EDGE self-consistency contact head (predict_contact) + BCE "
                         "supervision on top of the teacher-mask velocity loss")
    ap.add_argument("--foot_vel_weight", type=float, default=0.0,
                    help="E25: weight for matching teacher FK foot velocity (world frame); folds "
                         "low stance velocity into the distill objective (fixes skate; see E24)")
    ap.add_argument("--contact_bce_weight", type=float, default=0.0,
                    help="factorized contact-head BCE weight")
    ap.add_argument("--edge_velocity_weight", type=float, default=0.0,
                    help="factorized predicted-contact XY velocity (m/s) weight")
    ap.add_argument("--stance_velocity_weight", type=float, default=0.0,
                    help="factorized explicit-mask stance XY velocity (m/s) weight")
    ap.add_argument("--penetration_weight", type=float, default=0.0,
                    help="factorized absolute foot-penetration weight")
    ap.add_argument("--teacher_velocity_weight", type=float, default=0.0,
                    help="factorized teacher 3D foot-velocity (m/s) matching weight")
    ap.add_argument("--phase_balanced_velocity", action="store_true",
                    help="give stance and swing equal weight in --teacher_velocity_weight")
    ap.add_argument(
        "--contact_mask",
        choices=["teacher_height", "source_contact", "source_height", "teacher_legacy"],
        default="teacher_height",
        help="mask for factorized contact objectives; legacy flags always retain legacy masks",
    )
    ap.add_argument("--diag_every", type=int, default=1000,
                    help="loss/gradient diagnostic interval; 0 disables")
    ap.add_argument("--no_temporal", action="store_true",
                    help="ablation: disable the temporal transformer over frame latents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_dir = pathlib.Path(args.pairs_dir) if args.pairs_dir else data_root() / "pairs" / args.robot

    train_clips, val_clips = load_dataset(pairs_dir, args.device)
    print(f"train clips: {len(train_clips)}  val clips: {len(val_clips)}  device: {args.device}")

    rk = RobotKinematics(str(robot_mjcf(args.robot)), device=args.device)
    skel = lafan1_skeleton(device=args.device)
    # static features from a sample clip (bone lengths are constant across LAFAN1 subjects up to scale)
    static = human_static_features(skel, body_pos_sample=train_clips[0]["human_pos"])

    xy_scale = fit_xy_scale(train_clips)
    print(f"fitted robot/human xy trajectory scale: {xy_scale:.4f}")

    model = SNMR(
        SNMRConfig(latent_dim=args.latent_dim, enc_hidden=args.enc_hidden,
                   dec_hidden=args.dec_hidden, use_temporal=not args.no_temporal,
                   predict_contact=(
                       (args.edge_contact and args.contact_weight > 0)
                       or args.contact_bce_weight > 0
                       or args.edge_velocity_weight > 0
                   ))
    ).to(args.device)

    # foot-contact loss support: teacher contact masks + foot body indices (G1 by default)
    foot_names = FOOT_BODIES.get(args.robot)
    foot_idx = [rk.body_index(n) for n in foot_names] if foot_names else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.min_lr)

    start_step = 0
    ckpt_path = out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_step = state["step"]
        if "rng_state" in state:
            restore_rng_state(state["rng_state"])
        print(f"resumed from step {start_step}")

    split_paths = split_pair_paths(pairs_dir)
    split_paths["robot_assets"] = [robot_mjcf(args.robot)]
    print("hashing dataset for run manifest ...", flush=True)
    dataset = dataset_fingerprint(split_paths, root=pairs_dir)
    robot_exposures = {args.robot: start_step}
    manifest = RunManifest.start(
        out / "manifest.json",
        trainer="scripts/train_phase1.py",
        repo_root=ROOT,
        argv=[sys.executable, *sys.argv],
        config=vars(args),
        dataset=dataset,
        training={
            "seed": args.seed,
            "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": 1e-4},
            "lr_schedule": {
                "name": "CosineAnnealingLR",
                "t_max_steps": args.steps,
                "eta_min": args.min_lr,
            },
            "planned_optimizer_steps": args.steps,
            "planned_effective_robot_exposures": {args.robot: args.steps},
            "starting_step": start_step,
            "historical_exposures_reconstructable": True,
        },
        objectives=objective_manifest(args),
        resume=args.resume,
    )

    log_path = out / "log.jsonl"
    diagnostics_path = out / "diagnostics.jsonl"
    t0 = time.time()
    running = deque(maxlen=500)
    running_terms: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
    parameter_groups = model_parameter_groups(model)
    for step in range(start_step, args.steps):
        clip = random.choice(train_clips)
        hp, hq, teacher, (anchor_pos, anchor_quat), q = sample_window(clip, args.window, xy_scale)
        opt.zero_grad()
        pred = model.retarget_human_to_robot(hp, hq, skel, static, rk)
        terms = collect_loss_terms(pred, rk, teacher=teacher)
        weights = {name: DEFAULT_LOSS_WEIGHTS.get(name, 1.0) for name in terms}
        diagnostic_step = (
            args.diag_every > 0
            and ((step + 1) % args.diag_every == 0 or step + 1 == args.steps)
        )
        legacy_contact_active = args.contact_weight > 0
        factorized_contact_active = any([
            args.contact_bce_weight > 0,
            args.edge_velocity_weight > 0,
            args.stance_velocity_weight > 0,
            args.penetration_weight > 0,
            args.teacher_velocity_weight > 0,
        ])
        needs_world_prediction = (
            legacy_contact_active
            or args.foot_vel_weight > 0
            or args.stance_velocity_weight > 0
            or args.penetration_weight > 0
            or args.teacher_velocity_weight > 0
        )
        teacher_feet = None
        legacy_mask = None
        factorized_mask = None
        world_pred = None
        if foot_idx is not None and (
            legacy_contact_active or factorized_contact_active or diagnostic_step
        ):
            with torch.no_grad():
                t_body_w, _ = rk.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])
                teacher_feet = t_body_w[:, foot_idx, :]
                legacy_mask = detect_contact(teacher_feet, fps=clip["fps"])
                factorized_mask = factorized_contact_mask(
                    args.contact_mask, hp, teacher_feet, clip["fps"]
                )
        if foot_idx is not None and needs_world_prediction:
            wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
            world_pred = {"root_pos": wp, "root_quat": wq, "dof_pos": pred["dof_pos"]}

        # Historical flags retain their exact bundled loss semantics.
        if legacy_contact_active and foot_idx is not None:
            assert world_pred is not None and legacy_mask is not None
            terms["legacy_teacher_mask_contact"] = foot_contact_loss(
                world_pred, rk, foot_idx, legacy_mask
            )
            weights["legacy_teacher_mask_contact"] = args.contact_weight
            if args.edge_contact and "contact_logits" in pred:
                terms["legacy_edge_self_consistency"] = contact_self_consistency_loss(
                    pred, rk, foot_idx, anchor=(anchor_pos, anchor_quat)
                )
                terms["legacy_contact_bce"] = contact_prediction_loss(
                    pred["contact_logits"], foot_idx, legacy_mask
                )
                weights["legacy_edge_self_consistency"] = args.contact_weight
                weights["legacy_contact_bce"] = args.contact_weight
        if args.foot_vel_weight > 0 and foot_idx is not None:
            assert world_pred is not None
            terms["legacy_teacher_foot_displacement"] = foot_velocity_distill_loss(
                world_pred, rk, q[:, 0:3], q[:, 3:7], q[:, 7:], foot_idx
            )
            weights["legacy_teacher_foot_displacement"] = args.foot_vel_weight

        # New objectives are factorized, normalized in m/s, and independently weighted.
        if args.contact_bce_weight > 0 and foot_idx is not None:
            assert factorized_mask is not None and "contact_logits" in pred
            terms["contact_bce"] = contact_prediction_loss(
                pred["contact_logits"], foot_idx, factorized_mask
            )
            weights["contact_bce"] = args.contact_bce_weight
        if args.edge_velocity_weight > 0 and foot_idx is not None:
            terms["edge_velocity"] = contact_self_consistency_velocity_loss(
                pred,
                rk,
                foot_idx,
                fps=clip["fps"],
                anchor=(anchor_pos, anchor_quat),
            )
            weights["edge_velocity"] = args.edge_velocity_weight
        if args.stance_velocity_weight > 0 and foot_idx is not None:
            assert world_pred is not None and factorized_mask is not None
            terms["teacher_stance_velocity"] = teacher_stance_velocity_loss(
                world_pred, rk, foot_idx, factorized_mask, fps=clip["fps"]
            )
            weights["teacher_stance_velocity"] = args.stance_velocity_weight
        if args.penetration_weight > 0 and foot_idx is not None:
            assert world_pred is not None
            terms["penetration"] = foot_penetration_loss(world_pred, rk, foot_idx)
            weights["penetration"] = args.penetration_weight
        if args.teacher_velocity_weight > 0 and foot_idx is not None:
            assert world_pred is not None
            terms["teacher_velocity"] = teacher_foot_velocity_loss(
                world_pred,
                rk,
                q[:, 0:3],
                q[:, 3:7],
                q[:, 7:],
                foot_idx,
                fps=clip["fps"],
                contact_mask=factorized_mask,
                phase_balanced=args.phase_balanced_velocity,
            )
            weights["teacher_velocity"] = args.teacher_velocity_weight

        loss, parts = weighted_loss(terms, weights)
        if diagnostic_step:
            diagnostic = loss_gradient_diagnostics(terms, weights, parameter_groups)
            diagnostic.update({
                "step": step + 1,
                "robot": args.robot,
                "clip": clip["name"],
                "contact_definition": args.contact_mask,
            })
            if factorized_mask is not None:
                diagnostic["contact_labels"] = {
                    "prevalence": float(factorized_mask.float().mean()),
                    "samples": int(factorized_mask.sum()),
                }
                if "contact_logits" in pred:
                    diagnostic["contact_prediction"] = binary_contact_metrics(
                        pred["contact_logits"][:, foot_idx], factorized_mask
                    )
            append_jsonl(diagnostics_path, diagnostic)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(float(loss.detach()))
        robot_exposures[args.robot] += 1
        for name, value in parts.items():
            running_terms[f"{name}_raw"].append(value)
            running_terms[f"{name}_weighted"].append(value * weights.get(name, 1.0))

        if (step + 1) % args.eval_every == 0:
            mpjpe, dof_err = evaluate(model, rk, skel, static, val_clips, args.window, xy_scale)
            rec = {
                "step": step + 1,
                "train_loss": float(np.mean(running)),
                "val_mpjpe_m": mpjpe,
                "val_dof_err_rad": dof_err,
                "lr": sched.get_last_lr()[0],
                "elapsed_s": round(time.time() - t0, 1),
            }
            rec.update({
                f"train_{name}": float(np.mean(values))
                for name, values in running_terms.items()
                if values
            })
            append_jsonl(log_path, rec)
            print(f"step {step+1:6d}  loss {rec['train_loss']:.4f}  "
                  f"VAL mpjpe {mpjpe*100:.2f} cm  dof {dof_err:.4f} rad  "
                  f"({rec['elapsed_s']:.0f}s)", flush=True)

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "step": step + 1,
                 "xy_scale": xy_scale, "config": vars(args),
                 "robot_exposures": robot_exposures,
                 "rng_state": capture_rng_state()},
                ckpt_path,
            )
            manifest.update_progress(
                step=step + 1,
                robot_exposures=robot_exposures,
                checkpoint_path=ckpt_path,
            )

    mpjpe, dof_err = evaluate(model, rk, skel, static, val_clips, args.window, xy_scale, max_windows_per_clip=16)
    print(f"\nFINAL held-out: MPJPE {mpjpe*100:.2f} cm | dof err {dof_err:.4f} rad "
          f"| Gate G1 target < 3 cm")
    manifest.update_progress(
        step=args.steps,
        robot_exposures=robot_exposures,
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        complete=True,
    )


if __name__ == "__main__":
    main()
