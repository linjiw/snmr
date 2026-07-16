#!/usr/bin/env python
"""Phase-2 training: multi-robot shared latent (design step N4).

One SNMR model trained across multiple robots simultaneously:

  * Per step: sample a clip window; encode the human motion once -> z_h; decode z_h to K sampled
    robots and distill against each robot's GMR-teacher qpos.
  * **L_z (shared-space loss):** encode each robot's *teacher motion* (via that robot's graph +
    differentiable-FK pose features) -> z_r, and pull z_r and z_h together symmetrically. Since every
    robot's z_r is tied to the same z_h, all embodiments' encodings of the same motion converge to a
    single point in latent space — the SAME idea extended across robots.
  * **Leave-one-robot-out (LORO):** pass --holdout_robot to exclude a robot from ALL training
    (no decode, no distill, no L_z). Evaluation then decodes to it zero-shot from its MJCF-derived
    embodiment code alone. Its xy trajectory scale comes from the GMR IK config
    (human_scale_table[root] x height ratio) — a design constant, not fitted on held-out data
    (config-derived vs data-fitted scales agree to <=0.0013 across all 5 robots).

Robot roots are predicted in the scaled-human-root heading frame (Phase-1 lesson; per-robot scale).

    python scripts/train_phase2.py --steps 100000 --out runs/phase2_all5
    python scripts/train_phase2.py --steps 100000 --holdout_robot engineai_pm01 --out runs/phase2_loro_pm01
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

from snmr.data import (  # noqa: E402
    RobotMotion,
    local_root_to_world,
    robot_node_static_features,
    robot_pose_features,
    world_root_to_local,
)
from snmr.diagnostics import (  # noqa: E402
    binary_contact_metrics,
    loss_gradient_diagnostics,
    model_parameter_groups,
)
from snmr.experiment import (  # noqa: E402
    RunManifest,
    balanced_combination_schedule,
    capture_rng_state,
    dataset_fingerprint,
    restore_rng_state,
)
from snmr.human import (  # noqa: E402
    LAFAN1_BODY_NAMES,
    LAFAN1_CONTACT_BODIES,
    human_pose_features,
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
    foot_penetration_loss,
    latent_consistency_loss,
    teacher_foot_velocity_loss,
    teacher_stance_velocity_loss,
    weighted_loss,
)
from snmr.metrics import (  # noqa: E402
    FOOT_BODIES,
    detect_contact,
    detect_contact_height_hysteresis,
)
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402
from snmr.skeleton import SkeletonGraph  # noqa: E402

from snmr.paths import data_root, robot_mjcf  # noqa: E402

ALL_ROBOTS = ["unitree_g1", "booster_t1_29dof", "fourier_n1", "engineai_pm01", "stanford_toddy"]

# xy trajectory scale from the GMR IK config: human_scale_table[human_root] * (1.75 / height_assumption)
# (LAFAN1 loader hardcodes human height 1.75). Verified against least-squares data fits (diff <= 1.3e-3).
ROBOT_XY_SCALE_CONFIG = {
    "unitree_g1": 0.8750,
    "booster_t1_29dof": 0.5833,
    "fourier_n1": 0.7778,
    "engineai_pm01": 0.8264,
    "stanford_toddy": 0.3403,
}

VAL_CLIPS = [
    "walk1_subject5", "dance2_subject4", "fight1_subject3",
    "run2_subject1", "jumps1_subject2", "sprint1_subject4", "aiming2_subject3",
]


class RobotContext:
    """Everything static needed to decode to / encode from one robot."""

    def __init__(self, name: str, device: str):
        self.name = name
        self.kin = RobotKinematics(str(robot_mjcf(name)), device=device)
        self.static = robot_node_static_features(self.kin.graph)
        self.adj = _adjacency(SkeletonGraph.from_robot_graph(self.kin.graph))
        self.xy_scale = ROBOT_XY_SCALE_CONFIG[name]
        self.foot_idx = [self.kin.body_index(n) for n in FOOT_BODIES[name]]


def load_clips(pairs_root: pathlib.Path, robots: list[str], device: str):
    """Load all clips; human arrays deduplicated across robots (same LAFAN1 source clip)."""
    clip_names = sorted(p.stem for p in (pairs_root / robots[0]).glob("*.npz"))
    train, val = {}, {}
    for name in clip_names:
        entry = None
        for robot in robots:
            pair = load_pair_npz(str(pairs_root / robot / f"{name}.npz"))
            if entry is None:
                entry = {
                    "name": name,
                    "human_pos": pair["human_pos"].to(device),
                    "human_quat": pair["human_quat"].to(device),
                    "qpos": {},
                    "fps": pair["fps"],
                }
            entry["qpos"][robot] = pair["qpos"].to(device)
        (val if name in VAL_CLIPS else train)[name] = entry
    return train, val


def window_teacher(entry: dict, robot: str, s: int, e: int, xy_scale: float):
    """Teacher dict for one robot window, root in the scaled-human-heading frame."""
    q = entry["qpos"][robot][s:e]
    anchor_pos = entry["human_pos"][s:e, 0, :].clone()
    anchor_pos[:, :2] *= xy_scale
    anchor_quat = entry["human_quat"][s:e, 0, :]
    lp, lq = world_root_to_local(anchor_pos, anchor_quat, q[:, 0:3], q[:, 3:7])
    return {"root_pos": lp, "root_quat": lq, "dof_pos": q[:, 7:]}, (anchor_pos, anchor_quat), q


def encode_robot_teacher(model: SNMR, ctx: RobotContext, q: torch.Tensor) -> torch.Tensor:
    """z of a robot's teacher motion window (input features are constants — no grad through FK)."""
    with torch.no_grad():
        motion = RobotMotion(q[:, 0:3], q[:, 3:7], q[:, 7:], fps=30.0)
        feats = robot_pose_features(ctx.kin, motion)
    return model.encode(feats, ctx.static, ctx.adj)


def split_pair_paths(
    pairs_root: pathlib.Path,
    train_robots: list[str],
    eval_robots: list[str],
) -> dict[str, list[pathlib.Path]]:
    paths = {"train": [], "validation": [], "holdout_not_trained": []}
    for robot in eval_robots:
        for path in sorted((pairs_root / robot).glob("*.npz")):
            if path.stem in VAL_CLIPS:
                split = "validation"
            elif robot in train_robots:
                split = "train"
            else:
                split = "holdout_not_trained"
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
            "aggregation": "mean over sampled target robots",
            "semantics": "configuration-space root position, root rotation, and DOF MSE",
        },
        "limits": {"weight": 0.1, "semantics": "squared joint-limit excess"},
        "smooth": {"weight": 0.01, "semantics": "frame-difference acceleration and jerk"},
        "latent": {
            "weight": args.latent_weight,
            "semantics": "paired human/robot latent MSE",
        },
        "legacy_edge_self_consistency": {
            "weight": args.contact_weight,
            "active": args.contact_weight > 0,
            "semantics": "legacy predicted-contact displacement plus penetration",
        },
        "legacy_contact_bce": {
            "weight": args.contact_weight,
            "active": args.contact_weight > 0,
            "semantics": "BCE against legacy height-and-speed teacher mask",
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
        "robot_source_decode": {
            "probability": args.zr_decode_prob,
            "active": args.zr_decode_prob > 0,
        },
    }


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


@torch.no_grad()
def evaluate_robot(model, ctx: RobotContext, skel, h_static, clips: dict, window: int,
                   max_windows_per_clip: int = 4) -> tuple[float, float]:
    """Held-out MPJPE (m) + dof err for one robot (world-frame FK, scaled-anchor recomposition)."""
    model.eval()
    mpjpes, dof_errs = [], []
    for entry in clips.values():
        T = entry["qpos"][ctx.name].shape[0]
        starts = np.linspace(0, max(T - window, 0),
                             num=min(max_windows_per_clip, max(T // window, 1)), dtype=int)
        for s in starts:
            e = int(s) + window
            teacher, (anchor_pos, anchor_quat), q = window_teacher(entry, ctx.name, int(s), e, ctx.xy_scale)
            feats = human_pose_features(entry["human_pos"][int(s):e], entry["human_quat"][int(s):e])
            z = model.encode(feats, h_static, _HUMAN_ADJ)
            adapter_name = ctx.name if ctx.name in model.decoder.adapters else None
            pred = model.decoder(z, ctx.static, ctx.adj,
                                 model.embodiment_encoder(ctx.static), ctx.kin.graph,
                                 adapter_name=adapter_name)
            wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
            bp_p, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
            bp_t, _ = ctx.kin.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])
            mpjpes.append((bp_p - bp_t).norm(dim=-1).mean().item())
            dof_errs.append((pred["dof_pos"] - q[:, 7:]).abs().mean().item())
    model.train()
    return float(np.mean(mpjpes)), float(np.mean(dof_errs))


_HUMAN_ADJ = None  # set in main() after device is known


def main() -> None:
    global _HUMAN_ADJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=ALL_ROBOTS)
    ap.add_argument("--holdout_robot", default=None,
                    help="robot excluded from ALL training; evaluated zero-shot")
    ap.add_argument("--pairs_root", default=None, help="default: <data_root>/pairs")
    ap.add_argument("--out", default=str(ROOT / "runs" / "phase2"))
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--robots_per_step", type=int, default=2)
    ap.add_argument(
        "--robot_sampling",
        choices=["random", "balanced_combinations"],
        default="random",
        help="balanced_combinations cycles through every robot subset without using the data RNG",
    )
    ap.add_argument("--lr", type=float, default=3e-4)  # Phase-1 lesson: 8e-4 destabilizes this size
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--latent_weight", type=float, default=1.0)
    ap.add_argument("--contact_weight", type=float, default=0.0,
                    help="legacy N6 bundled self-consistency + contact-BCE weight")
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
        help="mask for factorized contact objectives; legacy flag retains its legacy mask",
    )
    ap.add_argument("--zr_decode_prob", type=float, default=0.0,
                    help="E21: probability of decoding a robot-teacher encoding z_r instead of the "
                         "human encoding z_h (0 = off; fixes robot->robot transfer OOD gap)")
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--enc_hidden", type=int, default=256)
    ap.add_argument("--dec_hidden", type=int, default=256)
    ap.add_argument(
        "--decoder_adapter_rank",
        type=int,
        default=0,
        help="rank of a zero-initialized per-training-robot decoder residual; 0 disables adapters",
    )
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--ckpt_every", type=int, default=10000)
    ap.add_argument("--diag_every", type=int, default=1000,
                    help="loss/gradient diagnostic interval; 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_robots = [r for r in args.robots if r != args.holdout_robot]
    eval_robots = list(args.robots)  # includes the holdout for zero-shot eval
    if not train_robots:
        raise SystemExit("at least one training robot is required")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs_root = pathlib.Path(args.pairs_root) if args.pairs_root else data_root() / "pairs"
    train_clips, val_clips = load_clips(pairs_root, eval_robots, args.device)
    print(f"train robots: {train_robots}  holdout: {args.holdout_robot}")
    print(f"train clips: {len(train_clips)}  val clips: {len(val_clips)}  device: {args.device}")

    ctxs = {r: RobotContext(r, args.device) for r in eval_robots}
    skel = lafan1_skeleton(device=args.device)
    sample_entry = next(iter(train_clips.values()))
    h_static = human_static_features(skel, body_pos_sample=sample_entry["human_pos"])
    _HUMAN_ADJ = _adjacency(skel)

    model = SNMR(SNMRConfig(latent_dim=args.latent_dim, enc_hidden=args.enc_hidden,
                            dec_hidden=args.dec_hidden,
                            decoder_adapter_rank=args.decoder_adapter_rank,
                            adapter_names=(
                                tuple(train_robots)
                                if args.decoder_adapter_rank > 0
                                else ()
                            ),
                            predict_contact=(
                                args.contact_weight > 0
                                or args.contact_bce_weight > 0
                                or args.edge_velocity_weight > 0
                            ))).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for p in model.decoder.adapters.parameters())
    print(
        f"model params: {n_params/1e6:.2f}M "
        f"({adapter_params/1e3:.1f}k robot-specific adapter)"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.min_lr)

    start_step = 0
    state = None
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

    split_paths = split_pair_paths(pairs_root, train_robots, eval_robots)
    split_paths["robot_assets"] = [robot_mjcf(robot) for robot in eval_robots]
    print("hashing dataset for run manifest ...", flush=True)
    dataset = dataset_fingerprint(split_paths, root=pairs_root)
    saved_exposures = state.get("robot_exposures") if state is not None else None
    robot_exposures = {
        robot: int(saved_exposures.get(robot, 0)) if saved_exposures else 0
        for robot in train_robots
    }
    exposures_known_from_step = 0 if start_step == 0 or saved_exposures else start_step
    robots_per_step = min(args.robots_per_step, len(train_robots))
    balanced_schedule = None
    if args.robot_sampling == "balanced_combinations":
        balanced_schedule = balanced_combination_schedule(
            train_robots,
            robots_per_step,
            seed=args.seed,
        )
        if args.steps % len(balanced_schedule) != 0:
            raise SystemExit(
                "balanced_combinations requires --steps to be a multiple of "
                f"the {len(balanced_schedule)}-step combination cycle"
            )
    if balanced_schedule is not None:
        planned_exposure = args.steps * robots_per_step // len(train_robots)
    else:
        planned_exposure = args.steps * robots_per_step / len(train_robots)
    manifest = RunManifest.start(
        out / "manifest.json",
        trainer="scripts/train_phase2.py",
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
            "robots_per_step": robots_per_step,
            "robot_sampling": {
                "strategy": args.robot_sampling,
                "cycle": (
                    [list(group) for group in balanced_schedule]
                    if balanced_schedule is not None
                    else None
                ),
            },
            "model_parameters": {
                "total": n_params,
                "robot_specific": adapter_params,
                "robot_specific_fraction": adapter_params / n_params,
            },
            "planned_effective_robot_exposures": {
                robot: planned_exposure for robot in train_robots
            },
            "starting_step": start_step,
            "observed_exposures_known_from_step": exposures_known_from_step,
        },
        objectives=objective_manifest(args),
        resume=args.resume,
    )

    log_path = out / "log.jsonl"
    diagnostics_path = out / "diagnostics.jsonl"
    clip_list = list(train_clips.values())
    t0 = time.time()
    running = deque(maxlen=500)
    running_lz = deque(maxlen=500)
    running_terms: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
    parameter_groups = model_parameter_groups(model)

    for step in range(start_step, args.steps):
        entry = random.choice(clip_list)
        T = entry["human_pos"].shape[0]
        s = 0 if T <= args.window else random.randint(0, T - args.window)
        e = min(s + args.window, T)

        if balanced_schedule is None:
            robots_k = random.sample(train_robots, robots_per_step)
        else:
            robots_k = list(balanced_schedule[step % len(balanced_schedule)])

        feats = human_pose_features(entry["human_pos"][s:e], entry["human_quat"][s:e])
        z_h = model.encode(feats, h_static, _HUMAN_ADJ)

        opt.zero_grad()
        loss = z_h.new_zeros(())
        lz_total = z_h.new_zeros(())
        diagnostic_step = (
            args.diag_every > 0
            and ((step + 1) % args.diag_every == 0 or step + 1 == args.steps)
        )
        factorized_contact_active = any([
            args.contact_bce_weight > 0,
            args.edge_velocity_weight > 0,
            args.stance_velocity_weight > 0,
            args.penetration_weight > 0,
            args.teacher_velocity_weight > 0,
        ])
        diagnostic_context = {}
        robot_totals = {}
        for robot in robots_k:
            ctx = ctxs[robot]
            teacher, anchor, q = window_teacher(entry, robot, s, e, ctx.xy_scale)
            # E21 (opt-in): decode-from-z_r augmentation. The decoder otherwise only ever consumes
            # HUMAN encodings; robot encodings z_r enter training solely through L_z, and the
            # residual z_h-z_r gap is out-of-distribution at robot->robot transfer time (measured:
            # 24-45cm vs 3-5cm on the human path, E19). With prob p we source the decoded latent
            # from a randomly chosen robot's teacher encoding instead (WITH grad, so the encoder
            # also learns to make robot encodings decodable), distilling to the same target.
            z_src = z_h
            if args.zr_decode_prob > 0 and random.random() < args.zr_decode_prob:
                src_robot = random.choice(train_robots)
                q_src = entry["qpos"][src_robot][s:e]
                src_ctx = ctxs[src_robot]
                motion_src = RobotMotion(
                    q_src[:, 0:3], q_src[:, 3:7], q_src[:, 7:], fps=entry["fps"]
                )
                with torch.no_grad():
                    feats_src = robot_pose_features(src_ctx.kin, motion_src)
                z_src = model.encode(feats_src, src_ctx.static, src_ctx.adj)
            pred = model.decoder(z_src, ctx.static, ctx.adj,
                                 model.embodiment_encoder(ctx.static), ctx.kin.graph,
                                 adapter_name=(
                                     robot if robot in model.decoder.adapters else None
                                 ))
            terms = collect_loss_terms(pred, ctx.kin, teacher=teacher)
            weights = {name: DEFAULT_LOSS_WEIGHTS.get(name, 1.0) for name in terms}
            legacy_mask = None
            factorized_mask = None
            if args.contact_weight > 0 or factorized_contact_active or diagnostic_step:
                with torch.no_grad():
                    teacher_body, _ = ctx.kin.forward_kinematics(
                        q[:, 0:3], q[:, 3:7], q[:, 7:]
                    )
                    teacher_feet = teacher_body[:, ctx.foot_idx, :]
                    legacy_mask = detect_contact(teacher_feet, fps=entry["fps"])
                    factorized_mask = factorized_contact_mask(
                        args.contact_mask,
                        entry["human_pos"][s:e],
                        teacher_feet,
                        entry["fps"],
                    )

            # Historical --contact_weight remains the exact bundled E10a/N6 objective.
            if args.contact_weight > 0 and "contact_logits" in pred:
                assert legacy_mask is not None
                terms["legacy_edge_self_consistency"] = contact_self_consistency_loss(
                    pred, ctx.kin, ctx.foot_idx, anchor=anchor
                )
                terms["legacy_contact_bce"] = contact_prediction_loss(
                    pred["contact_logits"], ctx.foot_idx, legacy_mask
                )
                weights["legacy_edge_self_consistency"] = args.contact_weight
                weights["legacy_contact_bce"] = args.contact_weight

            world_pred = None
            if any([
                args.stance_velocity_weight > 0,
                args.penetration_weight > 0,
                args.teacher_velocity_weight > 0,
            ]):
                world_pos, world_quat = local_root_to_world(
                    anchor[0], anchor[1], pred["root_pos"], pred["root_quat"]
                )
                world_pred = {
                    "root_pos": world_pos,
                    "root_quat": world_quat,
                    "dof_pos": pred["dof_pos"],
                }
            if args.contact_bce_weight > 0:
                assert factorized_mask is not None and "contact_logits" in pred
                terms["contact_bce"] = contact_prediction_loss(
                    pred["contact_logits"], ctx.foot_idx, factorized_mask
                )
                weights["contact_bce"] = args.contact_bce_weight
            if args.edge_velocity_weight > 0:
                terms["edge_velocity"] = contact_self_consistency_velocity_loss(
                    pred, ctx.kin, ctx.foot_idx, fps=entry["fps"], anchor=anchor
                )
                weights["edge_velocity"] = args.edge_velocity_weight
            if args.stance_velocity_weight > 0:
                assert world_pred is not None and factorized_mask is not None
                terms["teacher_stance_velocity"] = teacher_stance_velocity_loss(
                    world_pred,
                    ctx.kin,
                    ctx.foot_idx,
                    factorized_mask,
                    fps=entry["fps"],
                )
                weights["teacher_stance_velocity"] = args.stance_velocity_weight
            if args.penetration_weight > 0:
                assert world_pred is not None
                terms["penetration"] = foot_penetration_loss(
                    world_pred, ctx.kin, ctx.foot_idx
                )
                weights["penetration"] = args.penetration_weight
            if args.teacher_velocity_weight > 0:
                assert world_pred is not None
                terms["teacher_velocity"] = teacher_foot_velocity_loss(
                    world_pred,
                    ctx.kin,
                    q[:, 0:3],
                    q[:, 3:7],
                    q[:, 7:],
                    ctx.foot_idx,
                    fps=entry["fps"],
                    contact_mask=factorized_mask,
                    phase_balanced=args.phase_balanced_velocity,
                )
                weights["teacher_velocity"] = args.teacher_velocity_weight

            z_r = encode_robot_teacher(model, ctx, q)
            l_z = latent_consistency_loss(z_h, z_r)  # symmetric: grads flow into both encodings
            terms["latent"] = l_z
            weights["latent"] = args.latent_weight
            robot_total, parts = weighted_loss(terms, weights)
            loss = loss + robot_total
            lz_total = lz_total + l_z
            robot_totals[robot] = robot_total
            diagnostic_context[robot] = {
                "terms": terms,
                "weights": weights,
                "mask": factorized_mask,
                "logits": pred.get("contact_logits"),
                "foot_idx": ctx.foot_idx,
            }
            robot_exposures[robot] += 1
            for name, value in parts.items():
                running_terms[f"{robot}_{name}_raw"].append(value)
                running_terms[f"{robot}_{name}_weighted"].append(
                    value * weights.get(name, 1.0)
                )
        loss = loss / len(robots_k)

        if diagnostic_step:
            diagnostic = {
                "step": step + 1,
                "clip": entry["name"],
                "robots": {},
                "aggregation_divisor": len(robots_k),
                "contact_definition": args.contact_mask,
            }
            for robot, context in diagnostic_context.items():
                robot_diagnostic = loss_gradient_diagnostics(
                    context["terms"], context["weights"], parameter_groups
                )
                mask = context["mask"]
                if mask is not None:
                    robot_diagnostic["contact_labels"] = {
                        "prevalence": float(mask.float().mean()),
                        "samples": int(mask.sum()),
                    }
                    if context["logits"] is not None:
                        robot_diagnostic["contact_prediction"] = binary_contact_metrics(
                            context["logits"][:, context["foot_idx"]], mask
                        )
                diagnostic["robots"][robot] = robot_diagnostic
            diagnostic["cross_robot"] = loss_gradient_diagnostics(
                robot_totals,
                {robot: 1.0 / len(robots_k) for robot in robots_k},
                parameter_groups,
            )
            append_jsonl(diagnostics_path, diagnostic)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(float(loss.detach()))
        running_lz.append(float(lz_total.detach()) / len(robots_k))

        if (step + 1) % args.eval_every == 0:
            rec = {
                "step": step + 1,
                "train_loss": float(np.mean(running)),
                "train_lz": float(np.mean(running_lz)),
                "lr": sched.get_last_lr()[0],
                "elapsed_s": round(time.time() - t0, 1),
            }
            rec.update({
                f"train_{name}": float(np.mean(values))
                for name, values in running_terms.items()
                if values
            })
            msgs = []
            for robot in eval_robots:
                mpjpe, dof_err = evaluate_robot(model, ctxs[robot], skel, h_static, val_clips, args.window)
                tag = "ZS" if robot == args.holdout_robot else "  "
                rec[f"val_mpjpe_{robot}"] = mpjpe
                rec[f"val_dof_{robot}"] = dof_err
                msgs.append(f"{robot.split('_')[-1]}{tag}:{mpjpe*100:.1f}cm")
            append_jsonl(log_path, rec)
            print(f"step {rec['step']:6d} loss {rec['train_loss']:.4f} Lz {rec['train_lz']:.4f} | "
                  + " ".join(msgs) + f" | {rec['elapsed_s']:.0f}s", flush=True)

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "step": step + 1, "xy_scales": ROBOT_XY_SCALE_CONFIG, "config": vars(args),
                 "robot_exposures": robot_exposures,
                 "rng_state": capture_rng_state()},
                ckpt_path,
            )
            manifest.update_progress(
                step=step + 1,
                robot_exposures=robot_exposures,
                checkpoint_path=ckpt_path,
            )

    print("\n=== FINAL held-out eval (denser windows) ===")
    final = {}
    for robot in eval_robots:
        mpjpe, dof_err = evaluate_robot(model, ctxs[robot], skel, h_static, val_clips,
                                        args.window, max_windows_per_clip=16)
        tag = " [ZERO-SHOT]" if robot == args.holdout_robot else ""
        final[robot] = {"mpjpe_m": mpjpe, "dof_err_rad": dof_err}
        print(f"{robot:18s}{tag}: MPJPE {mpjpe*100:.2f} cm | dof err {dof_err:.4f} rad")
    with open(out / "final_eval.json", "w") as fh:
        json.dump(final, fh, indent=2)
    manifest.update_progress(
        step=args.steps,
        robot_exposures=robot_exposures,
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        complete=True,
    )


if __name__ == "__main__":
    main()
