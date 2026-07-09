#!/usr/bin/env python
"""N8: assemble the tracking-RL validation package (SNMR-retargeted vs GMR-teacher, matched clips).

The decisive "does better retargeting → better tracking" experiment (GMR-paper thesis, inverted).
Produces, for a set of held-out G1 clips:
  * `<out>/snmr/<clip>_mj.npz`     — SNMR retargeting exported to holosoma WBT format
  * `<out>/gmr/<clip>_mj.npz`      — the GMR teacher qpos exported through the SAME converter path
  * `<out>/WBT_COMMANDS.md`        — exact g1-29dof-wbt training + eval commands for an IsaacSim box
  * `<out>/manifest.json`          — clips, checkpoint, schema-validation status

Both branches go through `export_wbt_npz.mujoco_replay` so the ONLY difference the downstream WBT
policy sees is the retargeting source — a clean controlled comparison. NPZs are schema-validated
against the shipped holosoma sample so we know they load in `MotionLoader` without needing IsaacSim.

    python scripts/prepare_wbt_validation.py --ckpt runs/phase1_g1_large/ckpt_100k_final.pt \
        --clips walk1_subject5 dance2_subject4 --out runs/wbt_validation
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

from snmr.data import local_root_to_world  # noqa: E402
from snmr.human import human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.paths import data_root, g1_mjcf, holosoma_sample_npz  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

from export_wbt_npz import (  # noqa: E402
    mujoco_replay,
    resample_qpos,
    validate_against_reference,
)

DEFAULT_CLIPS = ["walk1_subject5", "dance2_subject4", "fight1_subject3",
                 "run2_subject1", "jumps1_subject2"]


def snmr_qpos(model, pair, rk, skel, h_static, xy_scale, device):
    from snmr.model import _adjacency
    from snmr.human import human_pose_features

    hp = pair["human_pos"].to(device)
    hq = pair["human_quat"].to(device)
    anchor = hp[:, 0, :].clone()
    anchor[:, :2] *= xy_scale
    with torch.no_grad():
        z = model.encode(human_pose_features(hp, hq), h_static, _adjacency(skel))
        pred = model.decode(z, rk)
        wp, wq = local_root_to_world(anchor, hq[:, 0, :], pred["root_pos"], pred["root_quat"])
    return torch.cat([wp, wq, pred["dof_pos"]], dim=-1).cpu().numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    ap.add_argument("--out", default=str(ROOT / "runs" / "wbt_validation"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--output_fps", type=float, default=50.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "snmr").mkdir(parents=True, exist_ok=True)
    (out / "gmr").mkdir(parents=True, exist_ok=True)

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    tc = state.get("config", {})
    model = SNMR(SNMRConfig(latent_dim=tc.get("latent_dim", 64),
                            enc_hidden=tc.get("enc_hidden", 128),
                            dec_hidden=tc.get("dec_hidden", 128))).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    # per-robot xy scale: phase-1 stores a scalar, phase-2 a dict
    xy_scale = float(state.get("xy_scale",
                     state.get("xy_scales", {}).get(args.robot, 0.875)))

    mjcf = str(g1_mjcf())
    rk = RobotKinematics(mjcf, device=args.device)
    skel = lafan1_skeleton(device=args.device)
    pairs = data_root() / "pairs" / args.robot
    reference = str(holosoma_sample_npz())

    manifest = {"ckpt": args.ckpt, "robot": args.robot, "xy_scale": xy_scale,
                "output_fps": args.output_fps, "clips": {}}
    for clip in args.clips:
        p = pairs / f"{clip}.npz"
        if not p.exists():
            print(f"skip {clip}: no pair NPZ at {p}")
            continue
        pair = load_pair_npz(str(p))
        h_static = human_static_features(skel, body_pos_sample=pair["human_pos"].to(args.device))

        # SNMR branch
        q_snmr = snmr_qpos(model, pair, rk, skel, h_static, xy_scale, args.device)
        q_snmr = resample_qpos(q_snmr, src_fps=pair["fps"], dst_fps=args.output_fps)
        snmr_npz = mujoco_replay(mjcf, q_snmr, args.output_fps)
        snmr_problems = validate_against_reference(snmr_npz, reference)
        np.savez_compressed(out / "snmr" / f"{clip}_mj.npz", **snmr_npz)

        # GMR teacher branch (same converter path)
        q_gmr = pair["qpos"].cpu().numpy().astype(np.float64)
        q_gmr = resample_qpos(q_gmr, src_fps=pair["fps"], dst_fps=args.output_fps)
        gmr_npz = mujoco_replay(mjcf, q_gmr, args.output_fps)
        gmr_problems = validate_against_reference(gmr_npz, reference)
        np.savez_compressed(out / "gmr" / f"{clip}_mj.npz", **gmr_npz)

        manifest["clips"][clip] = {
            "frames": int(q_snmr.shape[0]),
            "snmr_schema_ok": not snmr_problems,
            "gmr_schema_ok": not gmr_problems,
            "problems": {"snmr": snmr_problems, "gmr": gmr_problems},
        }
        print(f"{clip}: {q_snmr.shape[0]} frames  snmr_ok={not snmr_problems}  gmr_ok={not gmr_problems}")

    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    _write_commands(out, args)
    print(f"\nwrote {out}/  (snmr/, gmr/, manifest.json, WBT_COMMANDS.md)")


def _write_commands(out: pathlib.Path, args) -> None:
    clips = list(out.glob("snmr/*_mj.npz"))
    lines = [
        "# WBT tracking validation — run on an IsaacSim-capable machine",
        "",
        "SNMR-retargeted and GMR-teacher clip sets are in `snmr/` and `gmr/`, both in holosoma WBT",
        "format (schema-validated against the shipped sample — see manifest.json). The ONLY difference",
        "the tracking policy sees is the retargeting source, so this isolates retargeting → tracking",
        "quality (the GMR-paper thesis).",
        "",
        "## Per-clip policies (GMR-paper protocol: one policy per trajectory, identical config)",
        "```bash",
        "cd holosoma",
        "source scripts/source_isaacsim_setup.sh",
        "",
        "for SRC in snmr gmr; do",
        "  for CLIP in <clip names>; do",
        "    python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt logger:wandb \\",
        "      --command.setup_terms.motion_command.params.motion_config.motion_file="
        "<PATH>/$SRC/${CLIP}_mj.npz \\",
        "      --run_name ${SRC}_${CLIP}",
        "  done",
        "done",
        "```",
        "",
        "## Metrics to compare (SNMR vs GMR, paired by clip)",
        "- success rate (episode reaches end without early-termination), sim and sim-dr",
        "- E_g-mpbpe / E_mpbpe / E_mpjpe tracking errors (GMR-paper conventions)",
        "- sim2sim MuJoCo eval via holosoma_inference (deploy stack)",
        "",
        "## Local pre-check (no IsaacSim): confirm every NPZ loads in MotionLoader",
        "```bash",
        "python - <<'PY'",
        "from holosoma.managers.command.terms.wbt import MotionLoader  # adjust import if needed",
        "import glob",
        "for f in glob.glob('<PATH>/{snmr,gmr}/*_mj.npz'):",
        "    MotionLoader(f)  # should not raise",
        "PY",
        "```",
        f"\nClips exported: {sorted(p.stem.replace('_mj','') for p in clips)}",
    ]
    (out / "WBT_COMMANDS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
