#!/usr/bin/env python3
"""E56-D2: per-frame object-pose conditioning — the strong form of the E56-D question.

E56-D showed a 4-d filename-derived variant code recovers only 43% of the sibling spread
(R2 0.43 < 0.5) and barely steers between modes (R1 1.05 < 1.5). Live explanation (a):
the code was too coarse — the true hidden variable for object clips is the object pose
TRAJECTORY (7-d/frame, preserved in the converted pairs), and for terrain clips the
z-scale. E56-D2 conditions on exactly that:

  per-frame code (9-d) = [object_pose(7) or zeros, z_scale - 1.0, has_object]

broadcast into every decoder AdaLN alongside the embodiment code (MotionDecoder accepts
(T, cond_dim) since the E56-D2 model change). Dropout to the zero code with p=0.5
(HOVER precedent) so a marginal mode survives for comparison.

Pre-specified gates (same instrument as E56-D, held-out sibling GROUPS):
  R1 wrong/correct-code MPJPE ratio > 1.5  -> the code steers between modes
  R2 decode-spread / data-spread   >= 0.5  -> conditioning explains the variance
Decision rule: BOTH pass -> conditioning suffices, E56-C (MeanFlow) is dead again
(publishable: "interaction multimodality is hidden-variable-explained, given the right
variable"). Either fails -> the residual is real; E56-C proceeds with a measured target.

Note vs E56-D: rot/trans siblings differ by a rigid transform of the object AND of the
robot trajectory; the world-frame object pose is exactly the information that
disambiguates them, so this arm is the strongest conditioning we can build without
terrain geometry. Terrain siblings keep only the z-scale scalar (their object_pose is
absent) — R1/R2 are reported per family (object vs terrain) as well as pooled.

Run in .venv-snmr; fits beside the E52 v4 GPU job (<6 GB).
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_e55a_twoteacher import load_lafan, load_omni  # noqa: E402
from train_e56d_variant_code import holdout_sibling_groups  # noqa: E402
from snmr.human import human_pose_features  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.paths import robot_mjcf  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

COND_DIM = 9  # object_pose(7) + (z_scale - 1) + has_object


def attach_object_pose(pools, device):
    """Re-open the pair npz files and attach per-frame object_pose to each clip dict."""
    by_name = {pathlib.Path(f).stem: f
               for f in glob.glob("/home/ec2-user/work/retarget/data/pairs_omni/unitree_g1/omni_*.npz")}
    n_obj = 0
    for pool in pools:
        for c in pool.clips:
            f = by_name.get(c["name"])
            if f is None:
                c["object_pose"] = None
                continue
            d = np.load(f, allow_pickle=True)
            if "object_pose" in d and d["object_pose"].ndim == 2:
                c["object_pose"] = torch.from_numpy(
                    np.asarray(d["object_pose"], np.float32)).to(device)
                n_obj += 1
            else:
                c["object_pose"] = None
    return n_obj


def zscale_from_name(name: str) -> float:
    m = re.search(r"z_scale_([0-9.]+)", name)
    return float(m.group(1)) - 1.0 if m else 0.0


def frame_code(clip, s, e, device) -> torch.Tensor:
    """(e-s, 9) per-frame conditioning code."""
    T = e - s
    code = torch.zeros(T, COND_DIM, device=device)
    op = clip.get("object_pose")
    if op is not None:
        n = min(T, max(0, op.shape[0] - s))
        if n > 0:
            code[:n, :7] = op[s:s + n]
        code[:, 8] = 1.0
    code[:, 7] = zscale_from_name(clip["name"])
    return code


class ObjPoseSNMR(torch.nn.Module):
    """SNMR whose decoder AdaLN cond = [embodiment_code, per-frame 9-d code]."""

    def __init__(self, cfg: SNMRConfig):
        super().__init__()
        cfg_wide = SNMRConfig(**{**cfg.__dict__, "embodiment_dim": cfg.embodiment_dim + COND_DIM})
        self.core = SNMR(cfg_wide)
        self.embodiment_encoder = SNMR(cfg).embodiment_encoder

    def encode(self, feat, static, adj):
        return self.core.encode(feat, static, adj)

    def decode(self, z, kin, code: torch.Tensor):
        from snmr.data import robot_node_static_features
        from snmr.model import _adjacency
        from snmr.skeleton import SkeletonGraph

        static = robot_node_static_features(kin.graph)
        adj = _adjacency(SkeletonGraph.from_robot_graph(kin.graph))
        emb = self.embodiment_encoder(static)
        cond = torch.cat([emb.unsqueeze(0).expand(z.shape[0], -1), code.to(emb)], dim=-1)
        return self.core.decoder(z, static, adj, cond, kin.graph)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch_windows", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--code_dropout", type=float, default=0.5)
    args = ap.parse_args()
    out = pathlib.Path("runs/e56d2_objpose")
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    kin = RobotKinematics(str(robot_mjcf("unitree_g1")), device=device)
    lafan_tr, _ = load_lafan(device)
    omni_tr, omni_va = load_omni(device)
    n_obj = attach_object_pose(omni_tr + omni_va, device)
    sib_groups = holdout_sibling_groups(omni_tr, rng)
    n_sib = sum(len(g) for g in sib_groups.values())
    pools = [lafan_tr] + omni_tr
    frames = np.array([sum(c["human_pos"].shape[0] for c in p.clips) for p in pools], float)
    weights = np.sqrt(frames); weights /= weights.sum()
    print(f"pools: {[(p.name, len(p.clips)) for p in pools]} weights {weights.round(3)} | "
          f"{n_obj} clips carry object_pose | {n_sib} sibling groups held out", flush=True)

    cfg = SNMRConfig(latent_dim=128, enc_hidden=256, dec_hidden=256, use_temporal=False)
    model = ObjPoseSNMR(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = (out / "log.jsonl").open("a")

    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        total = 0.0
        for _ in range(args.batch_windows):
            pool = pools[rng.choice(len(pools), p=weights)]
            clip = pool.clips[rng.integers(len(pool.clips))]
            T = clip["human_pos"].shape[0]
            s = 0 if T <= args.window else int(rng.integers(0, T - args.window))
            e = min(T, s + args.window)
            feat = human_pose_features(clip["human_pos"][s:e], clip["human_quat"][s:e])
            z = model.encode(feat, pool.static, pool.adj)
            code = frame_code(clip, s, e, device)
            if rng.random() < args.code_dropout:
                code = torch.zeros_like(code)
            outp = model.decode(z, kin, code)
            qpos = clip["qpos"][s:e]
            loss = (outp["dof_pos"] - qpos[:, 7:]).square().mean() \
                 + 0.5 * (outp["root_pos"][:, 2] - qpos[:, 2]).square().mean()
            (loss / args.batch_windows).backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                fam = {"object": {"r1": [], "r2": []}, "terrain": {"r1": [], "r2": []}}
                for pool_name, groups in sib_groups.items():
                    pool = next(p for p in ([lafan_tr] + omni_tr + omni_va)
                                if p.name.startswith(pool_name.split("_val")[0]))
                    for stem_name, sibs in groups.items():
                        T = min(min(c["human_pos"].shape[0] for c in sibs), 128)
                        feat = human_pose_features(sibs[0]["human_pos"][:T],
                                                   sibs[0]["human_quat"][:T])
                        z = model.encode(feat, pool.static, pool.adj)
                        decs = [model.decode(z, kin, frame_code(c, 0, T, device))["dof_pos"]
                                for c in sibs]
                        errs = np.zeros((len(sibs), len(sibs)))
                        for i, d in enumerate(decs):
                            for j, c in enumerate(sibs):
                                errs[i, j] = float((d - c["qpos"][:T, 7:]).square().mean().sqrt())
                        correct = np.mean(np.diag(errs))
                        wrong = (errs.sum() - np.trace(errs)) / max(len(sibs)**2 - len(sibs), 1)
                        dec_spread = float(torch.stack(decs).std(dim=0).mean()) if len(decs) > 1 else 0.0
                        dat_spread = float(torch.stack([c["qpos"][:T, 7:] for c in sibs]).std(dim=0).mean())
                        family = "object" if any(c.get("object_pose") is not None for c in sibs) else "terrain"
                        fam[family]["r1"].append(wrong / max(correct, 1e-6))
                        fam[family]["r2"].append(dec_spread / max(dat_spread, 1e-6))
                allr1 = fam["object"]["r1"] + fam["terrain"]["r1"]
                allr2 = fam["object"]["r2"] + fam["terrain"]["r2"]
                rec = {"step": step, "loss": total / args.batch_windows,
                       "R1_resolution_ratio": float(np.mean(allr1)),
                       "R2_spread_recovery": float(np.mean(allr2)),
                       "R1_object": float(np.mean(fam["object"]["r1"])) if fam["object"]["r1"] else None,
                       "R2_object": float(np.mean(fam["object"]["r2"])) if fam["object"]["r2"] else None,
                       "R1_terrain": float(np.mean(fam["terrain"]["r1"])) if fam["terrain"]["r1"] else None,
                       "R2_terrain": float(np.mean(fam["terrain"]["r2"])) if fam["terrain"]["r2"] else None,
                       "elapsed_s": round(time.time() - t0)}
            model.train()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"model": model.state_dict(), "config": cfg.__dict__}, out / "ckpt.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
