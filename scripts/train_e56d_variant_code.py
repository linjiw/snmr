#!/usr/bin/env python3
"""E56-D: variant-code conditioning — does a hidden-variable code explain the sibling
multimodality, or is a generative decoder needed?

Design (docs/E55_E57_SCALING_AND_DISCRIMINATION.md Addendum 2 + INTERACTION_DATA_RISK doc):
OmniRetarget siblings share a bit-identical human motion but differ in robot trajectory
because of a hidden variable h (terrain z-scale / object rot-trans). Without h the decoder
must average siblings (mode averaging). E56-D conditions the decoder on a compact variant
code derived from h and asks whether that RESOLVES the siblings.

Variant code (4-d, broadcast alongside the embodiment code into every AdaLN):
  [z_scale - 1.0, is_rot, is_trans, variant_index/4]   (original = zeros)
Dropout to the zero code with p=0.5 during training (HOVER Bernoulli(0.5) precedent) so
the model retains a marginal (code-free) mode for fair comparison at eval.

Arms (matched budget, both on the FULL two-teacher pool with v3 frame-weighted sampling):
  nocode — plain E55-A v3 recipe (control; also serves as the v3 relaunch datapoint)
  code   — + variant code into AdaLN

Preregistered readouts (held-out sibling GROUPS — entire groups excluded from training):
  R1 sibling-resolution: per-group, MPJPE of code-conditioned decode against the CORRECT
     sibling vs against the WRONG sibling. Resolution ratio = wrong/correct (>1.5 = code
     steers between modes).
  R2 residual multimodality: dof-space spread of decodes across codes for one human input,
     vs the data's sibling spread (0.031-0.068 rad). If decodes with code reproduce >=50%
     of the spread, conditioning explains the variance -> E56-C (MeanFlow) DEAD again.
  R3 marginal fidelity: code-free decode MPJPE on omni-val vs the nocode arm (does the
     capacity spent on conditioning cost the marginal mode anything?).

Run in .venv-snmr, fits in <6 GB beside other jobs.
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

from train_e55a_twoteacher import (  # noqa: E402
    VAL_LAFAN,
    SourcePool,
    load_lafan,
    load_omni,
    sample_window,
)
from snmr.human import human_pose_features  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.paths import robot_mjcf  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

VARIANT_DIM = 4


def variant_code_from_name(name: str) -> np.ndarray:
    """4-d hidden-variable code from the clip filename."""
    code = np.zeros(VARIANT_DIM, dtype=np.float32)
    m = re.search(r"z_scale_([0-9.]+)", name)
    if m:
        code[0] = float(m.group(1)) - 1.0
    m = re.search(r"_rot_(\d+)", name)
    if m:
        code[1] = 1.0
        code[3] = int(m.group(1)) / 4.0
    m = re.search(r"_trans_(\d+)", name)
    if m:
        code[2] = 1.0
        code[3] = int(m.group(1)) / 4.0
    return code


class VariantSNMR(torch.nn.Module):
    """SNMR whose decoder AdaLN cond = [embodiment_code, variant_code]."""

    def __init__(self, cfg: SNMRConfig):
        super().__init__()
        cfg_wide = SNMRConfig(**{**cfg.__dict__, "embodiment_dim": cfg.embodiment_dim + VARIANT_DIM})
        self.core = SNMR(cfg_wide)
        self.base_embodiment_dim = cfg.embodiment_dim
        # embodiment encoder outputs base-dim; we re-project static feats with the stock
        # encoder from a plain cfg and append the variant code at call time.
        self.embodiment_encoder = SNMR(cfg).embodiment_encoder

    def encode(self, feat, static, adj):
        return self.core.encode(feat, static, adj)

    def decode(self, z, kin, variant: torch.Tensor):
        from snmr.data import robot_node_static_features
        from snmr.model import _adjacency
        from snmr.skeleton import SkeletonGraph

        static = robot_node_static_features(kin.graph)
        adj = _adjacency(SkeletonGraph.from_robot_graph(kin.graph))
        code = self.embodiment_encoder(static)
        cond = torch.cat([code, variant.to(code)], dim=-1)
        return self.core.decoder(z, static, adj, cond, kin.graph)


def holdout_sibling_groups(omni_pools, rng, n_groups=8):
    """Pick sibling groups (>=2 clips, same stem) and remove them from training pools."""
    stem = lambda n: re.sub(r"(_z_scale_[0-9.]+|_original|_rot_\d+|_trans_\d+)", "", n)
    groups = {}
    for pool in omni_pools:
        by = {}
        for c in pool.clips:
            by.setdefault(stem(c["name"]), []).append(c)
        multi = {k: v for k, v in by.items() if len(v) >= 2}
        take = list(multi)[: max(1, n_groups // len(omni_pools))]
        groups[pool.name] = {k: multi[k] for k in take}
        keep = {c["name"] for k in take for c in multi[k]}
        pool.clips = [c for c in pool.clips if c["name"] not in keep]
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["nocode", "code"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch_windows", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--code_dropout", type=float, default=0.5)
    args = ap.parse_args()
    out = pathlib.Path(f"runs/e56d_{args.arm}")
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    kin = RobotKinematics(str(robot_mjcf("unitree_g1")), device=device)
    lafan_tr, lafan_va = load_lafan(device)
    omni_tr, omni_va = load_omni(device)
    sib_groups = holdout_sibling_groups(omni_tr, rng)
    n_sib = sum(len(g) for g in sib_groups.values())
    pools = [lafan_tr] + omni_tr
    frames = np.array([sum(c["human_pos"].shape[0] for c in p.clips) for p in pools], float)
    weights = np.sqrt(frames); weights /= weights.sum()
    print(f"arm={args.arm} pools:", [(p.name, len(p.clips)) for p in pools],
          "weights", weights.round(3), f"| {n_sib} sibling groups held out")

    cfg = SNMRConfig(latent_dim=128, enc_hidden=256, dec_hidden=256, use_temporal=False)
    model = VariantSNMR(cfg).to(device) if args.arm == "code" else SNMR(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = (out / "log.jsonl").open("a")
    zero_code = torch.zeros(VARIANT_DIM, device=device)

    def decode(z, clip_name):
        if args.arm == "nocode":
            return model.decode(z, kin)
        code = torch.from_numpy(variant_code_from_name(clip_name)).to(device)
        if model.training and rng.random() < args.code_dropout:
            code = zero_code
        return model.decode(z, kin, code)

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
            outp = decode(z, clip["name"])
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
                # R1/R2 on held-out sibling groups
                res_ratios, spreads = [], []
                for pool_name, groups in sib_groups.items():
                    pool = next(p for p in ([lafan_tr] + omni_tr + omni_va) if p.name.startswith(pool_name.split("_val")[0]))
                    for stem_name, sibs in groups.items():
                        T = min(c["human_pos"].shape[0] for c in sibs)
                        T = min(T, 128)
                        feat = human_pose_features(sibs[0]["human_pos"][:T], sibs[0]["human_quat"][:T])
                        z = model.encode(feat, pool.static, pool.adj)
                        decs = []
                        for c in sibs:
                            if args.arm == "code":
                                code = torch.from_numpy(variant_code_from_name(c["name"])).to(device)
                                d = model.decode(z, kin, code)
                            else:
                                d = model.decode(z, kin)
                            decs.append(d["dof_pos"])
                        # R1: correct-vs-wrong assignment error
                        errs = np.zeros((len(sibs), len(sibs)))
                        for i, d in enumerate(decs):
                            for j, c in enumerate(sibs):
                                errs[i, j] = float((d - c["qpos"][:T, 7:]).square().mean().sqrt())
                        correct = np.mean(np.diag(errs))
                        wrong = (errs.sum() - np.trace(errs)) / max(len(sibs)**2 - len(sibs), 1)
                        res_ratios.append(wrong / max(correct, 1e-6))
                        # R2: decode spread vs data spread
                        dec_spread = float(torch.stack(decs).std(dim=0).mean()) if len(decs) > 1 else 0.0
                        dat_spread = float(torch.stack([c["qpos"][:T, 7:] for c in sibs]).std(dim=0).mean())
                        spreads.append(dec_spread / max(dat_spread, 1e-6))
                rec = {"step": step, "loss": total / args.batch_windows,
                       "R1_resolution_ratio": float(np.mean(res_ratios)),
                       "R2_spread_recovery": float(np.mean(spreads)),
                       "elapsed_s": round(time.time() - t0)}
            model.train()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"model": model.state_dict(), "arm": args.arm,
                        "config": cfg.__dict__}, out / "ckpt.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
