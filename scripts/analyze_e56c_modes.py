#!/usr/bin/env python3
"""E56-C v3 final analysis: is the sampled spread STRUCTURED (mode coverage) or noise?

G2's spread ratio alone cannot distinguish "the head reproduces the sibling modes" from
"the head adds isotropic noise of the right magnitude". Two discriminators, both on the
final checkpoint:

D1 (spread contrast): sample-std on held-out OMNI sibling frames vs on LAFAN val frames
   (near-Dirac conditional). Genuine conditional multimodality => omni >> lafan spread;
   uniform noise => equal.
D2 (mode coverage): for each held-out sibling group, decode K=16 samples from the
   SHARED human motion's z_ret; for each sibling trajectory, distance to the NEAREST
   sample (per-frame dof RMSE). Compare against the deterministic decoder's distance
   to each sibling. Coverage means every sibling has a nearby sample — the deterministic
   decoder by construction sits at the mean and misses off-mean siblings.

Run after runs/e56c_meanflow_v3/ckpt.pt exists (uses the same holdout split, seed 0).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_e55a_twoteacher import load_lafan, load_omni  # noqa: E402
from train_e56c_meanflow import MeanFlowHead, encode_pool, sample_nfe  # noqa: E402
from train_e56d_variant_code import holdout_sibling_groups  # noqa: E402
from snmr.human import human_pose_features  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.paths import robot_mjcf  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402


def main() -> None:
    device = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    ck = torch.load("runs/e56d_nocode/ckpt.pt", map_location=device)
    snmr = SNMR(SNMRConfig(**ck["config"])).to(device)
    snmr.load_state_dict(ck["model"]); snmr.eval()
    for p in snmr.parameters():
        p.requires_grad_(False)
    kin = RobotKinematics(str(robot_mjcf("unitree_g1")), device=device)

    hck = torch.load("runs/e56c_meanflow_v3/ckpt.pt", map_location=device)
    head = MeanFlowHead().to(device)
    head.load_state_dict(hck["head"]); head.eval()
    x_mean = hck["x_mean"].to(device); x_std = hck["x_std"].to(device)

    lafan_tr, lafan_va = load_lafan(device)
    omni_tr, omni_va = load_omni(device)
    sib_groups = holdout_sibling_groups(omni_tr, rng)  # same split as training run

    with torch.no_grad():
        # --- D1: spread contrast (sample std per frame, averaged) -----------------
        Z_la, _ = encode_pool(snmr, [lafan_va], device)
        idx = torch.from_numpy(rng.choice(Z_la.shape[0], 2000, replace=False))
        s_la = sample_nfe(head, Z_la[idx].to(device), x_mean, x_std, K=16, nfe=4)
        spread_lafan = float(s_la.std(0).mean())

        omni_frames, results = [], {"groups": []}
        for pool_name, groups in sib_groups.items():
            pool = next(p for p in ([lafan_tr] + omni_tr + omni_va)
                        if p.name.startswith(pool_name.split("_val")[0]))
            for stem, sibs in groups.items():
                T = min(min(c["human_pos"].shape[0] for c in sibs), 128)
                feat = human_pose_features(sibs[0]["human_pos"][:T],
                                           sibs[0]["human_quat"][:T])
                z = snmr.encode(feat, pool.static, pool.adj)
                omni_frames.append(z.cpu())
                draws = sample_nfe(head, z, x_mean, x_std, K=16, nfe=4)  # (16,T,29)
                det = snmr.decode(z, kin)["dof_pos"]                      # (T,29)
                grp = {"stem": stem, "n_sibs": len(sibs),
                       "data_spread": float(torch.stack(
                           [c["qpos"][:T, 7:] for c in sibs]).std(0).mean()),
                       "sample_spread": float(draws.std(0).mean())}
                near, detd = [], []
                for c in sibs:
                    gt = c["qpos"][:T, 7:]
                    d_each = (draws - gt[None]).square().mean(dim=(1, 2)).sqrt()  # (16,)
                    near.append(float(d_each.min()))
                    detd.append(float((det - gt).square().mean().sqrt()))
                grp["nearest_sample_rmse"] = near
                grp["deterministic_rmse"] = detd
                grp["coverage_gain"] = float(np.mean(detd) - np.mean(near))
                results["groups"].append(grp)

        Z_om = torch.cat(omni_frames)
        idx2 = torch.from_numpy(rng.choice(Z_om.shape[0], min(2000, Z_om.shape[0]), replace=False))
        s_om = sample_nfe(head, Z_om[idx2].to(device), x_mean, x_std, K=16, nfe=4)
        spread_omni = float(s_om.std(0).mean())

    results["D1"] = {"spread_lafan_val": spread_lafan, "spread_omni_sibs": spread_omni,
                     "ratio_omni_over_lafan": spread_omni / max(spread_lafan, 1e-9)}
    gains = [g["coverage_gain"] for g in results["groups"]]
    results["D2"] = {"mean_coverage_gain_rad": float(np.mean(gains)),
                     "groups_with_positive_gain": int(sum(g > 0 for g in gains)),
                     "n_groups": len(gains)}
    out = pathlib.Path("runs/e56c_meanflow_v3/mode_analysis.json")
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({"D1": results["D1"], "D2": results["D2"]}, indent=2))


if __name__ == "__main__":
    main()
