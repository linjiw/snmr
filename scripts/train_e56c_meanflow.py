#!/usr/bin/env python3
"""E56-C: MeanFlow generative head over dof-space — does sampling explain the
object-sibling multimodality that conditioning cannot?

Chain of evidence: E47 (single-teacher conditional is near-Dirac -> generative decoder
pointless) -> E55-A (two-teacher pool creates real multimodality: identical-human
siblings diverge 0.031-0.068 rad) -> E56-D (4-d variant code recovers 43% of spread,
gates fail) -> E56-D2 (exact per-frame object pose recovers only 6% of OBJECT-sibling
spread while a z-scale scalar recovers 52% of terrain spread -> object multimodality is
NOT hidden-variable-explained). E56-C is the registered response: model
p(dof | z_ret, embodiment) with a MeanFlow head (2505.13447) and ask whether SAMPLING
reproduces the sibling spread.

Why MeanFlow specifically (registered rationale): at a Dirac conditional the bootstrap
term (t-r)*du/dt vanishes identically and the objective collapses to regression — the
downside is bounded at deterministic-baseline behavior, the property that makes this
safe after E43-47.

Design:
  frozen two-teacher SNMR (runs/e56d_nocode/ckpt.pt) provides z_ret per frame (128-d);
  the head models x = dof_pos (29-d, standardized) with average-velocity field
  u_theta(z_t, r, t | z_ret): z_t = (1-t) x + t eps, v = eps - x,
  u_tgt = v - (t-r) * JVP(u, (z_t,r,t), (v,0,1)), loss ||u - sg(u_tgt)||^2,
  25% of samples with r<t (rest r=t, where u=v exactly).
  1-NFE sampling: x_hat = eps - u(eps, r=0, t=1).

Pre-specified gates (docs/E55_E57_SCALING_AND_DISCRIMINATION.md, E56-C section):
  G1 (implementation check): 1-NFE mean-sample dof-RMSE on val within 10% relative of
      the frozen deterministic decoder on the same frames (adaptation of the +0.3cm
      MPJPE gate to the dof-space instrument used by E56-D/D2).
  G2 (the question): K=8 samples per frame on held-out sibling groups reproduce >=50%
      of the data's sibling spread (E56-D2 conditioning floor to beat: 6% object /
      52% terrain). Report object/terrain split like E56-D2.
Decision: G2 pass -> generative head justified, interaction data needs it (paper
finding); G2 fail after G1 pass -> the generative line closes permanently per the
registered kill rule (the spread is neither conditionable nor samplable at this scale).

Runs in .venv-snmr, tiny model (<1 GB), coexists with the E53-2048 GPU job.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_e55a_twoteacher import load_lafan, load_omni  # noqa: E402
from train_e56d_variant_code import holdout_sibling_groups  # noqa: E402
from snmr.human import human_pose_features  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.paths import robot_mjcf  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

DOF = 29
ZDIM = 128


class FourierTime(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.register_buffer("freqs", 2 ** torch.arange(dim // 2).float() * math.pi)

    def forward(self, t):  # (B,) -> (B, dim)
        a = t[:, None] * self.freqs[None]
        return torch.cat([a.sin(), a.cos()], -1)


class MeanFlowHead(nn.Module):
    """v3: cond+time re-injected at every layer (FiLM-lite via concat), hidden 1024."""

    def __init__(self, hidden=1024, layers=4):
        super().__init__()
        self.temb = FourierTime(32)
        ctx = ZDIM + 2 * 32
        self.inp = nn.Linear(DOF + ctx, hidden)
        self.blocks = nn.ModuleList(
            [nn.Linear(hidden + ctx, hidden) for _ in range(layers - 1)])
        self.out = nn.Linear(hidden, DOF)

    def forward(self, z_t, cond, r, t):
        ctx = torch.cat([cond, self.temb(r), self.temb(t)], -1)
        h = torch.nn.functional.silu(self.inp(torch.cat([z_t, ctx], -1)))
        for blk in self.blocks:
            h = torch.nn.functional.silu(blk(torch.cat([h, ctx], -1)))
        return self.out(h)


@torch.no_grad()
def encode_pool(model, pools, device, window=64):
    """Frozen per-frame z for every clip; returns list of (z_cpu, x_cpu) tensors."""
    Z, X = [], []
    for pool in pools:
        for c in pool.clips:
            T = c["human_pos"].shape[0]
            zs = []
            for s in range(0, T, window):
                e = min(T, s + window)
                feat = human_pose_features(c["human_pos"][s:e], c["human_quat"][s:e])
                zs.append(model.encode(feat, pool.static, pool.adj))
            Z.append(torch.cat(zs).cpu())
            X.append(c["qpos"][:, 7:].cpu())
    return torch.cat(Z), torch.cat(X)


@torch.no_grad()
def sample_nfe(head, cond, x_mean, x_std, K=1, nfe=1, generator=None):
    """(F, ZDIM) cond -> (K, F, DOF); nfe uniform backward steps t: 1->0."""
    F = cond.shape[0]
    outs = []
    ts = torch.linspace(1.0, 0.0, nfe + 1, device=cond.device)
    for _ in range(K):
        z = torch.randn(F, DOF, device=cond.device, generator=generator)
        for i in range(nfe):
            t_hi, t_lo = ts[i], ts[i + 1]
            r = torch.full((F,), float(t_lo), device=cond.device)
            t = torch.full((F,), float(t_hi), device=cond.device)
            z = z - (t_hi - t_lo) * head(z, cond, r, t)
        outs.append(z * x_std + x_mean)
    return torch.stack(outs)


def sample_nfe1(head, cond, x_mean, x_std, K=1, generator=None):
    return sample_nfe(head, cond, x_mean, x_std, K=K, nfe=1, generator=generator)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--r_neq_t", type=float, default=0.25)
    ap.add_argument("--warmup_regression", type=int, default=20000,
                    help="steps with r=t only (pure regression) before the bootstrap term")
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = pathlib.Path("runs/e56c_meanflow_v3")
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # frozen two-teacher SNMR (E56-D nocode arm = plain recipe, full pool, 40k)
    ck = torch.load("runs/e56d_nocode/ckpt.pt", map_location=device)
    cfg = SNMRConfig(**{k: v for k, v in ck["config"].items()})
    snmr = SNMR(cfg).to(device)
    snmr.load_state_dict(ck["model"])
    snmr.eval()
    for p in snmr.parameters():
        p.requires_grad_(False)
    kin = RobotKinematics(str(robot_mjcf("unitree_g1")), device=device)

    lafan_tr, lafan_va = load_lafan(device)
    omni_tr, omni_va = load_omni(device)
    sib_groups = holdout_sibling_groups(omni_tr, rng)  # same instrument as D/D2
    print("encoding pools with frozen SNMR...", flush=True)
    Z_tr, X_tr = encode_pool(snmr, [lafan_tr] + omni_tr, device)
    Z_va, X_va = encode_pool(snmr, [lafan_va], device)
    x_mean = X_tr.mean(0).to(device)
    x_std = (X_tr.std(0) + 1e-6).to(device)
    N = Z_tr.shape[0]
    print(f"train frames {N}, val frames {Z_va.shape[0]}", flush=True)

    head = MeanFlowHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    log = (out / "log.jsonl").open("a")

    # deterministic baseline for G1: frozen decoder dof-RMSE on lafan val frames
    with torch.no_grad():
        base_err, seen = 0.0, 0
        for c in lafan_va.clips:
            feat = human_pose_features(c["human_pos"], c["human_quat"])
            z = snmr.encode(feat, lafan_va.static, lafan_va.adj)
            d = snmr.decode(z, kin)["dof_pos"]
            base_err += float((d - c["qpos"][:, 7:]).square().sum())
            seen += d.numel()
        baseline_rmse = math.sqrt(base_err / seen)
    print(f"G1 baseline (frozen deterministic decoder, lafan val): {baseline_rmse:.4f} rad",
          flush=True)

    def u_wrapped(z_t, r, t, cond):
        return head(z_t, cond, r, t)

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.from_numpy(rng.integers(0, N, args.batch))
        x = ((X_tr[idx].to(device) - x_mean) / x_std)
        cond = Z_tr[idx].to(device)
        eps = torch.randn_like(x)
        t = torch.rand(args.batch, device=device)
        # E56-C v2 stabilization (v1 diverged, loss 4e2 -> 1e13 by 10k):
        # (a) regression warmup: r=t (u_tgt=v exactly) for the first N steps;
        # (b) clamp the bootstrap target to a ball around v (JVP blowup guard);
        # (c) MeanFlow's own adaptive weighting w = 1/(err^2 + c)^p (Eq. 22, p=1).
        if step < args.warmup_regression:
            r = t
        else:
            r = torch.where(torch.rand(args.batch, device=device) < args.r_neq_t,
                            torch.rand(args.batch, device=device) * t, t)
        z_t = (1 - t)[:, None] * x + t[:, None] * eps
        v = eps - x
        u, dudt = jvp(lambda zz, rr, tt: u_wrapped(zz, rr, tt, cond),
                      (z_t, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        u_tgt = (v - (t - r)[:, None] * dudt).detach()
        u_tgt = v + (u_tgt - v).clamp(-5.0, 5.0)
        err2 = (u - u_tgt).square().mean(-1)
        w = (1.0 / (err2.detach() + 1e-3))
        loss = (w * err2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            head.eval()
            with torch.no_grad():
                # G1 at multiple NFE (v3): mean over 4 draws
                g1_by_nfe = {}
                for nfe in (1, 2, 4):
                    s = sample_nfe(head, Z_va.to(device), x_mean, x_std, K=4, nfe=nfe)
                    g1_by_nfe[nfe] = float((s.mean(0) - X_va.to(device)).square().mean().sqrt())
                g1 = g1_by_nfe[4]
                # G2: sibling-spread recovery on held-out groups (object/terrain split)
                fam = {"object": [], "terrain": []}
                for pool_name, groups in sib_groups.items():
                    pool = next(p for p in ([lafan_tr] + omni_tr + omni_va)
                                if p.name.startswith(pool_name.split("_val")[0]))
                    for stem, sibs in groups.items():
                        T = min(min(c["human_pos"].shape[0] for c in sibs), 128)
                        feat = human_pose_features(sibs[0]["human_pos"][:T],
                                                   sibs[0]["human_quat"][:T])
                        z = snmr.encode(feat, pool.static, pool.adj)
                        draws = sample_nfe(head, z, x_mean, x_std, K=8, nfe=4)
                        dec_spread = float(draws.std(dim=0).mean())
                        dat_spread = float(torch.stack(
                            [c["qpos"][:T, 7:] for c in sibs]).std(dim=0).mean())
                        family = "object" if "z_scale" not in sibs[0]["name"] else "terrain"
                        fam[family].append(dec_spread / max(dat_spread, 1e-6))
                rec = {"step": step, "loss": float(loss),
                       "G1_nfe1_rmse": g1_by_nfe[1], "G1_nfe2_rmse": g1_by_nfe[2],
                       "G1_nfe4_rmse": g1, "G1_baseline_rmse": baseline_rmse,
                       "G1_rel": g1 / baseline_rmse,
                       "G2_object": float(np.mean(fam["object"])) if fam["object"] else None,
                       "G2_terrain": float(np.mean(fam["terrain"])) if fam["terrain"] else None,
                       "elapsed_s": round(time.time() - t0)}
            head.train()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"head": head.state_dict(), "x_mean": x_mean.cpu(),
                        "x_std": x_std.cpu()}, out / "ckpt.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
