#!/usr/bin/env python
"""E81-B — can a *learned* motion prior fill an outage where every free fill fails?

E81-A established the regime boundary: model-free cycle continuation solves periodic motion
(0.08–0.10 rad, flat to 1.5 s) and fails on aperiodic motion (~0.30 rad at 1 s, the order of the
range of motion). This script asks whether the aperiodic regime is predictable *by anything* —
before asking whether SNMR's latent in particular is the right prior.

Design (deliberately generic, so a negative result is about the regime and not about SNMR):

* Predictor: a small MLP reading a causal history window of joint positions and velocities and
  emitting the reference at each horizon **directly** (multi-horizon regression rather than
  autoregression, so drift cannot be blamed for a failure).
* Split: clips, not frames — train and test never share a clip. Stratified over the periodic /
  aperiodic classes of E81-A.
* Baselines: the same three causal fills, evaluated on the identical test starts.

The gate this feeds (`docs/COMMAND_INTERFACE_SYNTHESIS_2026-08-16.md` Part IV, E81-B): a learned
prior must beat **0.30 rad at a 1 s outage on aperiodic clips**. If a generic learned prior cannot,
the ladder's rung 2 is impossible for anyone on that motion class and the framework says so; if it
can, SNMR has a concrete target to match with its own latent.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch



def collect_clips(directory: str, pattern: str, limit_per_class: int, classify) -> dict[str, list[str]]:
    import glob

    buckets: dict[str, list[str]] = {"periodic": [], "aperiodic": []}
    for path in sorted(glob.glob(str(pathlib.Path(directory) / pattern))):
        label = classify(pathlib.Path(path).name)
        if label in buckets and len(buckets[label]) < limit_per_class:
            buckets[label].append(path)
    return buckets


class HistoryDataset:
    """Windows of (history joints+velocities) -> (reference at each horizon)."""

    def __init__(self, clips: list[tuple[np.ndarray, np.ndarray, int]], history: int,
                 horizons: list[int], stride: int = 3):
        xs, ys, meta = [], [], []
        for ci, (q, qd, _fps) in enumerate(clips):
            hmax = max(horizons)
            for t in range(history, len(q) - hmax - 1, stride):
                hist_q = q[t - history + 1 : t + 1]
                hist_v = qd[t - history + 1 : t + 1]
                xs.append(np.concatenate([hist_q.ravel(), hist_v.ravel()]))
                ys.append(np.concatenate([q[t + h] - q[t] for h in horizons]))  # residual targets
                meta.append((ci, t))
        self.x = torch.tensor(np.asarray(xs), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(ys), dtype=torch.float32)
        self.meta = meta


def train_predictor(train: HistoryDataset, hidden: int, epochs: int, seed: int, device: str) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(train.x.shape[1], hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, train.y.shape[1]),
    ).to(device)
    mu, sd = train.x.mean(0), train.x.std(0) + 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = ((train.x - mu) / sd).to(device), train.y.to(device)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            idx = perm[i : i + 1024]
            loss = (model(x[idx]) - y[idx]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    model._norm = (mu.to(device), sd.to(device))  # type: ignore[attr-defined]
    return model


def evaluate(model, clips, history, horizons, device, samples_per_clip=150, seed=0) -> dict:
    """Per-horizon RMSE of the learned prior and the three causal baselines on the same starts."""
    rng = np.random.default_rng(seed)
    mu, sd = model._norm
    acc = {k: [[] for _ in horizons] for k in ("learned", "hold", "cv", "cycle")}
    for q, qd, fps in clips:
        hmax = max(horizons)
        min_lag, max_lag, match = int(0.5 * fps), int(1.6 * fps), 20
        lo, hi = max(history, max_lag + match), len(q) - hmax - 1
        if hi <= lo:
            continue
        t0s = rng.choice(np.arange(lo, hi), size=min(samples_per_clip, hi - lo), replace=False)
        # learned
        xs = np.stack([
            np.concatenate([q[t - history + 1 : t + 1].ravel(), qd[t - history + 1 : t + 1].ravel()])
            for t in t0s
        ])
        with torch.no_grad():
            x = ((torch.tensor(xs, dtype=torch.float32).to(device) - mu) / sd)
            pred = model(x).cpu().numpy()
        # cycle lag
        recent = np.stack([q[t0s - k] for k in range(match)], axis=1)
        lags = np.arange(min_lag, max_lag + 1)
        err = np.stack([((recent - np.stack([q[t0s - k - lag] for k in range(match)], axis=1)) ** 2)
                        .mean(axis=(1, 2)) for lag in lags], axis=1)
        best_lag = lags[err.argmin(axis=1)]
        for j, h in enumerate(horizons):
            target = q[t0s + h]
            d = q.shape[1]
            acc["learned"][j].append(np.sqrt(((target - (q[t0s] + pred[:, j * d : (j + 1) * d])) ** 2).mean(1)).mean())
            acc["hold"][j].append(np.sqrt(((target - q[t0s]) ** 2).mean(1)).mean())
            acc["cv"][j].append(np.sqrt(((target - (q[t0s] + qd[t0s] * (h / fps))) ** 2).mean(1)).mean())
            acc["cycle"][j].append(np.sqrt(((target - q[t0s + h - best_lag]) ** 2).mean(1)).mean())
    return {k: [float(np.mean(v)) if v else float("nan") for v in rows] for k, rows in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motions", required=True)
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--train-clips", type=int, default=80, help="per class")
    ap.add_argument("--test-clips", type=int, default=20, help="per class, disjoint from train")
    ap.add_argument("--history-ticks", type=int, default=20)
    ap.add_argument("--horizons-s", default="0.1,0.2,0.5,1.0,1.5")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "e81a", pathlib.Path(__file__).resolve().parent / "e81_fill_prediction_error.py"
    )
    e81a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e81a)

    # Collect generously and split *after* loading: many pool clips are shorter than the
    # window the evaluation needs, so a path-level limit would silently starve the test split.
    need = args.train_clips + args.test_clips
    buckets = collect_clips(args.motions, args.pattern, need * 6, e81a.classify)
    loaded = {k: [] for k in buckets}
    for label, paths in buckets.items():
        for p in paths:
            if len(loaded[label]) >= need:
                break
            got = e81a.load_joint_series(p)
            if got is not None:
                loaded[label].append(got)
        print(f"  {label}: {len(loaded[label])} clips usable of {len(paths)} candidates")
    fps = loaded["periodic"][0][2]
    horizons = [int(round(float(h) * fps)) for h in args.horizons_s.split(",")]

    splits = {}
    for label, clips in loaded.items():
        splits[label] = {"train": clips[: args.train_clips], "test": clips[args.train_clips :]}
    train_clips = splits["periodic"]["train"] + splits["aperiodic"]["train"]
    print(f"train clips {len(train_clips)}  test periodic {len(splits['periodic']['test'])} "
          f"aperiodic {len(splits['aperiodic']['test'])}")

    train = HistoryDataset(train_clips, args.history_ticks, horizons)
    print(f"training windows {len(train.x)}  input {train.x.shape[1]}  output {train.y.shape[1]}")
    model = train_predictor(train, args.hidden, args.epochs, seed=0, device=args.device)

    report = {"horizons_s": [float(h) for h in args.horizons_s.split(",")], "results": {}}
    for label in ("periodic", "aperiodic"):
        res = evaluate(model, splits[label]["test"], args.history_ticks, horizons, args.device)
        report["results"][label] = res
        print(f"\n{label} (held-out clips)")
        print(f"{'horizon':>9s}" + "".join(f"{k:>10s}" for k in ("hold", "cv", "cycle", "learned")))
        for i, h in enumerate(report["horizons_s"]):
            print(f"{h:8.2f}s" + "".join(f"{res[k][i]:10.3f}" for k in ("hold", "cv", "cycle", "learned")))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
