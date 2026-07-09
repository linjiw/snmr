#!/usr/bin/env python
"""Ablation-study driver (design doc §6 + risk table hypotheses).

Runs a defined grid of Phase-1 (single-robot G1) training variants sequentially, then benchmarks
each with `scripts/benchmark.py`, writing one summary table. GPU-friendly: waits for a free GPU
window (checks for other python training processes) before each run, and skips variants whose
output dir already has a final checkpoint (resume-safe: re-running the driver continues the grid).

Grid (each ablates one design decision against the Phase-1 winning recipe
z=128/hidden=256/lr 3e-4/50k steps — 50k not 100k to fit the grid in a night; the winning recipe's
own 50k row makes comparisons internally consistent):

  base           — winning recipe at 50k steps (the control row)
  no_temporal    — encoder without the temporal transformer (frame-independent latents)
  z32            — latent bottleneck 32 (SAME's original size) instead of 128
  small          — 0.41M model (z=64/hidden=128): capacity ablation at fixed steps
  contact        — + foot-contact loss (weight 0.1), tests the physical-plausibility term
  lr8e4          — lr 8e-4 (documents the instability finding as a grid row)

    python scripts/run_ablations.py --steps 50000            # run everything
    python scripts/run_ablations.py --only base no_temporal  # subset
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable

ABLATIONS: dict[str, list[str]] = {
    "base":        ["--latent_dim", "128", "--enc_hidden", "256", "--dec_hidden", "256", "--lr", "3e-4"],
    "no_temporal": ["--latent_dim", "128", "--enc_hidden", "256", "--dec_hidden", "256", "--lr", "3e-4",
                     "--no_temporal"],
    "z32":         ["--latent_dim", "32", "--enc_hidden", "256", "--dec_hidden", "256", "--lr", "3e-4"],
    "small":       ["--latent_dim", "64", "--enc_hidden", "128", "--dec_hidden", "128", "--lr", "3e-4"],
    "contact":     ["--latent_dim", "128", "--enc_hidden", "256", "--dec_hidden", "256", "--lr", "3e-4",
                     "--contact_weight", "0.1"],
    "lr8e4":       ["--latent_dim", "128", "--enc_hidden", "256", "--dec_hidden", "256", "--lr", "8e-4"],
}


def gpu_busy() -> bool:
    """True if another training process is using the GPU (avoid contending with Phase-2 runs)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        return False
    return bool(out)


def wait_for_gpu(poll_s: int = 300) -> None:
    while gpu_busy():
        print(f"[ablations] GPU busy; sleeping {poll_s}s ...", flush=True)
        time.sleep(poll_s)


def run_variant(name: str, extra: list[str], steps: int, out_root: pathlib.Path) -> dict | None:
    out = out_root / name
    ckpt = out / "ckpt.pt"
    bench_json = out / "benchmark.json"

    if not ckpt.exists():
        wait_for_gpu()
        print(f"[ablations] training '{name}' ...", flush=True)
        cmd = [PY, str(ROOT / "scripts" / "train_phase1.py"),
               "--steps", str(steps), "--eval_every", str(max(steps // 10, 1000)),
               "--ckpt_every", str(steps), "--out", str(out), *extra]
        res = subprocess.run(cmd, capture_output=True, text=True)
        (out / "train_stdout.log").parent.mkdir(parents=True, exist_ok=True)
        (out / "train_stdout.log").write_text(res.stdout[-20000:] + "\n--- stderr ---\n" + res.stderr[-20000:])
        if res.returncode != 0 or not ckpt.exists():
            print(f"[ablations] '{name}' FAILED (rc={res.returncode}); see {out}/train_stdout.log")
            return None
    else:
        print(f"[ablations] '{name}': checkpoint exists, skipping training")

    if not bench_json.exists():
        print(f"[ablations] benchmarking '{name}' ...", flush=True)
        res = subprocess.run(
            [PY, str(ROOT / "scripts" / "benchmark.py"), "--ckpt", str(ckpt),
             "--robots", "unitree_g1", "--out", str(out / "benchmark")],
            capture_output=True, text=True,
        )
        if res.returncode != 0 or not bench_json.exists():
            print(f"[ablations] benchmark for '{name}' FAILED: {res.stderr[-2000:]}")
            return None

    return json.loads(bench_json.read_text())["unitree_g1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--out_root", default=str(ROOT / "runs" / "ablations"))
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    names = args.only or list(ABLATIONS)

    results = {}
    for name in names:
        res = run_variant(name, ABLATIONS[name], args.steps, out_root)
        if res is not None:
            results[name] = res

    # summary table
    cols = [("mpjpe_m", "MPJPE (m)"), ("dof_err_rad", "dof err"),
            ("foot_skate_speed_ms", "skate (m/s)"), ("foot_slide_fraction", "slide frac"),
            ("dof_jerk", "dof jerk"), ("body_jerk", "body jerk")]
    lines = [f"# Ablation study — G1, {args.steps} steps each, held-out clips", "",
             "| variant | " + " | ".join(c[1] for c in cols) + " | fps |",
             "|---|" + "---|" * (len(cols) + 1)]
    for name, res in results.items():
        row = [f"{res['snmr'].get(k, float('nan')):.4f}" for k, _ in cols]
        lines.append(f"| {name} | " + " | ".join(row) + f" | {res['throughput_fps']:.0f} |")
    if results:
        first = next(iter(results.values()))
        t = first["teacher"]
        lines.append("| *teacher (GMR)* | — | — | "
                     f"{t['foot_skate_speed_ms']:.4f} | {t['foot_slide_fraction']:.4f} | "
                     f"{t['dof_jerk']:.1f} | {t['body_jerk']:.1f} | ~160 (CPU) |")
    md = "\n".join(lines)
    (out_root / "SUMMARY.md").write_text(md)
    print("\n" + md)
    print(f"\nwrote {out_root}/SUMMARY.md")


if __name__ == "__main__":
    main()
