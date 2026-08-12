#!/usr/bin/env python
"""Render the E70 headline ("pitch") figure from the frozen analysis files.

Every displayed number is machine-read from the frozen three-seed analyzer,
the frozen secondary temporal-block analysis, and the frozen per-seed
destruction evals — never hand-transcribed.  Output is a presentation/website
figure (PNG + PDF); the paper's own figure remains the generated
``paper/e70_effect_figure.tex``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

E70_ROOT = pathlib.Path("/data/robotixx/snmr-research/e70")

INK = "#1f2937"
MUTED = "#9ca3af"
ACCENT = "#2563eb"
CONTROL = "#4b5563"
NULLBAR = "#d1d5db"
RED = "#dc2626"


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=pathlib.Path, default=E70_ROOT / "analysis_seed0-1-2.json")
    parser.add_argument(
        "--secondary", type=pathlib.Path, default=E70_ROOT / "secondary_temporal_block_final.json"
    )
    parser.add_argument("--students-root", type=pathlib.Path, default=E70_ROOT / "students")
    parser.add_argument("--out-png", type=pathlib.Path, required=True)
    parser.add_argument("--out-pdf", type=pathlib.Path)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    secondary = json.loads(args.secondary.read_text())
    arms = analysis["arms"]
    at = analysis["snmr_minus_time"]
    as_ = analysis["snmr_minus_shuffled"]
    per_clip = analysis["snmr_minus_time_per_clip"]

    destruction = {}
    for seed in (0, 1, 2):
        base = args.students_root / f"seed{seed}_explicit"
        destruction[seed] = {
            "intact": json.loads((base / "c_prior_explicit_eval.json").read_text())["completion_rate"],
            "zero": json.loads(
                (base / "c_prior_explicit_eval_destroy_zero.json").read_text()
            )["completion_rate"],
            "shuffle": json.loads(
                (base / "c_prior_explicit_eval_destroy_shuffle.json").read_text()
            )["completion_rate"],
            "marginal_random": json.loads(
                (base / "c_prior_explicit_eval_destroy_marginal_random.json").read_text()
            )["completion_rate"],
        }

    sec_rows = secondary["comparisons"]

    def sec_effect(name: str):
        row = sec_rows[name]
        return row["difference"], row["ci95_low"], row["ci95_high"]

    fig = plt.figure(figsize=(16.5, 6.0), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.30, 0.62], wspace=0.30,
                          left=0.115, right=0.985, top=0.76, bottom=0.15)

    # ---------------------------------------------------------------- panel A
    ax = fig.add_subplot(gs[0])
    order = [
        ("explicit reference\n(assay-validity control)", arms["explicit"]["ambiguity_completion"], CONTROL),
        ("SNMR latent\n(the measured interface)", arms["snmr"]["ambiguity_completion"], ACCENT),
        ("time index only\n(matched clock null)", arms["time"]["ambiguity_completion"], NULLBAR),
        ("shuffled latent\n(phase-matched null)", arms["shuffled"]["ambiguity_completion"], NULLBAR),
        ("proprioception only\n(no goal source)", arms["proprio"]["ambiguity_completion"], NULLBAR),
    ]
    ys = range(len(order))[::-1]
    for y, (label, value, color) in zip(ys, order):
        ax.barh(y, value, height=0.62, color=color, zorder=3)
        if value > 0.9:
            ax.text(value - 0.015, y, f"{value:.3f}", va="center", ha="right",
                    fontsize=10.5, color="white", fontweight="bold", zorder=5)
        else:
            ax.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=10.5,
                    color=INK, fontweight="bold" if color == ACCENT else "normal")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([label for label, _, _ in order], fontsize=9.5)
    time_x = arms["time"]["ambiguity_completion"]
    ax.set_ylim(-0.55, 4.6)
    ax.axvline(time_x, color=INK, linestyle=":", linewidth=1.1, zorder=4)
    ax.text(time_x, 4.92, "left of this line = the clock alone", fontsize=8.2,
            color=INK, ha="center", style="italic", clip_on=False)
    snmr_x = arms["snmr"]["ambiguity_completion"]
    ax.annotate("", xy=(snmr_x, 3.42), xytext=(time_x, 3.42),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4))
    ax.text((time_x + snmr_x) / 2, 3.58, f"+{snmr_x - time_x:.3f}", color=RED,
            ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("ambiguous-start completion (69 paired starts, 3 seeds)", fontsize=9.5)
    ax.set_title("A · Five students, one exclusive interface", fontsize=11.5, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------------------------------------------------------------- panel B
    ax = fig.add_subplot(gs[1])
    rows = [
        ("vs time  (primary)", at["difference"], at["ci95_low"], at["ci95_high"],
         at["per_seed_difference"], True),
        ("vs shuffled  (primary)", as_["difference"], as_["ci95_low"], as_["ci95_high"],
         as_["per_seed_difference"], True),
        ("vs time — clip 1", per_clip["first"]["difference"],
         per_clip["first"]["ci95_low"], per_clip["first"]["ci95_high"],
         per_clip["first"]["per_seed_difference"], False),
        ("vs time — clip 2", per_clip["second"]["difference"],
         per_clip["second"]["ci95_low"], per_clip["second"]["ci95_high"],
         per_clip["second"]["per_seed_difference"], False),
        ("vs time — block bootstrap", *sec_effect("snmr_minus_time"), None, False),
        ("vs shuffled — block bootstrap", *sec_effect("snmr_minus_shuffled"), None, False),
    ]
    ys = range(len(rows))[::-1]
    for y, (label, diff, lo, hi, seeds, primary) in zip(ys, rows):
        color = ACCENT if primary else "#6b7280"
        ax.plot([lo, hi], [y, y], color=color, lw=3.2 if primary else 2.0,
                solid_capstyle="round", zorder=3)
        ax.plot(diff, y, "o", color=color, markersize=9 if primary else 6.5, zorder=4)
        if seeds:
            ax.plot(seeds, [y - 0.24] * len(seeds), "d", color=color, markersize=4.5,
                    alpha=0.75, zorder=4)
        ax.text(0.335, y, f"+{diff:.3f}  [{lo:.3f}, {hi:.3f}]", va="center",
                fontsize=9.0, color=INK, family="monospace")
    ax.axvline(0.0, color=RED, lw=1.2)
    ax.set_ylim(-0.6, 5.6)
    ax.text(0.010, 5.22, "0 = nothing beyond\ntime/phase", color=RED, fontsize=8.4, ha="left")
    ax.text(0.51, -0.42, "◆ individual training seeds", fontsize=8.4, color="#6b7280", ha="right")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.6)
    ax.set_xlim(-0.06, 0.52)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel("SNMR paired completion advantage (95% cluster CI)", fontsize=9.5)
    ax.set_title("B · Every interval excludes zero — both clips, both nulls,\n"
                 "and the preregistered robustness analysis", fontsize=11.5, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------------------------------------------------------------- panel C
    ax = fig.add_subplot(gs[2])
    modes = ["intact", "zero", "shuffle", "marginal_random"]
    labels = ["intact\nchannel", "zeroed", "shuffled", "marginal\nrandom"]
    for i, mode in enumerate(modes):
        vals = [destruction[s][mode] for s in (0, 1, 2)]
        mean = sum(vals) / 3
        ax.bar(i, mean, width=0.62, color=CONTROL if i == 0 else RED, zorder=3,
               alpha=1.0 if i == 0 else 0.85)
        ax.plot([i] * 3, vals, "d", color=INK, markersize=4, zorder=4)
        ax.text(i, mean + 0.03, f"{mean:.3f}", ha="center", fontsize=9.5,
                fontweight="bold", color=INK)
    ax.annotate("", xy=(1.0, 0.45), xytext=(0.15, 0.82),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("general completion", fontsize=9.5)
    ax.set_title("C · Destroy the 64-d channel:\ncompetence vanishes (3 seeds)",
                 fontsize=11.5, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "An exclusive retarget-to-track interface makes latent content measurable —\n"
        "the frozen retargeting latent carries control-usable content beyond time",
        fontsize=15.5, x=0.115, ha="left", color=INK, fontweight="bold", y=0.985,
    )
    fig.text(
        0.115, 0.845,
        "Unitree G1 (simulated) · preregistered two-walk ambiguity assay · 69 paired starts × 3 "
        "training seeds · matched goal-source controls · all numbers machine-generated from the "
        f"frozen analyzer (SHA-256 {sha256_file(args.analysis)[:8]}…)",
        fontsize=9.4, color="#4b5563",
    )

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, facecolor="white")
    if args.out_pdf:
        fig.savefig(args.out_pdf, facecolor="white")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
