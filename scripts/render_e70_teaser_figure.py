#!/usr/bin/env python
"""Render Figure 1 as three panels of the ARGUMENT, from frozen data.

The previous teaser showed the system (pipeline, architecture, a bar chart). It did not show
the experiment. The single most novel object this work builds -- matched present states with
divergent futures -- appeared in no panel of any figure.

This renders instead:
  (a) THE CONFOUND    on one deterministic clip, time indexes the entire target.
  (b) THE CONSTRUCTION the real 69 selected windows, present distance against future distance,
                       with both registered thresholds drawn. Every point is frozen data.
  (c) THE MEASUREMENT  the five arms and the two paired contrasts, with per-seed points.

Panels (b) and (c) are plotted from hash-stamped frozen artifacts; panel (a) is schematic and
carries no numbers, so no value in this figure is hand-transcribed. Numeric prose about the
single-clip diagnostic stays in the text, where its own lineage is cited.

Slots are left after panel (c) so a later result appends rather than forcing a redesign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


PRECHECK_PROTOCOL = "E70 reference-only ambiguity precheck v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scatter(windows: list[dict], thresholds: dict) -> tuple[list[str], dict]:
    """Panel (b): present distance (x) against future distance (y), in mm inside the panel."""
    xs = [float(w["state_distance"]) for w in windows]
    ys = [float(w["future_distance"]) for w in windows]
    x_max = float(thresholds["max_state_distance_rms_z"])
    y_min = float(thresholds["min_future_distance_rms_z"])

    # Frame the data rather than the origin: every present distance is >= 0.5 by construction,
    # so an axis anchored at zero would leave most of the panel empty.
    x_pad = (x_max - min(xs)) * 0.18
    x_lo, x_hi = min(xs) - x_pad, x_max + x_pad
    y_pad = (max(ys) - min(ys)) * 0.12
    y_lo, y_hi = min(y_min, min(ys)) - y_pad, max(ys) + y_pad
    W, H = 38.0, 28.0  # panel size in mm

    def px(v: float) -> float:
        return (v - x_lo) / (x_hi - x_lo) * W

    def py(v: float) -> float:
        return (v - y_lo) / (y_hi - y_lo) * H

    lines = []
    # Shade the registered admissible region: present <= threshold AND future >= threshold.
    lines.append(
        rf"\fill[washgreen] (0mm,{py(y_min):.2f}mm) rectangle "
        rf"({px(x_max):.2f}mm,{H:.2f}mm);"
    )
    # Threshold lines.
    lines.append(
        rf"\draw[corange, dashed, thin] ({px(x_max):.2f}mm,0mm) -- ({px(x_max):.2f}mm,{H:.2f}mm);"
    )
    lines.append(
        rf"\draw[corange, dashed, thin] (0mm,{py(y_min):.2f}mm) -- ({W:.2f}mm,{py(y_min):.2f}mm);"
    )
    # Axes.
    lines.append(rf"\draw[cink2!60] (0mm,0mm) -- ({W:.2f}mm,0mm);")
    lines.append(rf"\draw[cink2!60] (0mm,0mm) -- (0mm,{H:.2f}mm);")
    # The 69 windows.
    for x, y in zip(xs, ys):
        lines.append(rf"\fill[cgreen!80!black, opacity=0.72] ({px(x):.2f}mm,{py(y):.2f}mm) circle (0.62mm);")
    meta = {"W": W, "H": H, "x_max_mm": px(x_max), "y_min_mm": py(y_min), "n": len(xs)}
    return lines, meta


def _bars(arms: dict) -> list[str]:
    """Panel (c) upper: ambiguity completion per arm, drawn from the frozen analyzer."""
    order = [
        ("explicit", "explicit goal", "cblue"),
        ("snmr", r"frozen $z_{\mathrm{ret}}$", "cgreen!75!black"),
        ("time", "absolute time", "corange!65!black"),
        ("shuffled", "matched-phase shuffle", "corange"),
        ("proprio", "proprioception only", "cink2!55"),
    ]
    scale = 34.0  # mm per unit completion
    lines = []
    for i, (key, label, color) in enumerate(order):
        v = float(arms[key]["ambiguity_completion"])
        y = -5.0 * i
        lines.append(rf"\node[anchor=east, font=\scriptsize] at (0mm,{y:.1f}mm) {{{label}}};")
        lines.append(
            rf"\fill[{color}, rounded corners=0.4pt] (1mm,{y - 1.5:.1f}mm) rectangle "
            rf"({1.0 + v * scale:.2f}mm,{y + 1.5:.1f}mm);"
        )
        lines.append(
            rf"\node[anchor=west, font=\scriptsize\bfseries] at ({1.6 + v * scale:.2f}mm,{y:.1f}mm) "
            rf"{{{v:.3f}}};"
        )
    return lines


def _contrasts(analysis: dict) -> list[str]:
    """Panel (c) lower: the two paired contrasts with per-seed points."""
    rows = [
        ("A--T", "snmr_minus_time", "snmr_minus_time"),
        ("A--S", "snmr_minus_shuffled", "snmr_minus_shuffled"),
    ]
    per_seed = analysis["per_seed_differences"]
    seeds = sorted(int(k) for k in per_seed)
    lo_all, hi_all = 0.0, 0.34  # fixed axis so both rows share a scale
    W = 34.0

    def px(v: float) -> float:
        return (v - lo_all) / (hi_all - lo_all) * W

    lines = [
        rf"\draw[cgrid] ({px(0.0):.2f}mm,-8.5mm) -- ({px(0.0):.2f}mm,3mm);",
        rf"\node[font=\scriptsize, text=cink2, anchor=north] at ({px(0.0):.2f}mm,-8.5mm) {{0}};",
    ]
    for i, (label, agg_key, seed_key) in enumerate(rows):
        y = -6.0 * i
        agg = analysis[agg_key]
        point, lo, hi = (float(agg["difference"]), float(agg["ci95_low"]), float(agg["ci95_high"]))
        lines.append(rf"\node[anchor=east, font=\scriptsize] at (-1mm,{y:.1f}mm) {{{label}}};")
        lines.append(
            rf"\draw[cink2, thick] ({px(lo):.2f}mm,{y:.1f}mm) -- ({px(hi):.2f}mm,{y:.1f}mm);"
        )
        for cap in (lo, hi):
            lines.append(
                rf"\draw[cink2, thick] ({px(cap):.2f}mm,{y - 1.0:.1f}mm) -- "
                rf"({px(cap):.2f}mm,{y + 1.0:.1f}mm);"
            )
        for s in seeds:
            sv = float(per_seed[str(s)][seed_key])
            lines.append(
                rf"\fill[cink2!45] ({px(sv):.2f}mm,{y + 2.1:.1f}mm) circle (0.5mm);"
            )
        lines.append(
            rf"\fill[cgreen!70!black] ({px(point):.2f}mm,{y:.1f}mm) circle (0.85mm);"
        )
        lines.append(
            rf"\node[anchor=west, font=\scriptsize\bfseries] at ({px(hi) + 1.2:.2f}mm,{y:.1f}mm) "
            rf"{{{point:+.3f}}};"
        )
    return lines


def render(precheck_path: pathlib.Path, analysis_path: pathlib.Path) -> str:
    precheck = json.loads(precheck_path.read_text())
    if precheck.get("protocol") != PRECHECK_PROTOCOL:
        raise ValueError(f"unexpected precheck protocol: {precheck.get('protocol')!r}")
    analysis = json.loads(analysis_path.read_text())
    if sorted(analysis.get("seeds", [])) != [0, 1, 2]:
        raise ValueError("the teaser figure requires the frozen three-seed analyzer")

    report = precheck["pairs"][precheck["preferred_pair"]]
    windows = report["windows"]
    if len(windows) != 69:
        raise ValueError(f"expected the 69 frozen windows, found {len(windows)}")

    scatter, meta = _scatter(windows, precheck["thresholds"])
    bars = _bars(analysis["arms"])
    contrasts = _contrasts(analysis)

    digests = [(str(p), sha256_file(p)) for p in (precheck_path, analysis_path)]
    header = [
        "% Generated by scripts/render_e70_teaser_figure.py; do not edit by hand.",
        f"% inputs_sha256={hashlib.sha256(''.join(d for _, d in digests).encode()).hexdigest()}",
    ]
    header += [f"% input_sha256[{name}]={digest}" for name, digest in digests]

    body = rf"""
\begin{{figure*}}[t]
\centering
\begin{{tikzpicture}}[font=\small,
  box/.style={{draw=cink2!60, fill=white, rounded corners=2pt, align=center,
              inner sep=3pt, minimum height=7mm, font=\footnotesize}},
  arr/.style={{-{{Stealth[length=1.8mm]}}, thick, cink2}},
]

% ---------------- panel (a): the confound ----------------
\begin{{scope}}
\node[font=\bfseries\footnotesize, anchor=west] at (0mm, 30mm)
  {{(a) The confound: one clip is a clock}};
% a single deterministic reference clip
\draw[cink2!70, line width=0.9pt] (2mm, 20mm) -- (44mm, 20mm);
\foreach \x in {{2,8,14,20,26,32,38,44}}{{
  \draw[cink2!45] (\x mm, 18.6mm) -- (\x mm, 21.4mm);
}}
\node[anchor=west, font=\scriptsize, text=cink2] at (2mm, 23.4mm)
  {{one deterministic reference clip}};
\node[anchor=east, font=\scriptsize] at (1mm, 20mm) {{$t$}};
% time -> whole target
\node[box, fill=washorange, draw=corange] (clk) at (12mm, 11mm) {{absolute time}};
\node[box] (tgt) at (36mm, 11mm) {{the entire\\target}};
\draw[arr, corange] (clk) -- node[midway, above, font=\scriptsize, text=corange]
  {{indexes}} (tgt);
\node[anchor=west, font=\scriptsize, text=cink2, align=left] at (0mm, 3.2mm)
  {{tracking success cannot attribute\\credit to any command channel}};
\end{{scope}}

% ---------------- panel (b): the construction ----------------
\begin{{scope}}[xshift=62mm]
\node[font=\bfseries\footnotesize, anchor=west] at (-6mm, 30mm)
  {{(b) The construction}};
\begin{{scope}}[yshift=0mm]
{chr(10).join(scatter)}
\node[anchor=north, font=\scriptsize] at ({meta['W'] / 2:.1f}mm, -2mm)
  {{present distance (pooled SD)}};
\node[rotate=90, anchor=south, font=\scriptsize] at (-5.5mm, {meta['H'] / 2:.1f}mm)
  {{future distance}};
\node[anchor=west, font=\scriptsize, text=corange] at ({meta['x_max_mm'] + 0.8:.1f}mm, {meta['H'] - 1.5:.1f}mm)
  {{$\leq0.75$}};
\node[anchor=south west, font=\scriptsize, text=corange] at (0.6mm, {meta['y_min_mm'] + 0.4:.1f}mm)
  {{$\geq0.75$}};
\end{{scope}}
\node[anchor=west, font=\scriptsize, text=cink2, align=left] at (-6mm, -8.8mm)
  {{all {meta['n']} selected starts: similar now, different next---\\the clock can no longer identify the target}};
\end{{scope}}

% ---------------- panel (c): the measurement ----------------
\begin{{scope}}[xshift=132mm, yshift=27mm]
\node[font=\bfseries\footnotesize, anchor=west] at (-24mm, 4mm)
  {{(c) The measurement}};
\begin{{scope}}[yshift=-4mm]
{chr(10).join(bars)}
\end{{scope}}
\node[anchor=west, font=\scriptsize, text=cink2] at (-22mm, -29mm)
  {{ambiguity completion}};
\begin{{scope}}[yshift=-38mm]
{chr(10).join(contrasts)}
\end{{scope}}
\node[anchor=west, font=\scriptsize, text=cink2] at (-22mm, -52mm)
  {{paired contrast, 95\% CI; dots are training seeds}};
\end{{scope}}

\end{{tikzpicture}}
\caption{{\textbf{{What a learned command carries cannot be read off tracking success.}}
(a)~On a single deterministic clip, absolute time indexes the entire target, so completion
cannot attribute credit to any candidate command channel---and a clock-commanded policy in fact
outperforms the frozen retargeting latent (\S\ref{{sec:exp}}).
(b)~The repair: a policy-independent screen selects \ESelectedWindows{{}} start pairs whose present
states are close (${{\leq}}0.75$ pooled SD) but whose 1\,s futures diverge (${{\geq}}0.75$ SD);
every point is a frozen selected window, and the shaded region is the registered admissible zone.
Median future distance \ESelectedMedianFuture{{}}~SD.
(c)~Under that construction the frozen latent beats both an equally trained clock and a
matched-phase wrong-trajectory control; bars are the \EEndpointShort{{}}, and the two paired
contrasts show the 95\% interval with individual training seeds.}}
\label{{fig:teaser}}
\end{{figure*}}
"""
    return "\n".join(header) + body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck", type=pathlib.Path, required=True)
    parser.add_argument("--analysis", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rendered = render(args.precheck, args.analysis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.out)
    print(f"rendered E70 teaser figure -> {args.out}")


if __name__ == "__main__":
    main()
