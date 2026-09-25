"""Test the write-size rule: does quantized/fp32 performance depend only on how big
one write is in quantization steps?

    .venv/Scripts/python experiments/collapse.py results/writesize/*/summary.csv

Reads one or more summary.csv files from sweep.py, pairs every quantized config
with the fp32 run of the same (T, rule, lambda), and plots the ratio against the
mean write size in quantization steps. If the rule holds, points from INT8 and
INT4 across all lambdas and T fall on one curve.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {  # same entity colors as sweep.py
    (8, "nearest"): "#2a78d6",
    (8, "stochastic"): "#eb6834",
    (4, "nearest"): "#1baf7a",
    (4, "stochastic"): "#eda100",
}
MARKERS = {"hebbian": "o", "delta": "^"}
INK_2, MUTED, GRID, AXIS, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(paths, metric: str):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    fp = {(r["T"], r["rule"], r["lam"]): float(r[f"{metric}_mean"]) for r in rows if r["bits"] == "32"}
    points = []
    for r in rows:
        if r["rounding"] not in ("nearest", "stochastic"):  # fp32 reference and float formats (bf16)
            continue
        ref = fp[(r["T"], r["rule"], r["lam"])]
        points.append(dict(
            T=int(r["T"]), rule=r["rule"], lam=float(r["lam"]), bits=int(r["bits"]), rounding=r["rounding"],
            scale=r["scale"], write_lsb=float(r["write_lsb_mean"]), ratio=float(r[f"{metric}_mean"]) / ref,
        ))
    return points


def plot(points, metric: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for ax, rounding in zip(axes, ("nearest", "stochastic")):
        for p in points:
            if p["rounding"] != rounding:
                continue
            color = COLORS[(p["bits"], rounding)]
            filled = p["scale"] != "dynamic"
            ax.scatter(p["write_lsb"], p["ratio"], s=42, marker=MARKERS[p["rule"]], linewidths=1.3,
                       facecolors=color if filled else "none", edgecolors=color, zorder=3)
        ax.axhline(1.0, color=AXIS, lw=1, zorder=1)
        ax.axvline(0.5, color=MUTED, lw=1, ls=":", zorder=1)
        ax.text(0.5, 0.02, " ½ step", color=MUTED, fontsize=8, transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"{rounding} rounding", color=INK_2, fontsize=10)
        ax.set_xlabel("mean write size (quantization steps)", color=INK_2, fontsize=9)
    what = "items recalled" if metric == "recalled" else "readout fidelity (cosine)"
    axes[0].set_ylabel(f"{what}, quantized ÷ fp32", color=INK_2, fontsize=9)

    handles = [
        plt.Line2D([], [], ls="", marker="s", color=COLORS[(8, "nearest")], label="INT8 (nearest)"),
        plt.Line2D([], [], ls="", marker="s", color=COLORS[(4, "nearest")], label="INT4 (nearest)"),
        plt.Line2D([], [], ls="", marker="s", color=COLORS[(8, "stochastic")], label="INT8 (stochastic)"),
        plt.Line2D([], [], ls="", marker="s", color=COLORS[(4, "stochastic")], label="INT4 (stochastic)"),
        plt.Line2D([], [], ls="", marker="o", color=MUTED, label="Hebbian"),
        plt.Line2D([], [], ls="", marker="^", color=MUTED, label="delta rule"),
        plt.Line2D([], [], ls="", marker="o", color=MUTED, markerfacecolor=SURFACE, label="hollow = dynamic scale"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summaries", nargs="+", type=Path)
    p.add_argument("--metric", choices=("recalled", "fidelity"), default="recalled")
    p.add_argument("--out", type=Path, default=None, help="plot path (default: next to the first summary's parent)")
    args = p.parse_args()

    points = load(args.summaries, args.metric)
    out = args.out or args.summaries[0].parent.parent / f"collapse_{args.metric}.png"
    plot(points, args.metric, out)

    print(f"| write (steps) | ratio | bits | rounding | scale | rule | lam | T |")
    print("|---|---|---|---|---|---|---|---|")
    for q in sorted(points, key=lambda q: q["write_lsb"]):
        print(f"| {q['write_lsb']:.3f} | {q['ratio']:.2f} | {q['bits']} | {q['rounding']} | {q['scale']} | "
              f"{q['rule']} | {q['lam']} | {q['T']} |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
