"""Sweep precision x rounding x scale x rule x lambda on the associative recall task.

    .venv/Scripts/python experiments/sweep.py            # full grid (~a few minutes on CPU)
    .venv/Scripts/python experiments/sweep.py --quick    # smoke test

Writes to results/ (or --out): sweep.csv (one row per config), curves.csv
(accuracy by age), and recall_<scale>.png forgetting curves.
"""

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fastweight import MemoryConfig, RecallTask, calibrate, run_recall  # noqa: E402

RULES = ("hebbian", "delta")
LAMS = (0.9, 0.97, 0.99, 0.999, 1.0)
PRECISIONS = ((8, "nearest"), (8, "stochastic"), (4, "nearest"), (4, "stochastic"))

# Reference palette slots 1-4 (validated adjacent, light surface), fp32 in primary ink.
# Rounding is double-encoded as line style so identity never rests on color alone.
STYLE = {
    "fp32": dict(color="#0b0b0b", ls="-"),
    (8, "nearest"): dict(color="#2a78d6", ls="-"),
    (8, "stochastic"): dict(color="#eb6834", ls="--"),
    (4, "nearest"): dict(color="#1baf7a", ls="-"),
    (4, "stochastic"): dict(color="#eda100", ls="--"),
}
INK_2, MUTED, GRID, AXIS, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def run_grid(task: RecallTask, scales, lams, beta: float):
    results = []
    for rule in RULES:
        for lam in lams:
            t0 = time.perf_counter()
            base = MemoryConfig(rule=rule, lam=lam, beta=beta)
            results.append(run_recall(base, task))
            static_range = calibrate(base, task)
            for bits, rounding in PRECISIONS:
                for scale in scales:
                    cfg = replace(base, bits=bits, rounding=rounding, scale=scale)
                    results.append(run_recall(cfg, task, static_range))
            print(f"  {rule:8s} lam={lam:<6} {time.perf_counter() - t0:5.1f}s", flush=True)
    return results


def write_csvs(results, out: Path):
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "lam", "bits", "rounding", "scale", "recalled", "age50", "saturated_frac", "unchanged_frac"])
        for r in results:
            c = r.cfg
            w.writerow([c.rule, c.lam, c.bits or 32, c.rounding if c.bits else "", c.scale if c.bits else "",
                        f"{r.recalled:.2f}", r.age50(), f"{r.saturated_frac:.4f}", f"{r.unchanged_frac:.4f}"])
    with open(out / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "lam", "label", "age", "acc"])
        for r in results:
            for age, acc in enumerate(r.acc_by_age):
                w.writerow([r.cfg.rule, r.cfg.lam, r.cfg.label, age, f"{acc:.4f}"])


def print_table(results, scale: str, lams):
    by_key = {}
    for r in results:
        c = r.cfg
        if c.bits is None:
            by_key[(c.rule, c.lam, "fp32")] = r
        elif c.scale == scale:
            by_key[(c.rule, c.lam, (c.bits, c.rounding))] = r
    cols = ["fp32", *PRECISIONS]
    names = ["fp32"] + [f"int{b} {rd[:5]}" for b, rd in PRECISIONS]
    print(f"\nItems recalled (sum of accuracy over ages), scale={scale}")
    print("| rule | lam | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(cols))
    for rule in RULES:
        for lam in lams:
            cells = [f"{by_key[(rule, lam, k)].recalled:.1f}" for k in cols]
            print(f"| {rule} | {lam} | " + " | ".join(cells) + " |")


def plot(results, scale: str, lams, task: RecallTask, path: Path):
    fig, axes = plt.subplots(len(RULES), len(lams), figsize=(3.1 * len(lams), 5.6), sharex=True, sharey=True,
                             facecolor=SURFACE)
    ages = np.arange(task.T) + 1
    for r in results:
        c = r.cfg
        if c.bits is not None and c.scale != scale:
            continue
        key = "fp32" if c.bits is None else (c.bits, c.rounding)
        ax = axes[RULES.index(c.rule)][lams.index(c.lam)]
        label = "fp32 (reference)" if key == "fp32" else f"int{c.bits}, {c.rounding} rounding"
        ax.plot(ages, r.acc_by_age, lw=1.5, label=label, **STYLE[key])

    for i, rule in enumerate(RULES):
        for j, lam in enumerate(lams):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8)
            if i == 0:
                ax.set_title(f"λ = {lam}", color=INK_2, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{'Hebbian' if rule == 'hebbian' else 'Delta rule'}\nrecall accuracy", color=INK_2,
                              fontsize=9)
            if i == len(RULES) - 1:
                ax.set_xlabel("age (writes since stored)", color=INK_2, fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Forgetting curves, state stored with {scale} scale  (d={task.d}, vocab={task.vocab}, "
                 f"{task.batch} trials)", color=INK_2, fontsize=10, y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--row-scale", action="store_true", help="also run per-row static scales")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test grid")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = p.parse_args()

    if args.quick:
        task = RecallTask(d=32, T=64, vocab=64, batch=8, seed=args.seed)
        lams = (0.9, 0.99)
    else:
        task = RecallTask(d=args.d, T=args.T, vocab=args.vocab, batch=args.batch, seed=args.seed)
        lams = LAMS
    scales = ("static", "dynamic") + (("static_row",) if args.row_scale else ())
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"task: {task}")
    results = run_grid(task, scales, lams, args.beta)
    write_csvs(results, args.out)
    for scale in scales:
        print_table(results, scale, lams)
        plot(results, scale, lams, task, args.out / f"recall_{scale}.png")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
