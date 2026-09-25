"""Sweep precision x rounding x scale x rule x lambda on the associative recall task.

    .venv/Scripts/python experiments/sweep.py                    # full grid, one seed (~1 min on CPU)
    .venv/Scripts/python experiments/sweep.py --seeds 10         # repeat over seeds 0..9
    .venv/Scripts/python experiments/sweep.py --quick            # smoke test

Writes to results/ (or --out): sweep.csv (one row per config per seed),
summary.csv (mean/std over seeds), curves.csv (mean/std accuracy by age), and
recall_<scale>.png forgetting curves (mean line, +-1 std band across seeds).
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
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
FLOAT_FORMATS = ("bf16",)  # float storage formats; like fp32, they have no scale and appear in every scale's table

# Reference palette slots 1-5 (validated adjacent, light surface), fp32 in primary ink.
# Rounding / format is double-encoded as line style so identity never rests on color alone.
STYLE = {
    "fp32": dict(color="#0b0b0b", ls="-"),
    (8, "nearest"): dict(color="#2a78d6", ls="-"),
    (8, "stochastic"): dict(color="#eb6834", ls="--"),
    (4, "nearest"): dict(color="#1baf7a", ls="-"),
    (4, "stochastic"): dict(color="#eda100", ls="--"),
    "bf16": dict(color="#e87ba4", ls=":"),
}
INK_2, MUTED, GRID, AXIS, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def series_key(cfg: MemoryConfig):
    """Identity of a line in the plots and a column in the tables."""
    if cfg.bits is None:
        return cfg.float_format or "fp32"
    return (cfg.bits, cfg.rounding)


def series_label(cfg: MemoryConfig) -> str:
    key = series_key(cfg)
    if key == "fp32":
        return "fp32 (reference)"
    if isinstance(key, str):
        return f"{key} storage"
    return f"int{cfg.bits}, {cfg.rounding} rounding"


TABLE_COLS = ["fp32", *FLOAT_FORMATS, *PRECISIONS]
TABLE_NAMES = ["fp32", *FLOAT_FORMATS] + [f"int{b} {rd}" for b, rd in PRECISIONS]


def run_grid(task: RecallTask, scales, lams, beta: float):
    results = []
    for rule in RULES:
        for lam in lams:
            t0 = time.perf_counter()
            base = MemoryConfig(rule=rule, lam=lam, beta=beta)
            results.append(run_recall(base, task))
            for fmt in FLOAT_FORMATS:
                results.append(run_recall(replace(base, float_format=fmt), task))
            static_range = calibrate(base, task)
            for bits, rounding in PRECISIONS:
                for scale in scales:
                    cfg = replace(base, bits=bits, rounding=rounding, scale=scale)
                    results.append(run_recall(cfg, task, static_range))
            print(f"  seed {task.seed}  {rule:8s} lam={lam:<6} {time.perf_counter() - t0:5.1f}s", flush=True)
    return results


def group_by_config(results):
    """{MemoryConfig: [RecallResult per seed]}"""
    groups = defaultdict(list)
    for r in results:
        groups[r.cfg].append(r)
    return groups


def write_csvs(results, groups, T: int, out: Path):
    def cfg_cols(c):
        bits = c.bits or c.float_format or 32
        return [T, c.rule, c.lam, bits, c.rounding if c.bits else "", c.scale if c.bits else ""]

    cfg_header = ["T", "rule", "lam", "bits", "rounding", "scale"]
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*cfg_header, "seed", "recalled", "age50", "fidelity", "write_lsb", "saturated_frac",
                    "unchanged_frac"])
        for r in results:
            w.writerow([*cfg_cols(r.cfg), r.seed, f"{r.recalled:.2f}", r.age50(), f"{r.fidelity:.4f}",
                        f"{r.write_lsb:.4f}", f"{r.saturated_frac:.4f}", f"{r.unchanged_frac:.4f}"])
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*cfg_header, "n_seeds", "recalled_mean", "recalled_std", "age50_mean", "age50_std",
                    "fidelity_mean", "fidelity_std", "write_lsb_mean", "unchanged_frac_mean"])
        for cfg, rs in groups.items():
            rec = np.array([r.recalled for r in rs])
            a50 = np.array([r.age50() for r in rs])
            fid = np.array([r.fidelity for r in rs])
            w.writerow([*cfg_cols(cfg), len(rs), f"{rec.mean():.2f}", f"{rec.std():.2f}", f"{a50.mean():.1f}",
                        f"{a50.std():.1f}", f"{fid.mean():.4f}", f"{fid.std():.4f}",
                        f"{np.mean([r.write_lsb for r in rs]):.4f}", f"{np.mean([r.unchanged_frac for r in rs]):.4f}"])
    with open(out / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "lam", "label", "age", "acc_mean", "acc_std", "cos_mean"])
        for cfg, rs in groups.items():
            accs = np.stack([r.acc_by_age for r in rs])
            coss = np.stack([r.cos_by_age for r in rs]).mean(0)
            for age, (m, s, c) in enumerate(zip(accs.mean(0), accs.std(0), coss)):
                w.writerow([cfg.rule, cfg.lam, cfg.label, age, f"{m:.4f}", f"{s:.4f}", f"{c:.4f}"])


def print_table(groups, scale: str, lams, n_seeds: int):
    by_key = {}
    for cfg, rs in groups.items():
        if cfg.bits is None or cfg.scale == scale:
            by_key[(cfg.rule, cfg.lam, series_key(cfg))] = rs
    cols, names = TABLE_COLS, TABLE_NAMES

    def cell(rs):
        rec = np.array([r.recalled for r in rs])
        return f"{rec.mean():.1f}" if n_seeds == 1 else f"{rec.mean():.1f} ± {rec.std():.1f}"

    print(f"\nItems recalled (sum of accuracy over ages), scale={scale}, {n_seeds} seed(s), mean ± std")
    print("| rule | lam | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(cols))
    for rule in RULES:
        for lam in lams:
            print(f"| {rule} | {lam} | " + " | ".join(cell(by_key[(rule, lam, k)]) for k in cols) + " |")


def plot(groups, scale: str, lams, task: RecallTask, n_seeds: int, path: Path):
    ages = np.arange(task.T) + 1
    series = []
    for cfg, rs in groups.items():
        if cfg.bits is not None and cfg.scale != scale:
            continue
        accs = np.stack([r.acc_by_age for r in rs])
        series.append((cfg, ages, accs.mean(0), accs.std(0) if n_seeds > 1 else None))
    seeds_note = f"{n_seeds} seeds × {task.batch} trials, band = ±1 std across seeds" if n_seeds > 1 \
        else f"{task.batch} trials"
    plot_curves(series, lams, xlabel="age (writes since stored)", ylabel="recall accuracy", path=path,
                title=f"Forgetting curves, state stored with {scale} scale  (d={task.d}, vocab={task.vocab}, "
                      f"{seeds_note})")


def plot_curves(series, lams, *, xlabel: str, ylabel: str, title: str, path: Path):
    """series: iterable of (cfg, x, mean, std or None). One panel per (rule, lambda), log x."""
    fig, axes = plt.subplots(len(RULES), len(lams), figsize=(3.1 * len(lams), 5.6), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for cfg, x, mean, std in series:
        key = series_key(cfg)
        ax = axes[RULES.index(cfg.rule)][lams.index(cfg.lam)]
        ax.plot(x, mean, lw=1.5, label=series_label(cfg), **STYLE[key])
        if std is not None:
            ax.fill_between(x, mean - std, mean + std, color=STYLE[key]["color"], alpha=0.12, lw=0)

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
                ax.set_ylabel(f"{'Hebbian' if rule == 'hebbian' else 'Delta rule'}\n{ylabel}", color=INK_2,
                              fontsize=9)
            if i == len(RULES) - 1:
                ax.set_xlabel(xlabel, color=INK_2, fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=INK_2, fontsize=10, y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--seeds", type=int, default=1, help="number of seeds (seed, seed+1, ...)")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lams", type=float, nargs="+", default=None, help=f"decay values (default {LAMS})")
    p.add_argument("--row-scale", action="store_true", help="also run per-row static scales")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test grid")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the tables print "±"; Windows consoles default to cp1252

    if args.quick:
        task = RecallTask(d=32, T=64, vocab=64, batch=8, seed=args.seed)
        lams = (0.9, 0.99)
    else:
        task = RecallTask(d=args.d, T=args.T, vocab=args.vocab, batch=args.batch, seed=args.seed)
        lams = tuple(args.lams) if args.lams else LAMS
    scales = ("static", "dynamic") + (("static_row",) if args.row_scale else ())
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"task: {task}, seeds {args.seed}..{args.seed + args.seeds - 1}")
    results = []
    for seed in range(args.seed, args.seed + args.seeds):
        results += run_grid(replace(task, seed=seed), scales, lams, args.beta)
    groups = group_by_config(results)
    write_csvs(results, groups, task.T, args.out)
    for scale in scales:
        print_table(groups, scale, lams, args.seeds)
        plot(groups, scale, lams, task, args.seeds, args.out / f"recall_{scale}.png")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
