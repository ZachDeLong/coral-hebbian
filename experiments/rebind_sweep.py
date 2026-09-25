"""Rebinding sweep: does quantization make the memory give stale answers?

    .venv/Scripts/python experiments/rebind_sweep.py --seeds 3
    .venv/Scripts/python experiments/rebind_sweep.py --quick

Same grid as sweep.py, on the rebinding task (fastweight/rebind.py). Writes
summary.csv, a markdown table of current accuracy / stale rate, and
rebind_<metric>_<scale>.png curves against "writes since the key was last bound".
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from fastweight import MemoryConfig, RebindTask, calibrate_rebind, run_rebind  # noqa: E402
from sweep import LAMS, PRECISIONS, RULES, plot_curves  # noqa: E402


def run_grid(task: RebindTask, scales, lams, beta: float):
    results = []
    for rule in RULES:
        for lam in lams:
            t0 = time.perf_counter()
            base = MemoryConfig(rule=rule, lam=lam, beta=beta)
            results.append(run_rebind(base, task))
            static_range = calibrate_rebind(base, task)
            for bits, rounding in PRECISIONS:
                for scale in scales:
                    cfg = replace(base, bits=bits, rounding=rounding, scale=scale)
                    results.append(run_rebind(cfg, task, static_range))
            print(f"  seed {task.seed}  {rule:8s} lam={lam:<6} {time.perf_counter() - t0:5.1f}s", flush=True)
    return results


def age_bins(T: int) -> np.ndarray:
    """Edges 0, 1, 2, 4, ... up to T: log-spaced bins of writes since last binding."""
    edges = [0, 1]
    while edges[-1] < T:
        edges.append(edges[-1] * 2)
    return np.array(edges)


def binned(results, attr: str, edges: np.ndarray):
    """Per seed, the mean of `attr` in each age bin; returns (x, mean over seeds, std over seeds)."""
    per_seed = []
    for r in results:
        vals = getattr(r, attr)
        idx = np.digitize(r.age, edges) - 1
        per_seed.append([vals[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)])
    arr = np.array(per_seed, dtype=float)
    x = np.sqrt(np.maximum(edges[:-1], 0.5) * edges[1:])  # geometric bin centers
    return x, np.nanmean(arr, 0), np.nanstd(arr, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--pool", type=int, default=64)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lams", type=float, nargs="+", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "rebind")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.quick:
        task = RebindTask(d=32, pool=16, T=128, vocab=64, batch=8, seed=args.seed)
        lams = (0.9, 0.99)
    else:
        task = RebindTask(d=args.d, pool=args.pool, T=args.T, vocab=args.vocab, batch=args.batch, seed=args.seed)
        lams = tuple(args.lams) if args.lams else LAMS
    scales = ("static", "dynamic")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"task: {task}, seeds {args.seed}..{args.seed + args.seeds - 1}")
    results = []
    for seed in range(args.seed, args.seed + args.seeds):
        results += run_grid(replace(task, seed=seed), scales, lams, args.beta)
    groups = defaultdict(list)
    for r in results:
        groups[r.cfg].append(r)

    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "lam", "bits", "rounding", "scale", "n_seeds", "current_mean", "current_std",
                    "stale_mean", "stale_std", "write_lsb_mean", "unchanged_frac_mean"])
        for c, rs in groups.items():
            cur = np.array([r.current_acc for r in rs])
            stl = np.array([r.stale_rate for r in rs])
            w.writerow([c.rule, c.lam, c.bits or 32, c.rounding if c.bits else "", c.scale if c.bits else "",
                        len(rs), f"{cur.mean():.4f}", f"{cur.std():.4f}", f"{stl.mean():.4f}", f"{stl.std():.4f}",
                        f"{np.mean([r.write_lsb for r in rs]):.4f}", f"{np.mean([r.unchanged_frac for r in rs]):.4f}"])

    edges = age_bins(task.T)
    cols = ["fp32", *PRECISIONS]
    names = ["fp32"] + [f"int{b} {rd}" for b, rd in PRECISIONS]
    for scale in scales:
        pick = {}
        for c, rs in groups.items():
            if c.bits is None:
                pick[(c.rule, c.lam, "fp32")] = rs
            elif c.scale == scale:
                pick[(c.rule, c.lam, (c.bits, c.rounding))] = rs
        print(f"\nCurrent accuracy / stale rate (%), scale={scale}, {args.seeds} seed(s)")
        print("| rule | lam | " + " | ".join(names) + " |")
        print("|---|---|" + "---|" * len(cols))
        for rule in RULES:
            for lam in lams:
                cells = []
                for k in cols:
                    rs = pick[(rule, lam, k)]
                    cells.append(f"{100 * np.mean([r.current_acc for r in rs]):.0f} / "
                                 f"{100 * np.mean([r.stale_rate for r in rs]):.0f}")
                print(f"| {rule} | {lam} | " + " | ".join(cells) + " |")

        for attr, ylabel in (("current", "answers latest value"), ("stale", "answers an older value")):
            series = []
            for c, rs in groups.items():
                if c.bits is not None and c.scale != scale:
                    continue
                x, mean, std = binned(rs, attr, edges)
                series.append((c, x, mean, std if args.seeds > 1 else None))
            plot_curves(series, lams, xlabel="writes since key was last bound", ylabel=f"fraction {ylabel}",
                        path=args.out / f"rebind_{attr}_{scale}.png",
                        title=f"Rebinding ({task.pool} keys, {task.T} writes, d={task.d}), state stored with "
                              f"{scale} scale, {args.seeds} seed(s)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
