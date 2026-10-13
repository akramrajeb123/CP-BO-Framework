#!/usr/bin/env python3
"""
Ablation runner for CP-BO (Section 6.4, Table 4).

Runs the full method and 5 single-component ablations on CIFAR-10 and Adult
Income, 10 seeds each, 100-evaluation budget, then writes:
  - results_raw.csv      one row per (variant, dataset, seed)
  - results_table.csv    mean +/- std per (variant, dataset)
  - results_stats.csv    paired Wilcoxon vs. full CP-BO, Benjamini-Hochberg corrected
  - table4.tex           LaTeX rows ready to paste into the paper

IMPORTANT: `run_cpbo()` is a thin adapter. Connect it to YOUR CP-BO
implementation (it must accept the flags in AblationConfig and return the
three metrics). Nothing here produces numbers by itself.

Usage:
  python run_ablation.py --datasets cifar10 adult --seeds 10 --budget 100
  python run_ablation.py --only no_mab --datasets adult          # single variant
"""
import argparse
import csv
import itertools
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]  # as in Sec. 5.3


@dataclass(frozen=True)
class AblationConfig:
    use_ac3: bool = True          # CP preprocessing (AC-3 propagation)
    n_regions: int = 5            # K; K=1 -> no space decomposition
    use_pfeas: bool = True        # feasibility modeling (P_feas term)
    lam: float = 0.5              # boundary-distance weight; 0 -> removed
    use_mab: bool = True          # MAB (UCB1, alpha=2) region selection
    n_init: int = 10              # n0
    budget: int = 100


FULL = AblationConfig()
VARIANTS = {
    "full":        FULL,
    "no_ac3":      replace(FULL, use_ac3=False),
    "no_decomp":   replace(FULL, n_regions=1),
    "no_pfeas":    replace(FULL, use_pfeas=False),
    "no_boundary": replace(FULL, lam=0.0),
    "no_mab":      replace(FULL, use_mab=False),  # uniform random region choice
}
LABELS = {
    "full": "Full CP-BO",
    "no_ac3": "w/o CP preprocessing (AC-3 propagation)",
    "no_decomp": "w/o space decomposition (single region, $K=1$)",
    "no_pfeas": "w/o feasibility modeling ($P_{feas}$ term removed)",
    "no_boundary": "w/o boundary-distance term ($\\lambda=0$)",
    "no_mab": "w/o MAB region selection (uniform random region choice)",
}
# metric reported per dataset in Table 4
PRIMARY = {"cifar10": "accuracy", "adult": "f1"}


def run_cpbo(cfg: AblationConfig, dataset: str, seed: int) -> dict:
    """ADAPTER: call your CP-BO implementation here.

    Must return {"accuracy": float, "f1": float, "constraint_satisfaction": float}
    where constraint_satisfaction is the % of evaluated configs that were feasible
    (use the same definition as in Sec. 6.3), computed on the best config / the
    full evaluation trace exactly as in the main experiments.
    """
    raise NotImplementedError(
        "Connect run_cpbo() to your CP-BO code, e.g.\n"
        "  from cpbo import CPBO\n"
        "  opt = CPBO(dataset=dataset, seed=seed, use_ac3=cfg.use_ac3,\n"
        "             K=cfg.n_regions, use_pfeas=cfg.use_pfeas, lam=cfg.lam,\n"
        "             use_mab=cfg.use_mab, n0=cfg.n_init, budget=cfg.budget)\n"
        "  res = opt.run()\n"
        "  return dict(accuracy=res.acc, f1=res.f1,\n"
        "              constraint_satisfaction=res.feasible_pct)"
    )


def bh_correct(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[idx] * n / rank)
        adj[idx] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "adult"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--only", nargs="*", choices=list(VARIANTS), help="subset of variants (full always included)")
    ap.add_argument("--out", default="ablation_out")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    names = ["full"] + [v for v in (a.only or VARIANTS) if v != "full"]
    seeds = SEEDS[: a.seeds]

    raw = []
    for v, ds, s in itertools.product(names, a.datasets, seeds):
        cfg = replace(VARIANTS[v], budget=a.budget)
        m = run_cpbo(cfg, ds, s)
        raw.append(dict(variant=v, dataset=ds, seed=s, **m))
        print(f"[{v:11s}] {ds:7s} seed={s:5d} -> {m}", flush=True)

    with open(out / "results_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw[0].keys())
        w.writeheader()
        w.writerows(raw)

    def col(v, ds, key):
        rows = sorted((r for r in raw if r["variant"] == v and r["dataset"] == ds), key=lambda r: r["seed"])
        return np.array([r[key] for r in rows], float)

    # mean +/- std table
    summary = []
    for v in names:
        row = {"variant": v}
        for ds in a.datasets:
            for key in (PRIMARY[ds], "constraint_satisfaction"):
                x = col(v, ds, key)
                row[f"{ds}_{key}"] = f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"
        summary.append(row)
    with open(out / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    # paired Wilcoxon vs full, BH-corrected across all tests
    tests = []
    for v in names[1:]:
        for ds in a.datasets:
            for key in (PRIMARY[ds], "constraint_satisfaction"):
                x, y = col("full", ds, key), col(v, ds, key)
                try:
                    p = wilcoxon(x, y).pvalue
                except ValueError:  # all differences zero
                    p = 1.0
                tests.append((v, ds, key, p))
    adj = bh_correct([t[3] for t in tests]) if tests else []
    with open(out / "results_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "dataset", "metric", "p_raw", "p_BH"])
        for t, q in zip(tests, adj):
            w.writerow([t[0], t[1], t[2], f"{t[3]:.4g}", f"{q:.4g}"])

    # LaTeX rows for Table 4 (CIFAR-10 acc in %, Adult F1, constraint satisfaction on both)
    def fmt(v, ds, key, scale=1.0, nd=3):
        x = col(v, ds, key) * scale
        return f"{x.mean():.{nd}f} $\\pm$ {x.std(ddof=1):.{nd}f}"

    lines = []
    for v in names:
        cells = [LABELS[v]]
        cells.append(fmt(v, "cifar10", "accuracy", 1.0, 1) if "cifar10" in a.datasets else "--")
        cells.append(fmt(v, "adult", "f1", 1.0, 3) if "adult" in a.datasets else "--")
        cs = np.concatenate([col(v, ds, "constraint_satisfaction") for ds in a.datasets])
        cells.append(f"{cs.mean():.0f} $\\pm$ {cs.std(ddof=1):.0f}")
        lines.append(" & ".join(cells) + r" \\")
    (out / "table4.tex").write_text("\n".join(lines) + "\n")
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
