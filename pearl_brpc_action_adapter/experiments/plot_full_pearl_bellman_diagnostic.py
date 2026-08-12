from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_by_t(rows, key: str):
    by_t = defaultdict(list)
    for row in rows:
        by_t[int(row["t"])].append(float(row[key]))
    ts = sorted(by_t)
    return np.asarray(ts), np.asarray([np.mean(by_t[t]) for t in ts])


def main():
    parser = argparse.ArgumentParser(description="Plot full PEARL Bellman diagnostic time series.")
    parser.add_argument("--input-root", default="results/full_pearl_bellman_diag")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    rows = load_rows(input_root / "transition_bellman_residuals.csv")

    import matplotlib.pyplot as plt

    regimes = ["nominal", "ood_actuator_0.60", "sudden_actuator"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for regime in regimes:
        subset = [r for r in rows if r["regime"] == regime]
        t_abs, y_abs = mean_by_t(subset, "abs_eta")
        t_e, y_e = mean_by_t(subset, "E_p95_ratio")
        axes[0].plot(t_abs, y_abs, label=regime)
        axes[1].plot(t_e, y_e, label=regime)
    axes[0].axvline(100, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].axvline(100, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].set_ylabel("mean |eta|")
    axes[1].set_ylabel("mean |eta| / nominal P95")
    axes[1].set_xlabel("t")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    out = Path(args.output) if args.output else input_root / "bellman_residual_timeseries.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
