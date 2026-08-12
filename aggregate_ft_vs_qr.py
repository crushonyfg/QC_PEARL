"""Pivot the ft-vs-qr summary into per-regime tables (K on rows, method on cols).

Reads results/full_pearl_dynamics_lookahead_ft/summary_by_method_regime.json and prints
mean_return (+/- std across seeds) for value_shift_qr / finetune_lastlayer / finetune_full
across the warmup budget K, with the zero-shot full_pearl_only / q_greedy reference line.
Optionally writes return-vs-K line plots per regime (--plot).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAIN = ["value_shift_qr", "finetune_lastlayer", "finetune_full"]
REF = ["full_pearl_only", "q_greedy"]
KS = [0, 1, 3, 5]


def load(root: Path):
    """Scan every per-run aggregate.json (summary_by_method_regime.json is overwritten by
    each runner invocation, so it only holds the last K). Group by (regime, method, K) and
    average mean_return across seeds."""
    import numpy as np
    by_key = {}
    regimes = []
    for agg in root.rglob("aggregate.json"):
        with agg.open("r", encoding="utf-8") as f:
            r = json.load(f)
        key = (r["regime"], r["method"], r["warmup_episodes"])
        by_key.setdefault(key, []).append(r)
        if r["regime"] not in regimes:
            regimes.append(r["regime"])
    idx = {}
    for (regime, method, k), items in by_key.items():
        means = [x["mean_return"] for x in items]
        idx[(regime, method, k)] = {
            "regime": regime, "method": method, "warmup_episodes": k,
            "mean_return": float(np.mean(means)), "std_return": float(np.std(means)),
            "num_agents": items[0]["num_agents"], "n_seeds": len(items),
        }
    # Stable regime ordering.
    order = ["nominal", "ood_actuator_0.60", "sudden_actuator_0.40", "multi_sudden_actuator"]
    regimes = [r for r in order if r in regimes] + [r for r in regimes if r not in order]
    return idx, regimes


def fmt(row):
    if row is None:
        return "      -      "
    return f"{row['mean_return']:6.1f}+/-{row['std_return']:4.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/full_pearl_dynamics_lookahead_ft")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    idx, regimes = load(root)

    for regime in regimes:
        print(f"\n=== {regime} ===")
        # Reference (K-independent zero-shot): show whatever K they were run at.
        for m in REF:
            cand = [idx.get((regime, m, k)) for k in KS]
            cand = [c for c in cand if c is not None]
            if cand:
                print(f"  [ref] {m:18s} return={cand[0]['mean_return']:6.1f}+/-{cand[0]['std_return']:.0f} (M={cand[0]['num_agents']})")
        header = "  K     " + "".join(f"{m:>20s}" for m in MAIN)
        print(header)
        for k in KS:
            cells = "".join(f"{fmt(idx.get((regime, m, k))):>20s}" for m in MAIN)
            print(f"  {k:<5d} {cells}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return
        n = len(regimes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for ax, regime in zip(axes[0], regimes):
            for m in MAIN:
                ys, es, xs = [], [], []
                for k in KS:
                    r = idx.get((regime, m, k))
                    if r is not None:
                        xs.append(k); ys.append(r["mean_return"]); es.append(r["std_return"])
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=m)
            for m in REF:
                cand = [idx.get((regime, m, k)) for k in KS if idx.get((regime, m, k))]
                if cand:
                    ax.axhline(cand[0]["mean_return"], ls="--", alpha=0.6, label=f"{m} (zero-shot)")
            ax.set_title(regime); ax.set_xlabel("warmup episodes K"); ax.set_ylabel("return")
            ax.set_xticks(KS); ax.legend(fontsize=7)
        fig.tight_layout()
        out = root / "ft_vs_qr_returns.png"
        fig.savefig(out, dpi=130)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
