"""Aggregate the drift/tracking experiment (advantage B).

Tables: per (config, regime) mean return + regret vs zero-shot full_pearl_only.
Plots: time-resolved bin_mean_reward and bin_alive_frac vs step (averaged over seeds)
for the gradual / multi_sudden regimes -- the tracking curves. Shows whether vsqr with
a real forgetting factor (lower q_rho) tracks the drift, vs fine-tune (full vs recency buffer).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REGIMES = ["nominal", "gradual_actuator_0.20", "multi_sudden_actuator"]
TRACK_REGIMES = ["gradual_actuator_0.20", "multi_sudden_actuator"]
K, M = 3, 3

# (label, method, root, tag)
CONFIGS = [
    ("vsqr rho0.999",   "value_shift_qr",     "results/drift/rho0.999",  f"value_shift_qr_K{K}_M{M}"),
    ("vsqr rho0.99",    "value_shift_qr",     "results/drift/rho0.99",   f"value_shift_qr_K{K}_M{M}"),
    ("vsqr rho0.95",    "value_shift_qr",     "results/drift/rho0.95",   f"value_shift_qr_K{K}_M{M}"),
    ("vsqr rho0.90",    "value_shift_qr",     "results/drift/rho0.90",   f"value_shift_qr_K{K}_M{M}"),
    ("ft-LL fullbuf",   "finetune_lastlayer", "results/drift/ft_fullbuf", f"finetune_lastlayer_K{K}_M{M}"),
    ("ft-FU fullbuf",   "finetune_full",      "results/drift/ft_fullbuf", f"finetune_full_K{K}_M{M}"),
    ("ft-LL recency",   "finetune_lastlayer", "results/drift/ft_recency", f"finetune_lastlayer_K{K}_M{M}"),
    ("ft-FU recency",   "finetune_full",      "results/drift/ft_recency", f"finetune_full_K{K}_M{M}"),
]


def runs(root, regime, tag):
    return list((Path(root) / regime / tag).glob("seed*/aggregate.json"))


def load_runs(root, regime, tag):
    out = []
    for p in runs(root, regime, tag):
        with p.open("r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def mean_return(root, regime, tag):
    rs = load_runs(root, regime, tag)
    return (float(np.mean([r["mean_return"] for r in rs])), float(np.std([r["mean_return"] for r in rs]))) if rs else None


def baseline():
    out = {}
    for r in REGIMES:
        c = mean_return("results/drift/_ref", r, "full_pearl_only_K0")
        out[r] = c[0] if c else None
    return out


def avg_trace(root, regime, tag, key):
    rs = load_runs(root, regime, tag)
    arrs = [np.array(r[key], dtype=float) for r in rs if key in r]
    if not arrs:
        return None, None
    L = min(len(a) for a in arrs)
    A = np.vstack([a[:L] for a in arrs])
    return np.nanmean(A, axis=0), rs[0]["bin_centers"][:L]


def main():
    base = baseline()
    print("\n=== drift experiment: mean return (regret vs full_pearl_only) ===")
    print(f"baseline full_pearl_only: " + ", ".join(f"{r.split('_')[0]}={base[r]:.0f}" for r in REGIMES if base[r]))
    hdr = f"{'config':18s}" + "".join(f"{(r.split('_')[0] if r!='nominal' else 'nominal'):>22s}" for r in REGIMES)
    print(hdr)
    for label, method, root, tag in CONFIGS:
        row = f"{label:18s}"
        for r in REGIMES:
            c = mean_return(root, r, tag)
            if c and base.get(r):
                row += f"{c[0]:>10.0f} ({c[0]-base[r]:+5.0f})  "
            elif c:
                row += f"{c[0]:>10.0f}          "
            else:
                row += f"{'-':>22s}"
        print(row)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib unavailable; skipping plots)")
        return
    for key, ylab, fname in [("bin_mean_reward", "per-step reward", "drift_reward"),
                             ("bin_alive_frac", "alive fraction", "drift_alive")]:
        fig, axes = plt.subplots(1, len(TRACK_REGIMES), figsize=(7 * len(TRACK_REGIMES), 4.5), squeeze=False)
        for ax, regime in zip(axes[0], TRACK_REGIMES):
            for label, method, root, tag in CONFIGS:
                y, x = avg_trace(root, regime, tag, key)
                if y is None:
                    continue
                ls = "--" if label.startswith("ft") else "-"
                ax.plot(x, y, ls, marker=".", ms=4, label=label, alpha=0.85)
            ax.set_title(regime); ax.set_xlabel("env step"); ax.set_ylabel(ylab)
            ax.legend(fontsize=7)
        fig.tight_layout()
        out = Path("results/drift") / f"{fname}.png"
        fig.savefig(out, dpi=130)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
