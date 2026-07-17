"""Online / anytime analysis: cumulative return and online regret over the first E episodes
of deployment (K=0, M=1, every episode counts, continual adaptation). Tests whether vsqr's
early episodes are SAFER/CHEAPER than fine-tune's (which pays an unstable warmup cost), even
if fine-tune's asymptotic return is higher.
"""
from __future__ import annotations

import glob
import json

import numpy as np

REGIMES = ["nominal", "ood_actuator_0.60", "compound_mild", "multi_sudden_actuator"]
METHODS = [
    ("full_pearl_only", "frozen"),
    ("q_greedy", "q_greedy"),
    ("value_shift_qr", "vsqr"),
    ("finetune_lastlayer", "ft-LL"),
    ("finetune_full", "ft-FU"),
    ("qr_finetune_critic", "qr+ftC"),
    ("qr_finetune_full", "qr+ftF"),
]
ROOT = "results/online"


def curves(regime, method):
    """Stack per-seed episode_returns -> (n_seeds, E) array, or None."""
    fs = sorted(glob.glob(f"{ROOT}/{regime}/{method}_K0/seed*/aggregate.json"))
    rows = []
    for f in fs:
        d = json.load(open(f))
        if d.get("episode_returns"):
            rows.append(d["episode_returns"])
    if not rows:
        return None
    L = min(len(r) for r in rows)
    return np.array([r[:L] for r in rows], dtype=float)


def main():
    print("Online deployment (K=0, M=1, continual). Per-regime: mean per-episode return and")
    print("CUMULATIVE return at E=5 / E=10; cumulative regret vs frozen at E=10.\n")
    for r in REGIMES:
        frozen = curves(r, "full_pearl_only")
        fmean = frozen.mean(0) if frozen is not None else None
        print(f"=== {r} ===")
        print(f"  {'method':10s}{'ep1':>8s}{'ep3':>8s}{'ep10':>8s}{'cumret@5':>11s}{'cumret@10':>11s}{'cumReg@10':>11s}")
        for m, lab in METHODS:
            c = curves(r, m)
            if c is None:
                continue
            mean = c.mean(0)
            E = len(mean)
            cum5 = mean[:min(5, E)].sum()
            cum10 = mean[:min(10, E)].sum()
            creg = (fmean[:E] - mean).sum() if fmean is not None else float("nan")  # regret>0 => worse than frozen
            ep = lambda i: mean[i] if i < E else float("nan")
            print(f"  {lab:10s}{ep(0):8.0f}{ep(2):8.0f}{ep(9):8.0f}{cum5:11.0f}{cum10:11.0f}{creg:+11.0f}")
        print()

    # Plots: per-episode return curve + cumulative regret vs frozen.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping plots)")
        return
    for key, fname, ylab in [("ret", "online_return", "per-episode return"),
                             ("reg", "online_cumregret", "cumulative regret vs frozen")]:
        fig, axes = plt.subplots(1, len(REGIMES), figsize=(5 * len(REGIMES), 4), squeeze=False)
        for ax, r in zip(axes[0], REGIMES):
            frozen = curves(r, "full_pearl_only")
            fmean = frozen.mean(0) if frozen is not None else None
            for m, lab in METHODS:
                c = curves(r, m)
                if c is None:
                    continue
                mean = c.mean(0)
                x = np.arange(1, len(mean) + 1)
                ls = "--" if lab.startswith("ft") else "-"
                if key == "ret":
                    ax.plot(x, mean, ls, marker=".", label=lab)
                elif fmean is not None:
                    ax.plot(x, np.cumsum(fmean[:len(mean)] - mean), ls, marker=".", label=lab)
            ax.set_title(r); ax.set_xlabel("deployment episode"); ax.set_ylabel(ylab)
            if key == "reg":
                ax.axhline(0, color="k", lw=0.6, alpha=0.5)
            ax.legend(fontsize=7)
        fig.tight_layout()
        out = f"{ROOT}/{fname}.png"
        fig.savefig(out, dpi=130)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
