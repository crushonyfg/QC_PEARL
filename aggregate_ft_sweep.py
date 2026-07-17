"""Aggregate the K=3 hyperparameter robustness sweep.

Pulls the 5 sweep configs (results/ft_sweep/*) plus the two cells already in the main
run (ft lr3e-4/u200 and vsqr q_prior_var=0.25, at K3 M3). For each config x regime,
averages mean_return over seeds. Then answers the regime-agnostic-selection question:
since there is NO validation set for an unknown OOD shift, a config must be committed
to a priori -- so we report each config's WORST-regime return and its OOD-mean, not a
per-regime cherry-pick.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REGIMES = ["nominal", "ood_actuator_0.60", "sudden_actuator_0.40", "multi_sudden_actuator"]
OOD = REGIMES[1:]
K, M = 3, 3

# (config_label, method, results_root). Two cells reuse the main run.
SOURCES = [
    ("vsqr pv0.10",     "value_shift_qr",     "results/ft_sweep/vsqr_pv0.1"),
    ("vsqr pv0.25",     "value_shift_qr",     "results/full_pearl_dynamics_lookahead_ft"),
    ("vsqr pv1.00",     "value_shift_qr",     "results/ft_sweep/vsqr_pv1.0"),
    ("ft-LL lr1e-4 u50",  "finetune_lastlayer", "results/ft_sweep/ft_lr1e-4_u50"),
    ("ft-LL lr1e-4 u200", "finetune_lastlayer", "results/ft_sweep/ft_lr1e-4_u200"),
    ("ft-LL lr3e-4 u50",  "finetune_lastlayer", "results/ft_sweep/ft_lr3e-4_u50"),
    ("ft-LL lr3e-4 u200", "finetune_lastlayer", "results/full_pearl_dynamics_lookahead_ft"),
    ("ft-FU lr1e-4 u50",  "finetune_full",      "results/ft_sweep/ft_lr1e-4_u50"),
    ("ft-FU lr1e-4 u200", "finetune_full",      "results/ft_sweep/ft_lr1e-4_u200"),
    ("ft-FU lr3e-4 u50",  "finetune_full",      "results/ft_sweep/ft_lr3e-4_u50"),
    ("ft-FU lr3e-4 u200", "finetune_full",      "results/full_pearl_dynamics_lookahead_ft"),
]


def cell(root: str, regime: str, method: str, tag: str = None):
    d = Path(root) / regime / (tag or f"{method}_K{K}_M{M}")
    vals = []
    for agg in d.glob("seed*/aggregate.json"):
        with agg.open("r", encoding="utf-8") as f:
            vals.append(json.load(f)["mean_return"])
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def baseline():
    """Zero-shot 'do nothing' deploy reference (frozen policy) per regime, from the main run."""
    root = "results/full_pearl_dynamics_lookahead_ft"
    out = {}
    for r in REGIMES:
        c = cell(root, r, "full_pearl_only", tag="full_pearl_only_K0")
        out[r] = c[0] if c else None
    return out


def main():
    base = baseline()
    print(f"\nHyperparameter robustness @ K={K}, M={M} (mean return over seeds)")
    print(f"baseline (full_pearl_only, zero-shot deploy): " + ", ".join(f"{r.split('_')[0]}={base[r]:.0f}" for r in REGIMES if base[r]))

    print("\n--- absolute return ---")
    hdr = f"{'config':22s}" + "".join(f"{(r.split('_')[0] if r!='nominal' else 'nominal'):>10s}" for r in REGIMES)
    print(hdr + f"{'OODmean':>10s}")
    for label, method, root in SOURCES:
        cells = {r: cell(root, r, method) for r in REGIMES}
        if all(c is None for c in cells.values()):
            continue
        row = f"{label:22s}" + "".join(f"{cells[r][0]:10.0f}" if cells[r] else f"{'-':>10s}" for r in REGIMES)
        ood_vals = [cells[r][0] for r in OOD if cells[r]]
        row += f"{np.mean(ood_vals):10.0f}" if ood_vals else f"{'-':>10s}"
        print(row)

    print("\n--- REGRET vs zero-shot baseline (return - full_pearl_only); WORST = worst regret across regimes ---")
    print(hdr + f"{'WORST':>10s}")
    for label, method, root in SOURCES:
        cells = {r: cell(root, r, method) for r in REGIMES}
        if all(c is None for c in cells.values()):
            continue
        regrets = {r: (cells[r][0] - base[r]) for r in REGIMES if cells[r] and base[r]}
        row = f"{label:22s}" + "".join(f"{regrets[r]:+10.0f}" if r in regrets else f"{'-':>10s}" for r in REGIMES)
        row += f"{min(regrets.values()):+10.0f}" if regrets else f"{'-':>10s}"
        print(row)
    print("\nNo validation set => you must COMMIT one config for all regimes. The honest")
    print("criterion is the best WORST-case regret: a method that never badly hurts any regime.")


if __name__ == "__main__":
    main()
