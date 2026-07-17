"""n-step / lambda-return QR comparison. Per regime, return (seed-std) and regret vs the
zero-shot frozen policy, across n1 / n3 / n5 / lam0.9 / lam1.0(MC). Tests whether multi-step
real-return targets reduce the single-step bootstrap bias where 1-step is capped (multi_sudden,
gradual, compound) without hurting where 1-step is already near-ceiling (ood0.6) or nominal.
"""
from __future__ import annotations

import glob
import json

import numpy as np

REGIMES = ["nominal", "ood_actuator_0.60", "compound_hard", "body_mass_0.60",
           "gradual_actuator_0.20", "multi_sudden_actuator"]
VARIANTS = [("n1", "results/nstep/n1"), ("n3", "results/nstep/n3"), ("n5", "results/nstep/n5"),
            ("lam0.9", "results/nstep/lam09"), ("lam1.0", "results/nstep/lam10")]


def ret(root, regime, tag="value_shift_qr_K3_M3"):
    fs = glob.glob(f"{root}/{regime}/{tag}/seed*/aggregate.json")
    v = [json.load(open(f))["mean_return"] for f in fs]
    return (np.mean(v), np.std(v)) if v else None


def base(regime):
    # zero-shot frozen policy reference (results/zeroshot has all these regimes).
    fs = glob.glob(f"results/zeroshot/{regime}/full_pearl_only_K0/seed*/aggregate.json")
    v = [json.load(open(f))["mean_return"] for f in fs]
    return np.mean(v) if v else None


def main():
    print("n-step / lambda QR (value_shift_qr, K3 M3, 2 seeds) — return (seed-std) [regret vs frozen]")
    print(f"{'regime':22s}" + "".join(f"{v[0]:>18s}" for v in VARIANTS))
    for r in REGIMES:
        b = base(r)
        row = f"{r:22s}"
        for _, root in VARIANTS:
            c = ret(root, r)
            if c and b is not None:
                row += f"{c[0]:6.0f}(s{c[1]:3.0f},{c[0]-b:+4.0f})"
            elif c:
                row += f"{c[0]:6.0f}(s{c[1]:3.0f})   "
            else:
                row += f"{'-':>18s}"
        print(row)
    print("\nExpect: ood0.6 ~flat-or-worse (1-step already near oracle ceiling); nominal unhurt;")
    print("payoff (if any) on multi_sudden / gradual / compound where the frozen-V bias caps 1-step.")


if __name__ == "__main__":
    main()
