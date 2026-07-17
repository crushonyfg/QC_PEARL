"""Zero-shot (K=0, NO warmup) fleet-breadth comparison: vsqr at M=1/3/5 sharing one
calibrator. Tests whether M-fold per-step data averages out the action-noise corruption
(M=1 was -120 vs frozen, std 172) without any warmup. Shows return, seed-std, and regret
vs the frozen policy; flags the noisy regimes.
"""
from __future__ import annotations

import glob
import json

import numpy as np

ROOT = "results/zeroshot"
ORDER = [
    "nominal",
    "action_actuator_0.60", "action_actuator_0.40", "action_bias_0.20", "action_noise_0.20",
    "env_friction_0.60", "env_friction_1.50", "env_damping_0.60", "env_damping_1.50",
    "body_mass_1.50", "body_mass_0.60", "compound_mild", "compound_hard",
    "sudden_actuator_0.40", "gradual_actuator_0.20", "multi_sudden_actuator",
]
NOISY = {"action_noise_0.20", "action_bias_0.20", "body_mass_1.50"}  # M=1 high-variance regimes


def vsqr_cell(regime, M):
    tag = "value_shift_qr_K0" if M == 1 else f"value_shift_qr_K0_M{M}"
    fs = glob.glob(f"{ROOT}/{regime}/{tag}/seed*/aggregate.json")
    v = [json.load(open(f))["mean_return"] for f in fs]
    return (np.mean(v), np.std(v)) if v else None


def base_cell(regime):
    fs = glob.glob(f"{ROOT}/{regime}/full_pearl_only_K0/seed*/aggregate.json")
    v = [json.load(open(f))["mean_return"] for f in fs]
    return np.mean(v) if v else None


def main():
    print("Zero-shot (K=0, no warmup) vsqr at fleet M=1/3/5 — return (seed-std) [regret vs frozen]")
    print(f"{'regime':22s}{'M=1':>20s}{'M=3':>20s}{'M=5':>20s}")
    for r in ORDER:
        b = base_cell(r)
        row = f"{r:22s}{'*' if r in NOISY else ' '}"[:23]
        for M in (1, 3, 5):
            c = vsqr_cell(r, M)
            if c and b is not None:
                row += f"{c[0]:7.0f}(s{c[1]:3.0f},{c[0]-b:+4.0f})"
            elif c:
                row += f"{c[0]:7.0f}(s{c[1]:3.0f})    "
            else:
                row += f"{'-':>20s}"
        print(row)
    print("\n(* = regime that was high-variance at M=1.  Hypothesis: M-fold per-step data")
    print(" reduces the estimator variance most on the noisy targets — esp. action_noise.)")


if __name__ == "__main__":
    main()
