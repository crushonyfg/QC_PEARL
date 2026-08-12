"""Zero-shot (K=0, M=1) story: does value_shift_qr help with NO warmup episodes and a
SINGLE deployed agent (online adaptation from step 1 only)? Per regime, compare the two
zero-shot references (full_pearl_only = frozen policy; q_greedy = frozen-Q re-rank, no
adaptation) against value_shift_qr; report regret vs each.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = "results/zeroshot"
ORDER = [
    "nominal",
    "action_actuator_0.60", "action_actuator_0.40", "action_bias_0.20", "action_noise_0.20",
    "env_friction_0.60", "env_friction_1.50", "env_damping_0.60", "env_damping_1.50",
    "body_mass_1.50", "body_mass_0.60", "compound_mild", "compound_hard",
    "sudden_actuator_0.40", "gradual_actuator_0.20", "multi_sudden_actuator",
]
METHODS = ["full_pearl_only", "q_greedy", "value_shift_qr"]


def cell(regime, method):
    # K=0, M=1 -> tag is just "<method>_K0" (no _M suffix when M==1).
    fs = glob.glob(f"{ROOT}/{regime}/{method}_K0/seed*/aggregate.json")
    v = [json.load(open(f))["mean_return"] for f in fs]
    return (np.mean(v), np.std(v), len(v)) if v else None


def main():
    print("Zero-shot (K=0 warmup, M=1 single agent, 3 seeds) — mean return")
    print(f"{'regime':22s}{'full_pearl':>13s}{'q_greedy':>13s}{'vsqr':>13s}{'vs base':>9s}{'vs qg':>9s}")
    nwin_base = nwin_qg = nharm = 0
    for r in ORDER:
        c = {m: cell(r, m) for m in METHODS}
        if c["value_shift_qr"] is None:
            continue
        fp, qg, vs = c["full_pearl_only"], c["q_greedy"], c["value_shift_qr"]
        row = f"{r:22s}"
        row += f"{fp[0]:13.0f}" if fp else f"{'-':>13s}"
        row += f"{qg[0]:13.0f}" if qg else f"{'-':>13s}"
        row += f"{vs[0]:8.0f}(s{vs[1]:3.0f})"
        rb = (vs[0] - fp[0]) if fp else None
        rq = (vs[0] - qg[0]) if qg else None
        row += f"{rb:+9.0f}" if rb is not None else f"{'-':>9s}"
        row += f"{rq:+9.0f}" if rq is not None else f"{'-':>9s}"
        print(row)
        if rb is not None:
            if r != "nominal" and rb > 0: nwin_base += 1
            if r != "nominal" and rb < -5: nharm += 1
        if rq is not None and r != "nominal" and rq > 0: nwin_qg += 1
    n_ood = len([r for r in ORDER if r != "nominal"])
    print(f"\nvsqr beats frozen policy on {nwin_base}/{n_ood} OOD regimes; beats q_greedy on {nwin_qg}/{n_ood};")
    print(f"hurts (regret<-5 vs frozen) on {nharm}/{n_ood} OOD regimes. (zero-shot: no warmup, single agent)")


if __name__ == "__main__":
    main()
