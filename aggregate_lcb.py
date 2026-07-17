import glob, json
import numpy as np

REG = ["nominal", "ood_actuator_0.60", "sudden_actuator_0.40", "multi_sudden_actuator"]
base = {"nominal": 1465, "ood_actuator_0.60": 719, "sudden_actuator_0.40": 610, "multi_sudden_actuator": 649}
SRC = [("beta0", "results/full_pearl_dynamics_lookahead_ft"),
       ("beta1", "results/ft_sweep/vsqr_beta1"),
       ("beta2", "results/ft_sweep/vsqr_beta2")]


def cell(root, reg):
    fs = glob.glob(f"{root}/{reg}/value_shift_qr_K3_M3/seed*/aggregate.json")
    v = [json.load(open(f))["mean_return"] for f in fs]
    return (np.mean(v), np.std(v)) if v else None


print("LCB beta sweep (vsqr, K3 M3, 3 seeds). cell = return (seed-std, regret vs full_pearl_only)")
print(f"{'':8s}" + "".join(f"{r.split('_')[0]:>22s}" for r in REG) + f"{'WORSTreg':>10s}")
for lab, root in SRC:
    row = f"{lab:8s}"
    regrets = []
    for r in REG:
        c = cell(root, r)
        if c:
            row += f"{c[0]:8.0f}(s{c[1]:3.0f},{c[0]-base[r]:+5.0f})"
            regrets.append(c[0] - base[r])
        else:
            row += f"{'-':>22s}"
    row += f"{min(regrets):+10.0f}" if regrets else ""
    print(row)
