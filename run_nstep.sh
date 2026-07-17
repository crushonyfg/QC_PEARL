#!/usr/bin/env bash
# n-step / lambda-return QR variants vs 1-step, on regimes where 1-step may be bias-capped
# (multi_sudden/gradual/compound) + controls (nominal safety, ood0.6 where 1-step is already
# near the oracle ceiling). K=3, M=3, 2 seeds.
set -e
B="conda run -n bi-rl python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals --methods value_shift_qr --warmup 3 --num-agents 3 --skip-existing"
$B --config configs/nstep/base.json
for c in n3 n5 lam09 lam10; do $B --config configs/nstep/$c.json; done
echo "NSTEP DONE"
