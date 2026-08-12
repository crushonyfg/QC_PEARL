#!/usr/bin/env bash
set -e
RUN="conda run -n bi-rl python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals --num-agents 3 --warmup 3 --skip-existing --methods value_shift_qr"
for c in vsqr_beta1 vsqr_beta2; do
  $RUN --config configs/ft_sweep/$c.json
done
echo "LCB DONE"
