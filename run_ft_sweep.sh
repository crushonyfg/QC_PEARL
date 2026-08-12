#!/usr/bin/env bash
# Hyperparameter ROBUSTNESS sweep, K=3, M=3, all 4 regimes (incl nominal), 3 seeds.
# Protocol: every config is evaluated on ALL regimes; selection must stay regime-agnostic
# (no per-OOD-regime tuning -- there is no validation set for an unknown deployment shift).
# Question 1 (fine-tune): is there ONE lr/updates config that matches vsqr across regimes?
# Question 2 (vsqr): is vsqr sensitive to its own knob (q_prior_var)?
# Note: ft lr3e-4/u200 and vsqr q_prior_var=0.25 already exist in the main K=3 run
#       (results/full_pearl_dynamics_lookahead_ft, finetune_*_K3_M3 / value_shift_qr_K3_M3).
set -e
RUN="conda run -n bi-rl python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals --num-agents 3 --warmup 3 --skip-existing"

for c in ft_lr1e-4_u50 ft_lr1e-4_u200 ft_lr3e-4_u50; do
  $RUN --config configs/ft_sweep/$c.json --methods finetune_lastlayer finetune_full
done
for c in vsqr_pv0.1 vsqr_pv1.0; do
  $RUN --config configs/ft_sweep/$c.json --methods value_shift_qr
done
echo "SWEEP DONE"
