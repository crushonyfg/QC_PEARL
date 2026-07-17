#!/usr/bin/env bash
# Advantage B (tracking non-stationary drift). K=3, M=3, 3 seeds, regimes:
# nominal / gradual_actuator_0.20 (continuous drift) / multi_sudden_actuator.
# - vsqr forgetting-factor sweep q_rho in {0.999,0.99,0.95,0.90}: does a filter that
#   actually FORGETS track the drift and rescue multi_sudden? (0.999 ~ no forgetting.)
# - fine-tune with full (100k) vs recency (3k) replay buffer: the fair tracking baseline
#   (recency-weighting is the only way batch SGD can track a moving target).
# Per-step reward traces -> time-bins (bin_mean_reward / bin_alive_frac) for the curves.
set -e
RUN="conda run -n bi-rl python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals --num-agents 3 --skip-existing"

# Zero-shot regret reference on the new regimes.
$RUN --config configs/drift/base_drift.json --methods full_pearl_only q_greedy --warmup 0

# vsqr forgetting-factor sweep.
for rho in 0.999 0.99 0.95 0.90; do
  $RUN --config configs/drift/rho$rho.json --methods value_shift_qr --warmup 3
done

# Fine-tune: full-history vs recency buffer.
for c in ft_fullbuf ft_recency; do
  $RUN --config configs/drift/$c.json --methods finetune_lastlayer finetune_full --warmup 3
done
echo "DRIFT DONE"
