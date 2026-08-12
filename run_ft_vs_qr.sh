#!/usr/bin/env bash
# value_shift_qr (M=3) vs test-time fine-tune (lastlayer / full, M=3) across K in {0,1,3,5}.
# Plus zero-shot reference baselines (run once at K=0). Resumable via --skip-existing.
set -e
CFG=configs/full_pearl_dynamics_lookahead_ft.json
RUN="conda run -n bi-rl python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals --config $CFG --num-agents 3 --skip-existing"

# Zero-shot reference line (no fine-tune / no Q-residual): run once.
$RUN --methods full_pearl_only q_greedy --warmup 0

# Main few-shot comparison across the warmup-episode budget K.
for K in 0 1 3 5; do
  $RUN --methods value_shift_qr finetune_lastlayer finetune_full --warmup $K
done

echo "ALL DONE"
