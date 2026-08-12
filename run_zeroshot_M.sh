#!/usr/bin/env bash
# Zero-shot (K=0, NO warmup) but with fleet breadth M sharing one calibrator: does M-fold
# per-step data average out the action-noise corruption (std 172 at M=1) WITHOUT any warmup?
set -e
RUN="conda run -n bi-rl python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals --config configs/full_pearl_dynamics_lookahead_zeroshot.json --methods value_shift_qr --warmup 0 --skip-existing"
$RUN --num-agents 3
$RUN --num-agents 5
echo "ZEROSHOT_M DONE"
