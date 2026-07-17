# QC-PEARL

Code release for **Lightweight Test-Time Q Calibration for Frozen Meta-RL
under Dynamics Shift**.

QC-PEARL keeps a trained PEARL encoder, actor, twin critics, and offline world
model frozen at deployment. It fits a Bayesian linear Bellman-residual head on
the critics' penultimate features and uses the calibrated score to rerank a
small action set around the frozen policy. A dynamics-residual gate suppresses
the correction when there is little evidence of a deployment shift.

The implementation predates the paper name, so QC-PEARL appears in the code and
configs as `value_shift_qr` (also abbreviated VSQR).

## Repository layout

```text
pearl_brpc_action_adapter/
  pearl/                         # PEARL encoder, actor, critics, SAC losses
  dynamics/                      # frozen latent-conditioned world model
  eval/
    eval_full_pearl_dynamics_lookahead.py  # QC-PEARL implementation
    eval_full_pearl_finetune.py            # test-time fine-tuning baselines
  experiments/
    train_full_pearl.py          # train the frozen PEARL backbone
    train_full_pearl_dynamics.py # train the offline world model
    run_full_pearl_dynamics_lookahead_evals.py
brpc_latent_mpc/                 # small shared Bayesian/RFF/env utilities
configs/                         # paper, ablation, and smoke configurations
docs/value_shift_qr_methods.md   # method equations and implementation notes
results/                         # compact paper summaries and figures
qc_pearl.pdf                     # paper manuscript
```

`brpc_latent_mpc/` contains only the utility modules imported by QC-PEARL; the
separate latent-MPC research project and its experiment code are not included.

## Installation

Python 3.10 was used for the reported experiments.

```bash
conda create -n qc-pearl python=3.10 -y
conda activate qc-pearl
pip install -r requirements.txt
```

MuJoCo must be available through Gymnasium. For CUDA, install the PyTorch build
matching the local CUDA driver before installing the remaining requirements.

## Train the frozen models

Run commands from the repository root.

```bash
python -m pearl_brpc_action_adapter.experiments.train_full_pearl \
  --config configs/full_pearl_hopper.json --seed 0

python -m pearl_brpc_action_adapter.experiments.train_full_pearl_dynamics \
  --config configs/full_pearl_dynamics_lookahead_qr.json --seed 0
```

These commands create `checkpoints/full_pearl/full_pearl_best.pt` and
`checkpoints/full_pearl/full_pearl_dynamics.pt`. Checkpoints are intentionally
excluded from Git because they are generated artifacts.

## Evaluate QC-PEARL

The main OOD evaluation compares the frozen PEARL policy, Q-greedy reranking,
and QC-PEARL:

```bash
python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals \
  --config configs/full_pearl_dynamics_lookahead_oodtypes.json \
  --methods full_pearl_only q_greedy value_shift_qr \
  --num-agents 3 --warmup 3 --skip-existing
```

For the test-time fine-tuning comparison and online/anytime study:

```bash
bash run_ft_vs_qr.sh

python -m pearl_brpc_action_adapter.experiments.run_full_pearl_dynamics_lookahead_evals \
  --config configs/full_pearl_dynamics_lookahead_online.json \
  --num-agents 3 --warmup 0 --skip-existing
```

Additional scripts reproduce the forgetting-factor, LCB, fleet-size, n-step,
and hyperparameter studies. Existing compact summaries under `results/` match
the tables and plots in the manuscript.

## Method pointers

The core implementation is in
[`compute_scores`](pearl_brpc_action_adapter/eval/eval_full_pearl_dynamics_lookahead.py):

- candidate actions are sampled locally around the frozen PEARL policy;
- `QNetwork.features()` exposes the frozen critic representation;
- `BRPCResidualCalibrator` performs recursive Bayesian linear updates;
- `value_shift_qr` adds the calibrated Bellman-residual prediction, with an
  optional lower-confidence-bound penalty;
- the world-model residual controls the deployment-shift gate.

See [`docs/value_shift_qr_methods.md`](docs/value_shift_qr_methods.md) for the
full equations and experiment protocol.
