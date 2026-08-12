from __future__ import annotations

import argparse
import subprocess
import sys

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.experiments.train_full_pearl import train_full_pearl


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate standard full PEARL baseline.")
    parser.add_argument("--config", default="configs/full_pearl_hopper_smoke.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.skip_train:
        print("=== Training full PEARL ===", flush=True)
        train_full_pearl(cfg, seed=args.seed)
    if not args.skip_eval:
        print("=== Evaluating full PEARL ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pearl_brpc_action_adapter.experiments.run_full_pearl_evals",
                "--config",
                args.config,
                "--skip-existing",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
