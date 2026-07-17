from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.experiments.train_all import train_all


def main():
    parser = argparse.ArgumentParser(description="End-to-end PEARL+BRPC experiment.")
    parser.add_argument("--config", default="configs/pearl_full.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(cfg["checkpoints"]["dir"]) / "pearl_brpc_checkpoint.pt"

    if not args.skip_train:
        print("=== Training ===")
        train_all(cfg, seed=args.seed)

    if not args.skip_eval:
        print("=== Batch Evaluation ===")
        cmd = [
            sys.executable,
            "-m",
            "pearl_brpc_action_adapter.experiments.run_all_evals",
            "--config",
            args.config,
            "--skip-existing",
        ]
        subprocess.run(cmd, check=True)

    summary_path = Path(cfg["eval"]["output_root"]) / "summary_by_method_regime.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        print("\n=== Summary ===")
        for row in summary:
            print(
                f"{row['method']:16s} {row['regime']:20s} "
                f"return={row['mean_return']:.2f}±{row['std_return']:.2f} "
                f"e_base={row['mean_e_base']:.3f}"
            )


if __name__ == "__main__":
    main()
