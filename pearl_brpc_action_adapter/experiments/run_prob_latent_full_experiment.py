from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.experiments.train_prob_latent_policy import train_all


def main():
    parser = argparse.ArgumentParser(description="End-to-end probabilistic latent SAC + BRPC experiment.")
    parser.add_argument("--config", default="configs/prob_latent_sac_hopper.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not args.skip_train:
        print("=== Training stable probabilistic latent SAC pipeline ===", flush=True)
        train_all(cfg, seed=args.seed)

    if not args.skip_eval:
        print("=== Batch evaluation ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pearl_brpc_action_adapter.experiments.run_prob_latent_evals",
                "--config",
                args.config,
                "--skip-existing",
            ],
            check=True,
        )

    summary_path = Path(cfg["eval"]["output_root"]) / "summary_by_method_regime.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        print("\n=== Summary ===", flush=True)
        for row in summary:
            print(
                f"{row['method']:28s} {row['regime']:20s} "
                f"return={row['mean_return']:.1f} length={row['mean_length']:.1f} "
                f"e_base={row['mean_e_base']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
