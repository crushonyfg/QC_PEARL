from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.eval.eval_prob_latent_policy import METHODS, evaluate


def aggregate_results(rows):
    by_key = {}
    for row in rows:
        by_key.setdefault((row["method"], row["regime"]), []).append(row)
    summary = []
    for (method, regime), items in sorted(by_key.items()):
        summary.append(
            {
                "method": method,
                "regime": regime,
                "n_seeds": len(items),
                "mean_return": float(np.mean([x["mean_return"] for x in items])),
                "std_return": float(np.std([x["mean_return"] for x in items])),
                "mean_length": float(np.mean([x["mean_length"] for x in items])),
                "mean_e_base": float(np.mean([x["mean_mean_e_base"] for x in items])),
                "mean_correction_norm": float(np.mean([x["mean_correction_norm"] for x in items])),
                "mean_latent_error": float(np.mean([x["mean_mean_latent_error"] for x in items])),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch eval for probabilistic latent SAC + BRPC adapter.")
    parser.add_argument("--config", default="configs/prob_latent_sac_hopper.json")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = Path(args.checkpoint_dir or cfg["checkpoints"]["dir"])
    methods = args.methods or cfg.get("eval_methods_all", METHODS)
    seeds = args.seeds or cfg["eval"].get("seeds", [0])
    output_root = Path(args.output_root or cfg["eval"]["output_root"])

    all_agg = []
    for seed in seeds:
        for regime in cfg["eval_regimes"]:
            regime_name = regime["name"]
            for method in methods:
                out_dir = output_root / regime_name / method / f"seed{seed}"
                agg_path = out_dir / "aggregate.json"
                if args.skip_existing and agg_path.exists():
                    with agg_path.open("r", encoding="utf-8") as f:
                        all_agg.append(json.load(f))
                    print(f"Skip existing: {regime_name}/{method}/seed{seed}", flush=True)
                    continue
                print(f"=== Eval {method} | {regime_name} | seed={seed} ===", flush=True)
                all_agg.append(evaluate(cfg, method, regime, seed, ckpt_dir, out_dir))

    output_root.mkdir(parents=True, exist_ok=True)
    per_run_path = output_root / "all_runs.json"
    with per_run_path.open("w", encoding="utf-8") as f:
        json.dump(all_agg, f, indent=2)

    grouped = aggregate_results(all_agg)
    grouped_path = output_root / "summary_by_method_regime.json"
    with grouped_path.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)
    if grouped:
        csv_path = output_root / "summary_by_method_regime.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(grouped[0].keys()))
            writer.writeheader()
            writer.writerows(grouped)
    print(f"Saved summary to {grouped_path}", flush=True)


if __name__ == "__main__":
    main()
