from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.eval.eval_full_pearl import evaluate, load_checkpoint


def summarize(rows):
    by_key = {}
    for row in rows:
        by_key.setdefault(row["regime"], []).append(row)
    out = []
    for regime, items in sorted(by_key.items()):
        out.append(
            {
                "method": "full_pearl",
                "regime": regime,
                "n_seeds": len(items),
                "mean_return": float(np.mean([x["mean_return"] for x in items])),
                "std_return": float(np.std([x["mean_return"] for x in items])),
                "mean_length": float(np.mean([x["mean_length"] for x in items])),
                "std_length": float(np.std([x["mean_length"] for x in items])),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate standard full PEARL baseline.")
    parser.add_argument("--config", default="configs/full_pearl_hopper_smoke.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint or Path(cfg["checkpoints"]["dir"]) / "full_pearl_best.pt")
    output_root = Path(args.output_root or cfg["eval"]["output_root"])
    seeds = args.seeds or cfg["eval"].get("seeds", [0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_cfg, meta, models = load_checkpoint(ckpt, device)
    # Use runtime eval config for regimes/episodes, checkpoint config for model dims.
    ckpt_cfg["eval"] = cfg["eval"]
    ckpt_cfg["eval_regimes"] = cfg["eval_regimes"]
    rows = []
    for seed in seeds:
        for regime in cfg["eval_regimes"]:
            out_dir = output_root / regime["name"] / "full_pearl" / f"seed{seed}"
            agg_path = out_dir / "aggregate.json"
            if args.skip_existing and agg_path.exists():
                with agg_path.open("r", encoding="utf-8") as f:
                    rows.append(json.load(f))
                print(f"Skip existing: {regime['name']}/seed{seed}", flush=True)
                continue
            print(f"=== Eval full_pearl | {regime['name']} | seed={seed} ===", flush=True)
            rows.append(evaluate(ckpt_cfg, meta, models, regime, seed, out_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "all_runs.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    summary = summarize(rows)
    with (output_root / "summary_by_method_regime.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if summary:
        with (output_root / "summary_by_method_regime.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    print(f"Saved summary to {output_root / 'summary_by_method_regime.json'}", flush=True)


if __name__ == "__main__":
    main()
