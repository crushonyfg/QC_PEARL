from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.eval.eval_full_pearl import load_checkpoint
from pearl_brpc_action_adapter.eval.eval_full_pearl_z_reweight import METHODS, evaluate


def load_nominal_stats(cfg):
    path = cfg.get("z_reweight", {}).get("nominal_stats_path")
    if not path:
        return
    stats_path = Path(path)
    if not stats_path.exists():
        return
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    cfg["z_reweight"]["nominal_abs_p95"] = float(stats.get("nominal_abs_p95", cfg["z_reweight"].get("nominal_abs_p95", 1.0)))


def summarize(rows):
    by_key = {}
    for row in rows:
        by_key.setdefault((row["method"], row["regime"]), []).append(row)
    out = []
    for (method, regime), items in sorted(by_key.items()):
        out.append(
            {
                "method": method,
                "regime": regime,
                "n_seeds": len(items),
                "mean_return": float(np.mean([x["mean_return"] for x in items])),
                "std_return": float(np.std([x["mean_return"] for x in items])),
                "mean_length": float(np.mean([x["mean_length"] for x in items])),
                "std_length": float(np.std([x["mean_length"] for x in items])),
                "mean_abs_eta": float(np.mean([x["mean_mean_abs_eta"] for x in items])),
                "mean_p95_abs_eta": float(np.mean([x["mean_p95_abs_eta"] for x in items])),
                "mean_gate": float(np.mean([x["mean_mean_gate"] for x in items])),
                "mean_z_shift": float(np.mean([x["mean_mean_z_shift"] for x in items])),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate posterior z-reweighting on full PEARL.")
    parser.add_argument("--config", default="configs/full_pearl_z_reweight_smoke.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    load_nominal_stats(cfg)
    ckpt = Path(args.checkpoint or Path(cfg["checkpoints"]["dir"]) / "full_pearl_best.pt")
    output_root = Path(args.output_root or cfg["eval"]["output_root"])
    methods = args.methods or cfg.get("z_reweight", {}).get("methods", METHODS)
    seeds = args.seeds or cfg["eval"].get("seeds", [0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_cfg, meta, models = load_checkpoint(ckpt, device)
    ckpt_cfg["eval"] = cfg["eval"]
    ckpt_cfg["eval_regimes"] = cfg["eval_regimes"]
    ckpt_cfg["z_reweight"] = cfg["z_reweight"]
    rows = []
    for seed in seeds:
        for regime in cfg["eval_regimes"]:
            for method in methods:
                if method not in METHODS:
                    raise ValueError(f"Unknown method: {method}")
                out_dir = output_root / regime["name"] / method / f"seed{seed}"
                agg_path = out_dir / "aggregate.json"
                if args.skip_existing and agg_path.exists():
                    with agg_path.open("r", encoding="utf-8") as f:
                        rows.append(json.load(f))
                    print(f"Skip existing: {regime['name']}/{method}/seed{seed}", flush=True)
                    continue
                print(f"=== z-reweight eval {method} | {regime['name']} | seed={seed} ===", flush=True)
                rows.append(evaluate(ckpt_cfg, meta, models, method, regime, seed, out_dir))
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
    for row in summary:
        print(
            f"{row['method']:18s} {row['regime']:18s} "
            f"return={row['mean_return']:.1f} length={row['mean_length']:.1f} "
            f"eta={row['mean_abs_eta']:.3f} z_shift={row['mean_z_shift']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
