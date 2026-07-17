"""Evaluate value-shift action re-ranking (Q_pearl + gamma*dV) on full PEARL.

Loads a frozen PEARL checkpoint + an offline world model (full_pearl_dynamics.pt),
then runs few-shot continual evaluation per (regime, seed): a few warmup episodes
adapt the online BRPC dynamics-residual in the regime, then test episodes are
measured (BRPC keeps adapting).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.eval.eval_full_pearl import load_checkpoint
from pearl_brpc_action_adapter.eval.eval_full_pearl_dynamics_lookahead import (
    METHODS,
    evaluate,
    fleet_capable,
    load_dynamics,
)


def summarize(rows):
    by_key = {}
    for row in rows:
        key = (row["method"], row["regime"], row.get("warmup_episodes", 0), row.get("num_agents", 1))
        by_key.setdefault(key, []).append(row)
    out = []
    for (method, regime, warmup, n_agents), items in sorted(by_key.items()):
        out.append(
            {
                "method": method,
                "regime": regime,
                "n_seeds": len(items),
                "warmup_episodes": warmup,
                "num_agents": n_agents,
                "mean_return": float(np.mean([x["mean_return"] for x in items])),
                "std_return": float(np.std([x["mean_return"] for x in items])),
                "mean_length": float(np.mean([x["mean_length"] for x in items])),
                "mean_q_lift": float(np.mean([x["mean_mean_q_lift"] for x in items])),
                "mean_value_shift_term": float(np.mean([x["mean_mean_value_shift_term"] for x in items])),
                "mean_gate": float(np.mean([x["mean_mean_gate"] for x in items])),
                "mean_raw_resid_norm": float(np.mean([x["mean_mean_raw_resid_norm"] for x in items])),
                "mean_correction_sq": float(np.mean([x["mean_mean_correction_sq"] for x in items])),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate value-shift lookahead re-ranking on full PEARL.")
    parser.add_argument("--config", default="configs/full_pearl_dynamics_lookahead_smoke.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dynamics", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None, help="override lookahead.warmup_episodes (K sweep)")
    parser.add_argument("--num-agents", type=int, default=1, help="fleet size M (value_shift only); shares one BRPC")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["checkpoints"]["dir"])
    ckpt = Path(args.checkpoint or ckpt_dir / "full_pearl_best.pt")
    dyn_path = Path(args.dynamics or ckpt_dir / "full_pearl_dynamics.pt")
    output_root = Path(args.output_root or cfg["eval"]["output_root"])
    methods = args.methods or cfg.get("lookahead", {}).get("methods", METHODS)
    seeds = args.seeds or cfg["eval"].get("seeds", [0])

    ckpt_cfg, meta, models = load_checkpoint(ckpt, device)
    dyn, nominal_resid = load_dynamics(dyn_path, meta, device)
    # eval config drives regimes / latent / env / lookahead.
    ckpt_cfg["eval"] = cfg["eval"]
    ckpt_cfg["eval_regimes"] = cfg["eval_regimes"]
    ckpt_cfg["lookahead"] = cfg["lookahead"]
    if "finetune" in cfg:
        ckpt_cfg["finetune"] = cfg["finetune"]
    ckpt_cfg["latent"] = {**ckpt_cfg.get("latent", {}), **cfg.get("latent", {})}
    ckpt_cfg["env"] = {**ckpt_cfg.get("env", {}), **cfg.get("env", {})}

    rows = []
    for seed in seeds:
        for regime in cfg["eval_regimes"]:
            for method in methods:
                if method not in METHODS:
                    raise ValueError(f"Unknown method: {method}")
                # Fleet (M>1) only applies to online-adapting methods; baselines stay single-agent.
                eff_m = args.num_agents if fleet_capable(method) else 1
                # Tag the subfolder with K/M when sweeping so one output-root holds the grid.
                tag = method
                if args.warmup is not None:
                    tag += f"_K{args.warmup}"
                if eff_m > 1:
                    tag += f"_M{eff_m}"
                out_dir = output_root / regime["name"] / tag / f"seed{seed}"
                agg_path = out_dir / "aggregate.json"
                if args.skip_existing and agg_path.exists():
                    with agg_path.open("r", encoding="utf-8") as f:
                        rows.append(json.load(f))
                    print(f"Skip existing: {regime['name']}/{tag}/seed{seed}", flush=True)
                    continue
                print(f"=== Lookahead eval {tag} | {regime['name']} | seed={seed} ===", flush=True)
                rows.append(
                    evaluate(
                        ckpt_cfg, meta, models, dyn, nominal_resid, method, regime, seed, out_dir,
                        warmup_override=args.warmup, num_agents=eff_m,
                    )
                )

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
            f"{row['method']:14s} K={row['warmup_episodes']} M={row['num_agents']} {row['regime']:20s} "
            f"return={row['mean_return']:.1f} length={row['mean_length']:.1f} "
            f"dV_term={row['mean_value_shift_term']:.3f} "
            f"gate={row['mean_gate']:.3f} resid={row['mean_raw_resid_norm']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
