from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.envs.make_env import EvalDynamicsSchedule, make_env
from pearl_brpc_action_adapter.experiments.train_full_pearl import _context_tensor, _make_context_item
from pearl_brpc_action_adapter.eval.eval_full_pearl import load_checkpoint


def infer_z(encoder, context: List[np.ndarray], latent_dim: int, min_context: int, device: torch.device) -> torch.Tensor:
    if len(context) < min_context:
        return torch.zeros(1, latent_dim, device=device)
    with torch.no_grad():
        return encoder.infer_mean(_context_tensor(context, device))


def soft_value(actor, q1, q2, state: np.ndarray, z: torch.Tensor, alpha: float, n_actions: int, device: torch.device) -> float:
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    states = st.expand(n_actions, -1)
    z_batch = z.expand(n_actions, -1)
    with torch.no_grad():
        actions, log_probs = actor.sample_action_logprob(states, z_batch)
        q_min = torch.minimum(q1(states, actions, z_batch), q2(states, actions, z_batch))
        values = q_min - alpha * log_probs
    return float(values.mean().cpu().item())


def q_min_value(q1, q2, state: np.ndarray, action: np.ndarray, z: torch.Tensor, device: torch.device) -> float:
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    at = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        q_min = torch.minimum(q1(st, at, z), q2(st, at, z))
    return float(q_min.cpu().item())


def deterministic_action(actor, state: np.ndarray, z: torch.Tensor, device: torch.device) -> np.ndarray:
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return actor.deterministic_action(st, z)[0].cpu().numpy().astype(np.float32)


def run_episode(
    cfg: Dict,
    meta: Dict,
    models: Dict,
    regime: Dict,
    seed: int,
    episode_idx: int,
    n_value_actions: int,
    ewma_rho: float,
) -> Tuple[List[Dict], Dict]:
    device = next(models["actor"].parameters()).device
    actor, q1, q2, encoder = models["actor"], models["q1"], models["q2"], models["encoder"]
    env = make_env(cfg["env"]["name"], seed + episode_idx)
    schedule = EvalDynamicsSchedule(regime, cfg["dynamics_randomization"]["train_range"])
    context: Deque[np.ndarray] = deque(maxlen=int(cfg["latent"].get("eval_context_max", 50)))
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    max_steps = int(cfg["env"]["max_episode_steps"])
    gamma = float(cfg["pearl"]["gamma"])
    alpha = float(cfg["pearl"]["sac_alpha"])
    rows: List[Dict] = []
    shift_step = regime.get("shift_step")
    s, _ = env.reset(seed=seed + episode_idx)
    ep_return = 0.0
    ewma_abs_eta = 0.0
    for t in range(max_steps):
        xi = schedule.xi_at(t)
        env.set_dynamics(xi)
        z = infer_z(encoder, list(context), meta["latent_dim"], min_context, device)
        action = deterministic_action(actor, s, z, device)
        q_sa = q_min_value(q1, q2, s, action, z, device)
        s_next, reward, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        v_next = 0.0 if term else soft_value(actor, q1, q2, s_next, z, alpha, n_value_actions, device)
        eta = float(reward + gamma * (0.0 if term else 1.0) * v_next - q_sa)
        abs_eta = abs(eta)
        ewma_abs_eta = (1.0 - ewma_rho) * ewma_abs_eta + ewma_rho * abs_eta
        ep_return += float(reward)
        z_np = z.squeeze(0).detach().cpu().numpy()
        rows.append(
            {
                "regime": regime["name"],
                "seed": seed,
                "episode": episode_idx,
                "t": t,
                "reward": float(reward),
                "return_so_far": ep_return,
                "done": done,
                "terminated": bool(term),
                "truncated": bool(trunc),
                "eta": eta,
                "abs_eta": abs_eta,
                "eta_sq": eta * eta,
                "ewma_abs_eta": ewma_abs_eta,
                "q_sa": q_sa,
                "v_next": v_next,
                "action_norm": float(np.linalg.norm(action)),
                "z_norm": float(np.linalg.norm(z_np)),
                "xi_mass": float(xi["mass"]),
                "xi_friction": float(xi["friction"]),
                "xi_damping": float(xi["damping"]),
                "xi_actuator": float(xi["actuator"]),
                "phase": "post_shift" if shift_step is not None and t >= int(shift_step) else "pre_shift",
            }
        )
        context.append(_make_context_item(s, action, reward, s_next, done))
        s = s_next
        if done:
            break
    env.close()
    summary = {
        "regime": regime["name"],
        "seed": seed,
        "episode": episode_idx,
        "return": ep_return,
        "length": len(rows),
        "mean_abs_eta": float(np.mean([r["abs_eta"] for r in rows])) if rows else 0.0,
        "p90_abs_eta": float(np.percentile([r["abs_eta"] for r in rows], 90)) if rows else 0.0,
        "p95_abs_eta": float(np.percentile([r["abs_eta"] for r in rows], 95)) if rows else 0.0,
        "max_abs_eta": float(np.max([r["abs_eta"] for r in rows])) if rows else 0.0,
    }
    if shift_step is not None:
        pre = [r["abs_eta"] for r in rows if r["t"] < int(shift_step)]
        post = [r["abs_eta"] for r in rows if r["t"] >= int(shift_step)]
        summary["pre_shift_mean_abs_eta"] = float(np.mean(pre)) if pre else 0.0
        summary["post_shift_mean_abs_eta"] = float(np.mean(post)) if post else 0.0
        summary["post_shift_p95_abs_eta"] = float(np.percentile(post, 95)) if post else 0.0
        summary["post_shift_steps"] = len(post)
    else:
        summary["pre_shift_mean_abs_eta"] = summary["mean_abs_eta"]
        summary["post_shift_mean_abs_eta"] = 0.0
        summary["post_shift_p95_abs_eta"] = 0.0
        summary["post_shift_steps"] = 0
    return rows, summary


def add_nominal_normalization(rows: List[Dict], summaries: List[Dict], eps: float = 1e-8) -> Dict:
    nominal_abs = np.asarray([r["abs_eta"] for r in rows if r["regime"] == "nominal"], dtype=np.float64)
    if nominal_abs.size == 0:
        stats = {"nominal_abs_mean": 0.0, "nominal_abs_std": 1.0, "nominal_abs_p95": 1.0}
    else:
        stats = {
            "nominal_abs_mean": float(np.mean(nominal_abs)),
            "nominal_abs_std": float(np.std(nominal_abs)),
            "nominal_abs_p95": float(np.percentile(nominal_abs, 95)),
        }
    for row in rows:
        row["E_zscore"] = (row["abs_eta"] - stats["nominal_abs_mean"]) / (stats["nominal_abs_std"] + eps)
        row["E_p95_ratio"] = row["abs_eta"] / (stats["nominal_abs_p95"] + eps)
    for summary in summaries:
        vals = [r for r in rows if r["regime"] == summary["regime"] and r["seed"] == summary["seed"] and r["episode"] == summary["episode"]]
        summary["mean_E_zscore"] = float(np.mean([r["E_zscore"] for r in vals])) if vals else 0.0
        summary["p95_E_p95_ratio"] = float(np.percentile([r["E_p95_ratio"] for r in vals], 95)) if vals else 0.0
        post_vals = [r for r in vals if r["phase"] == "post_shift"]
        summary["post_shift_mean_E_zscore"] = float(np.mean([r["E_zscore"] for r in post_vals])) if post_vals else 0.0
        summary["post_shift_p95_E_p95_ratio"] = float(np.percentile([r["E_p95_ratio"] for r in post_vals], 95)) if post_vals else 0.0
    return stats


def aggregate_summaries(summaries: List[Dict]) -> List[Dict]:
    by_regime: Dict[str, List[Dict]] = {}
    for row in summaries:
        by_regime.setdefault(row["regime"], []).append(row)
    out = []
    for regime, items in sorted(by_regime.items()):
        out.append(
            {
                "regime": regime,
                "n_episodes": len(items),
                "mean_return": float(np.mean([x["return"] for x in items])),
                "mean_length": float(np.mean([x["length"] for x in items])),
                "mean_abs_eta": float(np.mean([x["mean_abs_eta"] for x in items])),
                "mean_p90_abs_eta": float(np.mean([x["p90_abs_eta"] for x in items])),
                "mean_p95_abs_eta": float(np.mean([x["p95_abs_eta"] for x in items])),
                "mean_E_zscore": float(np.mean([x["mean_E_zscore"] for x in items])),
                "mean_p95_E_p95_ratio": float(np.mean([x["p95_E_p95_ratio"] for x in items])),
                "post_shift_steps": int(np.sum([x["post_shift_steps"] for x in items])),
                "post_shift_mean_abs_eta": float(np.mean([x["post_shift_mean_abs_eta"] for x in items if x["post_shift_steps"] > 0]))
                if any(x["post_shift_steps"] > 0 for x in items)
                else 0.0,
                "post_shift_p95_abs_eta": float(np.mean([x["post_shift_p95_abs_eta"] for x in items if x["post_shift_steps"] > 0]))
                if any(x["post_shift_steps"] > 0 for x in items)
                else 0.0,
                "post_shift_mean_E_zscore": float(np.mean([x["post_shift_mean_E_zscore"] for x in items if x["post_shift_steps"] > 0]))
                if any(x["post_shift_steps"] > 0 for x in items)
                else 0.0,
                "post_shift_p95_E_p95_ratio": float(np.mean([x["post_shift_p95_E_p95_ratio"] for x in items if x["post_shift_steps"] > 0]))
                if any(x["post_shift_steps"] > 0 for x in items)
                else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Stage A Bellman residual diagnostic for full PEARL.")
    parser.add_argument("--config", default="configs/full_pearl_bellman_diag.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_root = Path(args.output_root or cfg["bellman_diag"]["output_root"])
    ckpt = Path(args.checkpoint or Path(cfg["checkpoints"]["dir"]) / "full_pearl_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_cfg, meta, models = load_checkpoint(ckpt, device)
    # Use diagnostic config for eval settings while preserving checkpoint model settings.
    ckpt_cfg["eval"] = cfg["eval"]
    ckpt_cfg["eval_regimes"] = cfg["eval_regimes"]
    ckpt_cfg["bellman_diag"] = cfg["bellman_diag"]
    all_rows: List[Dict] = []
    episode_summaries: List[Dict] = []
    for seed in cfg["eval"].get("seeds", [0]):
        for regime in cfg["eval_regimes"]:
            for ep in range(cfg["eval"]["num_episodes"]):
                print(f"=== Bellman diag {regime['name']} seed={seed} ep={ep} ===", flush=True)
                rows, summary = run_episode(
                    ckpt_cfg,
                    meta,
                    models,
                    regime,
                    seed,
                    ep,
                    int(cfg["bellman_diag"].get("value_action_samples", 16)),
                    float(cfg["bellman_diag"].get("ewma_rho", 0.05)),
                )
                all_rows.extend(rows)
                episode_summaries.append(summary)
                print(
                    f"  return={summary['return']:.1f} length={summary['length']} "
                    f"mean_abs_eta={summary['mean_abs_eta']:.3f} p95={summary['p95_abs_eta']:.3f}",
                    flush=True,
                )
    stats = add_nominal_normalization(all_rows, episode_summaries)
    aggregate = aggregate_summaries(episode_summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "transition_bellman_residuals.csv", all_rows)
    write_csv(output_root / "episode_summary.csv", episode_summaries)
    write_csv(output_root / "summary_by_regime.csv", aggregate)
    with (output_root / "summary_by_regime.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    with (output_root / "nominal_residual_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved Bellman diagnostic to {output_root}", flush=True)
    for row in aggregate:
        print(
            f"{row['regime']:20s} return={row['mean_return']:.1f} length={row['mean_length']:.1f} "
            f"abs_eta={row['mean_abs_eta']:.3f} E={row['mean_E_zscore']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
