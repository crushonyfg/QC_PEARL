"""Offline-train a latent-conditioned world model f(s,a,z)->delta_s on top of a
FROZEN full-PEARL checkpoint.

This is a non-invasive, post-hoc step: it never touches the PEARL actor / critic /
encoder weights. It reuses the SAME latent z that PEARL already learned (z is
inferred by the frozen encoder from a running context window, exactly as at eval
time), collects rollouts on the training-range tasks with the trained actor plus
exploration noise (so the model is accurate in the candidate-action neighborhood
used by the value-shift re-ranker), and fits f by MSE in a normalized space.

The world model is used at eval ONLY to provide the operating point
s'_nominal = s + f(s,a,z) at which the value-shift correction
  ΔV = V(s + f + Δf) - V(s + f)
is evaluated; the online BRPC residual Δf carries the distribution-shift signal,
and PEARL's own Q is the base score. Reward is not modeled (it cancels in ΔV up to
the s'-dependent forward term, which is handled analytically at eval from Δf).

It also measures the nominal one-step dynamics-residual scale (||s' - s - f(s,a,z)||),
which sets the BRPC obs-noise scale and the eval-time shift gate threshold.

Output: <checkpoints.dir>/full_pearl_dynamics.pt
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.dynamics.dynamics_model import NormalizedLatentDynamicsModel
from pearl_brpc_action_adapter.envs.make_env import LATENT_KEYS, make_env, sample_train_xi
from pearl_brpc_action_adapter.eval.eval_full_pearl import load_checkpoint
from pearl_brpc_action_adapter.experiments.train_full_pearl import (
    _context_tensor,
    _infer_z_for_action,
    _make_context_item,
)


def collect_transitions(
    cfg: Dict,
    meta: Dict,
    models: Dict,
    device: torch.device,
    tasks: List[Dict[str, float]],
    episodes_per_task: int,
    explore_std: float,
    seed: int,
    nominal_only: bool = False,
) -> Dict[str, np.ndarray]:
    """Roll out the trained actor (+Gaussian exploration noise) and record
    (s, a, z, delta_s). z is inferred online from the frozen encoder."""
    actor, encoder = models["actor"], models["encoder"]
    latent_dim = meta["latent_dim"]
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    context_max = int(cfg["latent"].get("eval_context_max", 50))
    max_steps = int(cfg["env"]["max_episode_steps"])
    states, actions, zs, deltas = [], [], [], []
    rng = np.random.default_rng(seed)
    for tid, xi in enumerate(tasks):
        task_xi = {k: 1.0 for k in LATENT_KEYS} if nominal_only else xi
        for ep in range(episodes_per_task):
            env = make_env(cfg["env"]["name"], seed + 7919 * tid + ep, dynamics=task_xi)
            context: Deque[np.ndarray] = deque(maxlen=context_max)
            s, _ = env.reset(seed=seed + 7919 * tid + ep)
            for _ in range(max_steps):
                if len(context) >= min_context:
                    z = _infer_z_for_action(encoder, list(context), latent_dim, device)
                else:
                    z = torch.zeros(1, latent_dim, device=device)
                st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    a = actor.deterministic_action(st, z)[0].cpu().numpy().astype(np.float32)
                if explore_std > 0.0:
                    a = a + rng.normal(0.0, explore_std, size=a.shape).astype(np.float32)
                a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
                s_next, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                states.append(s.astype(np.float32))
                actions.append(a)
                zs.append(z[0].detach().cpu().numpy().astype(np.float32))
                deltas.append((s_next - s).astype(np.float32))
                context.append(_make_context_item(s, a, r, s_next, done))
                s = s_next
                if done:
                    break
            env.close()
        if (tid + 1) % max(len(tasks) // 5, 1) == 0:
            print(f"  collected task {tid+1}/{len(tasks)}, transitions={len(states)}", flush=True)
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "zs": np.asarray(zs, dtype=np.float32),
        "deltas": np.asarray(deltas, dtype=np.float32),
    }


def compute_norm_stats(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    eps = 1e-6
    return {
        "state_mean": data["states"].mean(0),
        "state_std": data["states"].std(0) + eps,
        "delta_mean": data["deltas"].mean(0),
        "delta_std": data["deltas"].std(0) + eps,
    }


def fit_model(
    cfg: Dict,
    meta: Dict,
    model: NormalizedLatentDynamicsModel,
    data: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = data["states"].shape[0]
    rng = np.random.default_rng(seed)
    tens = {k: torch.from_numpy(v).to(device) for k, v in data.items()}
    # Pre-normalize the delta target once (model.forward_norm predicts in norm space).
    delta_norm = (tens["deltas"] - model.delta_mean) / model.delta_std
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        losses_d = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            bi = torch.from_numpy(idx).to(device)
            pred_d = model.forward_norm(tens["states"][bi], tens["actions"][bi], tens["zs"][bi])
            loss = nn.functional.mse_loss(pred_d, delta_norm[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses_d.append(float(loss.item()))
        if ep % max(epochs // 10, 1) == 0 or ep == epochs:
            print(f"  dyn epoch {ep}/{epochs} loss_delta={np.mean(losses_d):.4f}", flush=True)


def nominal_residual_scale(
    model: NormalizedLatentDynamicsModel, data: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, float]:
    """Per-dim std and aggregate norm of the dynamics residual on nominal data.
    Used (a) as BRPC obs_noise default scale and (b) for the eval gate threshold."""
    with torch.no_grad():
        pred_d = model.predict(
            torch.from_numpy(data["states"]).to(device),
            torch.from_numpy(data["actions"]).to(device),
            torch.from_numpy(data["zs"]).to(device),
        )
        resid = data["deltas"] - pred_d.cpu().numpy()
    resid_norm = np.linalg.norm(resid, axis=1)
    return {
        "resid_std_per_dim": resid.std(0).astype(np.float32).tolist(),
        "resid_norm_mean": float(np.mean(resid_norm)),
        "resid_norm_p95": float(np.percentile(resid_norm, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline-train dynamics+reward model on frozen full PEARL.")
    parser.add_argument("--config", default="configs/full_pearl_dynamics_lookahead_smoke.json")
    parser.add_argument("--checkpoint", default=None, help="frozen full PEARL checkpoint (default: <dir>/full_pearl_best.pt)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["checkpoints"]["dir"])
    ckpt = Path(args.checkpoint or ckpt_dir / "full_pearl_best.pt")
    print(f"Loading frozen PEARL checkpoint: {ckpt}", flush=True)
    ckpt_cfg, meta, models = load_checkpoint(ckpt, device)
    # eval-time latent settings come from the eval config; merge so collection matches.
    ckpt_cfg["latent"] = {**ckpt_cfg.get("latent", {}), **cfg.get("latent", {})}
    ckpt_cfg["env"] = {**ckpt_cfg.get("env", {}), **cfg.get("env", {})}

    dyn_cfg = cfg.get("dynamics_model", {})
    n_tasks = int(dyn_cfg.get("num_train_tasks", 24))
    eps_per_task = int(dyn_cfg.get("episodes_per_task", 2))
    explore_std = float(dyn_cfg.get("explore_std", 0.15))
    hidden = tuple(dyn_cfg.get("hidden", [256, 256, 256]))
    epochs = int(dyn_cfg.get("epochs", 60))
    batch_size = int(dyn_cfg.get("batch_size", 256))
    lr = float(dyn_cfg.get("lr", 1e-3))

    rng = np.random.default_rng(args.seed)
    tasks = [sample_train_xi(rng, cfg["dynamics_randomization"]["train_range"]) for _ in range(n_tasks)]

    print(f"Collecting train transitions ({n_tasks} tasks x {eps_per_task} eps, explore_std={explore_std})...", flush=True)
    data = collect_transitions(ckpt_cfg, meta, models, device, tasks, eps_per_task, explore_std, args.seed)
    print(f"  total transitions: {data['states'].shape[0]}", flush=True)

    model = NormalizedLatentDynamicsModel(meta["state_dim"], meta["action_dim"], meta["latent_dim"], hidden).to(device)
    stats = compute_norm_stats(data)
    model.set_norm_stats(stats)

    print("Fitting world model f(s,a,z)->delta_s...", flush=True)
    fit_model(cfg, meta, model, data, device, epochs, batch_size, lr, args.seed)

    # Nominal residual scale: collect a small nominal-only set with the trained actor (no noise).
    print("Measuring nominal dynamics-residual scale...", flush=True)
    nominal_tasks = tasks[: max(n_tasks // 3, 4)]
    nominal_data = collect_transitions(
        ckpt_cfg, meta, models, device, nominal_tasks, eps_per_task, 0.0, args.seed + 5, nominal_only=True
    )
    resid_stats = nominal_residual_scale(model, nominal_data, device)
    print(f"  nominal resid_norm mean={resid_stats['resid_norm_mean']:.4f} p95={resid_stats['resid_norm_p95']:.4f}", flush=True)

    out_path = ckpt_dir / "full_pearl_dynamics.pt"
    torch.save(
        {
            "meta": meta,
            "hidden": list(hidden),
            "model": model.state_dict(),
            "norm_stats": {k: np.asarray(v).tolist() for k, v in stats.items()},
            "nominal_resid": resid_stats,
            "source_checkpoint": str(ckpt),
        },
        out_path,
    )
    print(f"Saved world model: {out_path}", flush=True)


if __name__ == "__main__":
    main()
