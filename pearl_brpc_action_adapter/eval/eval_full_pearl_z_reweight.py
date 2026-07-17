from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch

from pearl_brpc_action_adapter.envs.make_env import EvalDynamicsSchedule, make_env
from pearl_brpc_action_adapter.experiments.train_full_pearl import _context_tensor, _make_context_item
from pearl_brpc_action_adapter.eval.eval_full_pearl_residual_q import bellman_residual, deterministic_action


METHODS = ["full_pearl_only", "z_reweight", "z_reweight_gated"]


def posterior_mean_samples(
    encoder,
    context: List[np.ndarray],
    latent_dim: int,
    min_context: int,
    n_samples: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(context) < min_context:
        z0 = torch.zeros(1, latent_dim, device=device)
        return z0, z0.expand(n_samples, -1)
    ctx = _context_tensor(context, device)
    with torch.no_grad():
        mu, logvar = encoder.forward_factors(ctx)
        z_mean, z_var = encoder.aggregate(mu, logvar)
        eps = torch.randn((n_samples, latent_dim), device=device, generator=generator)
        z_samples = z_mean.expand(n_samples, -1) + torch.sqrt(z_var).expand(n_samples, -1) * eps
    return z_mean, z_samples


def residual_loss_for_z(cfg: Dict, models: Dict, window: List[Tuple], z: torch.Tensor) -> float:
    vals = []
    for state, action, reward, next_state, terminated in window:
        eta = bellman_residual(cfg, models, state, action, reward, next_state, terminated, z)
        vals.append(eta * eta)
    return float(np.sum(vals)) if vals else 0.0


def choose_z(
    cfg: Dict,
    models: Dict,
    method: str,
    context: List[np.ndarray],
    window: List[Tuple],
    ewma_ratio: float,
    latent_dim: int,
    min_context: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    zr_cfg = cfg["z_reweight"]
    n_samples = int(zr_cfg.get("num_z_samples", 8))
    z_mean, z_samples = posterior_mean_samples(models["encoder"], context, latent_dim, min_context, n_samples, device, generator)
    gate_h = 1.0 / (1.0 + math.exp(-float(zr_cfg.get("gate_slope", 8.0)) * (ewma_ratio - float(zr_cfg.get("gate_threshold", 0.65)))))
    if method == "full_pearl_only" or len(window) < int(zr_cfg.get("min_window", 4)):
        return z_mean, {"gate": 0.0, "z_shift": 0.0, "loss_std": 0.0}

    losses = np.asarray([residual_loss_for_z(cfg, models, window, z_samples[i : i + 1]) for i in range(n_samples)], dtype=np.float64)
    tau = max(float(zr_cfg.get("tau_z", 5.0)), 1e-6)
    weights = np.exp(-(losses - np.min(losses)) / tau)
    weights = weights / max(np.sum(weights), 1e-12)
    w_t = torch.from_numpy(weights.astype(np.float32)).to(device).unsqueeze(-1)
    z_rw = torch.sum(w_t * z_samples, dim=0, keepdim=True)
    rho_max = float(zr_cfg.get("rho_z_max", 0.35))
    rho = rho_max if method == "z_reweight" else gate_h * rho_max
    z_final = (1.0 - rho) * z_mean + rho * z_rw
    z_shift = torch.norm(z_final - z_mean).detach().cpu().item()
    return z_final, {"gate": float(gate_h), "z_shift": float(z_shift), "loss_std": float(np.std(losses))}


def run_episode(cfg: Dict, meta: Dict, models: Dict, method: str, regime: Dict, seed: int, episode_idx: int) -> Dict:
    device = next(models["actor"].parameters()).device
    zr_cfg = cfg["z_reweight"]
    env = make_env(cfg["env"]["name"], seed + episode_idx)
    schedule = EvalDynamicsSchedule(regime, cfg["dynamics_randomization"]["train_range"])
    context: Deque[np.ndarray] = deque(maxlen=int(cfg["latent"].get("eval_context_max", 50)))
    window: Deque[Tuple] = deque(maxlen=int(zr_cfg.get("window_size", 8)))
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    max_steps = int(cfg["env"]["max_episode_steps"])
    nominal_p95 = max(float(zr_cfg.get("nominal_abs_p95", 1.0)), 1e-6)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 100003 * episode_idx)
    s, _ = env.reset(seed=seed + episode_idx)
    ret = 0.0
    ewma_ratio = 0.0
    eta_vals, gate_vals, z_shift_vals, loss_std_vals = [], [], [], []
    for t in range(max_steps):
        env.set_dynamics(schedule.xi_at(t))
        z, info = choose_z(cfg, models, method, list(context), list(window), ewma_ratio, meta["latent_dim"], min_context, device, generator)
        action = deterministic_action(models["actor"], s, z, device)
        s_next, reward, term, trunc, _ = env.step(action)
        terminated = bool(term)
        done = bool(term or trunc)
        eta = bellman_residual(cfg, models, s, action, float(reward), s_next, terminated, z)
        abs_eta = abs(eta)
        eta_vals.append(abs_eta)
        ratio = abs_eta / nominal_p95
        ewma_ratio = (1.0 - float(zr_cfg.get("ewma_rho", 0.05))) * ewma_ratio + float(zr_cfg.get("ewma_rho", 0.05)) * ratio
        ret += float(reward)
        gate_vals.append(info["gate"])
        z_shift_vals.append(info["z_shift"])
        loss_std_vals.append(info["loss_std"])
        window.append((s, action, float(reward), s_next, terminated))
        context.append(_make_context_item(s, action, reward, s_next, done))
        s = s_next
        if done:
            break
    env.close()
    return {
        "return": ret,
        "length": len(eta_vals),
        "mean_abs_eta": float(np.mean(eta_vals)) if eta_vals else 0.0,
        "p95_abs_eta": float(np.percentile(eta_vals, 95)) if eta_vals else 0.0,
        "mean_gate": float(np.mean(gate_vals)) if gate_vals else 0.0,
        "mean_z_shift": float(np.mean(z_shift_vals)) if z_shift_vals else 0.0,
        "mean_loss_std": float(np.mean(loss_std_vals)) if loss_std_vals else 0.0,
    }


def aggregate_episode_dicts(episodes: List[Dict]) -> Dict:
    out = {}
    for key in episodes[0]:
        vals = [x[key] for x in episodes]
        out[f"mean_{key}"] = float(np.mean(vals))
        out[f"std_{key}"] = float(np.std(vals))
    out["num_episodes"] = len(episodes)
    return out


def evaluate(cfg: Dict, meta: Dict, models: Dict, method: str, regime: Dict, seed: int, output_dir: Path) -> Dict:
    episodes = [run_episode(cfg, meta, models, method, regime, seed, ep) for ep in range(int(cfg["eval"]["num_episodes"]))]
    result = {"method": method, "regime": regime["name"], "seed": seed, **aggregate_episode_dicts(episodes)}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
