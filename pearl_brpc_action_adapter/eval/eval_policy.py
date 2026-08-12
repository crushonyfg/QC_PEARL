from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from pearl_brpc_action_adapter.brpc.action_adapter import (
    BRPCActionAdapter,
    ConstantBiasAdapter,
    RLSActionAdapter,
    build_rff,
    make_feature_input,
)
from pearl_brpc_action_adapter.dynamics.jacobian import action_jacobian, solve_delta_u
from pearl_brpc_action_adapter.envs.make_env import EvalDynamicsSchedule, make_env
from pearl_brpc_action_adapter.eval.metrics import EpisodeMetrics, aggregate_episodes
from pearl_brpc_action_adapter.experiments.train_all import load_checkpoints


METHODS = [
    "pearl_only",
    "latent_ema",
    "constant_bias",
    "rls_adapter",
    "brpc_adapter",
    "brpc_no_gate",
    "oracle_z",
]


def _context_to_tensor(context: List[np.ndarray], device: torch.device) -> torch.Tensor:
    if not context:
        return torch.zeros(1, 0, device=device)
    arr = np.stack(context, axis=0)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _encoder_latent_dim(encoder) -> int:
    if hasattr(encoder, "pearl"):
        return encoder.pearl.latent_dim
    return encoder.latent_dim


def _infer_z(encoder, context: List[np.ndarray], device: torch.device) -> np.ndarray:
    if not context:
        return np.zeros(_encoder_latent_dim(encoder), dtype=np.float32)
    ctx = _context_to_tensor(context, device)
    with torch.no_grad():
        z = encoder.infer_mean(ctx)[0].cpu().numpy()
    return z.astype(np.float32)


def _norm_state(state, norms):
    return norms["state"].transform(state)


def _norm_action(action, norms):
    return norms["action"].transform(action)


def _norm_z(z, norms):
    return norms["z"].transform(z)


def _predict_delta(dynamics, norms, state, action, z, device):
    s_n = torch.from_numpy(_norm_state(state, norms)).unsqueeze(0).to(device)
    a_n = torch.from_numpy(_norm_action(action, norms)).unsqueeze(0).to(device)
    z_n = torch.from_numpy(_norm_z(z, norms)).unsqueeze(0).to(device)
    with torch.no_grad():
        delta_n = dynamics(s_n, a_n, z_n)[0].cpu().numpy()
    return norms["delta"].inverse_transform(delta_n)


def _should_update(e: np.ndarray, cfg: Dict) -> bool:
    if not np.all(np.isfinite(e)):
        return False
    if np.linalg.norm(e) > cfg["brpc"].get("residual_clip", 5.0):
        return False
    return True


def run_episode(
    cfg: Dict,
    method: str,
    regime: Dict,
    models: Dict,
    seed: int,
    episode_idx: int = 0,
) -> EpisodeMetrics:
    device = next(models["actor"].parameters()).device
    meta = models["meta"]
    norms = models["norms"]
    actor = models["actor"]
    encoder = models["encoder"]
    dynamics = models["dynamics"]
    brpc_cfg = cfg["brpc"]
    train_range = cfg["dynamics_randomization"]["train_range"]

    schedule = EvalDynamicsSchedule(regime, train_range)
    env = make_env(cfg["env"]["name"], seed + episode_idx)
    max_steps = cfg["env"]["max_episode_steps"]
    context_max = cfg["latent"].get("eval_context_max", 50)
    min_context = cfg["latent"].get("eval_context_min", 5)
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)
    action_range = float(np.max(action_high - action_low))
    u_max = brpc_cfg.get("u_max_frac", 0.2) * action_range

    feat_dim = int(brpc_cfg["feature_dim"])
    input_dim = meta["state_dim"] + meta["action_dim"] + meta["latent_dim"]
    rff = build_rff(input_dim, brpc_cfg, seed=seed)

    brpc = BRPCActionAdapter(
        feature_dim=feat_dim,
        action_dim=meta["action_dim"],
        prior_var=brpc_cfg.get("prior_var", 1.0),
        obs_noise=brpc_cfg.get("obs_noise", 0.05),
        rho=brpc_cfg.get("rho", 0.99),
        q_alpha=brpc_cfg.get("q_alpha", 1e-4),
        u_max=u_max,
        kappa=brpc_cfg.get("kappa", 0.5),
    )
    rls = RLSActionAdapter(feat_dim, meta["action_dim"], prior_var=brpc_cfg.get("prior_var", 1.0))
    bias = ConstantBiasAdapter(meta["action_dim"], u_max=u_max)
    brpc.reset()
    rls.reset()
    bias.reset()

    z_ema = np.zeros(meta["latent_dim"], dtype=np.float32)
    ema_rho = cfg["baselines"].get("latent_ema_rho", 0.1)

    context: Deque[np.ndarray] = deque(maxlen=context_max)
    metrics = EpisodeMetrics()
    s, _ = env.reset()
    prev_a = np.zeros(meta["action_dim"], dtype=np.float32)
    u_exec_prev = np.zeros(meta["action_dim"], dtype=np.float32)

    for t in range(max_steps):
        if hasattr(env, "set_dynamics"):
            env.set_dynamics(schedule.xi_at(t))
        else:
            env.unwrapped.set_dynamics(schedule.xi_at(t))
        z_star = schedule.z_star_at(t)

        ctx_list = list(context)
        if method == "oracle_z":
            z = z_star.copy()
        else:
            z = _infer_z(encoder, ctx_list, device) if len(ctx_list) >= min_context else np.zeros(meta["latent_dim"], dtype=np.float32)
            if method == "latent_ema" and len(ctx_list) >= min_context:
                z_hat = z.copy()
                z = (1 - ema_rho) * z_ema + ema_rho * z_hat
                z_ema = z.copy()

        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device)
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            a0 = actor.deterministic_action(s_t, z_t)[0].cpu().numpy()

        u_exec = np.zeros(meta["action_dim"], dtype=np.float32)
        unc = 0.0
        if method in ("brpc_adapter", "brpc_no_gate"):
            x_feat = make_feature_input(s, a0, z)
            x_n = np.concatenate([_norm_state(s, norms), _norm_action(a0, norms), _norm_z(z, norms)])
            b = rff.transform(x_n)[0]
            if method == "brpc_no_gate":
                u_mean, unc = brpc.predict(b)
                u_exec = np.clip(u_mean, -u_max, u_max)
            else:
                u_exec, unc, _ = brpc.act_correction(b)
        elif method == "rls_adapter":
            x_n = np.concatenate([_norm_state(s, norms), _norm_action(a0, norms), _norm_z(z, norms)])
            b = rff.transform(x_n)[0]
            u_exec = rls.act_correction(b, u_max)
        elif method == "constant_bias":
            u_exec = bias.act_correction()

        a = np.clip(a0 + u_exec, action_low, action_high)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc

        delta_real = s_next - s
        delta_base = _predict_delta(dynamics, norms, s, a0, z, device)
        e = delta_real - delta_base

        if not done and method in ("brpc_adapter", "brpc_no_gate", "rls_adapter", "constant_bias") and _should_update(e, cfg):
            s_n = torch.from_numpy(_norm_state(s, norms)).to(device)
            a0_n = torch.from_numpy(_norm_action(a0, norms)).to(device)
            z_n = torch.from_numpy(_norm_z(z, norms)).to(device)
            J = action_jacobian(dynamics, s_n, a0_n, z_n)
            e_t = torch.from_numpy(e.astype(np.float32)).to(device)
            delta_u = solve_delta_u(J, e_t, lambda_J=brpc_cfg.get("lambda_J", 1e-3))
            eta_u = brpc_cfg.get("eta_u", 0.5)
            u_target = u_exec + eta_u * delta_u.cpu().numpy()
            u_target = np.clip(u_target, -u_max, u_max)

            if method in ("brpc_adapter", "brpc_no_gate"):
                x_n = np.concatenate([_norm_state(s, norms), _norm_action(a0, norms), _norm_z(z, norms)])
                b = rff.transform(x_n)[0]
                brpc.update(b, u_target)
            elif method == "rls_adapter":
                x_n = np.concatenate([_norm_state(s, norms), _norm_action(a0, norms), _norm_z(z, norms)])
                b = rff.transform(x_n)[0]
                rls.update(b, u_target)
            elif method == "constant_bias":
                bias.update(u_target)

        metrics.return_ += r
        metrics.length += 1
        metrics.action_move_cost += float(np.sum((a - prev_a) ** 2))
        metrics.correction_norm += float(np.sum(u_exec ** 2))
        metrics.e_base_sum += float(np.sum(e ** 2))
        metrics.e_base_count += 1
        metrics.e_norms.append(float(np.linalg.norm(e)))
        metrics.uncertainties.append(unc)
        if method != "oracle_z":
            metrics.latent_error_sum += float(np.sum((z - z_star) ** 2))
            metrics.latent_error_count += 1

        c = np.concatenate([s, a, [r], s_next, [float(done)]])
        context.append(c.astype(np.float32))
        s = s_next
        prev_a = a
        u_exec_prev = u_exec
        if done:
            break

    env.close()
    return metrics


def evaluate(
    cfg: Dict,
    method: str,
    regime: Dict,
    seed: int,
    ckpt_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_checkpoints(ckpt_path, device)
    episodes = []
    for ep in range(cfg["eval"]["num_episodes"]):
        episodes.append(run_episode(cfg, method, regime, models, seed, ep))
    agg = aggregate_episodes(episodes)
    result = {
        "method": method,
        "regime": regime["name"],
        "seed": seed,
        **agg,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
