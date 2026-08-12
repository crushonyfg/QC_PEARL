from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import torch
from stable_baselines3 import SAC

from brpc_latent_mpc.data.normalizer import Normalizer
from pearl_brpc_action_adapter.brpc.action_adapter import (
    BRPCActionAdapter,
    ConstantBiasAdapter,
    RLSActionAdapter,
    build_rff,
    make_feature_input,
)
from pearl_brpc_action_adapter.dynamics.dynamics_model import LatentDynamicsModel
from pearl_brpc_action_adapter.dynamics.jacobian import action_jacobian, solve_delta_u
from pearl_brpc_action_adapter.envs.latent_policy_env import make_policy_obs
from pearl_brpc_action_adapter.envs.make_env import EvalDynamicsSchedule, make_env
from pearl_brpc_action_adapter.eval.metrics import EpisodeMetrics, aggregate_episodes
from pearl_brpc_action_adapter.experiments.train_all import _make_encoder


METHODS = [
    "latent_sac_encoder",
    "latent_sac_z_projection",
    "latent_sac_constant_bias",
    "latent_sac_rls_adapter",
    "latent_sac_brpc_adapter",
    "latent_sac_zproj_brpc_adapter",
    "latent_sac_oracle_z",
]


def load_models(ckpt_dir: Path, device: torch.device) -> Dict[str, Any]:
    aux = torch.load(ckpt_dir / "latent_sac_aux.pt", map_location=device, weights_only=False)
    cfg = aux["cfg"]
    meta = aux["meta"]
    policy_path = ckpt_dir / "best_latent_sac.zip"
    if not policy_path.exists():
        policy_path = ckpt_dir / "final_latent_sac.zip"
    policy = SAC.load(policy_path, device=str(device))
    dynamics = LatentDynamicsModel(
        meta["state_dim"],
        meta["action_dim"],
        meta["latent_dim"],
        hidden_sizes=tuple(cfg["dynamics"]["hidden"]),
    ).to(device)
    encoder = _make_encoder(cfg, meta).to(device)
    dynamics.load_state_dict(aux["dynamics"])
    encoder.load_state_dict(aux["encoder"])
    dynamics.eval()
    encoder.eval()
    norms = {k: Normalizer.from_dict(v) for k, v in aux["norms"].items()}
    return {"policy": policy, "dynamics": dynamics, "encoder": encoder, "norms": norms, "meta": meta, "cfg": cfg}


def _context_to_tensor(context: List[np.ndarray], device: torch.device) -> torch.Tensor:
    if not context:
        return torch.zeros(1, 0, device=device)
    return torch.from_numpy(np.stack(context, axis=0)).unsqueeze(0).to(device)


def _infer_z(encoder, context: List[np.ndarray], latent_dim: int, device: torch.device) -> np.ndarray:
    if not context:
        return np.zeros(latent_dim, dtype=np.float32)
    with torch.no_grad():
        z = encoder.infer_mean(_context_to_tensor(context, device))[0].cpu().numpy()
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


def _torch_norm(value: torch.Tensor, normalizer, device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(normalizer.mean, device=device, dtype=torch.float32)
    std = torch.as_tensor(normalizer.std, device=device, dtype=torch.float32)
    return (value - mean) / torch.clamp(std, min=1e-8)


def _project_z(dynamics, norms, context: List[np.ndarray], z_prior: np.ndarray, meta: Dict, cfg: Dict, device):
    z_cfg = cfg.get("z_projection", {})
    window_len = int(z_cfg.get("window_len", 20))
    steps = int(z_cfg.get("steps", 10))
    lr = float(z_cfg.get("lr", 0.01))
    lambda_z = float(z_cfg.get("lambda_z", 1.0))
    z_clip = float(z_cfg.get("z_clip", 1.5))
    if not context or steps <= 0:
        return z_prior.astype(np.float32)
    arr = np.stack(context[-window_len:], axis=0)
    state_dim = meta["state_dim"]
    action_dim = meta["action_dim"]
    action_start = state_dim
    reward_idx = state_dim + action_dim
    next_start = reward_idx + 1
    states = arr[:, :state_dim].astype(np.float32)
    actions = arr[:, action_start : action_start + action_dim].astype(np.float32)
    next_states = arr[:, next_start : next_start + state_dim].astype(np.float32)
    deltas = next_states - states
    s_n = torch.from_numpy(norms["state"].transform(states)).to(device)
    a_n = torch.from_numpy(norms["action"].transform(actions)).to(device)
    ds_n = torch.from_numpy(norms["delta"].transform(deltas)).to(device)
    z0 = torch.from_numpy(z_prior.astype(np.float32)).to(device)
    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        z_n = _torch_norm(z, norms["z"], device).unsqueeze(0).expand(states.shape[0], -1)
        pred = dynamics(s_n, a_n, z_n)
        loss = ((pred - ds_n) ** 2).mean() + lambda_z * ((z - z0) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.clamp_(-z_clip, z_clip)
    return z.detach().cpu().numpy().astype(np.float32)


def _recent_model_error(dynamics, norms, context: List[np.ndarray], z: np.ndarray, meta: Dict, cfg: Dict, device) -> float:
    z_cfg = cfg.get("z_projection", {})
    window_len = int(z_cfg.get("window_len", 20))
    if not context:
        return 0.0
    arr = np.stack(context[-window_len:], axis=0)
    state_dim = meta["state_dim"]
    action_dim = meta["action_dim"]
    action_start = state_dim
    reward_idx = state_dim + action_dim
    next_start = reward_idx + 1
    states = arr[:, :state_dim].astype(np.float32)
    actions = arr[:, action_start : action_start + action_dim].astype(np.float32)
    next_states = arr[:, next_start : next_start + state_dim].astype(np.float32)
    deltas = next_states - states
    s_n = torch.from_numpy(norms["state"].transform(states)).to(device)
    a_n = torch.from_numpy(norms["action"].transform(actions)).to(device)
    z_n = torch.from_numpy(_norm_z(z, norms)).unsqueeze(0).expand(states.shape[0], -1).to(device)
    with torch.no_grad():
        pred_n = dynamics(s_n, a_n, z_n).cpu().numpy()
    pred = norms["delta"].inverse_transform(pred_n)
    return float(np.mean(np.sum((deltas - pred) ** 2, axis=-1)))


def _maybe_project_z(dynamics, norms, context: List[np.ndarray], z_prior: np.ndarray, meta: Dict, cfg: Dict, device) -> np.ndarray:
    z_cfg = cfg.get("z_projection", {})
    trigger_mse = float(z_cfg.get("trigger_mse", 0.0))
    blend = float(z_cfg.get("blend", 1.0))
    if trigger_mse > 0.0:
        err = _recent_model_error(dynamics, norms, context, z_prior, meta, cfg, device)
        if err < trigger_mse:
            return z_prior.astype(np.float32)
    z_proj = _project_z(dynamics, norms, context, z_prior, meta, cfg, device)
    blend = float(np.clip(blend, 0.0, 1.0))
    return ((1.0 - blend) * z_prior + blend * z_proj).astype(np.float32)


def _should_update(e: np.ndarray, cfg: Dict) -> bool:
    if not np.all(np.isfinite(e)):
        return False
    return np.linalg.norm(e) <= cfg["brpc"].get("residual_clip", 5.0)


def _uses_brpc(method: str) -> bool:
    return method in ("latent_sac_brpc_adapter", "latent_sac_zproj_brpc_adapter")


def _uses_adapter(method: str) -> bool:
    return method in (
        "latent_sac_constant_bias",
        "latent_sac_rls_adapter",
        "latent_sac_brpc_adapter",
        "latent_sac_zproj_brpc_adapter",
    )


def run_episode(cfg: Dict, method: str, regime: Dict, models: Dict, seed: int, episode_idx: int = 0) -> EpisodeMetrics:
    device = next(models["dynamics"].parameters()).device
    policy = models["policy"]
    dynamics = models["dynamics"]
    encoder = models["encoder"]
    norms = models["norms"]
    meta = models["meta"]
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

    context: Deque[np.ndarray] = deque(maxlen=context_max)
    metrics = EpisodeMetrics()
    s, _ = env.reset()
    prev_a = np.zeros(meta["action_dim"], dtype=np.float32)

    for t in range(max_steps):
        env.set_dynamics(schedule.xi_at(t))
        z_star = schedule.z_star_at(t)
        ctx_list = list(context)
        if method == "latent_sac_oracle_z":
            z = z_star.copy()
        else:
            z = _infer_z(encoder, ctx_list, meta["latent_dim"], device) if len(ctx_list) >= min_context else np.zeros(meta["latent_dim"], dtype=np.float32)
            if method in ("latent_sac_z_projection", "latent_sac_zproj_brpc_adapter") and len(ctx_list) >= min_context:
                z = _maybe_project_z(dynamics, norms, ctx_list, z, meta, cfg, device)

        a0, _ = policy.predict(make_policy_obs(s, z), deterministic=True)
        a0 = np.asarray(a0, dtype=np.float32)
        u_exec = np.zeros(meta["action_dim"], dtype=np.float32)
        unc = 0.0
        if _uses_brpc(method):
            x_n = np.concatenate([_norm_state(s, norms), _norm_action(a0, norms), _norm_z(z, norms)])
            b = rff.transform(x_n)[0]
            u_exec, unc, _ = brpc.act_correction(b)
        elif method == "latent_sac_rls_adapter":
            x_n = np.concatenate([_norm_state(s, norms), _norm_action(a0, norms), _norm_z(z, norms)])
            b = rff.transform(x_n)[0]
            u_exec = rls.act_correction(b, u_max)
        elif method == "latent_sac_constant_bias":
            u_exec = bias.act_correction()

        a = np.clip(a0 + u_exec, action_low, action_high)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        delta_real = s_next - s
        delta_base = _predict_delta(dynamics, norms, s, a0, z, device)
        e = delta_real - delta_base

        if not done and _uses_adapter(method) and _should_update(e, cfg):
            s_n = torch.from_numpy(_norm_state(s, norms)).to(device)
            a0_n = torch.from_numpy(_norm_action(a0, norms)).to(device)
            z_n = torch.from_numpy(_norm_z(z, norms)).to(device)
            J = action_jacobian(dynamics, s_n, a0_n, z_n)
            e_t = torch.from_numpy(e.astype(np.float32)).to(device)
            delta_u = solve_delta_u(J, e_t, lambda_J=brpc_cfg.get("lambda_J", 1e-3))
            eta_u = brpc_cfg.get("eta_u", 0.5)
            u_target = np.clip(u_exec + eta_u * delta_u.cpu().numpy(), -u_max, u_max)
            if _uses_brpc(method):
                x_n = np.concatenate([_norm_state(s, norms), _norm_action(a0, norms), _norm_z(z, norms)])
                b = rff.transform(x_n)[0]
                brpc.update(b, u_target)
            elif method == "latent_sac_rls_adapter":
                x_n = np.concatenate([_norm_state(s, norms), _norm_action(a0, norms), _norm_z(z, norms)])
                b = rff.transform(x_n)[0]
                rls.update(b, u_target)
            elif method == "latent_sac_constant_bias":
                bias.update(u_target)

        metrics.return_ += float(r)
        metrics.length += 1
        metrics.action_move_cost += float(np.sum((a - prev_a) ** 2))
        metrics.correction_norm += float(np.sum(u_exec ** 2))
        metrics.e_base_sum += float(np.sum(e ** 2))
        metrics.e_base_count += 1
        metrics.e_norms.append(float(np.linalg.norm(e)))
        metrics.uncertainties.append(unc)
        if method != "latent_sac_oracle_z":
            metrics.latent_error_sum += float(np.sum((z - z_star) ** 2))
            metrics.latent_error_count += 1
        c = np.concatenate([s, a, [r], s_next, [float(done)]]).astype(np.float32)
        context.append(c)
        s = s_next
        prev_a = a
        if done:
            break

    env.close()
    return metrics


def evaluate(cfg: Dict, method: str, regime: Dict, seed: int, ckpt_dir: Path, output_dir: Path) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(ckpt_dir, device)
    episodes = [
        run_episode(cfg, method, regime, models, seed, episode_idx=ep)
        for ep in range(cfg["eval"]["num_episodes"])
    ]
    agg = aggregate_episodes(episodes)
    result = {"method": method, "regime": regime["name"], "seed": seed, **agg}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
