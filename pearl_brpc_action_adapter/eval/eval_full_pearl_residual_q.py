from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch

from brpc_latent_mpc.brpc.bayes_linear_residual import BRPCResidualCalibrator
from pearl_brpc_action_adapter.brpc.action_adapter import build_rff
from pearl_brpc_action_adapter.envs.make_env import EvalDynamicsSchedule, make_env
from pearl_brpc_action_adapter.experiments.train_full_pearl import _context_tensor, _make_context_item
from pearl_brpc_action_adapter.eval.eval_full_pearl import load_checkpoint


METHODS = [
    "full_pearl_only",
    "q_sample",
    "brpc_q_mean",
    "brpc_q_conservative",
    "brpc_q_gated",
]


def infer_z(encoder, context: List[np.ndarray], latent_dim: int, min_context: int, device: torch.device) -> torch.Tensor:
    if len(context) < min_context:
        return torch.zeros(1, latent_dim, device=device)
    with torch.no_grad():
        return encoder.infer_mean(_context_tensor(context, device))


def q_min_batch(q1, q2, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z_batch = z.expand(states.shape[0], -1)
    return torch.minimum(q1(states, actions, z_batch), q2(states, actions, z_batch)).squeeze(-1)


def q_min_np(models: Dict, state: np.ndarray, action: np.ndarray, z: torch.Tensor) -> float:
    device = next(models["actor"].parameters()).device
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    at = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        q_val = torch.minimum(models["q1"](st, at, z), models["q2"](st, at, z))
    return float(q_val.cpu().item())


def soft_value(actor, q1, q2, state: np.ndarray, z: torch.Tensor, alpha: float, n_actions: int, device: torch.device) -> float:
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    states = st.expand(n_actions, -1)
    z_batch = z.expand(n_actions, -1)
    with torch.no_grad():
        actions, log_probs = actor.sample_action_logprob(states, z_batch)
        q_min = torch.minimum(q1(states, actions, z_batch), q2(states, actions, z_batch))
        values = q_min - alpha * log_probs
    return float(values.mean().cpu().item())


def deterministic_action(actor, state: np.ndarray, z: torch.Tensor, device: torch.device) -> np.ndarray:
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return actor.deterministic_action(st, z)[0].cpu().numpy().astype(np.float32)


def feature_inputs(states: np.ndarray, actions: np.ndarray, z: np.ndarray, rq_cfg: Dict, action_scale: float) -> np.ndarray:
    state_scale = float(rq_cfg.get("feature_state_scale", 5.0))
    z_scale = float(rq_cfg.get("feature_z_scale", 1.0))
    s = states / max(state_scale, 1e-6)
    a = actions / max(action_scale, 1e-6)
    zz = np.broadcast_to(z, (actions.shape[0], z.shape[0])) / max(z_scale, 1e-6)
    return np.concatenate([s, a, zz], axis=-1).astype(np.float32)


def brpc_predict_batch(brpc: BRPCResidualCalibrator, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b = np.asarray(features, dtype=np.float64)
    mean = (b @ brpc.M).reshape(-1)
    var = np.einsum("bi,ij,bj->b", b, brpc.P, b)
    return mean.astype(np.float32), np.maximum(var, 0.0).astype(np.float32)


def make_candidates(
    actor,
    state: np.ndarray,
    z: torch.Tensor,
    a0: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    rq_cfg: Dict,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    n = int(rq_cfg.get("num_candidates", 64))
    policy_frac = float(rq_cfg.get("policy_candidate_frac", 0.25))
    n_policy = int(round(max(n - 1, 0) * policy_frac))
    n_local = max(n - 1 - n_policy, 0)
    local_std = float(rq_cfg.get("local_action_std", 0.08))
    local = a0[None, :] + rng.normal(0.0, local_std, size=(n_local, a0.shape[0])).astype(np.float32)
    policy = np.zeros((0, a0.shape[0]), dtype=np.float32)
    if n_policy > 0:
        st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device).expand(n_policy, -1)
        z_batch = z.expand(n_policy, -1)
        with torch.no_grad():
            policy = actor.sample_action_logprob(st, z_batch)[0].cpu().numpy().astype(np.float32)
    candidates = np.concatenate([a0[None, :], local, policy], axis=0)
    return np.clip(candidates, action_low, action_high).astype(np.float32)


def select_action(
    method: str,
    cfg: Dict,
    models: Dict,
    brpc: BRPCResidualCalibrator,
    rff,
    state: np.ndarray,
    z: torch.Tensor,
    a0: np.ndarray,
    prev_action: np.ndarray,
    gate_h: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_scale: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if method == "full_pearl_only":
        return a0, {"gate": 0.0, "sigma": 0.0, "score_std": 0.0, "weighted_candidate_q": 0.0, "max_candidate_q": 0.0, "top_weight": 1.0}

    device = next(models["actor"].parameters()).device
    rq_cfg = cfg["residual_q"]
    actor, q1, q2 = models["actor"], models["q1"], models["q2"]
    candidates = make_candidates(actor, state, z, a0, action_low, action_high, rq_cfg, rng, device)
    states = torch.from_numpy(np.repeat(state[None, :].astype(np.float32), candidates.shape[0], axis=0)).to(device)
    actions = torch.from_numpy(candidates.astype(np.float32)).to(device)
    with torch.no_grad():
        q_scores = q_min_batch(q1, q2, states, actions, z).cpu().numpy()

    mu_delta = np.zeros_like(q_scores, dtype=np.float32)
    sigma_delta = np.zeros_like(q_scores, dtype=np.float32)
    if method.startswith("brpc_q"):
        z_np = z[0].detach().cpu().numpy().astype(np.float32)
        feats = rff.transform(feature_inputs(np.repeat(state[None, :], candidates.shape[0], axis=0), candidates, z_np, rq_cfg, action_scale))
        mu_delta, var_delta = brpc_predict_batch(brpc, feats)
        sigma_delta = np.sqrt(var_delta + 1e-8)

    lambda_a = float(rq_cfg.get("lambda_action", 10.0))
    lambda_sw = float(rq_cfg.get("lambda_switch", 1.0))
    trust = lambda_a * np.sum((candidates - a0[None, :]) ** 2, axis=1)
    switch = lambda_sw * np.sum((candidates - prev_action[None, :]) ** 2, axis=1)
    beta = float(rq_cfg.get("beta_unc", 1.0))
    if method == "q_sample":
        scores = q_scores - trust - switch
    elif method == "brpc_q_mean":
        scores = q_scores + mu_delta - trust - switch
    elif method == "brpc_q_conservative":
        scores = q_scores + mu_delta - beta * sigma_delta - trust - switch
    elif method == "brpc_q_gated":
        scores = q_scores + gate_h * (mu_delta - beta * sigma_delta) - trust - switch
    else:
        raise ValueError(f"Unknown residual-Q method: {method}")

    selection = rq_cfg.get("selection", "weighted")
    if selection == "argmax":
        weights = np.zeros_like(scores, dtype=np.float64)
        weights[int(np.argmax(scores))] = 1.0
    elif selection in ("weighted", "weighted_sample"):
        tau = max(float(rq_cfg.get("softmax_tau", 5.0)), 1e-6)
        weights = np.exp((scores - np.max(scores)) / tau)
        weights = weights / max(np.sum(weights), 1e-12)
    elif selection == "topk_sample":
        tau = max(float(rq_cfg.get("topk_softmax_tau", rq_cfg.get("softmax_tau", 1.0))), 1e-6)
        topk = max(1, min(int(rq_cfg.get("topk", 8)), scores.shape[0]))
        top_idx = np.argpartition(scores, -topk)[-topk:]
        top_scores = scores[top_idx]
        top_weights = np.exp((top_scores - np.max(top_scores)) / tau)
        top_weights = top_weights / max(np.sum(top_weights), 1e-12)
        weights = np.zeros_like(scores, dtype=np.float64)
        weights[top_idx] = top_weights
    else:
        raise ValueError(f"Unknown residual-Q selection mode: {selection}")
    if selection in ("weighted_sample", "topk_sample"):
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        selected = int(rng.choice(candidates.shape[0], p=weights))
        action = candidates[selected]
    else:
        action = np.sum(weights[:, None] * candidates, axis=0)
    return np.clip(action, action_low, action_high).astype(np.float32), {
        "gate": float(gate_h),
        "sigma": float(np.mean(sigma_delta)),
        "score_std": float(np.std(scores)),
        "weighted_candidate_q": float(np.sum(weights * q_scores)),
        "max_candidate_q": float(np.max(q_scores)),
        "selected_candidate_q": float(q_scores[selected]) if selection in ("weighted_sample", "topk_sample") else float(np.sum(weights * q_scores)),
        "top_weight": float(np.max(weights)),
    }


def bellman_residual(cfg: Dict, models: Dict, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, terminated: bool, z: torch.Tensor) -> float:
    device = next(models["actor"].parameters()).device
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    at = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)
    gamma = float(cfg["pearl"]["gamma"])
    alpha = float(cfg["pearl"]["sac_alpha"])
    value_cfg = cfg.get("residual_q", cfg.get("z_reweight", {}))
    with torch.no_grad():
        q_sa = torch.minimum(models["q1"](st, at, z), models["q2"](st, at, z))
    v_next = 0.0 if terminated else soft_value(
        models["actor"],
        models["q1"],
        models["q2"],
        next_state,
        z,
        alpha,
        int(value_cfg.get("value_action_samples", 8)),
        device,
    )
    return float(reward + gamma * (0.0 if terminated else 1.0) * v_next - q_sa.cpu().item())


def run_episode(cfg: Dict, meta: Dict, models: Dict, method: str, regime: Dict, seed: int, episode_idx: int) -> Dict:
    device = next(models["actor"].parameters()).device
    rq_cfg = cfg["residual_q"]
    env = make_env(cfg["env"]["name"], seed + episode_idx)
    schedule = EvalDynamicsSchedule(regime, cfg["dynamics_randomization"]["train_range"])
    context: Deque[np.ndarray] = deque(maxlen=int(cfg["latent"].get("eval_context_max", 50)))
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    max_steps = int(cfg["env"]["max_episode_steps"])
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)
    action_scale = float(np.max(np.abs(action_high)))
    input_dim = meta["state_dim"] + meta["action_dim"] + meta["latent_dim"]
    rff = build_rff(input_dim, rq_cfg, seed=seed + 1009 * episode_idx)
    brpc = BRPCResidualCalibrator(
        feature_dim=int(rq_cfg.get("feature_dim", 128)),
        output_dim=1,
        prior_var=float(rq_cfg.get("prior_var", 0.25)),
        obs_noise=float(rq_cfg.get("obs_noise", 1.0)),
        rho=float(rq_cfg.get("rho", 0.99)),
        q_alpha=float(rq_cfg.get("q_alpha", 1e-4)),
        residual_clip_sigma=float(rq_cfg.get("residual_clip_sigma", 5.0)),
    )
    rng = np.random.default_rng(seed + 100000 * episode_idx)
    s, _ = env.reset(seed=seed + episode_idx)
    prev_action = np.zeros(meta["action_dim"], dtype=np.float32)
    ret = 0.0
    ewma_ratio = 0.0
    nominal_p95 = max(float(rq_cfg.get("nominal_abs_p95", 1.0)), 1e-6)
    eta_vals, gate_vals, sigma_vals, correction_vals, score_std_vals = [], [], [], [], []
    signed_eta_vals, q_lift_vals, weighted_candidate_q_lift_vals, selected_candidate_q_lift_vals = [], [], [], []
    max_candidate_q_lift_vals, top_weight_vals = [], []
    for t in range(max_steps):
        env.set_dynamics(schedule.xi_at(t))
        z = infer_z(models["encoder"], list(context), meta["latent_dim"], min_context, device)
        a0 = deterministic_action(models["actor"], s, z, device)
        q_base = q_min_np(models, s, a0, z)
        gate_h = 1.0 / (1.0 + math.exp(-float(rq_cfg.get("gate_slope", 8.0)) * (ewma_ratio - float(rq_cfg.get("gate_threshold", 0.55)))))
        action, info = select_action(method, cfg, models, brpc, rff, s, z, a0, prev_action, gate_h, action_low, action_high, action_scale, rng)
        q_exec = q_min_np(models, s, action, z)
        s_next, reward, term, trunc, _ = env.step(action)
        terminated = bool(term)
        done = bool(term or trunc)
        eta = bellman_residual(cfg, models, s, action, float(reward), s_next, terminated, z)
        eta_vals.append(abs(eta))
        signed_eta_vals.append(float(eta))
        q_lift_vals.append(float(q_exec - q_base))
        if method == "full_pearl_only":
            weighted_candidate_q_lift_vals.append(0.0)
            max_candidate_q_lift_vals.append(0.0)
        else:
            weighted_candidate_q_lift_vals.append(float(info["weighted_candidate_q"] - q_base))
            selected_candidate_q_lift_vals.append(float(info["selected_candidate_q"] - q_base))
            max_candidate_q_lift_vals.append(float(info["max_candidate_q"] - q_base))
        top_weight_vals.append(float(info["top_weight"]))
        ratio = abs(eta) / nominal_p95
        ewma_ratio = (1.0 - float(rq_cfg.get("ewma_rho", 0.05))) * ewma_ratio + float(rq_cfg.get("ewma_rho", 0.05)) * ratio
        if method.startswith("brpc_q") and np.isfinite(eta):
            z_np = z[0].detach().cpu().numpy().astype(np.float32)
            feat = rff.transform(feature_inputs(s[None, :], action[None, :], z_np, rq_cfg, action_scale))[0]
            brpc.update(feat, np.asarray([eta], dtype=np.float32))
        ret += float(reward)
        correction_vals.append(float(np.sum((action - a0) ** 2)))
        gate_vals.append(info["gate"])
        sigma_vals.append(info["sigma"])
        score_std_vals.append(info["score_std"])
        context.append(_make_context_item(s, action, reward, s_next, done))
        s = s_next
        prev_action = action
        if done:
            break
    env.close()
    return {
        "return": ret,
        "length": len(eta_vals),
        "mean_abs_eta": float(np.mean(eta_vals)) if eta_vals else 0.0,
        "p95_abs_eta": float(np.percentile(eta_vals, 95)) if eta_vals else 0.0,
        "mean_signed_eta": float(np.mean(signed_eta_vals)) if signed_eta_vals else 0.0,
        "mean_q_lift": float(np.mean(q_lift_vals)) if q_lift_vals else 0.0,
        "p90_q_lift": float(np.percentile(q_lift_vals, 90)) if q_lift_vals else 0.0,
        "mean_weighted_candidate_q_lift": float(np.mean(weighted_candidate_q_lift_vals)) if weighted_candidate_q_lift_vals else 0.0,
        "mean_selected_candidate_q_lift": float(np.mean(selected_candidate_q_lift_vals)) if selected_candidate_q_lift_vals else 0.0,
        "mean_max_candidate_q_lift": float(np.mean(max_candidate_q_lift_vals)) if max_candidate_q_lift_vals else 0.0,
        "mean_top_weight": float(np.mean(top_weight_vals)) if top_weight_vals else 0.0,
        "mean_gate": float(np.mean(gate_vals)) if gate_vals else 0.0,
        "mean_sigma_delta": float(np.mean(sigma_vals)) if sigma_vals else 0.0,
        "mean_correction_sq": float(np.mean(correction_vals)) if correction_vals else 0.0,
        "mean_score_std": float(np.mean(score_std_vals)) if score_std_vals else 0.0,
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
