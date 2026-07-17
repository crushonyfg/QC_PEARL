from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch

from pearl_brpc_action_adapter.envs.make_env import EvalDynamicsSchedule, make_env
from pearl_brpc_action_adapter.eval.metrics import EpisodeMetrics, aggregate_episodes
from pearl_brpc_action_adapter.experiments.train_full_pearl import _context_tensor, _make_context_item, _make_models


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    meta = payload["meta"]
    models = _make_models(cfg, meta, device)
    models["actor"].load_state_dict(payload["actor"])
    models["q1"].load_state_dict(payload["q1"])
    models["q2"].load_state_dict(payload["q2"])
    models["encoder"].load_state_dict(payload["encoder"])
    for m in models.values():
        m.eval()
    return cfg, meta, models


def infer_z(encoder, context: List[np.ndarray], latent_dim: int, device: torch.device) -> torch.Tensor:
    if not context:
        return torch.zeros(1, latent_dim, device=device)
    with torch.no_grad():
        return encoder.infer_mean(_context_tensor(context, device))


def run_episode(cfg: Dict, meta: Dict, models: Dict, regime: Dict, seed: int, episode_idx: int) -> EpisodeMetrics:
    device = next(models["actor"].parameters()).device
    env = make_env(cfg["env"]["name"], seed + episode_idx)
    schedule = EvalDynamicsSchedule(regime, cfg["dynamics_randomization"]["train_range"])
    context: Deque[np.ndarray] = deque(maxlen=int(cfg["latent"].get("eval_context_max", 50)))
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    max_steps = int(cfg["env"]["max_episode_steps"])
    metrics = EpisodeMetrics()
    s, _ = env.reset(seed=seed + episode_idx)
    prev_a = np.zeros(meta["action_dim"], dtype=np.float32)
    for t in range(max_steps):
        env.set_dynamics(schedule.xi_at(t))
        ctx_list = list(context)
        z = infer_z(models["encoder"], ctx_list, meta["latent_dim"], device) if len(ctx_list) >= min_context else torch.zeros(1, meta["latent_dim"], device=device)
        st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            a = models["actor"].deterministic_action(st, z)[0].cpu().numpy().astype(np.float32)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        metrics.return_ += float(r)
        metrics.length += 1
        metrics.action_move_cost += float(np.sum((a - prev_a) ** 2))
        context.append(_make_context_item(s, a, r, s_next, done))
        s = s_next
        prev_a = a
        if done:
            break
    env.close()
    return metrics


def evaluate(cfg: Dict, meta: Dict, models: Dict, regime: Dict, seed: int, output_dir: Path):
    episodes = [run_episode(cfg, meta, models, regime, seed, ep) for ep in range(cfg["eval"]["num_episodes"])]
    agg = aggregate_episodes(episodes)
    result = {"method": "full_pearl", "regime": regime["name"], "seed": seed, **agg}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
