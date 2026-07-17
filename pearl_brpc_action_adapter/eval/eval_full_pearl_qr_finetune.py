"""Hybrid: alternate the QR head (fast, closed-form, safe corrector) with critic fine-tuning
(slow, SGD, raises the ceiling). Two-timescale:

  * within each episode the QR head h_w(phi_work) is fit (Bayesian RLS) to the residual of the
    CURRENT working critic and used to re-rank the candidate pool (the safe immediate fix);
  * at episode end a few SAC last-layer gradient steps lift the working critic (the ceiling),
    then the QR head is RESET (its basis phi_work and target both moved -> stale).

Two variants (full critic in both, so SGD can reshape phi -- complementing the last-layer QR):
  * qr_finetune_critic : actor FROZEN. a0 / pool stay the trusted frozen actions; the gate
    protects nominal. Only the Q used for re-ranking (+ QR mop-up) improves.
  * qr_finetune_full   : actor ALSO fine-tuned. a0 / candidate pool MOVE with the actor, so the
    search can escape the coverage ceiling (where re-ranking a frozen pool is capped) -- at the
    cost of nominal safety (a0 itself drifts, the gate can no longer fully fall back).
The pretrained checkpoint is never modified (the backbone is deep-copied per run).
"""
from __future__ import annotations

import copy
import json
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch

from pearl_brpc_action_adapter.buffers.task_replay_buffer import TaskReplayBuffer
from pearl_brpc_action_adapter.envs.make_env import make_env
from pearl_brpc_action_adapter.envs.action_perturb import maybe_wrap_action_perturb
from pearl_brpc_action_adapter.experiments.train_full_pearl import _make_context_item
from pearl_brpc_action_adapter.eval.eval_full_pearl_finetune import _finetune_updates
from pearl_brpc_action_adapter.eval.eval_full_pearl_dynamics_lookahead import (
    finalize_traces, infer_z, make_q_brpc, make_schedule, pick_action, q_min_single, step_adaptation,
)


def _reset_qr(q_brpc) -> None:
    """Reset the QR head to its prior (the working critic + its phi basis just moved under SGD)."""
    q_brpc.M = np.zeros_like(q_brpc.M)
    q_brpc.P = np.eye(q_brpc.feature_dim) * q_brpc.prior_var


def _build_hybrid_setup(cfg: Dict, frozen: Dict, train_actor: bool, device):
    """Deep-copied working backbone (full critic trainable; actor trainable iff train_actor;
    encoder always frozen). Originals / checkpoint untouched."""
    work = {k: copy.deepcopy(m).to(device) for k, m in frozen.items()}
    for p in work["encoder"].parameters():
        p.requires_grad_(False)
    q1_t = copy.deepcopy(work["q1"]).to(device)
    q2_t = copy.deepcopy(work["q2"]).to(device)
    ft = cfg.get("finetune", {})
    crit_params = list(work["q1"].parameters()) + list(work["q2"].parameters())
    opts = {"critic": torch.optim.Adam(crit_params, lr=float(ft.get("lr", cfg["pearl"]["critic_lr"])))}
    if train_actor:
        opts["actor"] = torch.optim.Adam(work["actor"].parameters(), lr=float(ft.get("actor_lr", cfg["pearl"]["actor_lr"])))
    else:
        for p in work["actor"].parameters():
            p.requires_grad_(False)
    return work, q1_t, q2_t, opts, crit_params


def _run_episode(cfg, meta, frozen, work, q1_t, q2_t, opts, crit_params, q_brpc, buf,
                 dyn, nominal_p95, regime, seed, episode_idx, n_updates, train_actor,
                 context: Deque, carry: Dict, device) -> Dict:
    lk_cfg = cfg["lookahead"]
    gamma = float(cfg["pearl"]["gamma"])
    env = make_env(cfg["env"]["name"], seed + episode_idx)
    env = maybe_wrap_action_perturb(env, regime, seed + episode_idx)
    schedule = make_schedule(regime, cfg["dynamics_randomization"]["train_range"])
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    max_steps = int(cfg["env"]["max_episode_steps"])
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)
    action_scale = float(np.max(np.abs(action_high)))
    rng = np.random.default_rng(seed + 100000 * episode_idx)
    use_gate = bool(lk_cfg.get("use_gate", True))
    gate_slope = float(lk_cfg.get("gate_slope", 8.0))
    gate_threshold = float(lk_cfg.get("gate_threshold", 0.55))
    ewma_rho = float(lk_cfg.get("ewma_rho", 0.1))
    p95 = max(nominal_p95, 1e-6)

    s, _ = env.reset(seed=seed + episode_idx)
    prev_action = np.zeros(meta["action_dim"], dtype=np.float32)
    ewma = float(carry.get("ewma", 0.0))
    ret = 0.0
    q_lift, dV_vals, gate_vals, resid_vals, corr_vals, rew = [], [], [], [], [], []

    for t in range(max_steps):
        env.set_dynamics(schedule.xi_at(t))
        z = infer_z(frozen["encoder"], list(context), meta["latent_dim"], min_context, device)
        with torch.no_grad():  # a0/pool from the WORKING actor: frozen when critic-only, MOVES when full
            a0 = work["actor"].deterministic_action(
                torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device), z
            )[0].cpu().numpy().astype(np.float32)
        q_base = q_min_single(work, s, a0, z)
        gate_h = 1.0 / (1.0 + math.exp(-gate_slope * (ewma - gate_threshold))) if use_gate else 1.0
        # Re-rank by the WORKING critic + QR head (reuse the value_shift_qr scoring on work models).
        action, dV = pick_action("value_shift_qr", cfg, lk_cfg, work, dyn, None, None, s, z, a0,
                                  prev_action, gate_h, action_low, action_high, action_scale, rng, device, q_brpc)
        q_exec = q_min_single(work, s, action, z)
        s_next, reward, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        # Gate signal (frozen world model) + fast QR update against the CURRENT working critic.
        resid_norm, pending, _ = step_adaptation("value_shift_qr", lk_cfg, work, dyn, None, q_brpc, None,
                                                  s, action, s_next, reward, bool(term), z, action_scale, gamma, device)
        ewma = (1.0 - ewma_rho) * ewma + ewma_rho * (resid_norm / p95)
        for cal, feat, target in pending:
            cal.update(feat, target)
        buf.add(s, action, float(reward), s_next, done, np.zeros(meta["latent_dim"], dtype=np.float32))
        context.append(_make_context_item(s, action, reward, s_next, done))
        ret += float(reward)
        q_lift.append(float(q_exec - q_base)); dV_vals.append(float(dV)); gate_vals.append(float(gate_h))
        resid_vals.append(resid_norm); corr_vals.append(float(np.sum((action - a0) ** 2))); rew.append(float(reward))
        s = s_next; prev_action = action
        if done:
            break
    env.close()

    # Slow learner: lift the working critic, then RESET the QR head (its basis/target moved).
    if n_updates > 0 and len(buf) >= int(cfg["pearl"]["rl_batch_size"]):
        z_ft = infer_z(frozen["encoder"], list(context), meta["latent_dim"], min_context, device)
        _finetune_updates(cfg, work, q1_t, q2_t, opts, crit_params, buf, z_ft,
                          update_actor=train_actor, n_updates=n_updates, rng=rng, device=device)
        _reset_qr(q_brpc)

    carry["ewma"] = ewma
    return {
        "return": ret, "length": len(q_lift),
        "mean_q_lift": float(np.mean(q_lift)) if q_lift else 0.0,
        "mean_value_shift_term": float(np.mean(dV_vals)) if dV_vals else 0.0,
        "mean_gate": float(np.mean(gate_vals)) if gate_vals else 0.0,
        "mean_raw_resid_norm": float(np.mean(resid_vals)) if resid_vals else 0.0,
        "mean_correction_sq": float(np.mean(corr_vals)) if corr_vals else 0.0,
        "reward_trace": rew,
    }


def evaluate_qr_finetune(cfg, meta, frozen, dyn, nominal_resid, method, regime, seed,
                         output_dir: Path, warmup_override: int = None, num_agents: int = 1) -> Dict:
    """Hybrid QR + fine-tune (single agent). method qr_finetune_critic (actor frozen) /
    qr_finetune_full (actor also fine-tuned -> candidate pool moves)."""
    device = next(frozen["actor"].parameters()).device
    lk_cfg = cfg["lookahead"]
    ft = cfg.get("finetune", {})
    warmup = int(lk_cfg.get("warmup_episodes", 0)) if warmup_override is None else int(warmup_override)
    n_test = int(cfg["eval"]["num_episodes"])
    n_updates = int(ft.get("updates_per_episode", 200))
    ctx_max = int(cfg["latent"].get("eval_context_max", 50))
    nominal_p95 = float(nominal_resid.get("resid_norm_p95", 1.0))
    train_actor = (method == "qr_finetune_full")

    work, q1_t, q2_t, opts, crit_params = _build_hybrid_setup(cfg, frozen, train_actor, device)
    q_brpc = make_q_brpc(cfg, meta)
    buf = TaskReplayBuffer(int(ft.get("buffer_capacity", 100000)), meta["state_dim"], meta["action_dim"], meta["latent_dim"])
    context: Deque[np.ndarray] = deque(maxlen=ctx_max)
    carry: Dict = {}

    ep = 0
    episodes: List[Dict] = []
    for _ in range(warmup):
        _run_episode(cfg, meta, frozen, work, q1_t, q2_t, opts, crit_params, q_brpc, buf,
                     dyn, nominal_p95, regime, seed, ep, n_updates, train_actor, context, carry, device)
        ep += 1
    for _ in range(n_test):
        episodes.append(_run_episode(cfg, meta, frozen, work, q1_t, q2_t, opts, crit_params, q_brpc, buf,
                                     dyn, nominal_p95, regime, seed, ep, n_updates, train_actor, context, carry, device))
        ep += 1

    out = {"method": method, "regime": regime["name"], "seed": seed, "warmup_episodes": warmup, "num_agents": 1}
    out.update(finalize_traces(episodes, int(cfg["env"]["max_episode_steps"]), int(lk_cfg.get("trace_bins", 20))))
    for key in episodes[0]:
        vals = [x[key] for x in episodes]
        out[f"mean_{key}"] = float(np.mean(vals)); out[f"std_{key}"] = float(np.std(vals))
    out["num_episodes"] = len(episodes)
    out["episode_returns"] = [float(e["return"]) for e in episodes]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
