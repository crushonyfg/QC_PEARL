"""Test-time fine-tuning baselines for the frozen full-PEARL backbone.

These are the natural apples-to-apples comparison for `value_shift_qr`: both adapt
the frozen backbone to a held-out regime from K warmup episodes of in-regime data
(fleet M agents, continual through the test episodes). `value_shift_qr` corrects Q
in its penultimate-feature basis with a closed-form Bayesian Bellman-residual head;
fine-tuning instead updates network weights with SAC gradient steps.

Two variants:
  finetune_lastlayer : SGD on ONLY Q's last linear layer (the same subspace
                       value_shift_qr corrects), actor frozen. Deploys like q_greedy
                       but re-ranks the local candidate pool with the fine-tuned Q.
                       => with K=0 / no updates this reduces to q_greedy.
  finetune_full      : standard SAC test-time fine-tune of actor + critic (strongest
                       fine-tune baseline), encoder frozen. Deploys the fine-tuned
                       deterministic actor directly.
                       => with K=0 / no updates this reduces to full_pearl_only.

Critical: the loaded frozen models are deep-copied per (regime, seed, K) run; the
pretrained checkpoint on disk is NEVER modified (it is reused across all runs).
The encoder is always frozen (z-adaptation alone is capped on OOD; fine-tuning adds
weight adaptation on top of the inferred regime latent z).
"""
from __future__ import annotations

import copy
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch

from pearl_brpc_action_adapter.buffers.task_replay_buffer import TaskReplayBuffer
from pearl_brpc_action_adapter.envs.make_env import make_env
from pearl_brpc_action_adapter.envs.action_perturb import maybe_wrap_action_perturb
from pearl_brpc_action_adapter.pearl.sac_losses import (
    compute_actor_loss,
    compute_q_loss,
    soft_update,
)
from pearl_brpc_action_adapter.experiments.train_full_pearl import _make_context_item

# Re-use the shared eval helpers (candidate pool / Q-min / argmax selection / schedule).
from pearl_brpc_action_adapter.eval.eval_full_pearl_dynamics_lookahead import (
    infer_z,
    make_candidates_local,
    make_schedule,
    q_min_single,
    select_index,
)


FINETUNE_METHODS = ("finetune_lastlayer", "finetune_full")


def is_finetune(method: str) -> bool:
    return method in FINETUNE_METHODS


def _build_finetune_setup(cfg: Dict, frozen: Dict, method: str, device: torch.device):
    """Deep-copy the frozen backbone into a trainable working copy and build the
    optimizers/targets for the chosen variant. The originals (and the on-disk
    checkpoint they came from) are untouched."""
    work = {k: copy.deepcopy(m).to(device) for k, m in frozen.items()}
    # Encoder is always frozen.
    for p in work["encoder"].parameters():
        p.requires_grad_(False)
    q1_t = copy.deepcopy(work["q1"]).to(device)
    q2_t = copy.deepcopy(work["q2"]).to(device)

    ft = cfg.get("finetune", {})
    critic_lr = float(ft.get("lr", cfg["pearl"]["critic_lr"]))
    actor_lr = float(ft.get("actor_lr", cfg["pearl"]["actor_lr"]))

    if method == "finetune_lastlayer":
        # Freeze everything, then re-enable ONLY the last linear layer of each Q.
        for m in (work["q1"], work["q2"], work["actor"]):
            for p in m.parameters():
                p.requires_grad_(False)
        crit_params = list(work["q1"].net[-1].parameters()) + list(work["q2"].net[-1].parameters())
        for p in crit_params:
            p.requires_grad_(True)
        opts = {"critic": torch.optim.Adam(crit_params, lr=critic_lr)}
        update_actor = False
        rerank = True
    elif method == "finetune_full":
        crit_params = list(work["q1"].parameters()) + list(work["q2"].parameters())
        opts = {
            "critic": torch.optim.Adam(crit_params, lr=critic_lr),
            "actor": torch.optim.Adam(work["actor"].parameters(), lr=actor_lr),
        }
        update_actor = True
        rerank = False
    else:
        raise ValueError(f"Not a fine-tune method: {method}")
    return work, q1_t, q2_t, opts, crit_params, update_actor, rerank


def _finetune_updates(
    cfg: Dict, work: Dict, q1_t, q2_t, opts: Dict, crit_params: List,
    buf: TaskReplayBuffer, z_ft: torch.Tensor, update_actor: bool,
    n_updates: int, rng: np.random.Generator, device: torch.device,
) -> None:
    """Run `n_updates` SAC gradient steps on the in-regime buffer, conditioned on the
    fixed regime latent z_ft (frozen encoder). Mirrors train_full_pearl's SAC update
    but with the encoder/KL removed and a single shared z."""
    gamma = float(cfg["pearl"]["gamma"])
    alpha = float(cfg["pearl"]["sac_alpha"])
    tau = float(cfg["pearl"]["tau"])
    rl_batch = int(cfg["pearl"]["rl_batch_size"])
    if len(buf) < max(rl_batch, 2):
        return
    for _ in range(n_updates):
        batch_np = buf.sample_batch(rl_batch, rng)
        batch = {k: torch.from_numpy(v).to(device) for k, v in batch_np.items()}
        batch["rewards"] = batch["rewards"].unsqueeze(-1)
        batch["dones"] = batch["dones"].unsqueeze(-1)
        q_loss = compute_q_loss(
            batch, z_ft, work["q1"], work["q2"], q1_t, q2_t, work["actor"], gamma, alpha
        )
        opts["critic"].zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(crit_params, 10.0)
        opts["critic"].step()
        if update_actor:
            actor_loss = compute_actor_loss(batch, z_ft, work["actor"], work["q1"], work["q2"], alpha)
            opts["actor"].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(work["actor"].parameters(), 10.0)
            opts["actor"].step()
        soft_update(work["q1"], q1_t, tau)
        soft_update(work["q2"], q2_t, tau)


def _pick_finetune_action(
    method: str, cfg: Dict, lk_cfg: Dict, work: Dict, frozen: Dict,
    s: np.ndarray, z: torch.Tensor, action_low, action_high,
    explore: bool, rng: np.random.Generator, device: torch.device,
):
    """Return (action, a0_frozen). finetune_full deploys the fine-tuned actor;
    finetune_lastlayer re-ranks a local candidate pool with the fine-tuned Q (like
    q_greedy, but with the fine-tuned critic)."""
    st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        a0_frozen = frozen["actor"].deterministic_action(st, z)[0].cpu().numpy().astype(np.float32)

    if method == "finetune_full":
        with torch.no_grad():
            if explore:
                a, _ = work["actor"].sample_action_logprob(st, z)
                action = a[0].cpu().numpy().astype(np.float32)
            else:
                action = work["actor"].deterministic_action(st, z)[0].cpu().numpy().astype(np.float32)
        return np.clip(action, action_low, action_high).astype(np.float32), a0_frozen

    # finetune_lastlayer: candidate pool from the FROZEN actor, re-ranked by fine-tuned Q.
    candidates = make_candidates_local(
        work["actor"], s, z, a0_frozen, action_low, action_high, lk_cfg, rng, device
    )
    n = candidates.shape[0]
    states_t = torch.from_numpy(np.repeat(s[None, :].astype(np.float32), n, axis=0)).to(device)
    actions_t = torch.from_numpy(candidates.astype(np.float32)).to(device)
    z_b = z.expand(n, -1)
    with torch.no_grad():
        q = torch.minimum(work["q1"](states_t, actions_t, z_b), work["q2"](states_t, actions_t, z_b)).squeeze(-1)
    q_scores = q.cpu().numpy()
    lambda_a = float(lk_cfg.get("lambda_action", 1.0))
    trust = lambda_a * np.sum((candidates - a0_frozen[None, :]) ** 2, axis=1)
    idx, _ = select_index(q_scores - trust, lk_cfg, rng)
    return np.clip(candidates[idx], action_low, action_high).astype(np.float32), a0_frozen


def _run_finetune_fleet_episode(
    cfg: Dict, meta: Dict, frozen: Dict, work: Dict, q1_t, q2_t, opts: Dict,
    crit_params: List, buf: TaskReplayBuffer, method: str, regime: Dict,
    seed: int, episode_idx: int, num_agents: int, update_actor: bool,
    context: Deque, explore: bool, n_updates: int, device: torch.device,
) -> List[Dict]:
    """One fleet episode: M agents act in lockstep in the same regime, all transitions
    feed one shared in-regime buffer + one shared context (M-fold data). After the
    episode, run `n_updates` SAC gradient steps. Returns one metric dict per agent."""
    lk_cfg = cfg["lookahead"]
    M = num_agents
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    max_steps = int(cfg["env"]["max_episode_steps"])
    rng = np.random.default_rng(seed + 100000 * episode_idx)

    envs = [
        maybe_wrap_action_perturb(make_env(cfg["env"]["name"], seed + 1000 * episode_idx + i), regime, seed + 1000 * episode_idx + i)
        for i in range(M)
    ]
    schedule = make_schedule(regime, cfg["dynamics_randomization"]["train_range"])
    action_low = envs[0].action_space.low.astype(np.float32)
    action_high = envs[0].action_space.high.astype(np.float32)
    states = [envs[i].reset(seed=seed + 1000 * episode_idx + i)[0] for i in range(M)]
    alive = [True for _ in range(M)]
    agg = [{"return": 0.0, "q_lift": [], "corr": [], "rew": []} for _ in range(M)]

    for t in range(max_steps):
        if not any(alive):
            break
        for i in range(M):
            if not alive[i]:
                continue
            env, s = envs[i], states[i]
            env.set_dynamics(schedule.xi_at(t))
            z = infer_z(frozen["encoder"], list(context), meta["latent_dim"], min_context, device)
            action, a0_frozen = _pick_finetune_action(
                method, cfg, lk_cfg, work, frozen, s, z, action_low, action_high, explore, rng, device
            )
            s_next, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            buf.add(s, action, float(reward), s_next, done, np.zeros(meta["latent_dim"], dtype=np.float32))
            context.append(_make_context_item(s, action, reward, s_next, done))
            # q_lift / correction measured against the FROZEN backbone (comparable to other methods).
            q_base = q_min_single(frozen, s, a0_frozen, z)
            q_exec = q_min_single(frozen, s, action, z)
            agg[i]["return"] += float(reward)
            agg[i]["q_lift"].append(float(q_exec - q_base))
            agg[i]["corr"].append(float(np.sum((action - a0_frozen) ** 2)))
            agg[i]["rew"].append(float(reward))
            states[i] = s_next
            if done:
                alive[i] = False
                env.close()
    for i in range(M):
        if alive[i]:
            envs[i].close()

    # Continual adaptation: SAC gradient steps after the episode, on the shared z.
    if n_updates > 0:
        z_ft = infer_z(frozen["encoder"], list(context), meta["latent_dim"], min_context, device)
        _finetune_updates(cfg, work, q1_t, q2_t, opts, crit_params, buf, z_ft, update_actor, n_updates, rng, device)

    out = []
    for i in range(M):
        a = agg[i]
        out.append({
            "return": a["return"],
            "length": len(a["q_lift"]),
            "mean_q_lift": float(np.mean(a["q_lift"])) if a["q_lift"] else 0.0,
            "mean_value_shift_term": 0.0,
            "mean_gate": 1.0,
            "mean_raw_resid_norm": 0.0,
            "mean_correction_sq": float(np.mean(a["corr"])) if a["corr"] else 0.0,
            "reward_trace": a["rew"],
        })
    return out


def evaluate_finetune(
    cfg: Dict, meta: Dict, frozen: Dict, method: str, regime: Dict, seed: int,
    output_dir: Path, warmup_override: int = None, num_agents: int = 1,
) -> Dict:
    """Test-time fine-tune of a deep-copied backbone: K warmup fleet-episodes adapt the
    weights (continual through the measured test episodes); the pretrained checkpoint on
    disk is never modified."""
    device = next(frozen["actor"].parameters()).device
    lk_cfg = cfg["lookahead"]
    ft = cfg.get("finetune", {})
    warmup = int(lk_cfg.get("warmup_episodes", 0)) if warmup_override is None else int(warmup_override)
    n_test = int(cfg["eval"]["num_episodes"])
    n_updates = int(ft.get("updates_per_episode", 200))
    ctx_max = int(cfg["latent"].get("eval_context_max", 50))
    capacity = int(ft.get("buffer_capacity", 100000))

    work, q1_t, q2_t, opts, crit_params, update_actor, _ = _build_finetune_setup(cfg, frozen, method, device)
    buf = TaskReplayBuffer(capacity, meta["state_dim"], meta["action_dim"], meta["latent_dim"])
    context: Deque[np.ndarray] = deque(maxlen=ctx_max)

    M = max(int(num_agents), 1)
    ep = 0
    episodes: List[Dict] = []
    # Warmup: explore (stochastic) to collect adaptation data; not measured.
    for _ in range(warmup):
        _run_finetune_fleet_episode(
            cfg, meta, frozen, work, q1_t, q2_t, opts, crit_params, buf, method, regime,
            seed, ep, M, update_actor, context, explore=True, n_updates=n_updates, device=device,
        )
        ep += 1
    # Test: deterministic deployment, still adapting continually (matches value_shift_qr).
    for _ in range(n_test):
        episodes.extend(
            _run_finetune_fleet_episode(
                cfg, meta, frozen, work, q1_t, q2_t, opts, crit_params, buf, method, regime,
                seed, ep, M, update_actor, context, explore=False, n_updates=n_updates, device=device,
            )
        )
        ep += 1

    out = {
        "method": method, "regime": regime["name"], "seed": seed,
        "warmup_episodes": warmup, "num_agents": M,
    }
    from pearl_brpc_action_adapter.eval.eval_full_pearl_dynamics_lookahead import finalize_traces
    out.update(finalize_traces(episodes, int(cfg["env"]["max_episode_steps"]), int(lk_cfg.get("trace_bins", 20))))
    for key in episodes[0]:
        vals = [x[key] for x in episodes]
        out[f"mean_{key}"] = float(np.mean(vals))
        out[f"std_{key}"] = float(np.std(vals))
    out["num_episodes"] = len(episodes)
    # Ordered per-episode returns (M=1 online deployment curve) for cumulative / regret analysis.
    out["episode_returns"] = [float(e["return"]) for e in episodes]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
