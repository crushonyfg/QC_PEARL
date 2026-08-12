from __future__ import annotations

import argparse
import copy
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from pearl_brpc_action_adapter.buffers.task_replay_buffer import MultiTaskBuffer
from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.envs.make_env import LATENT_KEYS, make_env, sample_train_xi
from pearl_brpc_action_adapter.pearl.actor import TanhGaussianPolicy
from pearl_brpc_action_adapter.pearl.context_encoder import PEARLContextEncoder
from pearl_brpc_action_adapter.pearl.q_network import QNetwork
from pearl_brpc_action_adapter.pearl.sac_losses import (
    compute_actor_loss,
    compute_q_loss,
    kl_divergence_to_standard_normal,
    soft_update,
)


def _device(cfg: Dict) -> torch.device:
    if cfg.get("device", "auto") == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def _context_tensor(context: List[np.ndarray], device: torch.device) -> torch.Tensor:
    if not context:
        return torch.zeros(1, 0, device=device)
    return torch.from_numpy(np.stack(context, axis=0).astype(np.float32)).unsqueeze(0).to(device)


def _infer_z_for_action(encoder: PEARLContextEncoder, context: List[np.ndarray], latent_dim: int, device: torch.device) -> torch.Tensor:
    if not context:
        return torch.zeros(1, latent_dim, device=device)
    with torch.no_grad():
        return encoder.infer_mean(_context_tensor(context, device))


def _make_context_item(s, a, r, s_next, done) -> np.ndarray:
    return np.concatenate([s, a, [float(r)], s_next, [float(done)]]).astype(np.float32)


def _make_models(cfg: Dict, meta: Dict, device: torch.device) -> Dict[str, nn.Module]:
    latent_dim = cfg["latent"]["dim"]
    actor = TanhGaussianPolicy(
        meta["state_dim"],
        meta["action_dim"],
        latent_dim,
        hidden_sizes=tuple(cfg["networks"]["actor_hidden"]),
        action_scale=float(cfg["pearl"].get("action_scale", 1.0)),
    ).to(device)
    q1 = QNetwork(meta["state_dim"], meta["action_dim"], latent_dim, tuple(cfg["networks"]["q_hidden"])).to(device)
    q2 = QNetwork(meta["state_dim"], meta["action_dim"], latent_dim, tuple(cfg["networks"]["q_hidden"])).to(device)
    encoder = PEARLContextEncoder(
        meta["state_dim"],
        meta["action_dim"],
        latent_dim,
        tuple(cfg["networks"]["encoder_hidden"]),
    ).to(device)
    return {"actor": actor, "q1": q1, "q2": q2, "encoder": encoder}


def sample_tasks(cfg: Dict, seed: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_tasks = int(cfg["full_pearl"]["num_train_tasks"])
    return [sample_train_xi(rng, cfg["dynamics_randomization"]["train_range"]) for _ in range(n_tasks)]


def init_buffers(cfg: Dict, seed: int) -> Tuple[MultiTaskBuffer, Dict]:
    probe = make_env(cfg["env"]["name"], seed)
    state_dim = probe.observation_space.shape[0]
    action_dim = probe.action_space.shape[0]
    probe.close()
    meta = {"state_dim": state_dim, "action_dim": action_dim, "latent_dim": cfg["latent"]["dim"]}
    mtb = MultiTaskBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=cfg["latent"]["dim"],
        capacity=int(cfg["meta"].get("buffer_capacity", 100000)),
    )
    return mtb, meta


def collect_rollout(
    cfg: Dict,
    task_xi: Dict[str, float],
    task_id: int,
    mtb: MultiTaskBuffer,
    actor: TanhGaussianPolicy,
    encoder: PEARLContextEncoder,
    device: torch.device,
    seed: int,
    random_policy: bool = False,
) -> Tuple[int, float]:
    env = make_env(cfg["env"]["name"], seed, dynamics=task_xi)
    max_steps = int(cfg["env"]["max_episode_steps"])
    context_max = int(cfg["latent"].get("eval_context_max", 50))
    min_context = int(cfg["latent"].get("eval_context_min", 5))
    latent_dim = int(cfg["latent"]["dim"])
    context: Deque[np.ndarray] = deque(maxlen=context_max)
    buf = mtb.get_or_create(task_id)
    s, _ = env.reset(seed=seed)
    ep_return = 0.0
    for t in range(max_steps):
        if random_policy:
            a = env.action_space.sample().astype(np.float32)
        else:
            z = _infer_z_for_action(encoder, list(context), latent_dim, device) if len(context) >= min_context else torch.zeros(1, latent_dim, device=device)
            st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                a_t, _ = actor.sample_action_logprob(st, z)
            a = a_t[0].cpu().numpy().astype(np.float32)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        # z_stars is unused by full PEARL; keep zeros for buffer compatibility.
        buf.add(s, a, float(r), s_next, done, np.zeros(latent_dim, dtype=np.float32))
        context.append(_make_context_item(s, a, r, s_next, done))
        ep_return += float(r)
        s = s_next
        if done:
            env.close()
            return t + 1, ep_return
    env.close()
    return max_steps, ep_return


def initial_random_collection(cfg: Dict, tasks: List[Dict[str, float]], mtb: MultiTaskBuffer, models: Dict[str, nn.Module], device: torch.device, seed: int) -> None:
    n = int(cfg["full_pearl"].get("initial_random_episodes_per_task", 2))
    for tid, xi in enumerate(tasks):
        for ep in range(n):
            collect_rollout(cfg, xi, tid, mtb, models["actor"], models["encoder"], device, seed + 1000 * tid + ep, random_policy=True)
        if (tid + 1) % max(len(tasks) // 5, 1) == 0:
            total = sum(len(b) for b in mtb.buffers.values())
            print(f"  initial collection task {tid+1}/{len(tasks)}, transitions={total}", flush=True)


def pearl_update_step(
    cfg: Dict,
    mtb: MultiTaskBuffer,
    models: Dict[str, nn.Module],
    targets: Dict[str, nn.Module],
    opts: Dict[str, torch.optim.Optimizer],
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, float]:
    actor, q1, q2, encoder = models["actor"], models["q1"], models["q2"], models["encoder"]
    q1_t, q2_t = targets["q1"], targets["q2"]
    meta_batch = int(cfg["pearl"]["meta_batch_size"])
    context_size = int(cfg["latent"]["context_batch_size"])
    rl_batch = int(cfg["pearl"]["rl_batch_size"])
    gamma = float(cfg["pearl"]["gamma"])
    alpha = float(cfg["pearl"]["sac_alpha"])
    beta_kl = float(cfg["pearl"]["beta_kl"])
    tau = float(cfg["pearl"]["tau"])
    task_ids = mtb.sample_task_ids(meta_batch, rng)
    if not task_ids:
        return {"q_loss": 0.0, "actor_loss": 0.0, "kl": 0.0}

    q_losses, actor_losses, kls = [], [], []
    for tid in task_ids:
        buf = mtb.buffers[tid]
        if len(buf) < max(rl_batch, 2):
            continue
        ctx_np = buf.sample_context(context_size, rng)
        batch_np = buf.sample_batch(rl_batch, rng)
        ctx = torch.from_numpy(ctx_np).unsqueeze(0).to(device)
        batch = {k: torch.from_numpy(v).to(device) for k, v in batch_np.items()}
        batch["rewards"] = batch["rewards"].unsqueeze(-1)
        batch["dones"] = batch["dones"].unsqueeze(-1)

        z, info = encoder.sample_z(ctx)
        q_loss = compute_q_loss(batch, z, q1, q2, q1_t, q2_t, actor, gamma, alpha)
        kl = kl_divergence_to_standard_normal(info["z_mean"], info["z_var"])
        critic_encoder_loss = q_loss + beta_kl * kl
        opts["critic_encoder"].zero_grad()
        critic_encoder_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()) + list(encoder.parameters()), 10.0)
        opts["critic_encoder"].step()

        with torch.no_grad():
            z_det = encoder.infer_mean(ctx)
        actor_loss = compute_actor_loss(batch, z_det, actor, q1, q2, alpha)
        opts["actor"].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        opts["actor"].step()

        soft_update(q1, q1_t, tau)
        soft_update(q2, q2_t, tau)
        q_losses.append(float(q_loss.item()))
        actor_losses.append(float(actor_loss.item()))
        kls.append(float(kl.item()))

    return {
        "q_loss": float(np.mean(q_losses)) if q_losses else 0.0,
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
    }


def evaluate_nominal(cfg: Dict, models: Dict[str, nn.Module], device: torch.device, seed: int, n_episodes: int) -> Tuple[float, float]:
    lengths, returns = [], []
    xi = {k: 1.0 for k in LATENT_KEYS}
    for ep in range(n_episodes):
        mtb_dummy, _ = init_buffers(cfg, seed + ep + 9999)
        length, ret = collect_rollout(
            cfg,
            xi,
            0,
            mtb_dummy,
            models["actor"],
            models["encoder"],
            device,
            seed + ep,
            random_policy=False,
        )
        lengths.append(length)
        returns.append(ret)
    return float(np.mean(lengths)), float(np.mean(returns))


def save_checkpoint(path: Path, cfg: Dict, meta: Dict, models: Dict[str, nn.Module], best_length: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": cfg,
            "meta": meta,
            "actor": models["actor"].state_dict(),
            "q1": models["q1"].state_dict(),
            "q2": models["q2"].state_dict(),
            "encoder": models["encoder"].state_dict(),
            "best_length": best_length,
        },
        path,
    )


def train_full_pearl(cfg: Dict, seed: int = 0) -> Path:
    device = _device(cfg)
    rng = np.random.default_rng(seed)
    tasks = sample_tasks(cfg, seed)
    mtb, meta = init_buffers(cfg, seed)
    models = _make_models(cfg, meta, device)
    targets = {"q1": copy.deepcopy(models["q1"]).to(device), "q2": copy.deepcopy(models["q2"]).to(device)}
    opts = {
        "critic_encoder": torch.optim.Adam(
            list(models["q1"].parameters()) + list(models["q2"].parameters()) + list(models["encoder"].parameters()),
            lr=float(cfg["pearl"]["critic_lr"]),
        ),
        "actor": torch.optim.Adam(models["actor"].parameters(), lr=float(cfg["pearl"]["actor_lr"])),
    }
    out_dir = Path(cfg["checkpoints"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "full_pearl_best.pt"
    final_path = out_dir / "full_pearl_final.pt"
    best_length = -np.inf

    print("Initial random collection...", flush=True)
    initial_random_collection(cfg, tasks, mtb, models, device, seed)

    epochs = int(cfg["full_pearl"]["epochs"])
    rollouts_per_epoch = int(cfg["full_pearl"].get("rollouts_per_epoch", 20))
    updates_per_epoch = int(cfg["full_pearl"].get("updates_per_epoch", 200))
    eval_interval = int(cfg["full_pearl"].get("eval_interval", 5))
    eval_eps = int(cfg["policy_verify"].get("num_rollouts", 3))

    for epoch in range(1, epochs + 1):
        lengths, returns = [], []
        for i in range(rollouts_per_epoch):
            tid = int(rng.integers(0, len(tasks)))
            length, ret = collect_rollout(
                cfg,
                tasks[tid],
                tid,
                mtb,
                models["actor"],
                models["encoder"],
                device,
                seed + epoch * 100000 + i,
                random_policy=False,
            )
            lengths.append(length)
            returns.append(ret)
        losses = []
        for _ in range(updates_per_epoch):
            losses.append(pearl_update_step(cfg, mtb, models, targets, opts, rng, device))
        mean_loss = {k: float(np.mean([x[k] for x in losses])) for k in ("q_loss", "actor_loss", "kl")}
        print(
            f"epoch {epoch}/{epochs} collect_len={np.mean(lengths):.1f} "
            f"collect_return={np.mean(returns):.1f} q={mean_loss['q_loss']:.3f} "
            f"actor={mean_loss['actor_loss']:.3f} kl={mean_loss['kl']:.3f}",
            flush=True,
        )
        if epoch % eval_interval == 0:
            mean_len, mean_ret = evaluate_nominal(cfg, models, device, seed + epoch * 777, eval_eps)
            print(f"  nominal_eval epoch={epoch}: length={mean_len:.1f} return={mean_ret:.1f}", flush=True)
            if mean_len > best_length:
                best_length = mean_len
                save_checkpoint(best_path, cfg, meta, models, best_length)
                print(f"  saved best full PEARL checkpoint: {best_path}", flush=True)

    save_checkpoint(final_path, cfg, meta, models, best_length)
    if not best_path.exists():
        save_checkpoint(best_path, cfg, meta, models, best_length)
    min_len = float(cfg["policy_verify"].get("min_episode_length", 100))
    if best_length < min_len and cfg["policy_verify"].get("fail_on_short_episode", True):
        raise RuntimeError(f"Full PEARL failed nominal gate: best_length={best_length:.1f} < {min_len}")
    print(f"Finished full PEARL training. best_length={best_length:.1f}", flush=True)
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train standard full PEARL joint SAC baseline.")
    parser.add_argument("--config", default="configs/full_pearl_hopper_smoke.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_full_pearl(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
