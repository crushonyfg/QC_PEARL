from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from brpc_latent_mpc.data.normalizer import Normalizer
from pearl_brpc_action_adapter.buffers.task_replay_buffer import MultiTaskBuffer
from pearl_brpc_action_adapter.dynamics.dynamics_model import LatentDynamicsModel
from pearl_brpc_action_adapter.envs.make_env import LATENT_KEYS, make_env, sample_train_xi, xi_to_z_star
from pearl_brpc_action_adapter.pearl.actor import TanhGaussianPolicy
from pearl_brpc_action_adapter.pearl.context_encoder import PEARLContextEncoder, SupervisedContextEncoder
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


def collect_meta_data(cfg: Dict, seed: int = 0) -> Tuple[MultiTaskBuffer, Dict]:
    rng = np.random.default_rng(seed)
    env_name = cfg["env"]["name"]
    train_range = cfg["dynamics_randomization"]["train_range"]
    max_steps = cfg["env"]["max_episode_steps"]
    num_tasks = cfg["meta"]["num_tasks_per_collect"]
    rollouts = cfg["meta"]["rollouts_per_task"]

    probe = make_env(env_name, seed)
    state_dim = probe.observation_space.shape[0]
    action_dim = probe.action_space.shape[0]
    latent_dim = cfg["latent"]["dim"]
    action_low = probe.action_space.low.astype(np.float32)
    action_high = probe.action_space.high.astype(np.float32)
    probe.close()

    mtb = MultiTaskBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        capacity=cfg["meta"]["buffer_capacity"],
    )

    policy_type = cfg["meta"].get("collect_policy", "random")
    actor = None
    device = _device(cfg)

    for task_idx in range(num_tasks):
        xi = sample_train_xi(rng, train_range)
        z_star = xi_to_z_star(xi, train_range).astype(np.float32)
        env = make_env(env_name, seed + task_idx, dynamics=xi)
        buf = mtb.get_or_create(task_idx)

        for _ in range(rollouts):
            s, _ = env.reset()
            for t in range(max_steps):
                if policy_type == "random":
                    a = env.action_space.sample()
                else:
                    raise NotImplementedError(f"collect_policy={policy_type}")
                s_next, r, term, trunc, _ = env.step(a)
                done = term or trunc
                buf.add(s, a, r, s_next, done, z_star)
                s = s_next
                if done:
                    break
        env.close()

    meta = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "action_low": action_low,
        "action_high": action_high,
    }
    return mtb, meta


def train_dynamics_model(cfg: Dict, mtb: MultiTaskBuffer, meta: Dict, device: torch.device) -> Tuple[LatentDynamicsModel, Normalizer, Normalizer]:
    data = mtb.concat_all()
    states = data["states"]
    actions = data["actions"]
    z_stars = data["z_stars"]
    deltas = data["next_states"] - data["states"]

    state_norm = Normalizer().fit(states)
    action_norm = Normalizer().fit(actions)
    z_norm = Normalizer().fit(z_stars)
    delta_norm = Normalizer().fit(deltas)

    xs = np.concatenate(
        [state_norm.transform(states), action_norm.transform(actions), z_norm.transform(z_stars)],
        axis=-1,
    )
    ys = delta_norm.transform(deltas)

    ds = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
    loader = DataLoader(ds, batch_size=cfg["dynamics"]["batch_size"], shuffle=True)

    model = LatentDynamicsModel(
        meta["state_dim"],
        meta["action_dim"],
        meta["latent_dim"],
        hidden_sizes=tuple(cfg["dynamics"]["hidden"]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["dynamics"]["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(cfg["dynamics"]["epochs"]):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            s_n = xb[:, : meta["state_dim"]]
            a_n = xb[:, meta["state_dim"] : meta["state_dim"] + meta["action_dim"]]
            z_n = xb[:, meta["state_dim"] + meta["action_dim"] :]
            pred = model(s_n, a_n, z_n)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % max(cfg["dynamics"]["epochs"] // 5, 1) == 0:
            print(f"  dynamics epoch {epoch+1}/{cfg['dynamics']['epochs']} loss={total/len(loader):.5f}")

    norm_bundle = {"state": state_norm, "action": action_norm, "z": z_norm, "delta": delta_norm}
    return model, norm_bundle


def _make_encoder(cfg: Dict, meta: Dict) -> nn.Module:
    hidden = tuple(cfg["networks"]["encoder_hidden"])
    if cfg["meta"].get("encoder_mode", "supervised") == "pearl":
        return PEARLContextEncoder(meta["state_dim"], meta["action_dim"], meta["latent_dim"], hidden)
    return SupervisedContextEncoder(meta["state_dim"], meta["action_dim"], meta["latent_dim"], hidden)


def train_universal_sac(
    cfg: Dict,
    mtb: MultiTaskBuffer,
    meta: Dict,
    device: torch.device,
) -> Dict[str, nn.Module]:
    """Train SAC with privileged z* (spec §13.2)."""
    latent_dim = meta["latent_dim"]
    action_scale = float(cfg["pearl"].get("action_scale", 1.0))

    actor = TanhGaussianPolicy(
        meta["state_dim"],
        meta["action_dim"],
        latent_dim,
        hidden_sizes=tuple(cfg["networks"]["actor_hidden"]),
        action_scale=action_scale,
    ).to(device)
    q1 = QNetwork(meta["state_dim"], meta["action_dim"], latent_dim, tuple(cfg["networks"]["q_hidden"])).to(device)
    q2 = QNetwork(meta["state_dim"], meta["action_dim"], latent_dim, tuple(cfg["networks"]["q_hidden"])).to(device)
    q1_t = copy.deepcopy(q1)
    q2_t = copy.deepcopy(q2)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["pearl"]["actor_lr"])
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg["pearl"]["critic_lr"])

    rng = np.random.default_rng(cfg["env"].get("seed", 0))
    meta_batch = cfg["pearl"]["meta_batch_size"]
    rl_batch = cfg["pearl"]["rl_batch_size"]
    num_updates = cfg["pearl"]["num_updates"]
    gamma = cfg["pearl"]["gamma"]
    alpha = cfg["pearl"]["sac_alpha"]
    tau = cfg["pearl"]["tau"]

    for step in range(num_updates):
        task_ids = mtb.sample_task_ids(meta_batch, rng)
        if not task_ids:
            break
        step_q = 0.0
        n_valid = 0

        for tid in task_ids:
            buf = mtb.buffers[tid]
            if len(buf) < 2:
                continue
            batch_np = buf.sample_batch(rl_batch, rng)
            batch = {k: torch.from_numpy(v).to(device) for k, v in batch_np.items()}
            z = batch["z_stars"]

            q_loss = compute_q_loss(batch, z, q1, q2, q1_t, q2_t, actor, gamma, alpha)
            actor_loss = compute_actor_loss(batch, z.detach(), actor, q1, q2, alpha)

            q_opt.zero_grad()
            actor_opt.zero_grad()
            q_loss.backward(retain_graph=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            q_opt.step()
            actor_opt.step()
            step_q += q_loss.item()
            n_valid += 1

        if n_valid == 0:
            continue
        soft_update(q1, q1_t, tau)
        soft_update(q2, q2_t, tau)
        if (step + 1) % max(num_updates // 5, 1) == 0:
            print(f"  SAC (oracle-z) step {step+1}/{num_updates} q_loss={step_q / n_valid:.4f}")

    return {"actor": actor, "q1": q1, "q2": q2}


def train_encoder(
    cfg: Dict,
    mtb: MultiTaskBuffer,
    meta: Dict,
    device: torch.device,
) -> nn.Module:
    """Train context encoder q(C)->z*."""
    encoder = _make_encoder(cfg, meta).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg["pearl"]["encoder_lr"])
    rng = np.random.default_rng(cfg["env"].get("seed", 0))
    context_size = cfg["latent"]["context_batch_size"]
    meta_batch = cfg["pearl"]["meta_batch_size"]
    num_updates = cfg["pearl"].get("encoder_updates", cfg["meta"].get("encoder_updates", max(cfg["pearl"]["num_updates"] // 5, 500)))
    encoder_mode = cfg["meta"].get("encoder_mode", "supervised")
    beta_kl = cfg["pearl"]["beta_kl"]

    for step in range(num_updates):
        task_ids = mtb.sample_task_ids(meta_batch, rng)
        if not task_ids:
            break
        loss_acc = 0.0
        n_valid = 0
        for tid in task_ids:
            buf = mtb.buffers[tid]
            if len(buf) < 2:
                continue
            ctx_np = buf.sample_context(context_size, rng)
            batch_np = buf.sample_batch(min(32, len(buf)), rng)
            ctx = torch.from_numpy(ctx_np).unsqueeze(0).to(device)
            z_target = torch.from_numpy(batch_np["z_stars"][0:1]).to(device)

            mu, logvar = encoder.forward_factors(ctx)
            z_mean, z_var = encoder.aggregate(mu, logvar)
            if encoder_mode == "pearl":
                loss = beta_kl * kl_divergence_to_standard_normal(z_mean, z_var) + ((z_mean - z_target) ** 2).mean()
            else:
                loss = ((z_mean - z_target) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += loss.item()
            n_valid += 1

        if n_valid and (step + 1) % max(num_updates // 5, 1) == 0:
            print(f"  encoder step {step+1}/{num_updates} loss={loss_acc / n_valid:.5f}")

    return encoder


def _sac_gradient_step(
    batch: Dict[str, torch.Tensor],
    actor: TanhGaussianPolicy,
    q1: QNetwork,
    q2: QNetwork,
    q1_t: QNetwork,
    q2_t: QNetwork,
    actor_opt: torch.optim.Optimizer,
    q_opt: torch.optim.Optimizer,
    gamma: float,
    alpha: float,
    tau: float,
) -> float:
    z = batch["z_stars"]
    q_loss = compute_q_loss(batch, z, q1, q2, q1_t, q2_t, actor, gamma, alpha)
    actor_loss = compute_actor_loss(batch, z.detach(), actor, q1, q2, alpha)
    q_opt.zero_grad()
    actor_opt.zero_grad()
    q_loss.backward(retain_graph=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    q_opt.step()
    actor_opt.step()
    soft_update(q1, q1_t, tau)
    soft_update(q2, q2_t, tau)
    return float(q_loss.item())


def train_online_sac(
    cfg: Dict,
    meta: Dict,
    device: torch.device,
    seed: int = 0,
    init_sac: Optional[Dict[str, nn.Module]] = None,
) -> Dict[str, nn.Module]:
    """Online SAC with privileged z* per episode."""
    env_name = cfg["env"]["name"]
    train_range = cfg["dynamics_randomization"]["train_range"]
    max_steps = cfg["env"]["max_episode_steps"]
    num_tasks = cfg["meta"].get("online_tasks", 20)
    episodes_per_task = cfg["meta"].get("online_episodes_per_task", 5)
    latent_dim = meta["latent_dim"]
    action_scale = float(cfg["pearl"].get("action_scale", 1.0))
    reward_scale = float(cfg["pearl"].get("reward_scale", 1.0))
    updates_per_episode = int(cfg["pearl"].get("sac_updates_per_episode", 1))
    rng = np.random.default_rng(seed)

    if init_sac is not None:
        actor = init_sac["actor"]
        q1 = init_sac["q1"]
        q2 = init_sac["q2"]
        q1_t = copy.deepcopy(q1)
        q2_t = copy.deepcopy(q2)
    else:
        actor = TanhGaussianPolicy(
            meta["state_dim"],
            meta["action_dim"],
            latent_dim,
            hidden_sizes=tuple(cfg["networks"]["actor_hidden"]),
            action_scale=action_scale,
        ).to(device)
        q1 = QNetwork(meta["state_dim"], meta["action_dim"], latent_dim, tuple(cfg["networks"]["q_hidden"])).to(device)
        q2 = QNetwork(meta["state_dim"], meta["action_dim"], latent_dim, tuple(cfg["networks"]["q_hidden"])).to(device)
        q1_t = copy.deepcopy(q1)
        q2_t = copy.deepcopy(q2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["pearl"]["actor_lr"])
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg["pearl"]["critic_lr"])

    gamma = cfg["pearl"]["gamma"]
    alpha = cfg["pearl"]["sac_alpha"]
    tau = cfg["pearl"]["tau"]
    batch_size = cfg["pearl"]["rl_batch_size"]
    replay_s, replay_a, replay_r, replay_ns, replay_d, replay_z = [], [], [], [], [], []

    ep_count = 0
    for task_idx in range(num_tasks):
        xi = sample_train_xi(rng, train_range)
        z_star = xi_to_z_star(xi, train_range).astype(np.float32)
        env = make_env(env_name, seed + task_idx, dynamics=xi)
        z_t = torch.from_numpy(z_star).unsqueeze(0).to(device)

        for _ in range(episodes_per_task):
            s, _ = env.reset()
            for _t in range(max_steps):
                s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    a_t, _ = actor.sample_action_logprob(s_t, z_t)
                a = a_t[0].cpu().numpy()
                s_next, r, term, trunc, _ = env.step(a)
                done = term or trunc
                replay_s.append(s)
                replay_a.append(a)
                replay_r.append(r * reward_scale)
                replay_ns.append(s_next)
                replay_d.append(float(done))
                replay_z.append(z_star)
                s = s_next
                if done:
                    break
            ep_count += 1

            if len(replay_s) >= batch_size:
                for _ in range(updates_per_episode):
                    idx = rng.integers(0, len(replay_s), size=batch_size)
                    batch = {
                        "states": torch.from_numpy(np.stack([replay_s[i] for i in idx]).astype(np.float32)).to(device),
                        "actions": torch.from_numpy(np.stack([replay_a[i] for i in idx]).astype(np.float32)).to(device),
                        "rewards": torch.from_numpy(np.array([replay_r[i] for i in idx], dtype=np.float32)).unsqueeze(-1).to(device),
                        "next_states": torch.from_numpy(np.stack([replay_ns[i] for i in idx]).astype(np.float32)).to(device),
                        "dones": torch.from_numpy(np.array([replay_d[i] for i in idx], dtype=np.float32)).unsqueeze(-1).to(device),
                        "z_stars": torch.from_numpy(np.stack([replay_z[i] for i in idx]).astype(np.float32)).to(device),
                    }
                    _sac_gradient_step(batch, actor, q1, q2, q1_t, q2_t, actor_opt, q_opt, gamma, alpha, tau)

        env.close()
        if (task_idx + 1) % max(num_tasks // 5, 1) == 0:
            print(f"  online SAC task {task_idx+1}/{num_tasks}, episodes={ep_count}")

    return {"actor": actor, "q1": q1, "q2": q2}


def train_sac_and_encoder(
    cfg: Dict,
    mtb: MultiTaskBuffer,
    meta: Dict,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, nn.Module]:
    if cfg["meta"].get("training_mode", "two_phase") == "joint":
        return _train_joint_pearl(cfg, mtb, meta, device)

    sac: Optional[Dict[str, nn.Module]] = None
    if cfg["meta"].get("warmstart_buffer_sac", False):
        print("Warm-starting SAC from meta-training buffer...")
        sac = train_universal_sac(cfg, mtb, meta, device)

    if cfg["meta"].get("use_online_sac", True):
        print("Training online universal SAC...")
        sac = train_online_sac(cfg, meta, device, seed, init_sac=sac)
    elif sac is None:
        sac = train_universal_sac(cfg, mtb, meta, device)
    print("Training context encoder...")
    encoder = train_encoder(cfg, mtb, meta, device)
    return {"actor": sac["actor"], "encoder": encoder, "q1": sac["q1"], "q2": sac["q2"]}


def _train_joint_pearl(
    cfg: Dict,
    mtb: MultiTaskBuffer,
    meta: Dict,
    device: torch.device,
) -> Dict[str, nn.Module]:
    """Full PEARL joint SAC + encoder training."""
    latent_dim = meta["latent_dim"]
    action_scale = float(cfg["pearl"].get("action_scale", 1.0))

    actor = TanhGaussianPolicy(
        meta["state_dim"],
        meta["action_dim"],
        latent_dim,
        hidden_sizes=tuple(cfg["networks"]["actor_hidden"]),
        action_scale=action_scale,
    ).to(device)
    q1 = QNetwork(meta["state_dim"], meta["action_dim"], latent_dim, tuple(cfg["networks"]["q_hidden"])).to(device)
    q2 = QNetwork(meta["state_dim"], meta["action_dim"], latent_dim, tuple(cfg["networks"]["q_hidden"])).to(device)
    q1_t = copy.deepcopy(q1)
    q2_t = copy.deepcopy(q2)
    encoder = _make_encoder(cfg, meta).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["pearl"]["actor_lr"])
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()) + list(encoder.parameters()), lr=cfg["pearl"]["critic_lr"])

    rng = np.random.default_rng(cfg["env"].get("seed", 0))
    context_size = cfg["latent"]["context_batch_size"]
    meta_batch = cfg["pearl"]["meta_batch_size"]
    rl_batch = cfg["pearl"]["rl_batch_size"]
    num_updates = cfg["pearl"]["num_updates"]
    gamma = cfg["pearl"]["gamma"]
    alpha = cfg["pearl"]["sac_alpha"]
    beta_kl = cfg["pearl"]["beta_kl"]
    tau = cfg["pearl"]["tau"]
    encoder_mode = cfg["meta"].get("encoder_mode", "supervised")

    for step in range(num_updates):
        task_ids = mtb.sample_task_ids(meta_batch, rng)
        if not task_ids:
            break

        step_q = 0.0
        n_valid = 0

        for tid in task_ids:
            buf = mtb.buffers[tid]
            if len(buf) < 2:
                continue
            ctx_np = buf.sample_context(context_size, rng)
            batch_np = buf.sample_batch(rl_batch, rng)

            ctx = torch.from_numpy(ctx_np).unsqueeze(0).to(device)
            batch = {k: torch.from_numpy(v).to(device) for k, v in batch_np.items()}

            z, info = encoder.sample_z(ctx)
            q_loss = compute_q_loss(batch, z, q1, q2, q1_t, q2_t, actor, gamma, alpha)

            z_det = info["z_mean"].detach()
            actor_loss = compute_actor_loss(batch, z_det, actor, q1, q2, alpha)

            enc_loss_task = q_loss
            if encoder_mode == "pearl":
                kl = kl_divergence_to_standard_normal(info["z_mean"], info["z_var"])
                enc_loss_task = enc_loss_task + beta_kl * kl
            else:
                z_target = batch["z_stars"][0:1]
                enc_loss_task = enc_loss_task + cfg["meta"].get("beta_sup", 1.0) * (
                    (info["z_mean"] - z_target) ** 2
                ).mean()

            q_opt.zero_grad()
            actor_opt.zero_grad()
            enc_loss_task.backward(retain_graph=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()) + list(encoder.parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            q_opt.step()
            actor_opt.step()

            step_q += enc_loss_task.item()
            n_valid += 1

        if n_valid == 0:
            continue

        soft_update(q1, q1_t, tau)
        soft_update(q2, q2_t, tau)

        if (step + 1) % max(num_updates // 5, 1) == 0:
            print(f"  SAC step {step+1}/{num_updates} q_loss={step_q / n_valid:.4f}")

    return {"actor": actor, "encoder": encoder, "q1": q1, "q2": q2}


def save_checkpoints(
    out_dir: Path,
    actor: nn.Module,
    encoder: nn.Module,
    dynamics: nn.Module,
    norm_bundle: Dict[str, Normalizer],
    meta: Dict,
    cfg: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": actor.state_dict(),
        "encoder": encoder.state_dict(),
        "dynamics": dynamics.state_dict(),
        "norms": {k: v.to_dict() for k, v in norm_bundle.items()},
        "meta": meta,
        "cfg": cfg,
    }
    torch.save(payload, out_dir / "pearl_brpc_checkpoint.pt")


def load_checkpoints(path: Path, device: torch.device) -> Dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload["meta"]
    cfg = payload["cfg"]
    actor = TanhGaussianPolicy(
        meta["state_dim"],
        meta["action_dim"],
        meta["latent_dim"],
        hidden_sizes=tuple(cfg["networks"]["actor_hidden"]),
        action_scale=float(cfg["pearl"].get("action_scale", 1.0)),
    ).to(device)
    encoder = _make_encoder(cfg, meta).to(device)
    dynamics = LatentDynamicsModel(
        meta["state_dim"],
        meta["action_dim"],
        meta["latent_dim"],
        hidden_sizes=tuple(cfg["dynamics"]["hidden"]),
    ).to(device)
    actor.load_state_dict(payload["actor"])
    encoder.load_state_dict(payload["encoder"])
    dynamics.load_state_dict(payload["dynamics"])
    norms = {k: Normalizer.from_dict(v) for k, v in payload["norms"].items()}
    return {
        "actor": actor,
        "encoder": encoder,
        "dynamics": dynamics,
        "norms": norms,
        "meta": meta,
        "cfg": payload["cfg"],
    }


def verify_policy_episode_length(
    cfg: Dict,
    actor: nn.Module,
    meta: Dict,
    device: torch.device,
    seed: int = 0,
) -> Tuple[float, bool]:
    """Roll out on nominal Hopper; return mean length and pass/fail vs threshold."""
    verify_cfg = cfg.get("policy_verify", {})
    min_length = float(verify_cfg.get("min_episode_length", 100))
    num_rollouts = int(verify_cfg.get("num_rollouts", 3))
    max_steps = cfg["env"]["max_episode_steps"]
    train_range = cfg["dynamics_randomization"]["train_range"]
    z_nominal = xi_to_z_star({k: 1.0 for k in LATENT_KEYS}, train_range).astype(np.float32)
    z_t = torch.from_numpy(z_nominal).unsqueeze(0).to(device)

    lengths = []
    env = make_env(cfg["env"]["name"], seed)
    for ep in range(num_rollouts):
        s, _ = env.reset(seed=seed + ep)
        for t in range(max_steps):
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                a = actor.deterministic_action(s_t, z_t)[0].cpu().numpy()
            s, _, term, trunc, _ = env.step(a)
            if term or trunc:
                lengths.append(t + 1)
                break
        else:
            lengths.append(max_steps)
    env.close()

    mean_len = float(np.mean(lengths))
    passed = mean_len >= min_length
    print(
        f"Policy verify (nominal): mean_episode_length={mean_len:.1f} "
        f"(min={min_length}, rollouts={num_rollouts}) -> {'PASS' if passed else 'FAIL'}"
    )
    return mean_len, passed


def train_all(cfg: Dict, seed: int = 0, skip_collect: bool = False, data_path: Optional[str] = None) -> Path:
    device = _device(cfg)
    ckpt_dir = Path(cfg["checkpoints"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if skip_collect and data_path:
        # reload buffer from npz
        raise NotImplementedError("Use full pipeline for now")
    print("Collecting meta-training data...")
    mtb, meta = collect_meta_data(cfg, seed)
    print(f"  tasks={len(mtb.task_ids())}, transitions={sum(len(b) for b in mtb.buffers.values())}")

    print("Training dynamics model...")
    dynamics, norm_bundle = train_dynamics_model(cfg, mtb, meta, device)

    print("Training SAC + context encoder...")
    models = train_sac_and_encoder(cfg, mtb, meta, device, seed=seed)

    if cfg.get("policy_verify"):
        _, passed = verify_policy_episode_length(cfg, models["actor"], meta, device, seed=seed)
        if not passed and cfg["policy_verify"].get("fail_on_short_episode", False):
            raise RuntimeError(
                f"Policy failed episode-length gate (need >= {cfg['policy_verify'].get('min_episode_length', 100)} steps on nominal)"
            )

    save_checkpoints(ckpt_dir, models["actor"], models["encoder"], dynamics, norm_bundle, meta, cfg)
    print(f"Saved checkpoint to {ckpt_dir / 'pearl_brpc_checkpoint.pt'}")
    return ckpt_dir / "pearl_brpc_checkpoint.pt"


if __name__ == "__main__":
    import argparse

    from pearl_brpc_action_adapter.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pearl_smoke.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_all(cfg, seed=args.seed)
