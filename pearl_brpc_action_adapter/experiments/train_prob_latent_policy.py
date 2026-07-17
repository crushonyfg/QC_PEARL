from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from pearl_brpc_action_adapter.buffers.task_replay_buffer import MultiTaskBuffer
from pearl_brpc_action_adapter.config import load_config
from pearl_brpc_action_adapter.envs.latent_policy_env import LatentPolicyEnv, make_policy_obs
from pearl_brpc_action_adapter.envs.make_env import LATENT_KEYS, make_env, sample_train_xi, xi_to_z_star
from pearl_brpc_action_adapter.experiments.train_all import (
    _device,
    train_dynamics_model,
    train_encoder,
)


class NominalLengthCallback(BaseCallback):
    """Save the SAC model with the best nominal Hopper survival length."""

    def __init__(self, cfg: Dict, out_dir: Path, seed: int, eval_freq: int, n_eval_episodes: int):
        super().__init__()
        self.cfg = cfg
        self.out_dir = out_dir
        self.seed = seed
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_mean_length = -np.inf
        self.best_path = out_dir / "best_latent_sac.zip"

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        mean_len, mean_return = evaluate_privileged_policy_length(
            self.model,
            self.cfg,
            seed=self.seed + self.num_timesteps,
            n_episodes=self.n_eval_episodes,
        )
        print(
            f"  eval@{self.num_timesteps}: mean_length={mean_len:.1f} "
            f"mean_return={mean_return:.1f} best_length={self.best_mean_length:.1f}",
            flush=True,
        )
        if mean_len > self.best_mean_length:
            self.best_mean_length = mean_len
            self.model.save(self.best_path)
            print(f"  saved best latent SAC to {self.best_path}", flush=True)
        return True


def evaluate_privileged_policy_length(
    model: SAC,
    cfg: Dict,
    seed: int,
    n_episodes: int,
) -> Tuple[float, float]:
    train_range = cfg["dynamics_randomization"]["train_range"]
    z_nom = xi_to_z_star({k: 1.0 for k in LATENT_KEYS}, train_range).astype(np.float32)
    env = make_env(cfg["env"]["name"], seed, dynamics={k: 1.0 for k in LATENT_KEYS})
    max_steps = int(cfg["env"]["max_episode_steps"])
    lengths, returns = [], []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        ret = 0.0
        for t in range(max_steps):
            obs = make_policy_obs(s, z_nom)
            a, _ = model.predict(obs, deterministic=True)
            s, r, term, trunc, _ = env.step(a)
            ret += float(r)
            if term or trunc:
                lengths.append(t + 1)
                returns.append(ret)
                break
        else:
            lengths.append(max_steps)
            returns.append(ret)
    env.close()
    return float(np.mean(lengths)), float(np.mean(returns))


def train_latent_sac(cfg: Dict, seed: int, out_dir: Path) -> SAC:
    sac_cfg = cfg["latent_sac"]
    train_env = Monitor(
        LatentPolicyEnv(
            cfg["env"]["name"],
            seed=seed,
            train_range=cfg["dynamics_randomization"]["train_range"],
            randomize_on_reset=True,
        )
    )
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
        buffer_size=int(sac_cfg.get("buffer_size", 1_000_000)),
        learning_starts=int(sac_cfg.get("learning_starts", 10_000)),
        batch_size=int(sac_cfg.get("batch_size", 256)),
        tau=float(sac_cfg.get("tau", 0.005)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        train_freq=(int(sac_cfg.get("train_freq", 1)), "step"),
        gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
        ent_coef=sac_cfg.get("ent_coef", "auto"),
        target_update_interval=int(sac_cfg.get("target_update_interval", 1)),
        verbose=int(sac_cfg.get("verbose", 1)),
        seed=seed,
        device=cfg.get("device", "auto"),
        tensorboard_log=str(out_dir / "tb"),
    )
    callback = NominalLengthCallback(
        cfg,
        out_dir,
        seed=seed,
        eval_freq=int(sac_cfg.get("eval_freq", 25_000)),
        n_eval_episodes=int(sac_cfg.get("eval_episodes", 3)),
    )
    total_timesteps = int(sac_cfg["total_timesteps"])
    print(f"Training latent SAC for {total_timesteps:,} steps...", flush=True)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    final_path = out_dir / "final_latent_sac.zip"
    model.save(final_path)
    train_env.close()
    best_path = out_dir / "best_latent_sac.zip"
    if best_path.exists():
        print(f"Loading best latent SAC from {best_path}", flush=True)
        return SAC.load(best_path, device=cfg.get("device", "auto"))
    print(f"No best callback checkpoint found; using final model {final_path}", flush=True)
    return SAC.load(final_path, device=cfg.get("device", "auto"))


def collect_policy_data(cfg: Dict, model: SAC, seed: int) -> Tuple[MultiTaskBuffer, Dict]:
    rng = np.random.default_rng(seed)
    train_range = cfg["dynamics_randomization"]["train_range"]
    data_cfg = cfg["adaptation_data"]
    probe = make_env(cfg["env"]["name"], seed)
    state_dim = probe.observation_space.shape[0]
    action_dim = probe.action_space.shape[0]
    probe.close()
    latent_dim = cfg["latent"]["dim"]
    mtb = MultiTaskBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        capacity=int(cfg["meta"].get("buffer_capacity", 100000)),
    )
    max_steps = int(cfg["env"]["max_episode_steps"])
    num_tasks = int(data_cfg.get("num_tasks", 50))
    rollouts = int(data_cfg.get("rollouts_per_task", 4))
    noise_std = float(data_cfg.get("action_noise", 0.05))

    for task_idx in range(num_tasks):
        xi = sample_train_xi(rng, train_range)
        z_star = xi_to_z_star(xi, train_range).astype(np.float32)
        env = make_env(cfg["env"]["name"], seed + task_idx, dynamics=xi)
        buf = mtb.get_or_create(task_idx)
        for ep in range(rollouts):
            s, _ = env.reset(seed=seed + 1000 * task_idx + ep)
            for _t in range(max_steps):
                obs = make_policy_obs(s, z_star)
                a, _ = model.predict(obs, deterministic=True)
                if noise_std > 0:
                    a = a + rng.normal(0.0, noise_std, size=a.shape).astype(np.float32)
                a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
                s_next, r, term, trunc, _ = env.step(a)
                done = term or trunc
                buf.add(s, a, r, s_next, done, z_star)
                s = s_next
                if done:
                    break
        env.close()
        if (task_idx + 1) % max(num_tasks // 5, 1) == 0:
            total = sum(len(b) for b in mtb.buffers.values())
            print(f"  collected task {task_idx+1}/{num_tasks}, transitions={total}", flush=True)

    meta = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
    }
    return mtb, meta


def save_aux_checkpoint(out_dir: Path, dynamics, encoder, norm_bundle, meta: Dict, cfg: Dict) -> Path:
    payload = {
        "dynamics": dynamics.state_dict(),
        "encoder": encoder.state_dict(),
        "norms": {k: v.to_dict() for k, v in norm_bundle.items()},
        "meta": meta,
        "cfg": cfg,
    }
    path = out_dir / "latent_sac_aux.pt"
    torch.save(payload, path)
    return path


def train_all(cfg: Dict, seed: int = 0) -> Path:
    out_dir = Path(cfg["checkpoints"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model = train_latent_sac(cfg, seed, out_dir)

    mean_len, mean_ret = evaluate_privileged_policy_length(
        model,
        cfg,
        seed=seed + 777,
        n_episodes=int(cfg["policy_verify"].get("num_rollouts", 5)),
    )
    min_len = float(cfg["policy_verify"].get("min_episode_length", 100))
    print(
        f"Policy verify privileged nominal: mean_episode_length={mean_len:.1f}, "
        f"mean_return={mean_ret:.1f}, min={min_len}",
        flush=True,
    )
    if mean_len < min_len and cfg["policy_verify"].get("fail_on_short_episode", True):
        raise RuntimeError(f"Latent SAC policy failed stability gate: {mean_len:.1f} < {min_len}")

    print("Collecting policy rollouts for dynamics model and context encoder...", flush=True)
    mtb, meta = collect_policy_data(cfg, model, seed)
    device = _device(cfg)
    print("Training latent-conditioned dynamics model...", flush=True)
    dynamics, norm_bundle = train_dynamics_model(cfg, mtb, meta, device)
    print("Training probabilistic context encoder...", flush=True)
    encoder = train_encoder(cfg, mtb, meta, device)
    aux_path = save_aux_checkpoint(out_dir, dynamics, encoder, norm_bundle, meta, cfg)
    print(f"Saved latent SAC model and aux checkpoint to {out_dir}", flush=True)
    return aux_path


def main():
    parser = argparse.ArgumentParser(description="Train stable probabilistic latent SAC + adaptation models.")
    parser.add_argument("--config", default="configs/prob_latent_sac_hopper.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_all(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
