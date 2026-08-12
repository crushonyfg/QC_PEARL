from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pearl_brpc_action_adapter.envs.make_env import (
    LATENT_KEYS,
    make_env,
    sample_train_xi,
    xi_to_z_star,
)


def default_xi() -> Dict[str, float]:
    return {k: 1.0 for k in LATENT_KEYS}


class LatentPolicyEnv(gym.Env):
    """Hopper env exposing privileged latent z in the policy observation.

    Training uses randomized dynamics and appends z_star to the physical state.
    Evaluation can pass a fixed dynamics dict. The policy still receives a z
    value, either z_star for privileged evaluation or an external estimate.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_name: str,
        seed: int,
        train_range: Dict[str, list],
        randomize_on_reset: bool = True,
        fixed_dynamics: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.env_name = env_name
        self.train_range = train_range
        self.randomize_on_reset = randomize_on_reset
        self.fixed_dynamics = {**default_xi(), **(fixed_dynamics or {})}
        self.rng = np.random.default_rng(seed)
        self.env = make_env(env_name, seed)
        self.state_dim = int(self.env.observation_space.shape[0])
        self.action_space = self.env.action_space
        obs_low = np.concatenate(
            [
                self.env.observation_space.low.astype(np.float32),
                -np.full(len(LATENT_KEYS), 2.0, dtype=np.float32),
            ]
        )
        obs_high = np.concatenate(
            [
                self.env.observation_space.high.astype(np.float32),
                np.full(len(LATENT_KEYS), 2.0, dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.current_xi = default_xi()
        self.current_z = np.zeros(len(LATENT_KEYS), dtype=np.float32)

    def _sample_or_fixed_xi(self) -> Dict[str, float]:
        if self.randomize_on_reset:
            return sample_train_xi(self.rng, self.train_range)
        return dict(self.fixed_dynamics)

    def _aug_obs(self, state: np.ndarray, z: Optional[np.ndarray] = None) -> np.ndarray:
        z_use = self.current_z if z is None else z
        return np.concatenate([state.astype(np.float32), z_use.astype(np.float32)], axis=-1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_xi = self._sample_or_fixed_xi()
        self.current_z = xi_to_z_star(self.current_xi, self.train_range).astype(np.float32)
        self.env.set_dynamics(self.current_xi)
        state, info = self.env.reset(seed=seed)
        info = dict(info)
        info["xi"] = dict(self.current_xi)
        info["z_star"] = self.current_z.copy()
        return self._aug_obs(state), info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["xi"] = dict(self.current_xi)
        info["z_star"] = self.current_z.copy()
        return self._aug_obs(next_state), reward, terminated, truncated, info

    def close(self):
        self.env.close()


def make_policy_obs(state: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.concatenate([state.astype(np.float32), z.astype(np.float32)], axis=-1)
