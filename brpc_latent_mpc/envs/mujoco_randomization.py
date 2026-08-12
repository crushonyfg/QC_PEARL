from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

LATENT_KEYS = ("mass", "friction", "damping", "actuator")


def _default_xi() -> Dict[str, float]:
    return {k: 1.0 for k in LATENT_KEYS}


def normalize_xi(xi: Dict[str, float], train_range: Dict[str, list]) -> np.ndarray:
    z = []
    for key in LATENT_KEYS:
        lo, hi = train_range[key]
        mu = 0.5 * (lo + hi)
        sigma = 0.5 * (hi - lo)
        z.append((xi[key] - mu) / max(sigma, 1e-8))
    return np.asarray(z, dtype=np.float32)


def xi_to_z_star(xi: Dict[str, float], train_range: Dict[str, list]) -> np.ndarray:
    return normalize_xi(xi, train_range)


def sample_train_xi(rng: np.random.Generator, train_range: Dict[str, list]) -> Dict[str, float]:
    return {key: float(rng.uniform(*train_range[key])) for key in LATENT_KEYS}


class DynamicsRandomizationWrapper(gym.Wrapper):
    """MuJoCo dynamics randomization with actuator scaling at step time."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        model = env.unwrapped.model
        self.orig_body_mass = np.array(model.body_mass, copy=True)
        self.orig_geom_friction = np.array(model.geom_friction, copy=True)
        self.orig_dof_damping = np.array(model.dof_damping, copy=True)
        self._xi = _default_xi()
        self._apply_dynamics()

    @property
    def model(self):
        return self.env.unwrapped.model

    def set_dynamics(self, xi: Dict[str, float]) -> None:
        self._xi = {**_default_xi(), **xi}
        self._apply_dynamics()

    def get_dynamics(self) -> Dict[str, float]:
        return dict(self._xi)

    def get_dynamics_vector(self) -> np.ndarray:
        return np.asarray([self._xi[k] for k in LATENT_KEYS], dtype=np.float32)

    def _apply_dynamics(self) -> None:
        model = self.model
        model.body_mass[:] = self.orig_body_mass * self._xi["mass"]
        model.geom_friction[:, 0] = self.orig_geom_friction[:, 0] * self._xi["friction"]
        model.dof_damping[:] = self.orig_dof_damping * self._xi["damping"]

    def reset(self, **kwargs):
        self._apply_dynamics()
        return self.env.reset(**kwargs)

    def step(self, action):
        action_exec = np.asarray(action, dtype=np.float32) * self._xi["actuator"]
        action_exec = np.clip(action_exec, self.action_space.low, self.action_space.high)
        return self.env.step(action_exec)


class EvalDynamicsSchedule:
    """Apply fixed, sudden, or gradual dynamics shifts during evaluation."""

    def __init__(
        self,
        regime: Dict,
        train_range: Dict[str, list],
    ):
        self.regime_type = regime.get("type", "fixed")
        self.train_range = train_range
        self.shift_step = int(regime.get("shift_step", 100))
        self.before = {**_default_xi(), **regime.get("before", {})}
        self.after = {**_default_xi(), **regime.get("after", {})}
        self.fixed = {**_default_xi(), **regime.get("fixed", {})}
        self.drift = regime.get("drift", {})

    def xi_at(self, t: int) -> Dict[str, float]:
        if self.regime_type == "nominal":
            return _default_xi()
        if self.regime_type == "fixed":
            return dict(self.fixed)
        if self.regime_type == "sudden":
            return dict(self.before if t < self.shift_step else self.after)
        if self.regime_type == "gradual":
            xi = _default_xi()
            for key, spec in self.drift.items():
                start = float(spec.get("start", 1.0))
                end = float(spec.get("end", 0.6))
                duration = int(spec.get("duration", 300))
                frac = min(1.0, t / max(duration, 1))
                xi[key] = start + frac * (end - start)
            return xi
        return _default_xi()

    def z_star_at(self, t: int) -> np.ndarray:
        return xi_to_z_star(self.xi_at(t), self.train_range)


def make_env(
    env_name: str,
    seed: int,
    dynamics: Optional[Dict[str, float]] = None,
) -> gym.Env:
    env = gym.make(env_name)
    env = DynamicsRandomizationWrapper(env)
    if dynamics is not None:
        env.set_dynamics(dynamics)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
