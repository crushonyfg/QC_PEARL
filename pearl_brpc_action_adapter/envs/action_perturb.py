"""Local action-side perturbation wrapper for OOD-type evaluation.

The shared DynamicsRandomizationWrapper already models an action GAIN (the `actuator`
key scales the action). This wrapper adds the other action-side shifts — additive bias
and actuation noise — that gain alone cannot express, so we can probe distinct
action-side OOD types. It wraps the env returned by make_env (so set_dynamics / unwrapped
still delegate through) and applies the perturbation before the inner step.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class ActionPerturbWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, gain: float = 1.0, bias=0.0, noise_std: float = 0.0, seed: int = 0):
        super().__init__(env)
        self.gain = float(gain)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        a = self.gain * np.asarray(action, dtype=np.float32) + self.bias
        if self.noise_std > 0.0:
            a = a + self._rng.normal(0.0, self.noise_std, size=a.shape).astype(np.float32)
        a = np.clip(a, self.action_space.low, self.action_space.high).astype(np.float32)
        return self.env.step(a)

    # Explicit passthrough (gymnasium.Wrapper does not auto-delegate custom methods).
    def set_dynamics(self, xi):
        return self.env.set_dynamics(xi)

    def get_dynamics(self):
        return self.env.get_dynamics()


def maybe_wrap_action_perturb(env: gym.Env, regime: dict, seed: int) -> gym.Env:
    """Wrap env if the regime declares an `action_perturb` block, e.g.
    {"action_perturb": {"bias": 0.2}} or {"action_perturb": {"noise_std": 0.15}}."""
    ap = regime.get("action_perturb")
    if not ap:
        return env
    return ActionPerturbWrapper(env, seed=seed, **ap)
