from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    z_star: np.ndarray
    task_id: int


class TaskReplayBuffer:
    """Per-task replay buffer with context sampling."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, latent_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
        self._z_stars = np.zeros((capacity, latent_dim), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        z_star: np.ndarray,
    ) -> None:
        i = self._ptr
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = float(done)
        self._z_stars[i] = z_star
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if self._size == 0:
            raise ValueError("Buffer is empty.")
        idx = rng.integers(0, self._size, size=batch_size)
        return {
            "states": self._states[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_states": self._next_states[idx],
            "dones": self._dones[idx],
            "z_stars": self._z_stars[idx],
        }

    def sample_context(self, context_size: int, rng: np.random.Generator) -> np.ndarray:
        """Return context transitions [N, context_dim]."""
        if self._size == 0:
            raise ValueError("Buffer is empty.")
        n = min(context_size, self._size)
        idx = rng.integers(0, self._size, size=n)
        ctx = np.concatenate(
            [
                self._states[idx],
                self._actions[idx],
                self._rewards[idx, None],
                self._next_states[idx],
                self._dones[idx, None],
            ],
            axis=-1,
        )
        return ctx.astype(np.float32)

    def all_transitions(self) -> Dict[str, np.ndarray]:
        n = self._size
        return {
            "states": self._states[:n].copy(),
            "actions": self._actions[:n].copy(),
            "rewards": self._rewards[:n].copy(),
            "next_states": self._next_states[:n].copy(),
            "dones": self._dones[:n].copy(),
            "z_stars": self._z_stars[:n].copy(),
        }


@dataclass
class MultiTaskBuffer:
    buffers: Dict[int, TaskReplayBuffer] = field(default_factory=dict)
    state_dim: int = 0
    action_dim: int = 0
    latent_dim: int = 0
    capacity: int = 100_000

    def get_or_create(self, task_id: int) -> TaskReplayBuffer:
        if task_id not in self.buffers:
            self.buffers[task_id] = TaskReplayBuffer(
                self.capacity, self.state_dim, self.action_dim, self.latent_dim
            )
        return self.buffers[task_id]

    def task_ids(self) -> List[int]:
        return list(self.buffers.keys())

    def sample_task_ids(self, n: int, rng: np.random.Generator) -> List[int]:
        ids = self.task_ids()
        if not ids:
            return []
        return list(rng.choice(ids, size=min(n, len(ids)), replace=len(ids) < n))

    def concat_all(self) -> Dict[str, np.ndarray]:
        parts = {k: [] for k in ("states", "actions", "rewards", "next_states", "dones", "z_stars")}
        for buf in self.buffers.values():
            if len(buf) == 0:
                continue
            data = buf.all_transitions()
            for k in parts:
                parts[k].append(data[k])
        if not parts["states"]:
            raise ValueError("No data in multi-task buffer.")
        return {k: np.concatenate(v, axis=0) for k, v in parts.items()}
