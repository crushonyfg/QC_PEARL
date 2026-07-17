from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class LatentDynamicsModel(nn.Module):
    """Deterministic latent-conditioned dynamics: f(s,a,z) -> delta_s."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes=(256, 256, 256),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim + latent_dim
        self.net = _mlp([input_dim, *hidden_sizes, state_dim])

    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions, z], dim=-1)
        return self.net(x)

    def predict(self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.forward(state, action, z)


class NormalizedLatentDynamicsModel(nn.Module):
    """Latent-conditioned world model f(s,a,z) -> delta_s, in a normalized space.

    Used only to provide the operating point s'_nominal = s + f(s,a,z) at which the
    value-shift correction ΔV = V(s + f + Δf) - V(s + f) is evaluated; the online
    BRPC residual Δf carries the actual distribution-shift signal. Inputs (state)
    and the delta_s target are standardized with stats stored as buffers so the
    same normalization is reapplied at eval. Actions and z are O(1) and fed raw.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes=(256, 256, 256),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim + latent_dim
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.trunk = nn.Sequential(*layers)  # ends in ReLU
        self.delta_head = nn.Linear(hidden_sizes[-1], state_dim)
        # Normalization buffers (filled by set_norm_stats before training/eval).
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("delta_mean", torch.zeros(state_dim))
        self.register_buffer("delta_std", torch.ones(state_dim))

    def set_norm_stats(self, stats: dict) -> None:
        for key in ("state_mean", "state_std", "delta_mean", "delta_std"):
            val = torch.as_tensor(stats[key], dtype=torch.float32, device=self.state_mean.device)
            getattr(self, key).copy_(val.reshape(getattr(self, key).shape))

    def _trunk_forward(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s_norm = (states - self.state_mean) / self.state_std
        x = torch.cat([s_norm, actions, z], dim=-1)
        return self.trunk(x)

    def forward_norm(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return delta_s in normalized space (for the training loss)."""
        return self.delta_head(self._trunk_forward(states, actions, z))

    def predict(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return delta_s in real units."""
        h = self._trunk_forward(states, actions, z)
        return self.delta_head(h) * self.delta_std + self.delta_mean
