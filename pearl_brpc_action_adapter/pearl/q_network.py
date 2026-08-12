from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes=(256, 256),
    ):
        super().__init__()
        in_dim = state_dim + action_dim + latent_dim
        layers = []
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions, z], dim=-1)
        return self.net(x)

    def features(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer activations (the basis in which Q is linear: Q = w·φ + b).
        Used as the feature map for an online Bellman-residual correction head."""
        x = torch.cat([states, actions, z], dim=-1)
        return self.net[:-1](x)
