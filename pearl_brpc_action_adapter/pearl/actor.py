from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class TanhGaussianPolicy(nn.Module):
    """SAC-style tanh-squashed Gaussian policy conditioned on z."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes=(256, 256),
        action_scale: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.action_scale = action_scale
        in_dim = state_dim + latent_dim
        layers = []
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Linear(in_dim, action_dim)

    def _forward_params(self, states: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([states, z], dim=-1)
        h = self.trunk(x)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def forward(
        self, states: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._forward_params(states, z)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mu) * self.action_scale
        return action, log_prob, mean_action

    def deterministic_action(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mu, _ = self._forward_params(state, z)
        return torch.tanh(mu) * self.action_scale

    def sample_action_logprob(
        self, states: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action, log_prob, _ = self.forward(states, z)
        return action, log_prob
