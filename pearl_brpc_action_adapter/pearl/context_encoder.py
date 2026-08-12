from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class PEARLContextEncoder(nn.Module):
    """Product-of-Gaussians context encoder (PEARL-style)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes=(200, 200, 200),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.context_dim = 2 * state_dim + action_dim + 2
        layers = []
        in_dim = self.context_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 2 * latent_dim)

    def forward_factors(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # context: [B, N, context_dim]
        b, n, _ = context.shape
        h = self.trunk(context.reshape(b * n, -1))
        out = self.out(h).reshape(b, n, 2 * self.latent_dim)
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        return mu, logvar

    def aggregate(self, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mu, logvar: [B, N, latent_dim]
        var = torch.exp(logvar)
        precision = 1.0 + (1.0 / var).sum(dim=1)
        weighted_mu = (mu / var).sum(dim=1)
        z_var = 1.0 / precision
        z_mean = z_var * weighted_mu
        return z_mean, z_var

    def sample_z(self, context: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        mu, logvar = self.forward_factors(context)
        z_mean, z_var = self.aggregate(mu, logvar)
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.sqrt(z_var) * eps
        return z, {"z_mean": z_mean, "z_var": z_var, "mu": mu, "logvar": logvar}

    def infer_mean(self, context: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.forward_factors(context)
        z_mean, _ = self.aggregate(mu, logvar)
        return z_mean


class SupervisedContextEncoder(nn.Module):
    """Supervised encoder: context -> z_star (UP-OSI style)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes=(200, 200, 200),
    ):
        super().__init__()
        self.pearl = PEARLContextEncoder(state_dim, action_dim, latent_dim, hidden_sizes)

    def forward_factors(self, context):
        return self.pearl.forward_factors(context)

    def aggregate(self, mu, logvar):
        return self.pearl.aggregate(mu, logvar)

    def sample_z(self, context):
        z_mean, z_var = self.infer_mean_var(context)
        eps = torch.randn_like(z_mean)
        return z_mean + torch.sqrt(z_var) * eps, {"z_mean": z_mean, "z_var": z_var}

    def infer_mean(self, context):
        mu, logvar = self.forward_factors(context)
        z_mean, _ = self.aggregate(mu, logvar)
        return z_mean

    def infer_mean_var(self, context):
        mu, logvar = self.forward_factors(context)
        return self.aggregate(mu, logvar)
