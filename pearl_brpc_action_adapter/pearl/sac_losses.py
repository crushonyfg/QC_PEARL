from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def kl_divergence_to_standard_normal(z_mean: torch.Tensor, z_var: torch.Tensor) -> torch.Tensor:
    return 0.5 * (z_mean.pow(2) + z_var - torch.log(z_var + 1e-8) - 1.0).sum(dim=-1).mean()


def _expand_z(z: torch.Tensor, batch_size: int) -> torch.Tensor:
    if z.shape[0] == batch_size:
        return z
    return z.expand(batch_size, -1)


def compute_q_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    actor: nn.Module,
    q_target1: nn.Module,
    q_target2: nn.Module,
    z: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    bs = next_states.shape[0]
    z_b = _expand_z(z, bs)
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample_action_logprob(next_states, z_b)
        q1 = q_target1(next_states, next_actions, z_b)
        q2 = q_target2(next_states, next_actions, z_b)
        min_q = torch.min(q1, q2) - alpha * next_log_probs
        targets = rewards + gamma * (1.0 - dones) * min_q
    return targets


def compute_q_loss(
    batch: Dict[str, torch.Tensor],
    z: torch.Tensor,
    q1: nn.Module,
    q2: nn.Module,
    q_target1: nn.Module,
    q_target2: nn.Module,
    actor: nn.Module,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    targets = compute_q_targets(
        batch["rewards"],
        batch["next_states"],
        batch["dones"],
        actor,
        q_target1,
        q_target2,
        z,
        gamma,
        alpha,
    )
    bs = batch["states"].shape[0]
    z_b = _expand_z(z, bs)
    q1_pred = q1(batch["states"], batch["actions"], z_b)
    q2_pred = q2(batch["states"], batch["actions"], z_b)
    loss = 0.5 * (q1_pred - targets).pow(2).mean() + 0.5 * (q2_pred - targets).pow(2).mean()
    return loss


def compute_actor_loss(
    batch: Dict[str, torch.Tensor],
    z: torch.Tensor,
    actor: nn.Module,
    q1: nn.Module,
    q2: nn.Module,
    alpha: float,
) -> torch.Tensor:
    bs = batch["states"].shape[0]
    z_b = _expand_z(z, bs)
    actions, log_probs = actor.sample_action_logprob(batch["states"], z_b)
    q1_val = q1(batch["states"], actions, z_b)
    q2_val = q2(batch["states"], actions, z_b)
    min_q = torch.min(q1_val, q2_val)
    return (alpha * log_probs - min_q).mean()


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
