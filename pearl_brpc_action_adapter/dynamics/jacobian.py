from __future__ import annotations

import torch


def action_jacobian(model, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute df/da at (state, action, z). Returns [state_dim, action_dim]."""
    action = action.clone().detach().requires_grad_(True)
    delta = model(state.unsqueeze(0), action.unsqueeze(0), z.unsqueeze(0))[0]
    rows = []
    for j in range(delta.shape[0]):
        grad_j = torch.autograd.grad(
            delta[j],
            action,
            retain_graph=True,
            create_graph=False,
        )[0]
        rows.append(grad_j)
    return torch.stack(rows, dim=0).detach()


def solve_delta_u(J: torch.Tensor, e: torch.Tensor, lambda_J: float = 1e-3) -> torch.Tensor:
    """Damped least squares: delta_u = -(J^T J + lambda I)^-1 J^T e."""
    jtj = J.T @ J
    action_dim = J.shape[1]
    reg = lambda_J * torch.eye(action_dim, device=J.device, dtype=J.dtype)
    rhs = J.T @ e
    try:
        delta_u = -torch.linalg.solve(jtj + reg, rhs)
    except RuntimeError:
        delta_u = -torch.linalg.lstsq(jtj + reg, rhs).solution
    return delta_u
