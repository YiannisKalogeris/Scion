"""
Loss functions for scientific ML training.

Includes standard data-driven losses and physics-informed losses
that embed PDE residuals into the training objective.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Standard mean squared error (wrapper for convenience)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class RelativeMSELoss(nn.Module):
    """Relative L2 loss: ||pred - target||² / ||target||²."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum((pred - target) ** 2) / (torch.sum(target ** 2) + 1e-12)


class WeightedLoss(nn.Module):
    """Weighted combination of multiple loss terms.

    Parameters
    ----------
    losses : dict[str, nn.Module]
        Named loss functions.
    weights : dict[str, float]
        Weight for each loss term.

    Example
    -------
    >>> loss = WeightedLoss(
    ...     losses={"data": MSELoss(), "physics": PhysicsInformedLoss(residual_fn)},
    ...     weights={"data": 1.0, "physics": 0.1},
    ... )
    """

    def __init__(self, losses: dict[str, nn.Module], weights: dict[str, float]):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=pred.device)
        for name, loss_fn in self.losses.items():
            total = total + self.weights[name] * loss_fn(pred, target)
        return total


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss that penalises PDE residuals.

    The residual function computes how well the predicted field
    satisfies the governing PDE (should be zero for exact solutions).

    Parameters
    ----------
    residual_fn : callable
        ``(pred, coords) -> residual_tensor`` where the residual is the
        PDE evaluated at collocation points using automatic differentiation.
    bc_loss_fn : callable or None
        Optional boundary-condition loss.
    lambda_pde : float
        Weight for the PDE residual term.
    lambda_bc : float
        Weight for the boundary condition term.
    """

    def __init__(
        self,
        residual_fn: Callable,
        bc_loss_fn: Callable | None = None,
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
    ):
        super().__init__()
        self.residual_fn = residual_fn
        self.bc_loss_fn = bc_loss_fn
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : network output (could be field values or collocation predictions)
        target : contains coordinates or other data needed for residual evaluation
        """
        residual = self.residual_fn(pred, target)
        loss = self.lambda_pde * torch.mean(residual ** 2)

        if self.bc_loss_fn is not None:
            loss = loss + self.lambda_bc * self.bc_loss_fn(pred, target)

        return loss
