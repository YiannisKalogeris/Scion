"""
Multi-Layer Perceptron with configurable architecture.

A general-purpose feedforward network used as a building block
for surrogates, encoders, decoders, and operator networks.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected network with flexible depth, width, and activation.

    Parameters
    ----------
    layer_sizes : sequence of int
        Sizes of each layer including input and output.
        E.g. ``[3, 128, 128, 1]`` for 3 inputs → 2 hidden layers of 128 → 1 output.
    activation : str
        Activation function: ``"relu"``, ``"tanh"``, ``"gelu"``, ``"silu"``.
    dropout : float
        Dropout probability (0 = no dropout).
    batch_norm : bool
        Whether to apply batch normalisation between layers.
    output_activation : nn.Module | None
        Optional activation on the final layer (e.g. ``nn.Softplus()`` for
        positive outputs).
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(self._ACTIVATIONS)}")

        act_cls = self._ACTIVATIONS[activation]
        layers: list[nn.Module] = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # No activation/norm/dropout on the last layer
            if i < len(layer_sizes) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(act_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        sizes = [m.in_features for m in self.net if isinstance(m, nn.Linear)]
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        sizes.append(last.out_features)
        return f"MLP({sizes}, params={self.n_parameters:,})"
