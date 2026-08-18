"""
Graph Neural Network for mesh-based data.

Operates on the mesh connectivity graph: nodes are mesh nodes,
edges are element connections. Natural for learning on unstructured meshes.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    """Single message-passing layer.

    Updates each node by aggregating features from its neighbours:
        h_i' = φ(h_i, AGG_{j ∈ N(i)} ψ(h_i, h_j, e_ij))

    Parameters
    ----------
    node_dim : int
        Node feature dimension.
    edge_dim : int
        Edge feature dimension (0 if no edge features).
    hidden_dim : int
        Hidden dimension for the message and update MLPs.
    """

    def __init__(self, node_dim: int, edge_dim: int = 0, hidden_dim: int = 128):
        super().__init__()
        message_input = 2 * node_dim + edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (n_nodes, node_dim)
        edge_index : (2, n_edges) source → target
        edge_attr : (n_edges, edge_dim) or None
        """
        src, dst = edge_index[0], edge_index[1]

        # Build messages
        msg_input = torch.cat([x[src], x[dst]], dim=-1)
        if edge_attr is not None:
            msg_input = torch.cat([msg_input, edge_attr], dim=-1)
        messages = self.message_mlp(msg_input)

        # Aggregate (sum) messages at destination nodes
        agg = torch.zeros_like(x[:, :messages.shape[-1]])
        agg = agg.new_zeros(x.shape[0], messages.shape[-1])
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Update node features
        x_updated = self.update_mlp(torch.cat([x, agg], dim=-1))
        return x + x_updated  # residual connection


class MeshGNN(nn.Module):
    """Graph neural network for mesh-based PDE learning.

    Parameters
    ----------
    input_dim : int
        Input features per node (e.g. coordinates + boundary flags).
    output_dim : int
        Output features per node (e.g. solution components).
    hidden_dim : int
        Hidden layer width.
    n_layers : int
        Number of message-passing layers.
    edge_dim : int
        Edge feature dimension (e.g. edge lengths, normals).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 6,
        edge_dim: int = 0,
    ):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr)
        return self.decoder(h)
