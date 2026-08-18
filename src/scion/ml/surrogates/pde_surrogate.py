"""
PDE surrogate model.

Wraps a trained neural network to act as a drop-in replacement for
a full PDE solver. Takes parameters (material properties, BCs, geometry)
and returns approximate solution fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PDESurrogate:
    """Surrogate that replaces a PDE solve with a neural network forward pass.

    Parameters
    ----------
    model : nn.Module
        Trained neural network.
    input_scaler : callable or None
        ``(raw_input) -> normalised_input`` for preprocessing.
    output_scaler : callable or None
        ``(normalised_output) -> physical_output`` for postprocessing.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        input_scaler: object | None = None,
        output_scaler: object | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    @torch.no_grad()
    def predict(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the surrogate at given parameter values.

        Parameters
        ----------
        params : (n_samples, n_params) or (n_params,) array
            Input parameters (e.g. material properties, BCs).

        Returns
        -------
        fields : np.ndarray
            Predicted solution field(s).
        """
        x = np.atleast_2d(params).astype(np.float32)

        if self.input_scaler is not None:
            x = self.input_scaler.transform(x)

        x_t = torch.from_numpy(x).to(self.device)
        y_t = self.model(x_t)
        y = y_t.cpu().numpy()

        if self.output_scaler is not None:
            y = self.output_scaler.inverse_transform(y)

        return y.squeeze()

    def __call__(self, params: np.ndarray) -> np.ndarray:
        return self.predict(params)

    @classmethod
    def load(cls, model_cls: type, checkpoint: str | Path, **kwargs) -> PDESurrogate:
        """Load a surrogate from a saved checkpoint.

        Parameters
        ----------
        model_cls : type
            The nn.Module class to instantiate.
        checkpoint : path
            Path to the saved state dict.
        **kwargs
            Arguments passed to ``model_cls()``.
        """
        model = model_cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        return cls(model=model)
