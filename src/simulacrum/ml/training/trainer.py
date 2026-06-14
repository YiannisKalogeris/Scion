"""
Generic training loop for PyTorch models.

Handles epoch iteration, logging, checkpointing, and early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simulacrum.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainHistory:
    """Record of training metrics."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


class Trainer:
    """General-purpose PyTorch trainer.

    Parameters
    ----------
    model : nn.Module
        Network to train.
    optimizer : torch.optim.Optimizer | None
        Optimizer (default: Adam with lr=1e-3).
    scheduler : learning rate scheduler or None
    loss_fn : callable
        Loss function ``(pred, target) -> scalar``.
    device : str
        ``"cpu"``, ``"cuda"``, or ``"mps"``.
    checkpoint_dir : path or None
        Where to save model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        loss_fn: Callable | None = None,
        device: str = "cpu",
        checkpoint_dir: str | Path | None = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.loss_fn = loss_fn or nn.MSELoss()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.history = TrainHistory()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 100,
        early_stopping_patience: int | None = None,
        log_every: int = 10,
    ) -> TrainHistory:
        """Run the training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader or None
        epochs : int
        early_stopping_patience : int or None
            Stop if val loss doesn't improve for this many epochs.
        log_every : int
            Print metrics every N epochs.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # --- Train ---
            train_loss = self._train_epoch(train_loader)
            self.history.train_loss.append(train_loss)
            self.history.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            # --- Validate ---
            val_loss = None
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                self.history.val_loss.append(val_loss)

            # --- Scheduler step ---
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss or train_loss)
                else:
                    self.scheduler.step()

            # --- Logging ---
            if epoch % log_every == 0 or epoch == 1:
                msg = f"Epoch {epoch:4d} | train_loss: {train_loss:.6e}"
                if val_loss is not None:
                    msg += f" | val_loss: {val_loss:.6e}"
                logger.info(msg)

            # --- Early stopping ---
            if early_stopping_patience and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint("best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _unpack_batch(self, batch: tuple | list) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch[0], batch[1]
        return inputs.to(self.device), targets.to(self.device)

    def _save_checkpoint(self, name: str) -> None:
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.checkpoint_dir / name
            torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
