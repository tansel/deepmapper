"""Plateau early-stopping, the pure decision logic (no torch).

Reproduces the Keras ``EarlyStopping(monitor, patience, min_delta,
restore_best_weights=True)`` behaviour that the original TF DeepMapper used and
that was lost in the PyTorch port: train until the monitored held-out metric
stops improving for ``patience`` epochs, then restore the best-epoch weights.
A ``min_epochs`` floor keeps the run out of the undertrained zone where
attribution is unreliable (the 12-epoch 88% artefact).

Kept torch-free and side-effect-free so it is unit-tested without a GPU, the
same split the project uses for ``accumulate``.
"""
from __future__ import annotations

from dataclasses import dataclass


def is_improvement(value: float, best: float, min_delta: float, mode: str = "min") -> bool:
    """True if ``value`` beats ``best`` by more than ``min_delta`` (pure)."""
    if mode == "min":
        return value < best - min_delta
    return value > best + min_delta


def should_stop(num_bad: int, patience: int, epoch: int, min_epochs: int) -> bool:
    """True if training has plateaued: past the ``min_epochs`` floor AND
    ``patience`` consecutive non-improving epochs have elapsed (pure)."""
    return (epoch + 1) >= min_epochs and num_bad >= patience


@dataclass
class EarlyStopper:
    """Track a monitored metric and decide when training has plateaued.

    Call :meth:`step` once per epoch with the held-out value. It records whether
    this epoch was a new best (``self.improved``, the caller saves weights then)
    and returns whether training should stop. ``mode='min'`` for a loss,
    ``'max'`` for an accuracy.
    """
    patience: int = 20
    min_delta: float = 1e-3
    min_epochs: int = 40
    mode: str = "min"

    best: float = float("inf")
    best_epoch: int = -1
    num_bad: int = 0
    improved: bool = False

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.best = float("inf") if self.mode == "min" else float("-inf")

    def step(self, epoch: int, value: float) -> bool:
        """Record ``value`` at ``epoch`` (0-based); return True to stop now.

        Before ``min_epochs`` (the warmup) we neither record a best epoch nor
        count patience, so the restored best-epoch model has always trained at
        least ``min_epochs``. The floor gates which epoch we keep, not only when
        we stop, on an easy task whose validation loss bottoms out very early
        this keeps attribution off a barely-trained model.
        """
        if (epoch + 1) < self.min_epochs:
            self.improved = False
            return False
        self.improved = is_improvement(value, self.best, self.min_delta, self.mode)
        if self.improved:
            self.best, self.best_epoch, self.num_bad = value, epoch, 0
        else:
            self.num_bad += 1
        return should_stop(self.num_bad, self.patience, epoch, self.min_epochs)
