"""
When to stop training, and which checkpoints to keep.

TrainingStopper
    - early stopping on validation RMSE (optional; off by default so
      existing comparisons are unchanged): stop after `patience`
      epochs without improvement;
    - time budget (optional): do not START an epoch that is expected to
      end after `max_seconds` (expected length = longest epoch so far).
      Used on Kaggle, where a session that exceeds its limit loses
      every output file.

MilestoneTracker
    Best validation RMSE among the epochs up to each milestone epoch
    (e.g. 20 in a 30-epoch run). A separate checkpoint per milestone
    lets one long run also answer "what if training had stopped at
    epoch 20?". The first `milestone` epochs of a longer run follow the
    same learning-rate schedule as a run of exactly `milestone` epochs.
"""

from __future__ import annotations

import math


class TrainingStopper:

    def __init__(
        self,
        early_stopping_patience: int | None = None,
        max_seconds: float | None = None,
    ) -> None:

        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")

        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive")

        self.patience = early_stopping_patience
        self.max_seconds = max_seconds
        self.best = math.inf
        self.epochs_without_improvement = 0
        self.longest_epoch = 0.0
        self.reason: str | None = None

    def after_epoch(self, val_rmse: float, epoch_seconds: float) -> None:

        self.longest_epoch = max(self.longest_epoch, float(epoch_seconds))

        if val_rmse < self.best:
            self.best = float(val_rmse)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def should_stop(self, elapsed_seconds: float) -> bool:
        """Call before starting the next epoch."""

        if (
            self.patience is not None
            and self.epochs_without_improvement >= self.patience
        ):
            self.reason = (
                f"early stopping: {self.epochs_without_improvement} epochs "
                "without validation improvement"
            )
            return True

        if (
            self.max_seconds is not None
            and elapsed_seconds + self.longest_epoch > self.max_seconds
        ):
            self.reason = (
                f"time budget: {elapsed_seconds:.0f}s used, next epoch "
                f"~{self.longest_epoch:.0f}s, limit {self.max_seconds:.0f}s"
            )
            return True

        return False


class MilestoneTracker:

    def __init__(self, milestones) -> None:

        self.milestones = sorted({int(m) for m in milestones or []})

        if any(m < 1 for m in self.milestones):
            raise ValueError("milestones must be positive epochs")

        self.best = {m: math.inf for m in self.milestones}
        self.best_epoch = {m: None for m in self.milestones}

    def update(self, epoch: int, val_rmse: float) -> list[int]:
        """
        Record an epoch (1-based). Returns the milestones whose best
        checkpoint must be saved now.
        """

        improved = []

        for m in self.milestones:
            if epoch <= m and val_rmse < self.best[m]:
                self.best[m] = float(val_rmse)
                self.best_epoch[m] = int(epoch)
                improved.append(m)

        return improved
