"""
Per-run experiment records: epoch_metrics.csv and run_summary.json.

epoch_metrics.csv (one row per completed epoch, VALIDATION metrics):

    epoch, RMSE, MAE, Masked_MAPE, R2, Accuracy, learning_rate,
    is_best_epoch, best_val_RMSE_so_far, train_loss, elapsed_seconds,
    Accuracy_TGCN

The first seven columns are fixed in this order; extra columns are only
appended after them. Metrics are in original units (vehicles / 5 min),
see training.metrics.evaluate_metrics for the definitions.

learning_rate is the rate USED DURING the epoch (before the scheduler
step that follows the epoch's validation).

is_best_epoch marks the epoch with the lowest validation RMSE so far
(ties keep the earlier epoch, as the checkpoint rule does). The whole
file is rewritten atomically after every epoch, so the flag is always
correct and an interrupted run keeps every completed epoch.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
from pathlib import Path

from .metrics import METRIC_COLUMNS


REQUIRED_COLUMNS = ("epoch",) + METRIC_COLUMNS + ("learning_rate",)

EXTRA_COLUMNS = (
    "is_best_epoch",
    "best_val_RMSE_so_far",
    "train_loss",
    "elapsed_seconds",
    "Accuracy_TGCN",
)

COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS


def _write_atomic(path: Path, text: str) -> None:

    temporary = path.with_name(path.name + ".tmp")

    with open(temporary, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


class EpochMetricsLogger:
    """Writes epoch_metrics.csv for one run."""

    def __init__(self, path) -> None:

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.best_epoch: int | None = None
        self.best_rmse = math.inf

        self._flush()

    def log_epoch(
        self,
        epoch: int,
        metrics: dict,
        learning_rate: float,
        train_loss: float,
        elapsed_seconds: float,
    ) -> bool:
        """
        Record one completed epoch. Returns True if it is the new best.
        """

        if self.rows and epoch <= self.rows[-1]["epoch"]:
            raise ValueError("epochs must be logged in increasing order")

        missing = [name for name in METRIC_COLUMNS if name not in metrics]
        if missing:
            raise ValueError(f"missing metrics: {missing}")

        rmse = float(metrics["RMSE"])

        is_best = math.isfinite(rmse) and rmse < self.best_rmse

        if is_best:
            self.best_rmse = rmse
            self.best_epoch = int(epoch)

        row = {
            "epoch": int(epoch),
            **{name: float(metrics[name]) for name in METRIC_COLUMNS},
            "learning_rate": float(learning_rate),
            "train_loss": float(train_loss),
            "elapsed_seconds": float(elapsed_seconds),
            "Accuracy_TGCN": float(metrics.get("Accuracy_TGCN", math.nan)),
        }

        self.rows.append(row)
        self._flush()

        return is_best

    def best_row(self) -> dict | None:

        for row in self.rows:
            if row["epoch"] == self.best_epoch:
                return self._with_flags(row)

        return None

    def _with_flags(self, row: dict) -> dict:

        so_far = min(
            r["RMSE"] for r in self.rows if r["epoch"] <= row["epoch"]
        )

        return {
            **row,
            "is_best_epoch": row["epoch"] == self.best_epoch,
            "best_val_RMSE_so_far": so_far,
        }

    def _flush(self) -> None:

        lines = [",".join(COLUMNS)]

        for row in self.rows:
            full = self._with_flags(row)
            lines.append(",".join(_format(full[name]) for name in COLUMNS))

        _write_atomic(self.path, "\n".join(lines) + "\n")


def _format(value) -> str:

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else str(value)

    return repr(float(value))


def read_epoch_metrics(path) -> list[dict]:
    """Read an epoch_metrics.csv back (numbers as float, flags as bool)."""

    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out = []

    for row in rows:
        parsed = {}
        for key, value in row.items():
            if key == "epoch":
                parsed[key] = int(value)
            elif key == "is_best_epoch":
                parsed[key] = value == "True"
            else:
                parsed[key] = float(value)
        out.append(parsed)

    return out


def software_versions() -> dict:

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    for module in ("numpy", "scipy", "tensorflow", "h5py", "pandas", "lingam"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = None

    return versions


def write_run_summary(path, summary: dict) -> None:

    _write_atomic(Path(path), json.dumps(summary, indent=2, default=str) + "\n")
