"""
Tests for the per-run records (Phase 1):

    1. evaluate_metrics on known values; NaN (not 0) for empty input
       or an empty MAPE mask; non-finite input rejected.
    2. epoch_metrics.csv: exact column order, one row per epoch,
       is_best_epoch moves to a later improving epoch, exactly one best,
       ties keep the earlier epoch.
    3. An interrupted run keeps every completed epoch on disk.
    4. learning_rate is the rate used DURING the epoch (before the
       scheduler step), with the project's ValidationLRScheduler.
    5. summarize_runs.py: consolidated column order, smoke runs kept
       out of multi_seed_summary, mean / std over seeds.

Run from the implementation directory:

    python -m tests.test_run_logger
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
import math
import subprocess
import tempfile

import numpy as np

from training.metrics import evaluate_metrics
from training.run_logger import (
    COLUMNS,
    EpochMetricsLogger,
    read_epoch_metrics,
    write_run_summary,
)
from training.scheduler import ValidationLRScheduler


IMPLEMENTATION_DIR = Path(__file__).resolve().parent.parent

REQUIRED = ["epoch", "RMSE", "MAE", "Masked_MAPE", "R2", "Accuracy", "learning_rate"]


def metrics_for(rmse_value):
    return {"RMSE": rmse_value, "MAE": rmse_value / 2, "Masked_MAPE": 10.0,
            "R2": 0.9, "Accuracy": 0.5, "Accuracy_TGCN": 0.8}


def header(path):
    with open(path, encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def test_metrics():

    y = np.array([[10.0, 20.0], [0.0, 40.0]])
    p = np.array([[11.0, 18.0], [0.0, 40.0]])

    m = evaluate_metrics(y, p, mape_min_flow=10.0)

    ss_res = 1 + 4
    ss_tot = np.sum((y - y.mean()) ** 2)

    assert math.isclose(m["RMSE"], math.sqrt(5 / 4))
    assert math.isclose(m["MAE"], 3 / 4)
    assert math.isclose(m["Masked_MAPE"], 100 * (0.1 + 0.1 + 0.0) / 3)   # y=0 masked out
    assert math.isclose(m["R2"], 1 - ss_res / ss_tot)
    assert m["Accuracy"] == 1.0                                          # all within 10 %
    assert math.isclose(m["Accuracy_TGCN"], 1 - math.sqrt(5) / np.linalg.norm(y))
    assert list(m)[:5] == ["RMSE", "MAE", "Masked_MAPE", "R2", "Accuracy"]

    # 12 vs 10 is 20 % off -> not accurate; zero target only exact.
    m2 = evaluate_metrics(np.array([10.0, 0.0]), np.array([12.0, 0.5]))
    assert m2["Accuracy"] == 0.0

    # MAPE mask empty -> NaN, not 0.
    assert math.isnan(evaluate_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))["Masked_MAPE"])

    # Empty input -> NaN everywhere.
    empty = evaluate_metrics(np.empty((0, 3)), np.empty((0, 3)))
    assert all(math.isnan(v) for v in empty.values())

    for bad in (np.array([1.0, np.nan]), np.array([1.0, np.inf])):
        try:
            evaluate_metrics(np.array([1.0, 2.0]), bad)
        except ValueError:
            continue
        raise AssertionError("non-finite predictions accepted")

    print("[PASS] 1. metric values, NaN for empty input / mask, non-finite rejected")


def test_logger_rows_and_best(tmp):

    path = Path(tmp) / "run" / "epoch_metrics.csv"
    logger = EpochMetricsLogger(path)

    assert header(path) == list(COLUMNS)
    assert header(path)[:7] == REQUIRED
    assert header(path)[7] == "is_best_epoch"
    assert read_epoch_metrics(path) == []

    sequence = [30.0, 28.0, 29.0, 27.5, 27.5, 31.0]
    expected_best = [1, 2, 2, 4, 4, 4]

    for epoch, value in enumerate(sequence, start=1):
        is_best = logger.log_epoch(epoch, metrics_for(value), 0.001, 1.0, float(epoch))
        rows = read_epoch_metrics(path)

        assert len(rows) == epoch, "a row must be written after every epoch"
        assert is_best == (expected_best[epoch - 1] == epoch)
        assert [r["epoch"] for r in rows if r["is_best_epoch"]] == [expected_best[epoch - 1]]
        assert rows[-1]["best_val_RMSE_so_far"] == min(sequence[:epoch])

    assert logger.best_row()["epoch"] == 4          # tie at epoch 5 keeps epoch 4

    try:
        logger.log_epoch(3, metrics_for(1.0), 0.001, 1.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-order epoch accepted")

    print("[PASS] 2. column order, one row per epoch, best flag moves, one best, ties keep earlier")


def test_interrupted_run(tmp):

    path = Path(tmp) / "interrupted" / "epoch_metrics.csv"

    def run_until_crash():
        logger = EpochMetricsLogger(path)
        for epoch, value in enumerate([5.0, 4.0, 4.5], start=1):
            logger.log_epoch(epoch, metrics_for(value), 0.001, 1.0, 1.0)
        raise KeyboardInterrupt

    try:
        run_until_crash()
    except KeyboardInterrupt:
        pass

    rows = read_epoch_metrics(path)
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    assert [r["is_best_epoch"] for r in rows] == [False, True, False]
    assert not path.with_name(path.name + ".tmp").exists()

    print("[PASS] 3. interrupted run keeps all completed epochs")


def test_learning_rate_convention(tmp):

    path = Path(tmp) / "lr" / "epoch_metrics.csv"
    logger = EpochMetricsLogger(path)

    scheduler = ValidationLRScheduler(initial_lr=0.001, patience=2, factor=0.5, min_lr=1e-5)

    used = []
    for epoch, value in enumerate([10.0, 11.0, 12.0, 13.0, 9.0], start=1):
        lr_used = scheduler.current_lr          # same order as main.py
        scheduler.step(value)
        logger.log_epoch(epoch, metrics_for(value), lr_used, 1.0, 1.0)
        used.append(lr_used)

    rows = read_epoch_metrics(path)
    # No improvement in epochs 2 and 3 -> halved AFTER epoch 3, so the
    # halved rate is the one used during epoch 4.
    assert [r["learning_rate"] for r in rows] == [0.001, 0.001, 0.001, 0.0005, 0.0005]
    assert used == [r["learning_rate"] for r in rows]

    print("[PASS] 4. learning_rate = rate used during the epoch")


def make_run(root, experiment, seed, test_rmse, smoke=None, selection_only=False):

    run_dir = Path(root) / experiment / f"seed_{seed}"
    logger = EpochMetricsLogger(run_dir / "epoch_metrics.csv")
    for epoch, value in enumerate([test_rmse + 2, test_rmse + 1], start=1):
        logger.log_epoch(epoch, metrics_for(value), 0.001, 1.0, 1.0)

    test = "not evaluated (selection run)" if selection_only else {
        "RMSE": test_rmse, "MAE": test_rmse / 2, "Masked_MAPE": 12.0,
        "R2": 0.95, "Accuracy": 0.6, "Accuracy_TGCN": 0.85,
    }

    write_run_summary(run_dir / "run_summary.json", {
        "experiment": experiment, "seed": seed, "smoke_windows": smoke,
        "best_epoch": logger.best_row()["epoch"],
        "best_validation_metrics": logger.best_row(),
        "test_metrics": test, "epochs_completed": 2, "training_seconds": 1.0,
        "checkpoint_check": {"consistent": True},
    })


def test_summarize_runs(tmp):

    root = Path(tmp) / "results"
    make_run(root, "E0", 42, 30.0)
    make_run(root, "E0", 7, 32.0)
    make_run(root, "E1", 42, 28.0)
    make_run(root, "E1", 7, 99.0, smoke=50)            # must not count
    make_run(root, "E2", 42, 1.0, selection_only=True)  # must not count

    out = Path(tmp) / "summary"
    subprocess.run(
        [sys.executable, str(IMPLEMENTATION_DIR / "summarize_runs.py"),
         "--results", str(root), "--out", str(out)],
        check=True, capture_output=True, text=True, cwd=IMPLEMENTATION_DIR,
    )

    all_header = header(out / "all_epoch_metrics.csv")
    assert all_header[:4] == ["experiment", "seed", "horizon", "run"]
    assert all_header[4:11] == REQUIRED

    with open(out / "all_epoch_metrics.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 10     # 5 runs x 2 epochs

    with open(out / "multi_seed_summary.csv", encoding="utf-8") as f:
        multi = {r["experiment"]: r for r in csv.DictReader(f)}

    assert set(multi) == {"E0", "E1"}
    assert multi["E0"]["n_seeds"] == "2"
    assert math.isclose(float(multi["E0"]["test_RMSE_mean"]), 31.0)
    assert math.isclose(float(multi["E0"]["test_RMSE_std"]), float(np.std([30, 32], ddof=1)))
    assert multi["E1"]["n_seeds"] == "1" and math.isnan(float(multi["E1"]["test_RMSE_std"]))

    print("[PASS] 5. summarize_runs: column order, smoke / selection runs excluded, mean and std")


def main():

    print("=" * 70)
    print("RUN LOGGER AND METRICS TESTS")
    print("=" * 70)

    test_metrics()

    with tempfile.TemporaryDirectory() as tmp:
        test_logger_rows_and_best(tmp)
        test_interrupted_run(tmp)
        test_learning_rate_convention(tmp)
        test_summarize_runs(tmp)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
