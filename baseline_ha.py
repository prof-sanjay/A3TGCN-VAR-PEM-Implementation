"""
Historical Average (HA) baseline.

Prediction for sensor i at time t:

    mean of the OBSERVED TRAINING values of sensor i on the same day of
    the week and the same 5-minute slot of the day as t

(falling back to the sensor's mean for that slot over all days, then to
the sensor's overall training mean, if a cell has no observation). Only
training rows are used; there is nothing to tune.

The test score uses EXACTLY the test targets of a reference A3T-GCN run
(--reference: a run folder containing test_predictions.npz), so HA is
compared on identical samples and sensors.

Validation metrics (no model selection happens) are reported on every
validation row from row seq_len onward whose sensors are all observed,
i.e. the rows that can be targets; this matches main.py's validation
targets except for windows whose filled inputs were still incomplete.

Outputs (default results/HA/):
    run_summary.json, test_predictions.npz

Run from the implementation directory (traffic-gpu or traffic-causal):

    python baseline_ha.py --reference results/kaggle/output_E1_final
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from config.config import ExperimentConfig
from data.loader import load_gba_dataset
from data.sensor_filter import cached_duplicate_filter, select_sensors
from data.splitting import chronological_split
from training.metrics import evaluate_metrics
from training.run_logger import software_versions, write_run_summary


IMPLEMENTATION_DIR = Path(__file__).resolve().parent

MAPE_MIN_FLOW = 10.0
SLOTS_PER_DAY = 288


def day_and_slot(timestamps_ns):
    """Day of week (0 = Monday) and 5-minute slot of each timestamp."""

    times = np.asarray(timestamps_ns, dtype=np.int64).astype("datetime64[ns]")
    days = times.astype("datetime64[D]")
    slot = ((times - days).astype("timedelta64[m]").astype(np.int64)) // 5
    # 1970-01-01 was a Thursday (weekday 3).
    weekday = (days.astype(np.int64) + 3) % 7
    return weekday.astype(np.int64), slot.astype(np.int64)


def fit_ha(train, weekday, slot):
    """Mean of observed values per (weekday, slot, sensor), with fallbacks."""

    num_nodes = train.shape[1]
    observed = np.isfinite(train)
    values = np.where(observed, train, 0.0).astype(np.float64)

    cell = weekday * SLOTS_PER_DAY + slot

    total = np.zeros((7 * SLOTS_PER_DAY, num_nodes))
    count = np.zeros((7 * SLOTS_PER_DAY, num_nodes))
    np.add.at(total, cell, values)
    np.add.at(count, cell, observed)

    table = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)

    by_slot_total = total.reshape(7, SLOTS_PER_DAY, num_nodes).sum(axis=0)
    by_slot_count = count.reshape(7, SLOTS_PER_DAY, num_nodes).sum(axis=0)
    slot_mean = np.divide(by_slot_total, by_slot_count,
                          out=np.full_like(by_slot_total, np.nan), where=by_slot_count > 0)
    sensor_mean = total.sum(axis=0) / np.maximum(count.sum(axis=0), 1)

    fallback_slot = np.tile(slot_mean, (7, 1))
    fallbacks = {
        "cells_without_observation": int(np.isnan(table).sum()),
    }
    table = np.where(np.isnan(table), fallback_slot, table)
    fallbacks["cells_using_sensor_mean"] = int(np.isnan(table).sum())
    table = np.where(np.isnan(table), sensor_mean[None, :], table)

    return table, fallbacks


def main():

    parser = argparse.ArgumentParser(description="Historical Average baseline")
    parser.add_argument("--reference", required=True,
                        help="Run folder with test_predictions.npz (same test targets).")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "results" / "HA"))
    args = parser.parse_args()

    config = ExperimentConfig()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(Path(args.reference) / "test_predictions.npz") as ref:
        target_rows = ref["target_rows"]
        ref_sensors = ref["sensors"]
        ref_targets = ref["targets"].astype(np.float64)
        sensor_ids = ref["sensor_ids"]
        timestamps_ns = ref["timestamps_ns"]
        input_observed = ref["input_fully_observed"]

    traffic, adjacency = load_gba_dataset(
        config.traffic_path, config.adjacency_path,
        num_timesteps=config.num_timesteps, num_nodes=config.num_nodes,
    )

    train, val, test = chronological_split(traffic, config.train_rate, config.val_rate)
    train_end, val_end = len(train), len(train) + len(val)

    sensors, _ = cached_duplicate_filter(
        train, IMPLEMENTATION_DIR / "outputs" / "sensor_filter"
    )

    if not np.array_equal(sensors, ref_sensors):
        raise ValueError("Reference run used a different sensor set.")

    traffic, _ = select_sensors(traffic, adjacency, sensors)
    del train, val, test, adjacency

    with h5py.File(config.traffic_path, "r") as f:
        all_timestamps = np.asarray(f["timestamps"][: config.num_timesteps]).astype(np.int64)

    weekday, slot = day_and_slot(all_timestamps)

    # ---- fit on TRAINING rows only
    table, fallbacks = fit_ha(traffic[:train_end], weekday[:train_end], slot[:train_end])

    def predict(rows):
        return table[weekday[rows] * SLOTS_PER_DAY + slot[rows]]

    # ---- test: the reference run's exact targets
    if not np.all(target_rows >= val_end):
        raise ValueError("Reference target rows are not all in the test split.")

    test_targets = traffic[target_rows].astype(np.float64)

    if not np.allclose(test_targets, ref_targets, rtol=0, atol=1e-3):
        raise ValueError("Reference targets do not match the data at target_rows.")

    test_pred = predict(target_rows)
    test_metrics = evaluate_metrics(test_targets, test_pred, MAPE_MIN_FLOW)
    test_metrics["mape_min_flow"] = MAPE_MIN_FLOW

    # ---- validation: every fully observed potential target row
    val_rows = np.arange(train_end + config.seq_len, val_end)
    val_rows = val_rows[np.all(np.isfinite(traffic[val_rows]), axis=1)]
    val_metrics = evaluate_metrics(traffic[val_rows].astype(np.float64),
                                   predict(val_rows), MAPE_MIN_FLOW)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        targets=test_targets.astype(np.float32),
        predictions=test_pred.astype(np.float32),
        target_rows=target_rows,
        timestamps_ns=timestamps_ns,
        sensors=sensors,
        sensor_ids=sensor_ids,
        input_fully_observed=input_observed,
    )

    write_run_summary(out_dir / "run_summary.json", {
        "experiment": "HA",
        "seed": 0,
        "method": "historical average per (weekday, 5-minute slot, sensor), "
                  "observed training values only",
        "smoke_windows": None,
        "best_epoch": None,
        "best_validation_metrics": val_metrics,
        "validation_rows": int(len(val_rows)),
        "test_metrics": test_metrics,
        "test_targets_from": str(args.reference),
        "test_windows": int(len(target_rows)),
        "fallbacks": fallbacks,
        "train_end": int(train_end),
        "software": software_versions(),
    })

    print(json.dumps({"validation": val_metrics, "test": test_metrics,
                      "fallbacks": fallbacks}, indent=2))
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
