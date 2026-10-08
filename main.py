"""
E0 / E1 / E2 ablation runner for A3T-GCN on LargeST-GBA 2021.

    E0 : A3T-GCN on the road-network adjacency.
    E1 : A3T-GCN on the topology-aware VAR graph.
    E2 : A3T-GCN on the VAR + PEM probabilistic causal graph.

Only the graph differs between E0, E1 and E2. Everything else
(data, split, preprocessing, missing-value policy, model, training,
validation, checkpointing, metrics) is shared via ExperimentConfig.

The E1 / E2 graphs are built OFFLINE from the TRAINING split only
(VAR: run_var.py, PEM: run_local_pem.py) and passed in as an .npz
graph artifact with the keys:

    adjacency   float array (num_nodes, num_nodes), G[target, source]
    experiment  "E1" or "E2"
    train_end   number of training timesteps used to build the graph
    num_nodes   number of sensors
    sensors     original GBA indices of the sensors (after filtering)

VAR and PEM are therefore never executed inside this process, and
E0 cannot use them by construction.

Data handling (shared by E0 / E1 / E2, set in ExperimentConfig):

    sensor_filter = "duplicates":
        sensors with copied series (identified on TRAIN only) are
        excluded (data/sensor_filter.py).

    imputation = "gap_bands":
        gap-length-band filling of A3T-GCN INPUTS
        (data/imputation.py; look-ahead only inside TRAIN).
        A window is used only when its inputs are finite after
        filling AND its target timestep was originally observed for
        every sensor, so every scored target is a real observation.

    imputation = "none":
        no filling; a window is used only when every value in its
        input and target is finite.

Run from the repository root:

    python -m implementation.main --experiment E0
    python -m implementation.main --experiment E2 --graph <graph.npz>
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from .config.config import ExperimentConfig

from .data.loader import load_gba_dataset
from .data.splitting import chronological_split
from .data.preprocessing import (
    fit_max_scaler,
    normalize_data,
)
from .data.windows import (
    finite_row_mask,
    valid_window_starts_from_rows,
    iterate_window_batches,
    missing_value_report,
)
from .data.sensor_filter import (
    cached_duplicate_filter,
    select_sensors,
)
from .data.imputation import (
    fit_neighbour_models,
    fill_split,
    gap_length_counts,
)

from .causal.graph_builders import to_model_orientation

from .models.a3tgcn import A3TGCNModel

from .training.trainer import A3TGCNTrainer
from .training.scheduler import ValidationLRScheduler
from .training.metrics import (
    rmse,
    mae,
    r2_score,
    accuracy,
    masked_mape,
)

# Targets below this flow are excluded from MAPE (vehicles / 5 min).
MAPE_MIN_FLOW = 10.0


# ============================================================
# Reproducibility
# ============================================================

def set_seeds(seed):

    np.random.seed(seed)
    tf.set_random_seed(seed)


# ============================================================
# Graph selection (the only E0 / E1 / E2 difference)
# ============================================================

def graph_statistics(adjacency):
    """
    Summary statistics of the graph given to A3T-GCN.
    """

    off_diagonal = adjacency.copy()
    np.fill_diagonal(off_diagonal, 0.0)

    edges = off_diagonal != 0
    weights = off_diagonal[edges]

    num_nodes = adjacency.shape[0]

    return {
        "num_nodes": int(num_nodes),
        "num_edges_off_diagonal": int(edges.sum()),
        "density_off_diagonal": float(
            edges.sum() / (num_nodes * (num_nodes - 1))
        ),
        "num_self_loops": int(np.count_nonzero(np.diag(adjacency))),
        "num_reciprocal_pairs": int((edges & edges.T).sum() // 2),
        "isolated_nodes": int(
            np.sum(~edges.any(axis=0) & ~edges.any(axis=1))
        ),
        "is_symmetric": bool(np.array_equal(adjacency, adjacency.T)),
        "weight_min": float(weights.min()) if weights.size else 0.0,
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "weight_max": float(weights.max()) if weights.size else 0.0,
    }


def load_graph(config, road_adjacency, train_end, sensors):
    """
    Return the adjacency matrix used by A3T-GCN.

    E0:
        Road-network adjacency, unchanged.

    E1 / E2:
        Offline graph artifact. Its metadata must show that it was
        built from exactly this training split and sensor set.
    """

    if config.experiment == "E0":

        return np.asarray(road_adjacency, dtype=np.float32), {
            "source": "road_adjacency",
            "path": config.adjacency_path,
        }

    with np.load(config.graph_path, allow_pickle=False) as artifact:

        adjacency = np.asarray(
            artifact["adjacency"],
            dtype=np.float32,
        )

        graph_experiment = str(artifact["experiment"])
        graph_train_end = int(artifact["train_end"])
        graph_num_nodes = int(artifact["num_nodes"])
        graph_sensors = np.asarray(artifact["sensors"], dtype=np.int64)

    if not np.array_equal(graph_sensors, sensors):
        raise ValueError(
            "Graph artifact was built for a different sensor set "
            f"({len(graph_sensors)} sensors) than this run "
            f"({len(sensors)} sensors)."
        )

    if graph_experiment != config.experiment:
        raise ValueError(
            f"Graph artifact was built for {graph_experiment}, "
            f"but the run is {config.experiment}."
        )

    if graph_train_end != train_end:
        raise ValueError(
            "Graph artifact was not built from this training split: "
            f"train_end {graph_train_end} != {train_end}."
        )

    if adjacency.shape != road_adjacency.shape or (
        graph_num_nodes != road_adjacency.shape[0]
    ):
        raise ValueError(
            f"Graph shape {adjacency.shape} does not match "
            f"{road_adjacency.shape}."
        )

    if not np.all(np.isfinite(adjacency)):
        raise ValueError("Graph artifact contains non-finite values.")

    return adjacency, {
        "source": f"{config.experiment}_artifact",
        "path": str(config.graph_path),
        "train_end": graph_train_end,
    }


# ============================================================
# Prediction
# ============================================================

def predict_dataset(
    trainer,
    model,
    data,
    starts,
    config,
):

    predictions = []
    targets = []

    for X_batch, y_batch in iterate_window_batches(
        data,
        starts,
        config.seq_len,
        config.pre_len,
        config.batch_size,
    ):

        feed_dict = {
            model.inputs: X_batch,
            model.labels: y_batch,
        }

        pred = trainer.session.run(
            model.y_pred,
            feed_dict=feed_dict,
        )

        predictions.append(pred)
        targets.append(
            y_batch.reshape(
                -1,
                model.num_nodes,
            )
        )

    if not predictions:

        return (
            np.empty(
                (0, model.num_nodes),
                dtype=np.float32,
            ),
            np.empty(
                (0, model.num_nodes),
                dtype=np.float32,
            ),
        )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    targets = np.concatenate(
        targets,
        axis=0,
    )

    return targets, predictions


# ============================================================
# Detailed outputs
# ============================================================

def _error_summary(targets, predictions):

    return {
        "windows": int(len(targets)),
        "rmse": float(rmse(targets, predictions)) if len(targets) else None,
        "mae": float(mae(targets, predictions)) if len(targets) else None,
        "masked_mape": (
            masked_mape(targets, predictions, MAPE_MIN_FLOW)
            if len(targets) else None
        ),
    }


def save_test_outputs(
    run_dir,
    results,
    config,
    sensors,
    test_starts,
    val_end,
    test_row_observed,
    targets,
    predictions,
):
    """
    Save everything needed to compare runs on identical test targets:

        test_predictions.npz   targets / predictions (vehicles), target
                               timestamps and rows, sensors, and whether
                               the input window was fully observed
        per_sensor_metrics.csv MAE / RMSE / masked MAPE per sensor
        time_of_day_metrics.csv MAE / RMSE per 5-minute slot
        history.csv            per-epoch training / validation curve

    and add the fully-observed vs filled-input breakdown to results.
    """

    num_nodes = targets.shape[1]
    pre_len = config.pre_len

    # Rows of the full year for every target row (window-major,
    # then forecast step, as produced by predict_dataset).
    target_rows = (
        val_end
        + np.repeat(test_starts, pre_len)
        + config.seq_len
        + np.tile(np.arange(pre_len), len(test_starts))
    ).astype(np.int64)

    with h5py.File(config.traffic_path, "r") as f:
        timestamps = np.asarray(f["timestamps"][:])[target_rows]
        sensor_ids = np.asarray(f["sensor_ids"][:])[sensors]

    # Was every INPUT value of the window originally observed?
    # (False = the window used gap-band filled inputs.)
    input_observed = np.array([
        bool(np.all(test_row_observed[s : s + config.seq_len]))
        for s in test_starts
    ])
    input_observed = np.repeat(input_observed, pre_len)

    np.savez_compressed(
        run_dir / "test_predictions.npz",
        targets=targets.astype(np.float32),
        predictions=predictions.astype(np.float32),
        target_rows=target_rows,
        timestamps_ns=timestamps.astype(np.int64),
        sensors=np.asarray(sensors, dtype=np.int64),
        sensor_ids=sensor_ids.astype(np.int64),
        input_fully_observed=input_observed,
    )

    # Per-sensor metrics
    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    valid_mape = np.abs(targets) >= MAPE_MIN_FLOW

    with np.errstate(invalid="ignore", divide="ignore"):
        sensor_mape = 100.0 * np.nanmean(
            np.where(valid_mape, np.abs(errors) / np.abs(targets), np.nan),
            axis=0,
        )

    with open(run_dir / "per_sensor_metrics.csv", "w", encoding="utf-8") as f:
        f.write("sensor_index,sensor_id,mae,rmse,masked_mape,mean_flow\n")
        for j in range(num_nodes):
            f.write(
                f"{int(sensors[j])},{int(sensor_ids[j])},"
                f"{np.mean(np.abs(errors[:, j])):.6f},"
                f"{np.sqrt(np.mean(errors[:, j] ** 2)):.6f},"
                f"{sensor_mape[j]:.6f},"
                f"{np.mean(targets[:, j]):.6f}\n"
            )

    # Time-of-day metrics (5-minute slots, local time of the data)
    times = timestamps.astype(np.int64).astype("datetime64[ns]")

    minutes = (
        (times - times.astype("datetime64[D]"))
        .astype("timedelta64[m]").astype(np.int64)
    )
    slots = minutes // 5

    with open(run_dir / "time_of_day_metrics.csv", "w", encoding="utf-8") as f:
        f.write("slot,time,count,mae,rmse\n")
        for slot in range(288):
            rows = slots == slot
            if not np.any(rows):
                continue
            f.write(
                f"{slot},{slot * 5 // 60:02d}:{slot * 5 % 60:02d},"
                f"{int(rows.sum())},"
                f"{np.mean(np.abs(errors[rows])):.6f},"
                f"{np.sqrt(np.mean(errors[rows] ** 2)):.6f}\n"
            )

    # Training curve
    with open(run_dir / "history.csv", "w", encoding="utf-8") as f:
        keys = list(results["history"][0].keys())
        f.write(",".join(keys) + "\n")
        for row in results["history"]:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    results["test_breakdown"] = {
        "inputs_fully_observed": _error_summary(
            targets[input_observed], predictions[input_observed]
        ),
        "inputs_partly_filled": _error_summary(
            targets[~input_observed], predictions[~input_observed]
        ),
    }

    results["test_outputs"] = [
        "test_predictions.npz",
        "per_sensor_metrics.csv",
        "time_of_day_metrics.csv",
        "history.csv",
    ]


# ============================================================
# Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="E0 / E1 / E2 A3T-GCN ablation on LargeST-GBA."
    )

    parser.add_argument(
        "--experiment",
        choices=["E0", "E1", "E2"],
        default="E0",
    )

    parser.add_argument(
        "--graph",
        default=None,
        help="Graph artifact (.npz) for E1 / E2.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training_epoch (smoke tests only).",
    )

    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Model selection run: train + validate only, the test "
             "set is NOT evaluated.",
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Output folder under results/ (default: experiment).",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = ExperimentConfig(
        experiment=args.experiment,
        graph_path=args.graph,
    )

    if args.epochs is not None:
        config.training_epoch = args.epochs

    config.validate()

    run_dir = Path(config.results_dir) / (
        args.run_name or config.experiment
    )

    checkpoint_path = str(run_dir / "checkpoints" / "best_model")

    os.makedirs(run_dir / "checkpoints", exist_ok=True)

    set_seeds(config.seed)

    results = {
        "experiment": config.experiment,
        "run_dir": str(run_dir),
        "config": asdict(config),
        "epochs_overridden": args.epochs is not None,
        "selection_only": args.selection_only,
    }

    print("=" * 70)
    print(f"{config.experiment}: A3T-GCN ABLATION RUN")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data (float32, NaNs preserved)
    # --------------------------------------------------------

    print("\nLoading traffic data...")

    traffic, road_adjacency = load_gba_dataset(
        config.traffic_path,
        config.adjacency_path,
        num_timesteps=config.num_timesteps,
        num_nodes=config.num_nodes,
    )

    print(f"Traffic shape   : {traffic.shape} {traffic.dtype}")
    print(f"Adjacency shape : {road_adjacency.shape}")

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train_data, val_data, test_data = chronological_split(
        traffic,
        config.train_rate,
        config.val_rate,
    )

    train_end = len(train_data)
    val_end = train_end + len(val_data)

    results["split"] = {
        "train_rows": [0, train_end],
        "val_rows": [train_end, val_end],
        "test_rows": [val_end, len(traffic)],
    }

    print(
        f"\nSplit rows: train [0, {train_end}), "
        f"val [{train_end}, {val_end}), "
        f"test [{val_end}, {len(traffic)})"
    )

    # --------------------------------------------------------
    # Sensor filter (decision D9), defined on TRAIN rows only.
    # Values are not modified; sensors are only selected.
    # --------------------------------------------------------

    if config.sensor_filter == "duplicates":

        sensors, filter_report = cached_duplicate_filter(
            train_data,
            Path(config.results_dir).parent / "outputs" / "sensor_filter",
        )

        del train_data, val_data, test_data

        traffic, road_adjacency = select_sensors(
            traffic,
            road_adjacency,
            sensors,
        )

        train_data, val_data, test_data = chronological_split(
            traffic,
            config.train_rate,
            config.val_rate,
        )

    else:

        sensors = np.arange(traffic.shape[1], dtype=np.int64)
        filter_report = {"rule": "none"}

    num_nodes = traffic.shape[1]

    results["sensor_filter"] = filter_report
    results["sensors"] = sensors.tolist()

    print(
        f"Sensor filter ({config.sensor_filter}): "
        f"{num_nodes} sensors used"
    )

    # --------------------------------------------------------
    # Missing-value statistics (raw data, nothing modified)
    # --------------------------------------------------------

    results["missing_values"] = {
        name: missing_value_report(
            split,
            config.seq_len,
            config.pre_len,
        )
        for name, split in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]
    }

    for name, report in results["missing_values"].items():
        print(
            f"{name:5s}: missing {report['missing_values']} "
            f"({report['missing_percentage']:.4f}%), "
            f"valid windows {report['valid_windows']} / "
            f"{report['total_windows']}"
        )

    # --------------------------------------------------------
    # Graph (the only difference between E0 / E1 / E2)
    # --------------------------------------------------------

    adjacency, graph_info = load_graph(
        config,
        road_adjacency,
        train_end,
        sensors,
    )

    graph_info["statistics"] = graph_statistics(adjacency)
    graph_info["orientation"] = (
        "stored as G[target, source]; transposed once for A3T-GCN "
        "(decision D2, identical for E0/E1/E2)"
    )
    results["graph"] = graph_info

    print(f"\nGraph source : {graph_info['source']}")
    print(f"Graph stats  : {graph_info['statistics']}")

    # --------------------------------------------------------
    # Fit scaler on observed TRAIN data only.
    # --------------------------------------------------------

    max_value = fit_max_scaler(train_data)

    print(f"\nTraining max value : {max_value}")

    results["max_value"] = max_value

    # Target validity always refers to the ORIGINAL observations.
    target_row_ok = {
        "train": finite_row_mask(train_data),
        "val": finite_row_mask(val_data),
        "test": finite_row_mask(test_data),
    }

    if config.imputation == "gap_bands":

        # ----------------------------------------------------
        # Gap-band filling of A3T-GCN inputs. Neighbour models
        # are fitted on observed TRAIN data; only the training
        # split may look ahead. Each split is filled into a new
        # array and then normalized in place (identical values
        # to normalize_data, without a second copy).
        # ----------------------------------------------------

        print("\nFitting neighbour models on training data...")

        neighbour_models = fit_neighbour_models(
            train_data,
            road_adjacency,
            num_neighbours=config.imputation_neighbours,
        )

        imputation_report = {
            "method": "gap_bands",
            "num_neighbours": config.imputation_neighbours,
            "sensors_with_neighbour_model": len(neighbour_models),
        }

        normalized = {}

        for name, split, look_ahead in [
            ("train", train_data, True),
            ("val", val_data, False),
            ("test", test_data, False),
        ]:

            filled, counts = fill_split(
                split,
                neighbour_models,
                look_ahead,
            )

            filled /= max_value

            normalized[name] = filled

            imputation_report[name] = {
                "gaps": gap_length_counts(split),
                "filled": counts,
            }

            print(f"{name:5s}: filled {counts}")

        del neighbour_models, train_data, val_data, test_data, traffic

        train_norm = normalized.pop("train")
        val_norm = normalized.pop("val")
        test_norm = normalized.pop("test")

    else:

        # ----------------------------------------------------
        # No filling. The whole array is divided once and the
        # splits are views, which gives exactly the same values
        # as normalizing each split separately.
        # ----------------------------------------------------

        imputation_report = {"method": "none"}

        del train_data, val_data, test_data

        traffic_norm = normalize_data(
            traffic,
            max_value,
        )

        del traffic

        train_norm, val_norm, test_norm = chronological_split(
            traffic_norm,
            config.train_rate,
            config.val_rate,
        )

    results["imputation"] = imputation_report

    # --------------------------------------------------------
    # Valid windows: inputs finite (after filling, if any) and
    # target timestep originally observed for every sensor.
    # Without filling this equals "no NaN anywhere in the window".
    # --------------------------------------------------------

    train_starts = valid_window_starts_from_rows(
        finite_row_mask(train_norm), target_row_ok["train"],
        config.seq_len, config.pre_len,
    )
    val_starts = valid_window_starts_from_rows(
        finite_row_mask(val_norm), target_row_ok["val"],
        config.seq_len, config.pre_len,
    )
    test_starts = valid_window_starts_from_rows(
        finite_row_mask(test_norm), target_row_ok["test"],
        config.seq_len, config.pre_len,
    )

    results["valid_sequences"] = {
        "train": int(len(train_starts)),
        "val": int(len(val_starts)),
        "test": int(len(test_starts)),
    }

    print(f"\nValid sequences : {results['valid_sequences']}")

    # --------------------------------------------------------
    # TensorFlow graph
    # --------------------------------------------------------

    tf.reset_default_graph()

    # Re-establish reproducibility after graph reset.
    set_seeds(config.seed)

    print("\nBuilding A3T-GCN model...")

    model = A3TGCNModel(
        seq_len=config.seq_len,
        num_nodes=num_nodes,
        gru_units=config.gru_units,
        pre_len=config.pre_len,
        adjacency=to_model_orientation(adjacency),
        lambda_loss=config.lambda_loss,
    )

    trainer = A3TGCNTrainer(
        loss=model.loss,
        learning_rate=config.learning_rate,
        optimizer="adam",
    )

    trainer.initialize()

    scheduler = ValidationLRScheduler(
        initial_lr=config.learning_rate,
        patience=config.lr_patience,
        factor=config.lr_factor,
        min_lr=config.lr_min,
    )

    best_val_rmse = float("inf")
    best_epoch = -1

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("A3T-GCN TRAINING")
    print("=" * 70)

    training_start = time.perf_counter()

    for epoch in range(config.training_epoch):

        epoch_start = time.perf_counter()

        batch_losses = []

        for X_batch, y_batch in iterate_window_batches(
            train_norm,
            train_starts,
            config.seq_len,
            config.pre_len,
            config.batch_size,
        ):

            feed_dict = {
                model.inputs: X_batch,
                model.labels: y_batch,
            }

            batch_losses.append(
                trainer.train_batch(feed_dict)
            )

        train_loss = (
            float(np.mean(batch_losses))
            if batch_losses
            else float("nan")
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_targets, val_predictions = predict_dataset(
            trainer,
            model,
            val_norm,
            val_starts,
            config,
        )

        val_rmse = rmse(val_targets, val_predictions)
        val_mae = mae(val_targets, val_predictions)

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        current_lr = scheduler.step(val_rmse)

        trainer.set_learning_rate(current_lr)

        # ----------------------------------------------------
        # Save BEST model using validation RMSE.
        # Test data is NOT used here.
        # ----------------------------------------------------

        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse
            best_epoch = epoch

            trainer.save(checkpoint_path)

            checkpoint_status = " * BEST"

        else:

            checkpoint_status = ""

        epoch_seconds = time.perf_counter() - epoch_start

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "val_rmse_original": float(val_rmse * max_value),
            "val_mae_original": float(val_mae * max_value),
            "learning_rate": float(current_lr),
            "seconds": epoch_seconds,
        })

        print(
            f"Epoch {epoch + 1:03d}/{config.training_epoch} | "
            f"Loss {train_loss:.6f} | "
            f"Val RMSE {val_rmse:.6f} | "
            f"Val MAE {val_mae:.6f} | "
            f"LR {current_lr:.8f} | "
            f"{epoch_seconds:.0f}s"
            f"{checkpoint_status}"
        )

    training_seconds = time.perf_counter() - training_start

    results["history"] = history
    results["training_seconds"] = training_seconds
    results["best_epoch"] = best_epoch + 1
    results["best_val_rmse"] = float(best_val_rmse)
    results["best_val_rmse_original"] = float(best_val_rmse * max_value)

    # --------------------------------------------------------
    # Restore best validation checkpoint
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RESTORING BEST MODEL")
    print("=" * 70)

    print("Best epoch :", best_epoch + 1)
    print("Best validation RMSE :", best_val_rmse)

    trainer.restore(checkpoint_path)

    if args.selection_only:

        # Model-selection run: the test set is never evaluated.
        trainer.close()

        results["test_metrics"] = "not evaluated (selection run)"

        results_path = run_dir / "results.json"

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\nSelection run: test set NOT evaluated.")
        print(f"Results saved to {results_path}")

        return

    # --------------------------------------------------------
    # Final TEST evaluation (exactly once)
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    assert "test_metrics" not in results, "Test set evaluated twice."

    test_targets, test_predictions = predict_dataset(
        trainer,
        model,
        test_norm,
        test_starts,
        config,
    )

    # Inverse scaling
    test_targets_original = test_targets * max_value
    test_predictions_original = test_predictions * max_value

    results["test_metrics"] = {
        "rmse": float(rmse(test_targets_original, test_predictions_original)),
        "mae": float(mae(test_targets_original, test_predictions_original)),
        "r2": float(r2_score(test_targets_original, test_predictions_original)),
        "accuracy": float(
            accuracy(test_targets_original, test_predictions_original)
        ),
    }

    results["test_metrics"]["masked_mape"] = masked_mape(
        test_targets_original,
        test_predictions_original,
        MAPE_MIN_FLOW,
    )
    results["test_metrics"]["mape_min_flow"] = MAPE_MIN_FLOW

    for name, value in results["test_metrics"].items():
        print(f"{name.upper():12s}: {value:.6f}")

    trainer.close()

    # --------------------------------------------------------
    # Detailed test outputs for the E0 / E1 / E2 comparison
    # --------------------------------------------------------

    save_test_outputs(
        run_dir,
        results,
        config,
        sensors,
        test_starts,
        val_end,
        target_row_ok["test"],
        test_targets_original,
        test_predictions_original,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_path = run_dir / "results.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    print("\n")
    print("=" * 70)
    print(f"{config.experiment} COMPLETE")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
