"""
E0-E7 ablation runner for A3T-GCN on LargeST-GBA 2021.

The experiments are defined in experiments/registry.py (docs/
architecture.md): E0 road graph, E1 VAR, E2 VAR-LiNGAM, E3 VAR + PEM
(the former E2), E4 VAR-LiNGAM + PEM, E5 horizon graphs, E6
data-driven neighbourhoods, E7 road prior.

Only the graph differs between experiments. Everything else
(data, split, preprocessing, missing-value policy, model, training,
validation, checkpointing, metrics) is shared via ExperimentConfig.

Graphs other than the road graph are built OFFLINE from the TRAINING split only
(VAR: run_var.py, PEM: run_local_pem.py) and passed in as an .npz
graph artifact with the keys:

    adjacency   float array (num_nodes, num_nodes), G[target, source]
    method      graph method (registry graph_method; legacy artifacts
                carry only the old label "E1" / "E2" = var / var_pem)
    experiment  label of the experiment it was built for
    orientation "target_source"
    train_end   number of training timesteps used to build the graph
    num_nodes   number of sensors
    sensors     original GBA indices of the sensors (after filtering)

VAR and PEM are therefore never executed inside this process, and
E0 cannot use them by construction.

Data handling (shared by every experiment, set in ExperimentConfig):

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

from .config.config import GRAPH_DIR, ExperimentConfig

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
from .experiments.registry import EXPERIMENTS, artifact_method, get_experiment
from .graph.adjacency import check_model_graph, graph_diagnostics

from .models.a3tgcn import A3TGCNModel

from .training.trainer import A3TGCNTrainer
from .training.scheduler import ValidationLRScheduler
from .training.metrics import (
    rmse,
    mae,
    r2_score,
    accuracy,
    masked_mape,
    evaluate_metrics,
)
from .training.run_logger import (
    EpochMetricsLogger,
    read_epoch_metrics,
    software_versions,
    write_run_summary,
)
from .training.stopping import MilestoneTracker, TrainingStopper

# Targets below this flow are excluded from MAPE (vehicles / 5 min).
MAPE_MIN_FLOW = 10.0


# ============================================================
# Reproducibility
# ============================================================

def set_seeds(seed):

    np.random.seed(seed)
    tf.set_random_seed(seed)


# ============================================================
# Graph selection (the only difference between experiments)
# ============================================================

def load_graph(config, road_adjacency, train_end, sensors):
    """
    Return the graph for A3T-GCN in the [target, source] convention,
    with its checks and diagnostics.

    graph_method "road" (E0):
        Road-network adjacency, unchanged (self-loops allowed).

    any other method:
        Offline graph artifact. Its method must be the experiment's
        method (experiments/registry.py), and it must have been built
        from exactly this training split and sensor set, in the
        [target, source] orientation. Causal graphs may not contain
        self-loops. Directed graphs and weights are used unchanged.
    """

    spec = get_experiment(config.experiment)

    if spec.graph_method == "none":

        # No-graph baseline: A3T-GCN adds the identity, so each
        # sensor only sees its own history.
        adjacency = np.zeros((len(sensors), len(sensors)), dtype=np.float32)

        return adjacency, {
            "source": "no_graph",
            "method": "none",
            "path": None,
            "statistics": graph_diagnostics(adjacency),
        }

    if spec.graph_method == "road":

        adjacency = np.asarray(road_adjacency, dtype=np.float32)

        return adjacency, {
            "source": "road_adjacency",
            "method": "road",
            "path": config.adjacency_path,
            "statistics": check_model_graph(
                adjacency, len(sensors), allow_self_loops=True,
            ),
        }

    with np.load(config.graph_path, allow_pickle=False) as artifact:

        files = {name: artifact[name] for name in artifact.files}

    adjacency = np.asarray(files["adjacency"], dtype=np.float32)
    method = artifact_method(files)
    graph_train_end = int(files["train_end"])
    graph_sensors = np.asarray(files["sensors"], dtype=np.int64)
    orientation = str(files.get("orientation", "target_source"))

    if method != spec.graph_method:
        raise ValueError(
            f"Graph artifact method is {method!r}, but {config.experiment} "
            f"needs {spec.graph_method!r}."
        )

    if spec.graph_method.endswith("_horizon"):

        graph_horizon = int(files["horizon"]) if "horizon" in files else None

        if graph_horizon != config.horizon:
            raise ValueError(
                f"Graph artifact is for horizon {graph_horizon}, "
                f"but the run forecasts horizon {config.horizon}."
            )

    if orientation != "target_source":
        raise ValueError(
            f"Graph artifact orientation {orientation!r}; expected "
            "'target_source'."
        )

    if not np.array_equal(graph_sensors, sensors):
        raise ValueError(
            "Graph artifact was built for a different sensor set "
            f"({len(graph_sensors)} sensors) than this run "
            f"({len(sensors)} sensors)."
        )

    if graph_train_end != train_end:
        raise ValueError(
            "Graph artifact was not built from this training split: "
            f"train_end {graph_train_end} != {train_end}."
        )

    statistics = check_model_graph(
        adjacency, len(sensors), allow_self_loops=False,
    )

    return adjacency, {
        "source": f"{config.experiment}_artifact",
        "method": method,
        "artifact_label": str(files["experiment"]),
        "path": str(config.graph_path),
        "train_end": graph_train_end,
        "statistics": statistics,
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
        config.horizon - 1,
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
        + (config.horizon - 1)
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
        description="E0-E7 A3T-GCN ablation on LargeST-GBA."
    )

    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="E0",
        help="E0-E7 (experiments/registry.py). The former E2 "
             "(VAR + PEM) is E3.",
    )

    parser.add_argument(
        "--graph",
        default=None,
        help="Graph artifact (.npz); default FINAL_GRAPHS[experiment].",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training_epoch (smoke tests only).",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning_rate (stability checks; must then be "
             "identical for every experiment compared).",
    )

    parser.add_argument(
        "--clip-norm",
        type=float,
        default=None,
        help="Gradient clipping by global norm (default: none).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: config.seed = 42). Only the "
             "random initialisation changes; report every seed run.",
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
        help="Output folder under results/ "
             "(default: <experiment>/seed_<seed>).",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast this many 5-minute steps ahead (default 1). E5 "
             "uses the graph of that horizon; other experiments keep "
             "their graph (the matched no-conditioning comparison).",
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Stop after N epochs without validation improvement "
             "(default: config.early_stopping_patience = off).",
    )

    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Wall-clock budget for the whole process (Kaggle: 11). "
             "No epoch is started that would end after it; the best "
             "checkpoint is then evaluated as usual.",
    )

    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=None,
        help="Also keep the best checkpoint among epochs <= M for each "
             "M (e.g. 20 in a 30-epoch selection run).",
    )

    parser.add_argument(
        "--evaluate-only",
        default=None,
        metavar="RUN_DIR",
        help="Evaluate the TEST set once for an existing selection run "
             "(its saved settings and checkpoint; no training).",
    )

    parser.add_argument(
        "--milestone",
        type=int,
        default=None,
        help="With --evaluate-only: use the best checkpoint among "
             "epochs <= MILESTONE (must have been saved with --milestones).",
    )

    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        help="Smoke test only: use the first N train / val / test "
             "windows. Results are marked as a smoke run.",
    )

    return parser.parse_args()


def main():

    process_start = time.perf_counter()

    args = parse_args()

    previous = None

    if args.evaluate_only:

        # --------------------------------------------------------
        # Test evaluation of an existing selection run: its own
        # settings, data pipeline and checkpoint; nothing else.
        # --------------------------------------------------------

        eval_dir = Path(args.evaluate_only).resolve()

        with open(eval_dir / "results.json", encoding="utf-8") as f:
            previous = json.load(f)

        if isinstance(previous.get("test_metrics"), dict):
            raise RuntimeError(
                f"{eval_dir}: the test set was already evaluated; "
                "it is evaluated only once per run."
            )

        args.smoke = previous.get("smoke_windows")

        # A run made elsewhere (e.g. Kaggle) saved that machine's
        # paths: use this machine's files, which must have the same
        # names (the graph artifact is checked again when loaded).
        saved = dict(previous["config"])
        defaults = ExperimentConfig()

        for key in ("traffic_path", "adjacency_path", "results_dir"):
            if not Path(saved[key]).exists():
                if Path(saved[key]).name != Path(getattr(defaults, key)).name:
                    raise FileNotFoundError(f"{key} {saved[key]} not available here")
                saved[key] = getattr(defaults, key)

        if saved.get("graph_path") and not Path(saved["graph_path"]).exists():
            saved["graph_path"] = str(GRAPH_DIR / Path(saved["graph_path"]).name)

        config = ExperimentConfig(**saved)

    else:

        if args.milestone is not None:
            raise ValueError("--milestone needs --evaluate-only")

        config = ExperimentConfig(
            experiment=args.experiment,
            graph_path=args.graph,
        )

    if args.epochs is not None:
        config.training_epoch = args.epochs

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    if args.clip_norm is not None:
        config.grad_clip_norm = args.clip_norm

    if args.seed is not None:
        config.seed = args.seed

    if args.early_stopping is not None:
        config.early_stopping_patience = args.early_stopping

    if args.horizon is not None:
        config.horizon = args.horizon

    if previous is not None:
        overridden = [
            name for name in ("epochs", "learning_rate", "clip_norm", "seed", "horizon",
                              "early_stopping", "graph", "run_name", "milestones")
            if getattr(args, name) is not None
        ]
        if overridden:
            raise ValueError(
                f"--evaluate-only uses the run's saved settings; remove {overridden}"
            )

    config.validate()

    if previous is not None:
        run_dir = eval_dir
    else:
        run_dir = Path(config.results_dir) / (
            args.run_name or (
                f"{config.experiment}/"
                + ("" if config.horizon == 1 else f"h{config.horizon}/")
                + f"seed_{config.seed}"
            )
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
        "smoke_windows": args.smoke,
        "software": software_versions(),
    }

    if previous is not None:
        # Keep everything recorded during training; only the test
        # evaluation is added below.
        results = {
            **previous,
            "software_evaluation": results["software"],
        }
        results.pop("test_metrics", None)
        args.selection_only = False

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
    # Graph (the only difference between experiments)
    # --------------------------------------------------------

    adjacency, graph_info = load_graph(
        config,
        road_adjacency,
        train_end,
        sensors,
    )

    graph_info["orientation"] = (
        "stored as G[target, source]; transposed once for A3T-GCN "
        "(decision D2, identical for every experiment)"
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

    # Target = `horizon` steps after the last input (horizon 1 = next step).
    train_starts = valid_window_starts_from_rows(
        finite_row_mask(train_norm), target_row_ok["train"],
        config.seq_len, config.pre_len, config.horizon - 1,
    )
    val_starts = valid_window_starts_from_rows(
        finite_row_mask(val_norm), target_row_ok["val"],
        config.seq_len, config.pre_len, config.horizon - 1,
    )
    test_starts = valid_window_starts_from_rows(
        finite_row_mask(test_norm), target_row_ok["test"],
        config.seq_len, config.pre_len, config.horizon - 1,
    )

    if args.smoke is not None:
        train_starts = train_starts[: args.smoke]
        val_starts = val_starts[: args.smoke]
        test_starts = test_starts[: args.smoke]
        print(f"\nSMOKE RUN: first {args.smoke} windows per split")

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
        clip_norm=config.grad_clip_norm,
    )

    trainer.initialize()

    scheduler = ValidationLRScheduler(
        initial_lr=config.learning_rate,
        patience=config.lr_patience,
        factor=config.lr_factor,
        min_lr=config.lr_min,
    )

    if previous is not None:

        # --------------------------------------------------------
        # Evaluate-only: best epoch from the run's epoch_metrics.csv,
        # among epochs <= --milestone if given.
        # --------------------------------------------------------

        rows = read_epoch_metrics(run_dir / "epoch_metrics.csv")

        if args.milestone is None:
            checkpoint_name = "best_model"
            candidates = rows
        else:
            checkpoint_name = f"best_until_epoch_{args.milestone}"
            candidates = [r for r in rows if r["epoch"] <= args.milestone]

        if not candidates:
            raise ValueError("No epochs to choose from.")

        best_row = min(candidates, key=lambda r: (r["RMSE"], r["epoch"]))
        best_row = {**best_row, "is_best_epoch": True}
        best_epoch = best_row["epoch"] - 1
        best_val_rmse = best_row["RMSE"] / max_value

        checkpoint_path = str(run_dir / "checkpoints" / checkpoint_name)
        history = previous["history"]
        training_seconds = previous["training_seconds"]

        results["evaluated_checkpoint"] = {
            "checkpoint": checkpoint_name,
            "milestone": args.milestone,
            "best_epoch": best_row["epoch"],
            "epochs_trained": len(rows),
        }
        results["best_epoch"] = best_row["epoch"]
        results["best_val_rmse"] = float(best_val_rmse)
        results["best_val_rmse_original"] = float(best_row["RMSE"])

        print(f"\nEvaluate-only: {checkpoint_name} (epoch {best_row['epoch']})")

    else:

        best_val_rmse = float("inf")
        best_epoch = -1

        history = []

        epoch_logger = EpochMetricsLogger(run_dir / "epoch_metrics.csv")

        stopper = TrainingStopper(
            early_stopping_patience=config.early_stopping_patience,
            # 15 minutes kept for the final validation / test pass.
            max_seconds=(
                None if args.max_hours is None
                else args.max_hours * 3600 - 900
            ),
        )

        milestones = MilestoneTracker(args.milestones)

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        print("\n")
        print("=" * 70)
        print("A3T-GCN TRAINING")
        print("=" * 70)

        training_start = time.perf_counter()

        for epoch in range(config.training_epoch):

            if stopper.should_stop(time.perf_counter() - process_start):
                print(f"\nTraining stopped before epoch {epoch + 1}: {stopper.reason}")
                break

            epoch_start = time.perf_counter()

            batch_losses = []

            for X_batch, y_batch in iterate_window_batches(
                train_norm,
                train_starts,
                config.seq_len,
                config.pre_len,
                config.batch_size,
                config.horizon - 1,
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

            # All validation metrics in original units (vehicles).
            val_metrics = evaluate_metrics(
                val_targets * max_value,
                val_predictions * max_value,
                MAPE_MIN_FLOW,
            )

            # ----------------------------------------------------
            # Scheduler
            # ----------------------------------------------------

            # Rate used during this epoch (epoch_metrics.csv convention);
            # history.csv keeps the rate after the scheduler step.
            lr_used = scheduler.current_lr

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

                with open(run_dir / "checkpoints" / "best_model_epoch.json",
                          "w", encoding="utf-8") as f:
                    json.dump({
                        "epoch": epoch + 1,
                        "val_rmse_original": float(val_rmse * max_value),
                    }, f)

                checkpoint_status = " * BEST"

            else:

                checkpoint_status = ""

            for milestone in milestones.update(epoch + 1, val_rmse):

                name = f"best_until_epoch_{milestone}"
                trainer.save(str(run_dir / "checkpoints" / name))

                with open(run_dir / "checkpoints" / f"{name}.json",
                          "w", encoding="utf-8") as f:
                    json.dump({
                        "epoch": epoch + 1,
                        "val_rmse_original": float(val_rmse * max_value),
                    }, f)

            epoch_seconds = time.perf_counter() - epoch_start

            stopper.after_epoch(val_rmse, epoch_seconds)

            logged_best = epoch_logger.log_epoch(
                epoch=epoch + 1,
                metrics=val_metrics,
                learning_rate=lr_used,
                train_loss=train_loss,
                elapsed_seconds=time.perf_counter() - training_start,
            )

            if logged_best != bool(checkpoint_status):
                raise RuntimeError(
                    "epoch_metrics best epoch disagrees with the checkpoint rule"
                )

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
        results["stopped_early"] = stopper.reason
        results["milestones"] = {
            str(m): milestones.best_epoch[m] for m in milestones.milestones
        }

        if best_epoch < 0:
            raise RuntimeError(
                f"No epoch was completed ({stopper.reason}); nothing to evaluate."
            )

        checkpoint_name = "best_model"
        best_row = epoch_logger.best_row()

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

    # --------------------------------------------------------
    # Check that the restored weights ARE the best epoch: the
    # marked row, the checkpoint record and a fresh validation
    # pass must agree.
    # --------------------------------------------------------

    restored_targets, restored_predictions = predict_dataset(
        trainer, model, val_norm, val_starts, config,
    )

    restored_rmse = float(
        rmse(restored_targets, restored_predictions) * max_value
    )

    record_name = (
        "best_model_epoch.json" if checkpoint_name == "best_model"
        else f"{checkpoint_name}.json"
    )

    with open(run_dir / "checkpoints" / record_name, encoding="utf-8") as f:
        checkpoint_record = json.load(f)

    checkpoint_check = {
        "epoch_metrics_best_epoch": best_row["epoch"],
        "checkpoint_epoch": checkpoint_record["epoch"],
        "results_best_epoch": best_epoch + 1,
        "best_row_val_rmse": best_row["RMSE"],
        "restored_val_rmse": restored_rmse,
    }

    checkpoint_check["consistent"] = bool(
        best_row["epoch"] == checkpoint_record["epoch"] == best_epoch + 1
        and abs(restored_rmse - best_row["RMSE"])
        <= 1e-4 * max(1.0, best_row["RMSE"])
    )

    results["checkpoint_check"] = checkpoint_check

    print(f"Checkpoint check : {checkpoint_check}")

    if not checkpoint_check["consistent"]:
        raise RuntimeError(f"Best-checkpoint mismatch: {checkpoint_check}")

    summary = {
        "experiment": config.experiment,
        "seed": config.seed,
        "run_dir": str(run_dir),
        "smoke_windows": args.smoke,
        "best_epoch": best_row["epoch"],
        "best_validation_metrics": best_row,
        "checkpoint_path": checkpoint_path,
        "checkpoint_check": checkpoint_check,
        "epochs_completed": len(history),
        "stopped_early": results.get("stopped_early"),
        "evaluated_checkpoint": results.get("evaluated_checkpoint"),
        "training_seconds": training_seconds,
        "valid_sequences": results["valid_sequences"],
        "graph": {
            "source": graph_info["source"],
            "path": graph_info.get("path"),
            "statistics": graph_info["statistics"],
        },
        "config": asdict(config),
        "software": results["software"],
        "test_metrics": None,
    }

    if args.selection_only:

        # Model-selection run: the test set is never evaluated.
        trainer.close()

        results["test_metrics"] = "not evaluated (selection run)"

        summary["test_metrics"] = "not evaluated (selection run)"
        write_run_summary(run_dir / "run_summary.json", summary)

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

    summary["test_metrics"] = evaluate_metrics(
        test_targets_original,
        test_predictions_original,
        MAPE_MIN_FLOW,
    )
    summary["test_metrics"]["mape_min_flow"] = MAPE_MIN_FLOW

    write_run_summary(run_dir / "run_summary.json", summary)

    trainer.close()

    # --------------------------------------------------------
    # Detailed test outputs for the comparison of experiments
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
