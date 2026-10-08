"""
Topology-aware VAR on the LargeST-GBA TRAINING split (E1 / E2).

Pipeline:
    GBA traffic + road-network adjacency
            ↓
    Chronological split, sensor filter (shared ExperimentConfig)
            ↓
    Topology-aware VAR fitted on TRAINING rows only
            ↓
    VAR coefficients (+ training residuals and valid mask)
            ↓
    Out-of-sample one-step residuals on the VALIDATION split
    (for lag-order / ridge selection; the test split is never read)
            ↓
    Saved outputs for the E1 graph and for PEM (E2)

Missing values are NOT imputed. VAR uses observed values only:
a regression sample is used only when its target and every lagged
predictor are finite (the gap-band filling of A3T-GCN inputs is
never applied here).

Run from the implementation directory (traffic-gpu environment):

    python run_var.py --lag 1 --alpha 1e-5
    python run_var.py --lag 1 --alpha 1e4 --save-residuals
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config.config import VAR_ALPHA, VAR_LAG, ExperimentConfig
from data.loader import load_gba_dataset
from data.splitting import chronological_split
from data.sensor_filter import cached_duplicate_filter, select_sensors
from causal.var import TopologyAwareVAR
from causal.var_forecast import (
    residual_summary,
    var_one_step_residuals,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "var_train"

USE_GPU = True


def parse_args():

    parser = argparse.ArgumentParser(
        description="Topology-aware VAR on the GBA training split."
    )

    parser.add_argument("--lag", type=int, default=VAR_LAG)
    parser.add_argument("--alpha", type=float, default=VAR_ALPHA)

    parser.add_argument(
        "--save-residuals",
        action="store_true",
        help="Also save training residuals and valid mask (~0.9 GB).",
    )

    return parser.parse_args()


# ============================================================
# OUTPUT
# ============================================================

def save_results(
    output_dir: Path,
    result,
    sensors: np.ndarray,
    metadata: dict,
    save_residuals: bool,
) -> None:
    """
    Save VAR outputs required by later stages.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "coefficients.npy", result.coefficients)
    np.save(output_dir / "sensors.npy", sensors)

    if save_residuals:
        np.save(output_dir / "residuals.npy", result.residuals)
        np.save(output_dir / "valid_mask.npy", result.valid_mask)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    config = ExperimentConfig()
    config.validate()

    output_dir = OUTPUT_ROOT / f"lag{args.lag}_alpha{args.alpha:g}"

    print("=" * 70)
    print("TOPOLOGY-AWARE VAR (TRAINING SPLIT)")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Traffic file  : {config.traffic_path}")
    print(f"  Adjacency     : {config.adjacency_path}")
    print(f"  Sensor filter : {config.sensor_filter}")
    print(f"  Lag order     : {args.lag}")
    print(f"  Ridge alpha   : {args.alpha:g}")
    print(f"  GPU           : {USE_GPU}")
    print(f"  Output        : {output_dir}")

    # --------------------------------------------------------
    # Load data, split, filter sensors (same as main.py)
    # --------------------------------------------------------

    print("\nLoading traffic data...")

    load_start = time.perf_counter()

    traffic, adjacency = load_gba_dataset(
        config.traffic_path,
        config.adjacency_path,
        num_timesteps=config.num_timesteps,
        num_nodes=config.num_nodes,
    )

    train_data, val_data, _ = chronological_split(
        traffic,
        config.train_rate,
        config.val_rate,
    )

    train_end = len(train_data)
    val_end = train_end + len(val_data)

    if config.sensor_filter == "duplicates":
        sensors, filter_report = cached_duplicate_filter(
            train_data,
            OUTPUT_ROOT.parent / "sensor_filter",
        )
    else:
        sensors = np.arange(traffic.shape[1], dtype=np.int64)
        filter_report = {"rule": "none"}

    # Keep only TRAIN and VALIDATION rows; the test rows are dropped
    # here and never used by this script.
    traffic, adjacency = select_sensors(
        traffic[:val_end],
        adjacency,
        sensors,
    )

    train_data = traffic[:train_end]
    val_data = traffic[train_end:val_end]

    load_time = time.perf_counter() - load_start

    num_nodes = train_data.shape[1]

    print(f"Training shape   : {train_data.shape}")
    print(f"Validation shape : {val_data.shape}")
    print(f"Adjacency shape  : {adjacency.shape}")
    print(f"Loading time     : {load_time:.3f} seconds")

    # --------------------------------------------------------
    # Missing-data diagnostics (training rows)
    # --------------------------------------------------------

    missing_count = np.count_nonzero(~np.isfinite(train_data))
    total_count = train_data.size
    missing_percentage = 100.0 * missing_count / total_count

    print("\nINPUT DATA (training rows)")
    print(f"  Total values       : {total_count}")
    print(f"  Missing/non-finite : {missing_count}")
    print(f"  Missing percentage : {missing_percentage:.6f}%")

    # --------------------------------------------------------
    # Adjacency diagnostics
    # --------------------------------------------------------

    nonzero_adjacency = np.count_nonzero(adjacency)
    density = nonzero_adjacency / adjacency.size
    self_loops = np.count_nonzero(np.diag(adjacency))

    print("\nADJACENCY")
    print(f"  Non-zero entries : {nonzero_adjacency}")
    print(f"  Density          : {density:.6f}")
    print(f"  Self-loops       : {self_loops}")

    # --------------------------------------------------------
    # Fit VAR on TRAINING rows only
    # --------------------------------------------------------

    print("\nFitting VAR on training rows...")

    model = TopologyAwareVAR(
        lag_order=args.lag,
        ridge_alpha=args.alpha,
        use_gpu=USE_GPU,
    )

    start_time = time.perf_counter()

    result = model.fit(
        data=train_data,
        adjacency=adjacency,
    )

    runtime = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("VAR RESULT")
    print("=" * 70)

    print(f"Runtime              : {runtime:.3f} seconds")
    print(f"Runtime              : {runtime / 60:.3f} minutes")
    print(f"Coefficient shape    : {result.coefficients.shape}")
    print(f"Residual shape       : {result.residuals.shape}")
    print(f"Valid-mask shape     : {result.valid_mask.shape}")
    print(f"Lag order            : {result.lag_order}")

    # --------------------------------------------------------
    # Residual coverage
    # --------------------------------------------------------

    valid_counts = np.sum(result.valid_mask, axis=0)

    coverage = (valid_counts / train_end) * 100.0

    print("\nRESIDUAL COVERAGE (training)")
    print(f"  Minimum valid residuals : {valid_counts.min()}")
    print(f"  Maximum valid residuals : {valid_counts.max()}")
    print(f"  Mean valid residuals    : {valid_counts.mean():.3f}")
    print(f"  Minimum coverage (%)    : {coverage.min():.3f}")
    print(f"  Maximum coverage (%)    : {coverage.max():.3f}")
    print(f"  Mean coverage (%)       : {coverage.mean():.6f}")

    # --------------------------------------------------------
    # In-sample and out-of-sample residuals
    # --------------------------------------------------------

    train_summary = residual_summary(result.residuals)

    val_residuals, _ = var_one_step_residuals(
        val_data,
        result.coefficients,
        adjacency,
    )

    val_summary = residual_summary(val_residuals)

    del val_residuals

    print("\nRESIDUALS (vehicles per 5 min)")
    print(
        f"  Training   (in-sample)     : RMSE {train_summary['rmse']:.4f} | "
        f"MAE {train_summary['mae']:.4f} | mean {train_summary['mean']:.4f}"
    )
    print(
        f"  Validation (out-of-sample) : RMSE {val_summary['rmse']:.4f} | "
        f"MAE {val_summary['mae']:.4f} | mean {val_summary['mean']:.4f} | "
        f"coverage {val_summary['coverage_percentage']:.3f}%"
    )

    # --------------------------------------------------------
    # Coefficient statistics
    # --------------------------------------------------------

    coefficients = result.coefficients

    nonzero_coefficients = np.count_nonzero(coefficients)

    print("\nCOEFFICIENTS")
    print(f"  Non-zero coefficients : {nonzero_coefficients}")
    print(f"  Minimum               : {coefficients.min()}")
    print(f"  Maximum               : {coefficients.max()}")
    print(f"  Mean                  : {coefficients.mean()}")
    print(f"  Std                   : {coefficients.std()}")
    print(f"  Max |coefficient|     : {np.abs(coefficients).max()}")

    # --------------------------------------------------------
    # Topology consistency
    # --------------------------------------------------------

    allowed = (adjacency != 0) | np.eye(num_nodes, dtype=bool)

    topology_preserved = bool(
        np.all(result.coefficients[:, ~allowed] == 0)
    )

    print("\nTOPOLOGY CONSISTENCY")
    print(f"  Non-zero adjacency entries : {nonzero_adjacency}")
    print(f"  Non-zero VAR coefficients  : {nonzero_coefficients}")
    print(
        f"  Topology restriction       : "
        f"{'PRESERVED' if topology_preserved else 'VIOLATED'}"
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    metadata = {
        "traffic_file": str(config.traffic_path),
        "adjacency_file": str(config.adjacency_path),
        "train_rows": [0, train_end],
        "val_rows": [train_end, val_end],
        "train_end": train_end,
        "sensor_filter": config.sensor_filter,
        "num_sensors_excluded": int(
            filter_report.get("num_excluded", 0)
        ),
        "num_nodes": int(num_nodes),
        "lag_order": args.lag,
        "ridge_alpha": args.alpha,
        "use_gpu": USE_GPU,
        "imputation": "none (observed values only)",
        "coefficient_shape": list(result.coefficients.shape),
        "residuals_saved": bool(args.save_residuals),
        "runtime_seconds": runtime,
        "train_missing_count": int(missing_count),
        "train_missing_percentage": float(missing_percentage),
        "mean_residual_coverage": float(coverage.mean()),
        "minimum_residual_coverage": float(coverage.min()),
        "train_residuals": train_summary,
        "val_residuals": val_summary,
        "max_abs_coefficient": float(np.abs(coefficients).max()),
        "nonzero_adjacency": int(nonzero_adjacency),
        "nonzero_coefficients": int(nonzero_coefficients),
        "topology_restriction_preserved": topology_preserved,
    }

    save_results(
        output_dir,
        result,
        sensors,
        metadata,
        args.save_residuals,
    )

    print("\nOUTPUTS SAVED")
    print(f"  Directory      : {output_dir}")
    print("  coefficients   : coefficients.npy")
    print("  sensors        : sensors.npy")
    if args.save_residuals:
        print("  residuals      : residuals.npy")
        print("  valid mask     : valid_mask.npy")
    print("  metadata       : metadata.json")

    print("\n" + "=" * 70)
    print("VAR RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
