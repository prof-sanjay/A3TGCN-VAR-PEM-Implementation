"""
Combine every run's epoch_metrics.csv and run_summary.json into:

    <out>/all_epoch_metrics.csv    experiment, seed, horizon, run, epoch, RMSE, MAE,
                                   Masked_MAPE, R2, Accuracy, learning_rate, ...
    <out>/best_epoch_summary.csv   one row per run: best epoch, its
                                   validation metrics, final test metrics
    <out>/multi_seed_summary.csv   per experiment and horizon: mean and std of the
                                   test metrics over seeds (final runs only)

A run is any folder under --results containing run_summary.json.
Smoke runs and selection-only runs are listed in the first two files
(columns smoke / selection_only) but never enter multi_seed_summary.

Run from the implementation directory:

    python summarize_runs.py [--results results] [--out results/summary]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from training.metrics import METRIC_COLUMNS
from training.run_logger import COLUMNS, read_epoch_metrics


IMPLEMENTATION_DIR = Path(__file__).resolve().parent

TEST_COLUMNS = METRIC_COLUMNS + ("Accuracy_TGCN",)


def find_runs(results_dir: Path):

    for summary_path in sorted(results_dir.rglob("run_summary.json")):
        yield summary_path.parent, json.loads(
            summary_path.read_text(encoding="utf-8")
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(IMPLEMENTATION_DIR / "results"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else results_dir / "summary"

    epoch_rows = []
    best_rows = []

    for run_dir, summary in find_runs(results_dir):

        if out_dir in run_dir.parents or run_dir == out_dir:
            continue

        run = run_dir.relative_to(results_dir).as_posix()

        ids = {
            "experiment": summary["experiment"],
            "seed": summary["seed"],
            # Forecast horizon in 5-minute steps (1 for runs before Phase 7).
            "horizon": int((summary.get("config") or {}).get("horizon", 1)),
            "run": run,
        }

        flags = {
            "smoke": summary.get("smoke_windows") is not None,
            "selection_only": not isinstance(summary.get("test_metrics"), dict),
        }

        epoch_file = run_dir / "epoch_metrics.csv"

        if epoch_file.exists():
            for row in read_epoch_metrics(epoch_file):
                epoch_rows.append({**ids, **{c: row.get(c) for c in COLUMNS}, **flags})

        best = summary["best_validation_metrics"]
        test = summary["test_metrics"] if flags["selection_only"] is False else {}

        best_rows.append({
            **ids,
            "best_epoch": summary["best_epoch"],
            **{f"val_{m}": best.get(m) for m in METRIC_COLUMNS},
            **{f"test_{m}": test.get(m, math.nan) for m in TEST_COLUMNS},
            "epochs_completed": summary.get("epochs_completed"),
            "training_seconds": summary.get("training_seconds"),
            "checkpoint_consistent": summary.get("checkpoint_check", {}).get("consistent"),
            **flags,
        })

    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = pd.DataFrame(epoch_rows)
    best = pd.DataFrame(best_rows)

    epochs.to_csv(out_dir / "all_epoch_metrics.csv", index=False, na_rep="nan")
    best.to_csv(out_dir / "best_epoch_summary.csv", index=False, na_rep="nan")

    multi = []

    if not best.empty:

        final = best[~best["smoke"] & ~best["selection_only"]]

        for (experiment, horizon), group in final.groupby(
            ["experiment", "horizon"], sort=True
        ):

            row = {
                "experiment": experiment,
                "horizon": int(horizon),
                "n_seeds": int(group["seed"].nunique()),
                "seeds": " ".join(str(s) for s in sorted(group["seed"].unique())),
            }

            for m in ("val_RMSE",) + tuple(f"test_{m}" for m in TEST_COLUMNS):
                values = group[m].astype(float).to_numpy()
                row[f"{m}_mean"] = float(np.mean(values))
                # Sample std; undefined (NaN) with a single seed.
                row[f"{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan

            multi.append(row)

    pd.DataFrame(multi).to_csv(out_dir / "multi_seed_summary.csv", index=False, na_rep="nan")

    print(f"Runs found          : {len(best_rows)}")
    print(f"Epoch rows          : {len(epoch_rows)}")
    print(f"Experiments (final) : {[r['experiment'] for r in multi]}")
    print(f"Saved               : {out_dir}")


if __name__ == "__main__":
    main()
