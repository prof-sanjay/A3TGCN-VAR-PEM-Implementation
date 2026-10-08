"""
Local PEM on the VAR TRAINING residuals, one causal unit at a time (E2).

Pipeline:
    outputs/var_train/<run>/residuals.npy   (training rows only)
            ↓
    topology causal units  U_i = {i} ∪ N_in(i) ∪ N_out(i)
            ↓
    LocalPEM (unchanged) on each unit, in parallel worker processes
            ↓
    per-unit results cached on disk (resumable)
            ↓
    aggregation: P = max over units (symmetric inclusion probability)
                 D = OR over units  (directed PEM-LBN edges)

Each unit is passed to LocalPEM with ONLY its own residual columns
(local indices 0..p-1) and mapped back afterwards. The aggregation
(element-wise max for P, logical OR for D) is order-independent and
therefore identical to running LocalPEM.fit on all units at once.

Missing values are NOT imputed: LocalPEM keeps only the rows where
every residual of the unit is finite.

A failed unit is recorded with its error and is NOT silently treated
as "no edges"; aggregation refuses to run while failures exist unless
--allow-failed is given (the failures are then reported).

Run from the implementation directory (traffic-causal environment):

    python run_local_pem.py --var-run lag1_alpha10000 --workers 10
    python run_local_pem.py --var-run lag1_alpha10000 --sample-units 60
"""

import os

# One BLAS thread per worker process (set before NumPy is imported).
for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config.config import ExperimentConfig
from causal.causal_units import CausalUnit, build_causal_units
from causal.local_pem import LocalPEM


IMPLEMENTATION_DIR = Path(__file__).resolve().parent

VAR_ROOT = IMPLEMENTATION_DIR / "outputs" / "var_train"
PEM_ROOT = IMPLEMENTATION_DIR / "outputs" / "pem"


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Local PEM on VAR training residuals."
    )

    parser.add_argument("--var-run", required=True,
                        help="Folder under outputs/var_train/.")

    parser.add_argument("--nu0", type=float, default=0.05)
    parser.add_argument("--nu1", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prior-inclusion", type=float, default=0.5)
    # Optimizer for the unchanged PEM-MAP objective (decision D4).
    # SLSQP is the original solver but is infeasible for large units;
    # L-BFGS-B with the same objective-value accuracy (tolerance is
    # converted to L-BFGS-B's relative ftol inside PEM) gives the same
    # graphs and causal orders (test_pem_speedup.py, part 5).
    parser.add_argument("--optimizer", default="L-BFGS-B",
                        choices=["L-BFGS-B", "SLSQP"])
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--tolerance", type=float, default=1e-6)

    parser.add_argument("--workers", type=int, default=10)

    parser.add_argument(
        "--sample-units", type=int, default=None,
        help="Run only a reproducible sample of units (sweeps).",
    )
    parser.add_argument(
        "--max-unit-size", type=int, default=None,
        help="Skip larger units (diagnostics only; reported).",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--row-start", type=int, default=None,
        help="Use only training residual rows [row-start, row-end) "
             "(split-half stability checks; still training data).",
    )
    parser.add_argument("--row-end", type=int, default=None)

    parser.add_argument("--tag", default=None,
                        help="Output folder name (default from settings).")

    parser.add_argument("--allow-failed", action="store_true")

    return parser.parse_args()


def settings_tag(args):

    return (
        f"{args.var_run}"
        f"__nu0_{args.nu0:g}_nu1_{args.nu1:g}_tau_{args.tau:g}"
        f"_thr_{args.threshold:g}_eta_{args.prior_inclusion:g}"
        f"_{args.optimizer}"
    )


# ============================================================
# One unit (runs in a worker process)
# ============================================================

def run_unit(
    unit_index: int,
    target: int,
    nodes: np.ndarray,
    local_residuals: np.ndarray,
    pem_kwargs: dict,
    output_file: str,
) -> dict:
    """
    Run the unchanged LocalPEM on one unit and save the result.

    local_residuals holds only this unit's columns, in the order of
    ``nodes``; the unit is therefore re-indexed to 0..p-1.

    It may also be a tuple (residual_file, row_start, row_end): the
    worker then reads the unit's columns itself from the memory-mapped
    residual file. This keeps the task small, so joblib does not copy
    every unit's residuals to temporary files on disk.
    """

    if isinstance(local_residuals, tuple):

        residual_file, row_start, row_end = local_residuals

        local_residuals = np.ascontiguousarray(
            np.load(residual_file, mmap_mode="r")[row_start:row_end, nodes],
            dtype=np.float64,
        )

    p = len(nodes)

    local_unit = CausalUnit(
        target=int(np.flatnonzero(nodes == target)[0]),
        nodes=np.arange(p, dtype=np.int64),
        incoming_neighbors=np.empty(0, dtype=np.int64),
        outgoing_neighbors=np.empty(0, dtype=np.int64),
        neighbors=np.empty(0, dtype=np.int64),
    )

    start = time.perf_counter()

    status = "ok"
    message = ""
    slsqp_retries = 0

    try:

        local_pem = LocalPEM(
            units=[local_unit],
            **pem_kwargs,
        )

        result = local_pem.fit(local_residuals)

        slsqp_retries = int(getattr(local_pem, "slsqp_retries", 0))

        probabilities = result.global_inclusion_probabilities
        adjacency = result.global_adjacency
        causal_order = np.asarray(
            result.unit_causal_orders[0], dtype=np.int64
        )
        complete_rows = int(result.valid_sample_counts[0])

        if p >= 2 and complete_rows <= p:
            status = "too_few_samples"

    except Exception as exc:  # recorded, not hidden

        status = "failed"
        message = f"{type(exc).__name__}: {exc}"

        probabilities = np.zeros((p, p))
        adjacency = np.zeros((p, p))
        causal_order = np.empty(0, dtype=np.int64)
        complete_rows = int(
            np.all(np.isfinite(local_residuals), axis=1).sum()
        )

    seconds = time.perf_counter() - start

    np.savez(
        output_file,
        unit_index=unit_index,
        target=target,
        nodes=nodes,
        inclusion_probabilities=probabilities,
        adjacency=adjacency,
        causal_order=causal_order,
        complete_rows=complete_rows,
        seconds=seconds,
        slsqp_retries=slsqp_retries,
        status=status,
        message=message,
    )

    return {
        "unit_index": unit_index,
        "size": p,
        "status": status,
        "seconds": seconds,
        "message": message,
    }


# ============================================================
# Aggregation
# ============================================================

def aggregate(unit_files, num_nodes):
    """
    P = element-wise max of local inclusion probabilities,
    D = logical OR of local directed edges (D[target, source]).
    Identical to the aggregation inside LocalPEM.fit.
    """

    P = np.zeros((num_nodes, num_nodes))
    D = np.zeros((num_nodes, num_nodes))

    for path in unit_files:

        with np.load(path) as unit:

            nodes = unit["nodes"]

            index = np.ix_(nodes, nodes)

            P[index] = np.maximum(P[index], unit["inclusion_probabilities"])
            D[index] = np.maximum(D[index], (unit["adjacency"] > 0).astype(float))

    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(D, 0.0)

    return P, D


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    config = ExperimentConfig()

    var_dir = VAR_ROOT / args.var_run

    with open(var_dir / "metadata.json", encoding="utf-8") as f:
        var_metadata = json.load(f)

    if not var_metadata.get("residuals_saved", False):
        raise FileNotFoundError(
            f"{var_dir} has no saved residuals (run run_var.py "
            "with --save-residuals)."
        )

    sensors = np.load(var_dir / "sensors.npy")

    residuals = np.load(var_dir / "residuals.npy", mmap_mode="r")

    if residuals.shape != (var_metadata["train_end"], len(sensors)):
        raise ValueError("Residuals do not match the VAR metadata.")

    row_slice = slice(args.row_start, args.row_end)
    row_range = range(residuals.shape[0])[row_slice]

    full_adjacency = np.load(config.adjacency_path, mmap_mode="r")

    adjacency = np.asarray(
        full_adjacency[np.ix_(sensors, sensors)],
        dtype=np.float32,
    )

    units = build_causal_units(adjacency)

    indices = np.arange(len(units))

    if args.sample_units is not None:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(
            rng.choice(len(units), size=args.sample_units, replace=False)
        )

    skipped_large = []

    if args.max_unit_size is not None:
        skipped_large = [
            int(i) for i in indices
            if len(units[i].nodes) > args.max_unit_size
        ]
        indices = np.array(
            [i for i in indices if len(units[i].nodes) <= args.max_unit_size],
            dtype=np.int64,
        )

    tag = args.tag or settings_tag(args)

    if args.sample_units is not None:
        tag += f"__sample{args.sample_units}_seed{args.seed}"

    if args.row_start is not None or args.row_end is not None:
        tag += f"__rows{row_range.start}-{row_range.stop}"

    output_dir = PEM_ROOT / tag
    unit_dir = output_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)

    pem_kwargs = {
        "nu0": args.nu0,
        "nu1": args.nu1,
        "tau": args.tau,
        "threshold": args.threshold,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
        "prior_inclusion": args.prior_inclusion,
        "optimizer": args.optimizer,
    }

    sizes = np.array([len(units[i].nodes) for i in indices])

    print("=" * 70)
    print("LOCAL PEM ON VAR TRAINING RESIDUALS")
    print("=" * 70)
    print(f"VAR run        : {args.var_run}")
    print(f"Residuals      : {residuals.shape}, rows used "
          f"[{row_range.start}, {row_range.stop})")
    print(f"Units selected : {len(indices)} of {len(units)}")
    print(f"Unit sizes     : min {sizes.min()}, median {np.median(sizes):.0f}, "
          f"max {sizes.max()}")
    print(f"PEM settings   : {pem_kwargs}")
    print(f"Workers        : {args.workers}")
    print(f"Output         : {output_dir}")

    # Largest units first so the slowest ones do not finish last.
    order = indices[np.argsort(-sizes, kind="stable")]

    pending = [
        int(i) for i in order
        if not (unit_dir / f"unit_{i:05d}.npz").exists()
    ]

    print(f"Pending units  : {len(pending)} "
          f"({len(indices) - len(pending)} cached)")

    def tasks():
        for i in pending:
            nodes = np.asarray(units[i].nodes, dtype=np.int64)
            yield (
                i,
                int(units[i].target),
                nodes,
                (
                    str(var_dir / "residuals.npy"),
                    row_range.start,
                    row_range.stop,
                ),
                pem_kwargs,
                str(unit_dir / f"unit_{i:05d}.npz"),
            )

    start = time.perf_counter()

    if pending:

        from joblib import Parallel, delayed

        done = 0

        for record in Parallel(
            n_jobs=args.workers,
            return_as="generator_unordered",
        )(delayed(run_unit)(*task) for task in tasks()):

            done += 1

            if record["status"] != "ok" or done % 50 == 0 or done == len(pending):
                print(
                    f"[{done}/{len(pending)}] unit {record['unit_index']} "
                    f"(size {record['size']}): {record['status']} "
                    f"{record['seconds']:.1f}s "
                    f"| elapsed {time.perf_counter() - start:.0f}s "
                    f"{record['message']}",
                    flush=True,
                )

    wall_seconds = time.perf_counter() - start

    # --------------------------------------------------------
    # Summary of all selected units (cached + new)
    # --------------------------------------------------------

    unit_files = [unit_dir / f"unit_{i:05d}.npz" for i in indices]

    records = []

    for path in unit_files:
        with np.load(path) as unit:
            records.append({
                "unit_index": int(unit["unit_index"]),
                "size": int(len(unit["nodes"])),
                "status": str(unit["status"]),
                "seconds": float(unit["seconds"]),
                "complete_rows": int(unit["complete_rows"]),
                "edges": int(np.count_nonzero(unit["adjacency"])),
                "slsqp_retries": int(unit["slsqp_retries"])
                if "slsqp_retries" in unit.files else 0,
                "message": str(unit["message"]),
            })

    failed = [r for r in records if r["status"] == "failed"]
    too_few = [r for r in records if r["status"] == "too_few_samples"]

    summary = {
        "var_run": args.var_run,
        "train_end": var_metadata["train_end"],
        "num_nodes": int(len(sensors)),
        "pem_settings": pem_kwargs,
        "units_selected": int(len(indices)),
        "units_total": len(units),
        "units_skipped_too_large": skipped_large,
        "sample_units": args.sample_units,
        "seed": args.seed,
        "rows": [row_range.start, row_range.stop],
        "wall_seconds_this_call": wall_seconds,
        "cpu_seconds_all_units": float(sum(r["seconds"] for r in records)),
        "failed_units": failed,
        "units_with_slsqp_retry": int(
            sum(1 for r in records if r["slsqp_retries"] > 0)
        ),
        "slsqp_retries_total": int(
            sum(r["slsqp_retries"] for r in records)
        ),
        "too_few_samples_units": [r["unit_index"] for r in too_few],
        "local_edge_density": float(
            sum(r["edges"] for r in records)
            / max(1, sum(r["size"] * (r["size"] - 1) for r in records))
        ),
    }

    print("\nSUMMARY")
    print(f"  Units ok / failed / too few samples : "
          f"{len(records) - len(failed) - len(too_few)} / {len(failed)} / "
          f"{len(too_few)}")
    print(f"  Units with SLSQP retry              : "
          f"{summary['units_with_slsqp_retry']} "
          f"({summary['slsqp_retries_total']} optimizations)")
    print(f"  CPU seconds (sum over units)        : "
          f"{summary['cpu_seconds_all_units']:.0f}")
    print(f"  Wall seconds (this call)            : {wall_seconds:.0f}")
    print(f"  Local directed edge density         : "
          f"{summary['local_edge_density']:.4f}")

    # --------------------------------------------------------
    # Aggregate into the global P and D (full runs only)
    # --------------------------------------------------------

    full_rows = len(row_range) == residuals.shape[0]

    if args.sample_units is None and args.max_unit_size is None and full_rows:

        if failed and not args.allow_failed:
            print(f"\n{len(failed)} failed units: aggregation skipped "
                  "(use --allow-failed to aggregate and report them).")

        else:

            P, D = aggregate(unit_files, len(sensors))

            np.savez(
                output_dir / "pem_result.npz",
                inclusion_probabilities=P,
                directed_adjacency=D,
                sensors=sensors,
                train_end=var_metadata["train_end"],
                num_nodes=len(sensors),
            )

            summary["aggregated"] = {
                "symmetric_pairs_with_probability_ge_threshold": int(
                    np.count_nonzero(np.triu(P >= args.threshold, 1))
                ),
                "directed_edges": int(np.count_nonzero(D)),
                "two_way_directed_pairs": int(
                    np.count_nonzero(np.triu((D > 0) & (D.T > 0), 1))
                ),
            }

            print(f"  Aggregated                          : "
                  f"{summary['aggregated']}")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
