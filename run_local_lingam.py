"""
Local VAR-LiNGAM on the VAR TRAINING residuals (E2 / E4, Phase 5).

VAR-LiNGAM (Hyvarinen, Zhang, Shimizu & Hoyer 2010):

    x(t) = sum_k A_k x(t-k) + e(t)                (reduced-form VAR, run_var.py)
    e(t) = B_0 e(t) + n(t),  n non-Gaussian       (DirectLiNGAM on residuals)
    B_k  = (I - B_0) A_k                          (lagged structural effects)

DirectLiNGAM on all 2,056 sensors at once is infeasible, so it is run
on the same topology causal units as PEM, U_i = {i} + N_in(i) + N_out(i),
each on its own residual columns (complete rows only; no imputation).

Global B_0 (G[target, source] convention, as lingam's adjacency_matrix_):
row i of B_0 -- the effects INTO sensor i -- is taken from unit U_i,
the unit centred on i ("home unit" rule). Every row therefore comes from
exactly one estimate; entries for sources outside U_i are 0.

Outputs: outputs/lingam/<var_run>/
    units/unit_XXXXX.npz   local B_0, causal order, rows used, time, status
    lingam_result.npz      global B_0 (full runs only), sensors, train_end
    summary.json           timing, failures, edge counts

B_k and the E2 / E4 graphs are built by build_graph.py.

Run from the implementation directory (traffic-causal environment):

    python run_local_lingam.py --workers 10
    python run_local_lingam.py --sample-units 20        # timing test
"""

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config.config import LINGAM_MAX_ROWS, VAR_RUN, ExperimentConfig
from causal.candidates import load_candidates, unit_options
from causal.causal_units import build_causal_units
from causal.var_lingam import VARLiNGAM


IMPLEMENTATION_DIR = Path(__file__).resolve().parent
VAR_ROOT = IMPLEMENTATION_DIR / "outputs" / "var_train"
LINGAM_ROOT = IMPLEMENTATION_DIR / "outputs" / "lingam"


def parse_args():

    parser = argparse.ArgumentParser(description="Local VAR-LiNGAM on VAR residuals.")
    parser.add_argument("--var-run", default=VAR_RUN)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sample-units", type=int, default=None,
                        help="Only a reproducible sample of units (timing).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Unit sample and DirectLiNGAM random_state.")
    parser.add_argument("--max-rows", type=int, default=LINGAM_MAX_ROWS,
                        help="Use at most this many complete rows per unit "
                             "(evenly spaced; only if full rows are infeasible).")
    parser.add_argument("--row-start", type=int, default=None,
                        help="Use only training residual rows [row-start, row-end) "
                             "(split-half stability checks; still training data).")
    parser.add_argument("--row-end", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--allow-failed", action="store_true")
    return parser.parse_args()


def run_unit(unit_index, target, nodes, source, seed, max_rows, output_file):
    """DirectLiNGAM on one unit; local indices 0..p-1 follow ``nodes``."""

    residual_file, row_start, row_end = source
    local = np.ascontiguousarray(
        np.load(residual_file, mmap_mode="r")[row_start:row_end, nodes],
        dtype=np.float64,
    )

    complete = np.all(np.isfinite(local), axis=1)
    rows_used = int(complete.sum())

    if max_rows is not None and rows_used > max_rows:
        keep = np.flatnonzero(complete)
        keep = keep[np.linspace(0, len(keep) - 1, max_rows).astype(int)]
        complete = np.zeros_like(complete)
        complete[keep] = True
        rows_used = max_rows

    p = len(nodes)
    status, message = "ok", ""
    b0 = np.zeros((p, p))
    order = np.empty(0, dtype=np.int64)

    start = time.perf_counter()

    try:
        if p < 2:
            status = "single_node"
        elif rows_used <= p:
            status = "too_few_samples"
        else:
            local_input = np.where(complete[:, None], local, np.nan)
            result = VARLiNGAM(random_state=seed).fit(local_input)
            b0 = result.adjacency_matrix
            order = np.asarray(result.causal_order, dtype=np.int64)
    except Exception as exc:  # recorded, not hidden
        status, message = "failed", f"{type(exc).__name__}: {exc}"

    seconds = time.perf_counter() - start

    np.savez(output_file, unit_index=unit_index, target=target, nodes=nodes,
             b0=b0, causal_order=order, rows_used=rows_used, seconds=seconds,
             status=status, message=message)

    return {"unit_index": unit_index, "size": p, "status": status,
            "seconds": seconds, "message": message}


def aggregate_home_rows(unit_files, num_nodes):
    """Global B_0: row i from the unit centred on sensor i."""

    b0 = np.zeros((num_nodes, num_nodes))
    filled = np.zeros(num_nodes, dtype=bool)

    for path in unit_files:
        with np.load(path) as unit:
            nodes = unit["nodes"]
            target = int(unit["target"])
            local_target = int(np.flatnonzero(nodes == target)[0])
            b0[target, nodes] = unit["b0"][local_target]
            filled[target] = True

    np.fill_diagonal(b0, 0.0)
    return b0, filled


def main():

    args = parse_args()

    var_dir = VAR_ROOT / args.var_run
    meta = json.loads((var_dir / "metadata.json").read_text(encoding="utf-8"))
    sensors = np.load(var_dir / "sensors.npy")
    residuals = np.load(var_dir / "residuals.npy", mmap_mode="r")

    if residuals.shape != (meta["train_end"], len(sensors)):
        raise ValueError("Residuals do not match the VAR metadata.")

    config = ExperimentConfig()
    road = np.asarray(
        np.load(config.adjacency_path, mmap_mode="r")[np.ix_(sensors, sensors)],
        dtype=np.float32,
    )
    # Local units from the VAR run's candidate links (road unless E6).
    candidates = meta.get("candidates", "road")
    units = build_causal_units(
        load_candidates(candidates, sensors, meta["train_end"], road),
        **unit_options(candidates),
    )

    indices = np.arange(len(units))
    if args.sample_units is not None:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(len(units), size=args.sample_units, replace=False))

    row_range = range(residuals.shape[0])[slice(args.row_start, args.row_end)]
    partial_rows = len(row_range) != residuals.shape[0]

    tag = args.tag or args.var_run
    if args.max_rows is not None:
        tag += f"__rows{args.max_rows}"
    if partial_rows:
        tag += f"__range{row_range.start}-{row_range.stop}"
    if args.sample_units is not None:
        tag += f"__sample{args.sample_units}_seed{args.seed}"

    out_dir = LINGAM_ROOT / tag
    unit_dir = out_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)

    sizes = np.array([len(units[i].nodes) for i in indices])
    order = indices[np.argsort(-sizes, kind="stable")]
    pending = [int(i) for i in order if not (unit_dir / f"unit_{i:05d}.npz").exists()]

    print(f"VAR run        : {args.var_run}, residuals {residuals.shape}, "
          f"rows [{row_range.start}, {row_range.stop})")
    print(f"Units selected : {len(indices)} of {len(units)} "
          f"(sizes min {sizes.min()}, median {np.median(sizes):.0f}, max {sizes.max()})")
    print(f"Pending units  : {len(pending)}  | workers {args.workers}  | output {out_dir}")

    start = time.perf_counter()

    if pending:
        from joblib import Parallel, delayed

        tasks = (
            delayed(run_unit)(
                i, int(units[i].target), np.asarray(units[i].nodes, dtype=np.int64),
                (str(var_dir / "residuals.npy"), row_range.start, row_range.stop),
                args.seed, args.max_rows, str(unit_dir / f"unit_{i:05d}.npz"),
            )
            for i in pending
        )

        for done, record in enumerate(
            Parallel(n_jobs=args.workers, return_as="generator_unordered")(tasks), start=1
        ):
            if record["status"] != "ok" or done % 50 == 0 or done == len(pending):
                print(f"[{done}/{len(pending)}] unit {record['unit_index']} "
                      f"(size {record['size']}): {record['status']} "
                      f"{record['seconds']:.1f}s | elapsed {time.perf_counter() - start:.0f}s "
                      f"{record['message']}", flush=True)

    unit_files = [unit_dir / f"unit_{i:05d}.npz" for i in indices]
    records = []
    for path in unit_files:
        with np.load(path) as unit:
            records.append({
                "unit_index": int(unit["unit_index"]), "size": int(len(unit["nodes"])),
                "status": str(unit["status"]), "seconds": float(unit["seconds"]),
                "rows_used": int(unit["rows_used"]),
                "edges": int(np.count_nonzero(unit["b0"])),
                "message": str(unit["message"]),
            })

    failed = [r for r in records if r["status"] == "failed"]
    seconds = np.array([r["seconds"] for r in records])
    sizes_all = np.array([r["size"] for r in records])

    summary = {
        "var_run": args.var_run, "candidates": candidates, "train_end": meta["train_end"],
        "num_nodes": int(len(sensors)), "seed": args.seed, "max_rows": args.max_rows,
        "units_selected": len(records), "units_total": len(units),
        "sample_units": args.sample_units,
        "rows": [row_range.start, row_range.stop],
        "failed_units": failed,
        "status_counts": {s: sum(r["status"] == s for r in records)
                          for s in sorted({r["status"] for r in records})},
        "cpu_seconds_all_units": float(seconds.sum()),
        "wall_seconds_this_call": time.perf_counter() - start,
        "seconds_vs_size": {
            "size": sizes_all.tolist(), "seconds": seconds.tolist(),
        },
        "local_b0_edges": int(sum(r["edges"] for r in records)),
    }

    print(f"\nUnits ok / failed : {len(records) - len(failed)} / {len(failed)}  "
          f"{summary['status_counts']}")
    print(f"CPU seconds       : {summary['cpu_seconds_all_units']:.0f} "
          f"(mean {seconds.mean():.1f}s per unit, max {seconds.max():.1f}s)")

    if args.sample_units is None:
        if failed and not args.allow_failed:
            print(f"{len(failed)} failed units: aggregation skipped.")
        else:
            b0, filled = aggregate_home_rows(unit_files, len(sensors))
            np.savez(out_dir / "lingam_result.npz", b0=b0, home_row_filled=filled,
                     candidates=candidates,
                     sensors=sensors, train_end=meta["train_end"],
                     num_nodes=len(sensors), var_run=args.var_run)
            summary["aggregated"] = {
                "b0_edges": int(np.count_nonzero(b0)),
                "rows_without_home_unit": int((~filled).sum()),
            }
            print(f"Aggregated B_0    : {summary['aggregated']}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
