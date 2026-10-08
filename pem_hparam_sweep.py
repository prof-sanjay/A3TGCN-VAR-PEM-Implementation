"""
Phase 5, Stage A: PEM hyperparameter sweep on a sample of causal units.

Uses TRAINING residuals only (no validation or test data):

    for each (nu0, nu1, prior_inclusion, threshold) setting:
        run_local_pem.py on the same sampled units with
            (a) all training rows
            (b) first half of the training rows
            (c) second half of the training rows
        and report
            - local directed edge density and edges per unit
            - failed units, CPU seconds
            - split-half stability: Jaccard overlap of the directed
              edges found on (b) and (c)

Results: outputs/pem/sweep_stage_a.json and a printed table.

Run from the implementation directory (traffic-causal environment):

    python pem_hparam_sweep.py --var-run "lag3_alpha1e+06"
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


IMPLEMENTATION_DIR = Path(__file__).resolve().parent

PEM_ROOT = IMPLEMENTATION_DIR / "outputs" / "pem"


def run(var_run, setting, sample_units, seed, workers, rows=None):

    command = [
        sys.executable, str(IMPLEMENTATION_DIR / "run_local_pem.py"),
        "--var-run", var_run,
        "--nu0", f"{setting['nu0']:g}",
        "--nu1", f"{setting['nu1']:g}",
        "--prior-inclusion", f"{setting['prior_inclusion']:g}",
        "--threshold", f"{setting['threshold']:g}",
        "--sample-units", str(sample_units),
        "--seed", str(seed),
        "--workers", str(workers),
    ]

    if rows is not None:
        command += ["--row-start", str(rows[0]), "--row-end", str(rows[1])]

    output = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=IMPLEMENTATION_DIR,
    )

    if output.returncode != 0:
        raise RuntimeError(output.stdout[-2000:] + output.stderr[-2000:])

    saved = [
        line for line in output.stdout.splitlines()
        if line.startswith("Saved: ")
    ][-1]

    return Path(saved[len("Saved: "):].strip())


def directed_edges(run_dir):
    """Set of global directed edges (source, target) over all units."""

    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    edges = set()
    sizes = []
    edge_counts = []
    seconds = 0.0

    for path in sorted((run_dir / "units").glob("unit_*.npz")):

        with np.load(path) as unit:

            nodes = unit["nodes"]
            adjacency = unit["adjacency"]

            targets, sources = np.nonzero(adjacency > 0)

            edges.update(
                zip(nodes[sources].tolist(), nodes[targets].tolist())
            )

            sizes.append(len(nodes))
            edge_counts.append(len(targets))
            seconds += float(unit["seconds"])

    return edges, summary, np.array(sizes), np.array(edge_counts), seconds


def jaccard(a, b):

    if not a and not b:
        return float("nan")

    return len(a & b) / len(a | b)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--var-run", required=True)
    parser.add_argument("--sample-units", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--nu0", type=float, nargs="+",
                        default=[3e-5, 1e-4, 3e-4])
    parser.add_argument("--nu1-ratio", type=float, nargs="+",
                        default=[20.0, 100.0])
    parser.add_argument("--prior-inclusion", type=float, nargs="+",
                        default=[0.5])
    parser.add_argument("--threshold", type=float, nargs="+",
                        default=[0.5])
    parser.add_argument("--train-rows", type=int, default=84_096)
    args = parser.parse_args()

    half = args.train_rows // 2

    results = []

    print(f"{'nu0':>8} {'nu1':>8} {'eta':>4} {'thr':>4} | {'density':>8} "
          f"{'edges/unit':>10} {'fail':>4} {'cpu_s':>6} | "
          f"{'split-half J':>12} {'edges h1':>8} {'edges h2':>8}",
          flush=True)

    for nu0, ratio, eta, threshold in itertools.product(
        args.nu0, args.nu1_ratio, args.prior_inclusion, args.threshold
    ):

        setting = {
            "nu0": nu0,
            "nu1": nu0 * ratio,
            "prior_inclusion": eta,
            "threshold": threshold,
        }

        full_dir = run(args.var_run, setting, args.sample_units,
                       args.seed, args.workers)
        h1_dir = run(args.var_run, setting, args.sample_units,
                     args.seed, args.workers, rows=(0, half))
        h2_dir = run(args.var_run, setting, args.sample_units,
                     args.seed, args.workers, rows=(half, args.train_rows))

        edges, summary, sizes, counts, seconds = directed_edges(full_dir)
        edges_h1, *_ = directed_edges(h1_dir)
        edges_h2, *_ = directed_edges(h2_dir)

        record = {
            **setting,
            "units": int(len(sizes)),
            "failed_units": len(summary["failed_units"]),
            "local_density": summary["local_edge_density"],
            "edges_per_unit_mean": float(counts.mean()),
            "edges_per_unit_median": float(np.median(counts)),
            "cpu_seconds": seconds,
            "split_half_jaccard": jaccard(edges_h1, edges_h2),
            "edges_half1": len(edges_h1),
            "edges_half2": len(edges_h2),
            "edges_full": len(edges),
            "run_dir": str(full_dir),
        }

        results.append(record)

        print(f"{nu0:8.0e} {setting['nu1']:8.0e} {eta:4.2f} {threshold:4.2f} | "
              f"{record['local_density']:8.4f} "
              f"{record['edges_per_unit_mean']:10.1f} "
              f"{record['failed_units']:4d} {seconds:6.0f} | "
              f"{record['split_half_jaccard']:12.3f} "
              f"{record['edges_half1']:8d} {record['edges_half2']:8d}",
              flush=True)

        with open(PEM_ROOT / "sweep_stage_a.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved: {PEM_ROOT / 'sweep_stage_a.json'}")


if __name__ == "__main__":
    main()
