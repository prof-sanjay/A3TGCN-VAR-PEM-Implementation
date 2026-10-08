"""
Equivalence of run_local_pem.py with a direct LocalPEM.fit.

The per-unit runner (local columns only, parallel workers, results
on disk) followed by aggregate() must give exactly the same global
inclusion probabilities P and directed adjacency D as one
LocalPEM.fit over all units on the full residual matrix.

Synthetic residuals with NaNs and overlapping causal units are used.

Run from the implementation directory:

    python test_run_local_pem.py
"""

import tempfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from causal.causal_units import build_causal_units
from causal.local_pem import LocalPEM
from run_local_pem import aggregate, run_unit


def main():

    print("=" * 70)
    print("RUN_LOCAL_PEM EQUIVALENCE TEST")
    print("=" * 70)

    rng = np.random.default_rng(7)

    num_nodes = 14
    n = 3000

    # Sparse directed chain-like topology with some extra links.
    adjacency = np.eye(num_nodes, dtype=np.float32)
    for i in range(num_nodes - 1):
        adjacency[i, i + 1] = 1.0
    for i in range(0, num_nodes - 3, 3):
        adjacency[i + 3, i] = 0.5

    # Residuals with contemporaneous dependence and some NaNs.
    noise = rng.laplace(size=(n, num_nodes))
    B = np.zeros((num_nodes, num_nodes))
    for i in range(1, num_nodes):
        B[i, i - 1] = 0.7
    residuals = noise @ np.linalg.inv(np.eye(num_nodes) - B).T
    residuals[rng.random(residuals.shape) < 0.002] = np.nan

    units = build_causal_units(adjacency)

    pem_kwargs = {
        "nu0": 0.05, "nu1": 1.0, "tau": 0.01, "threshold": 0.5,
        "max_iter": 20000, "tolerance": 1e-6, "prior_inclusion": 0.5,
        "optimizer": "L-BFGS-B",
    }

    # Reference: one LocalPEM.fit over all units (original path).
    reference = LocalPEM(units=units, **{
        k: v for k, v in pem_kwargs.items()
    }).fit(residuals)

    with tempfile.TemporaryDirectory() as tmp:

        files = [str(Path(tmp) / f"unit_{i:05d}.npz")
                 for i in range(len(units))]

        records = Parallel(n_jobs=4)(
            delayed(run_unit)(
                i,
                int(unit.target),
                np.asarray(unit.nodes, dtype=np.int64),
                np.ascontiguousarray(residuals[:, unit.nodes]),
                pem_kwargs,
                files[i],
            )
            for i, unit in enumerate(units)
        )

        statuses = {r["status"] for r in records}

        P, D = aggregate(files, num_nodes)

    same_P = np.array_equal(P, reference.global_inclusion_probabilities)
    same_D = np.array_equal(D, reference.global_adjacency)

    print(f"Units: {len(units)}, sizes "
          f"{sorted(len(u.nodes) for u in units)}")
    print(f"Unit statuses        : {statuses}")
    print(f"P identical          : {same_P}")
    print(f"D identical          : {same_D}")
    print(f"Directed edges in D  : {int(np.count_nonzero(D))}")

    assert statuses == {"ok"}
    assert same_P and same_D

    print("\n[PASS] per-unit runner + aggregation == LocalPEM.fit")


if __name__ == "__main__":
    main()
