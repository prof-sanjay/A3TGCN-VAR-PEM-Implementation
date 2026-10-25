"""
Tests for the informative road prior and residual standardisation
used by run_local_pem.py (--prior road, --standardise).

    1. road_prior_eta: symmetric eta from an ASYMMETRIC road graph
       (eta = high if a link exists in either direction, else low).
    2. LocalPEM with an explicit uniform eta matrix == the original
       uniform prior_inclusion path (no regression).
    3. Per-unit runner with the road prior + aggregation ==
       LocalPEM.fit(eta=global road prior), and the prior is used.
    4. residual_scaling: training mean/std (NaN-aware); standardised
       residuals have mean 0 / std 1; the runner with scaling (also on
       a row subset read from a memory-mapped file) == LocalPEM.fit on
       residuals standardised with the FULL-training parameters.

Run from the implementation directory:

    python -m tests.test_pem_road_prior
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from causal.causal_units import build_causal_units
from causal.local_pem import LocalPEM
from causal.pem_prior import road_prior_eta
from run_local_pem import aggregate, residual_scaling, run_unit


PEM_KWARGS = {
    "nu0": 0.05, "nu1": 1.0, "tau": 0.01, "threshold": 0.5,
    "max_iter": 20000, "tolerance": 1e-6, "prior_inclusion": 0.5,
    "optimizer": "L-BFGS-B",
}


def synthetic(rng, num_nodes=12, n=3000):
    """Directed road-like topology and scaled, dependent residuals."""

    adjacency = np.eye(num_nodes, dtype=np.float32)
    for i in range(num_nodes - 1):
        adjacency[i, i + 1] = 1.0            # one-way links
    for i in range(0, num_nodes - 3, 3):
        adjacency[i + 3, i] = 0.5
        adjacency[i, i + 3] = 0.5            # two-way links

    noise = rng.laplace(size=(n, num_nodes))
    B = np.zeros((num_nodes, num_nodes))
    for i in range(1, num_nodes):
        B[i, i - 1] = 0.7
    residuals = noise @ np.linalg.inv(np.eye(num_nodes) - B).T

    # Very different sensor scales, like raw traffic residuals.
    residuals = residuals * np.linspace(4.0, 60.0, num_nodes) + 3.0
    residuals[rng.random(residuals.shape) < 0.002] = np.nan

    return adjacency, residuals


def run_units(units, residual_source, eta, scaling, num_nodes, tmp):
    """Run every unit through run_unit (as run_local_pem.main does)."""

    files = [str(Path(tmp) / f"unit_{i:05d}.npz") for i in range(len(units))]

    def task(i, unit):
        nodes = np.asarray(unit.nodes, dtype=np.int64)
        kwargs = PEM_KWARGS if eta is None else {
            **PEM_KWARGS, "eta": eta[np.ix_(nodes, nodes)]
        }
        local = (
            residual_source if isinstance(residual_source, tuple)
            else np.ascontiguousarray(residual_source[:, nodes])
        )
        return run_unit(
            i, int(unit.target), nodes, local, kwargs, files[i],
            None if scaling is None else (scaling[0][nodes], scaling[1][nodes]),
        )

    records = Parallel(n_jobs=4)(
        delayed(task)(i, unit) for i, unit in enumerate(units)
    )

    assert {r["status"] for r in records} == {"ok"}, records

    return aggregate(files, num_nodes)


def test_road_prior_eta():

    A = np.zeros((4, 4))
    A[0, 0] = 1.0            # self-loop: ignored
    A[1, 0] = 1.0            # 0 -> 1 only (one-way)
    A[2, 3] = A[3, 2] = 2.0  # two-way, any non-zero weight

    eta = road_prior_eta(A, high=0.7, low=0.3)

    expected = np.full((4, 4), 0.3)
    expected[0, 1] = expected[1, 0] = 0.7
    expected[2, 3] = expected[3, 2] = 0.7
    np.fill_diagonal(expected, 0.0)

    assert np.array_equal(eta, expected), eta
    assert np.array_equal(eta, eta.T)

    for high, low in [(0.3, 0.7), (1.0, 0.3), (0.7, 0.0)]:
        try:
            road_prior_eta(A, high=high, low=low)
        except ValueError:
            continue
        raise AssertionError(f"accepted high={high}, low={low}")

    print("[PASS] 1. road_prior_eta: symmetric max rule, diagonal 0, "
          "one-way link -> 0.7 both ways")


def test_uniform_eta_matches_original(adjacency, residuals):

    units = build_causal_units(adjacency)
    n = residuals.shape[1]

    original = LocalPEM(units=units, **PEM_KWARGS).fit(residuals)

    uniform = np.full((n, n), 0.5)
    np.fill_diagonal(uniform, 0.0)
    explicit = LocalPEM(units=units, eta=uniform, **PEM_KWARGS).fit(residuals)

    assert np.array_equal(original.global_inclusion_probabilities,
                          explicit.global_inclusion_probabilities)
    assert np.array_equal(original.global_adjacency, explicit.global_adjacency)

    print("[PASS] 2. explicit uniform eta == original prior_inclusion path")


def test_road_prior_runner(adjacency, residuals):

    units = build_causal_units(adjacency)
    eta = road_prior_eta(adjacency)

    reference = LocalPEM(units=units, eta=eta, **PEM_KWARGS).fit(residuals)
    uniform = LocalPEM(units=units, **PEM_KWARGS).fit(residuals)

    with tempfile.TemporaryDirectory() as tmp:
        P, D = run_units(units, residuals, eta, None, residuals.shape[1], tmp)

    assert np.array_equal(P, reference.global_inclusion_probabilities)
    assert np.array_equal(D, reference.global_adjacency)

    changed = not np.array_equal(
        reference.global_inclusion_probabilities,
        uniform.global_inclusion_probabilities,
    )
    assert changed, "road prior had no effect"

    print("[PASS] 3. runner + aggregation with road prior == "
          "LocalPEM.fit(eta=road prior); differs from uniform prior")


def test_standardisation(adjacency, residuals):

    mean, std = residual_scaling(residuals, chunk_rows=700)

    assert np.allclose(mean, np.nanmean(residuals, axis=0), rtol=1e-12, atol=1e-10)
    assert np.allclose(std, np.nanstd(residuals, axis=0), rtol=1e-12)

    z = (residuals - mean) / std
    assert np.allclose(np.nanmean(z, axis=0), 0.0, atol=1e-10)
    assert np.allclose(np.nanstd(z, axis=0), 1.0, atol=1e-10)
    assert np.array_equal(np.isnan(z), np.isnan(residuals))

    units = build_causal_units(adjacency)

    # Full rows, residuals passed in memory.
    reference = LocalPEM(units=units, **PEM_KWARGS).fit(z)

    with tempfile.TemporaryDirectory() as tmp:
        P, D = run_units(units, residuals, None, (mean, std), residuals.shape[1], tmp)

    assert np.array_equal(P, reference.global_inclusion_probabilities)
    assert np.array_equal(D, reference.global_adjacency)

    # Row subset from a memory-mapped file: still the FULL-training
    # parameters (as for split-half runs).
    with tempfile.TemporaryDirectory() as tmp:

        path = str(Path(tmp) / "residuals.npy")
        np.save(path, residuals.astype(np.float32))
        stored = np.load(path).astype(np.float64)
        mean32, std32 = residual_scaling(np.load(path, mmap_mode="r"))

        rows = (500, 2200)
        z_rows = ((stored - mean32) / std32)[rows[0]:rows[1]]
        reference_rows = LocalPEM(units=units, **PEM_KWARGS).fit(z_rows)

        P, D = run_units(units, (path, rows[0], rows[1]), None,
                         (mean32, std32), residuals.shape[1], tmp)

    assert np.allclose(P, reference_rows.global_inclusion_probabilities,
                       rtol=0, atol=1e-9)
    assert np.array_equal(D, reference_rows.global_adjacency)

    print("[PASS] 4. residual_scaling (NaN-aware) -> mean 0 / std 1; runner "
          "with scaling == LocalPEM.fit on standardised residuals "
          "(full rows and memory-mapped row subset)")


def main():

    print("=" * 70)
    print("PEM ROAD PRIOR + RESIDUAL STANDARDISATION TESTS")
    print("=" * 70)

    adjacency, residuals = synthetic(np.random.default_rng(11))

    test_road_prior_eta()
    test_uniform_eta_matches_original(adjacency, residuals)
    test_road_prior_runner(adjacency, residuals)
    test_standardisation(adjacency, residuals)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
