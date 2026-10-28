"""
Tests for the VAR-LiNGAM graphs (Phase 5, E2 / E4):

    1. B_k = (I - B_0) A_k; E2 / E4 formulas; diagonal 0; E4 with
       P = 1 equals E2, with P = 0 it drops only the B_0 term.
    2. On a synthetic structural VAR with non-Gaussian noise,
       DirectLiNGAM on the reduced-form residuals recovers B_0, and
       (I - B_0_hat) A_k recovers the true lagged structural effects.
    3. run_local_lingam: per-unit results from a memory-mapped residual
       file and the home-unit rule (row i of B_0 only from unit i).

Run from the implementation directory:

    python -m tests.test_var_lingam_graph
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

import numpy as np

from causal.causal_units import build_causal_units
from causal.graph_builders import (
    structural_lag_matrices, var_lingam_graph, var_lingam_pem_graph, var_pem_mask_graph,
)
from causal.var_lingam import VARLiNGAM
from run_local_lingam import aggregate_home_rows, run_unit


def test_formulas():

    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 4, 4))
    B0 = np.zeros((4, 4)); B0[1, 0] = 0.5; B0[3, 2] = -0.4

    Bk = structural_lag_matrices(A, B0)
    for k in range(2):
        assert np.allclose(Bk[k], (np.eye(4) - B0) @ A[k])

    e2 = var_lingam_graph(A, B0)
    expected = np.abs(Bk).sum(axis=0) + np.abs(B0)
    np.fill_diagonal(expected, 0)
    assert np.allclose(e2, expected, atol=1e-6) and np.all(np.diag(e2) == 0)

    assert np.allclose(var_lingam_pem_graph(A, B0, np.ones((4, 4))), e2, atol=1e-6)

    no_b0 = np.abs(Bk).sum(axis=0); np.fill_diagonal(no_b0, 0)
    assert np.allclose(var_lingam_pem_graph(A, B0, np.zeros((4, 4))), no_b0, atol=1e-6)

    # E3 (amended): VAR weights where D has an edge, 0 elsewhere and on the diagonal.
    D = np.zeros((4, 4)); D[1, 0] = D[2, 3] = D[0, 0] = 1
    e3 = var_pem_mask_graph(A, D)
    w = np.abs(A).sum(axis=0)
    assert np.isclose(e3[1, 0], w[1, 0]) and np.isclose(e3[2, 3], w[2, 3])
    assert np.count_nonzero(e3) == 2

    print("[PASS] 1. B_k = (I - B_0) A_k, E2 / E3 / E4 formulas, diagonal 0")


def simulate(n=20000, seed=1):
    """Structural VAR(1): x = B0 x + B1 x(t-1) + n, Laplace noise."""

    rng = np.random.default_rng(seed)
    B0 = np.zeros((3, 3)); B0[1, 0] = 0.8; B0[2, 1] = -0.6        # 0 -> 1 -> 2
    B1 = np.diag([0.5, 0.4, 0.3]); B1[2, 0] = 0.2
    inv = np.linalg.inv(np.eye(3) - B0)
    A1 = inv @ B1                                                  # reduced form

    x = np.zeros((n, 3))
    noise = rng.laplace(size=(n, 3))
    for t in range(1, n):
        x[t] = A1 @ x[t - 1] + inv @ noise[t]

    return x, A1, B0, B1


def test_recovery():

    x, A1, B0, B1 = simulate()

    # Reduced-form VAR(1) by least squares, then residuals.
    X, Y = x[:-1], x[1:]
    A1_hat = np.linalg.lstsq(X, Y, rcond=None)[0].T
    residuals = Y - X @ A1_hat.T

    result = VARLiNGAM(random_state=0).fit(residuals)
    B0_hat = result.adjacency_matrix

    assert result.causal_order == [0, 1, 2], result.causal_order
    assert np.allclose(B0_hat, B0, atol=0.05), B0_hat

    B1_hat = structural_lag_matrices(A1_hat[None], B0_hat)[0]
    assert np.allclose(B1_hat, B1, atol=0.05), B1_hat

    print("[PASS] 2. synthetic SVAR: B_0, causal order and B_1 recovered")


def test_runner():

    x, *_ = simulate(n=6000, seed=3)
    residuals = np.concatenate([x, x[:, :1] * 0.5 + np.random.default_rng(4).laplace(size=(len(x), 1))], axis=1)
    residuals[5, 2] = np.nan                     # one incomplete row

    adjacency = np.eye(4, dtype=np.float32)
    adjacency[1, 0] = adjacency[2, 1] = adjacency[3, 0] = 1.0
    units = build_causal_units(adjacency)

    with tempfile.TemporaryDirectory() as tmp:

        path = str(Path(tmp) / "residuals.npy")
        np.save(path, residuals.astype(np.float32))

        files = []
        for i, unit in enumerate(units):
            f = str(Path(tmp) / f"unit_{i:05d}.npz")
            record = run_unit(i, int(unit.target), np.asarray(unit.nodes, dtype=np.int64),
                              (path, 0, len(residuals)), 0, None, f)
            assert record["status"] == "ok", record
            files.append(f)

        b0, filled = aggregate_home_rows(files, 4)

        for unit, f in zip(units, files):
            with np.load(f) as u:
                nodes, target = u["nodes"], int(u["target"])
                local = int(np.flatnonzero(nodes == target)[0])
                expected = np.zeros(4); expected[nodes] = u["b0"][local]; expected[target] = 0
                assert np.allclose(b0[target], expected), target
                assert int(u["rows_used"]) == len(residuals) - (1 if 2 in nodes else 0)

    assert filled.all() and np.all(np.diag(b0) == 0)
    print("[PASS] 3. runner: memory-mapped unit columns, complete rows, home-unit rows")


def main():

    print("=" * 70)
    print("VAR-LiNGAM GRAPH TESTS")
    print("=" * 70)

    test_formulas()
    test_recovery()
    test_runner()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
