"""
Validation of the computational PEM speed-up (decision D3).

Compares causal/pem.py against the ORIGINAL implementation from the
baseline commit (c55c1a4, written to a temporary module):

    1. _vector_to_precision: identical matrices.
    2. _map_objective: identical values (vectorized prior), including
       hard exclusion (eta = 0) and hard inclusion (eta = 1) entries.
    3. Exact gradient vs central finite differences.
    4. Full PEM fits on synthetic DAG data: same adjacency, same causal
       order, inclusion probabilities and precision within tolerance,
       plus runtime of both versions.

Run from the implementation directory:

    python test_pem_speedup.py
"""

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from causal.pem import PEM


BASELINE_COMMIT = "c55c1a4"


def load_baseline_pem():

    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:implementation/causal/pem.py"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent,
    ).stdout

    path = Path(tempfile.gettempdir()) / "pem_baseline_c55c1a4.py"
    path.write_text(source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("pem_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pem_baseline"] = module
    spec.loader.exec_module(module)

    return module.PEM


def random_eta(rng, p):

    eta = rng.uniform(0.05, 0.95, (p, p))
    eta = (eta + eta.T) / 2.0
    eta[0, 1] = eta[1, 0] = 0.0      # hard exclusion
    eta[1, 2] = eta[2, 1] = 1.0      # hard inclusion
    np.fill_diagonal(eta, 0.0)

    return eta


def test_objective_and_precision(BaselinePEM):

    print("\n1-2. PRECISION MAPPING AND OBJECTIVE")

    rng = np.random.default_rng(0)

    worst_precision = 0.0
    worst_objective = 0.0

    for trial in range(50):

        p = int(rng.integers(3, 15))

        eta = random_eta(rng, p)

        X = rng.laplace(size=(500, p))
        S = np.cov(X, rowvar=False, bias=True)

        parameters = rng.normal(0, 0.5, p * (p + 1) // 2)

        new, old = PEM(eta), BaselinePEM(eta)

        P_new = new._vector_to_precision(parameters, p)
        P_old = old._vector_to_precision(parameters, p)

        worst_precision = max(worst_precision, np.abs(P_new - P_old).max())

        f_new = new._map_objective(P_new, S, eta, 500)
        f_old = old._map_objective(P_old, S, eta, 500)

        worst_objective = max(
            worst_objective,
            abs(f_new - f_old) / max(1.0, abs(f_old)),
        )

    assert worst_precision == 0.0, worst_precision
    assert worst_objective < 1e-12, worst_objective

    print(f"  [PASS] precision identical (max |diff| {worst_precision})")
    print(f"  [PASS] objective equal (max rel diff {worst_objective:.1e}) "
          "over 50 random cases incl. eta = 0 and eta = 1")


def test_gradient():

    print("\n3. EXACT GRADIENT vs FINITE DIFFERENCES")

    rng = np.random.default_rng(1)

    worst = 0.0

    for trial in range(20):

        p = int(rng.integers(3, 10))

        eta = random_eta(rng, p)

        X = rng.laplace(size=(2000, p))
        S = np.cov(X, rowvar=False, bias=True)

        model = PEM(eta)

        theta = rng.normal(0, 0.5, p * (p + 1) // 2)

        _, gradient = model._objective_and_gradient(theta, S, eta, 2000, p)

        h = 1e-6
        numeric = np.empty_like(theta)

        for i in range(len(theta)):
            e = np.zeros_like(theta)
            e[i] = h
            f_plus = model._objective_and_gradient(theta + e, S, eta, 2000, p)[0]
            f_minus = model._objective_and_gradient(theta - e, S, eta, 2000, p)[0]
            numeric[i] = (f_plus - f_minus) / (2 * h)

        error = np.abs(gradient - numeric).max() / max(1.0, np.abs(numeric).max())

        worst = max(worst, error)

    assert worst < 1e-5, worst

    print(f"  [PASS] max relative error {worst:.1e} over 20 random cases")


def generate_dag_data(rng, p, n):

    B = np.zeros((p, p))

    for target in range(1, p):
        for source in range(target):
            if rng.random() < 0.3:
                B[target, source] = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])

    noise = rng.laplace(size=(n, p))

    return noise @ np.linalg.inv(np.eye(p) - B).T


def test_full_fits(BaselinePEM):

    print("\n4. FULL PEM FITS: ORIGINAL vs SPEED-UP")

    rng = np.random.default_rng(2)

    print(f"  {'p':>3s} {'n':>6s} {'adjacency':>10s} {'order':>6s} "
          f"{'max|dP|':>9s} {'max|dOmega|rel':>15s} "
          f"{'t_orig':>8s} {'t_new':>7s} {'speed-up':>8s}")

    for p, n in [(4, 3000), (6, 3000), (8, 5000), (10, 5000), (12, 5000)]:

        X = generate_dag_data(rng, p, n)

        eta = np.full((p, p), 0.5)
        np.fill_diagonal(eta, 0.0)

        start = time.perf_counter()
        old = BaselinePEM(eta).fit(X)
        t_old = time.perf_counter() - start

        start = time.perf_counter()
        new = PEM(eta).fit(X)
        t_new = time.perf_counter() - start

        same_adjacency = np.array_equal(old.adjacency_matrix, new.adjacency_matrix)
        same_order = old.causal_order == new.causal_order

        d_prob = np.abs(old.inclusion_probabilities - new.inclusion_probabilities).max()
        d_prec = (
            np.abs(old.precision_matrix - new.precision_matrix).max()
            / np.abs(old.precision_matrix).max()
        )

        print(f"  {p:3d} {n:6d} {str(same_adjacency):>10s} {str(same_order):>6s} "
              f"{d_prob:9.2e} {d_prec:15.2e} "
              f"{t_old:7.2f}s {t_new:6.2f}s {t_old / t_new:7.1f}x")

        assert same_adjacency and same_order
        assert d_prob < 1e-3 and d_prec < 1e-3

    print("  [PASS] identical graphs and causal orders; probabilities "
          "and precision within tolerance")


def test_lbfgsb_vs_slsqp():
    """
    Decision D4: the same PEM-MAP objective minimised with L-BFGS-B
    (same objective-value accuracy as SLSQP) must reach an objective no
    worse than SLSQP (1e-8 relative) with the same graph and order.
    """

    print("\n5. OPTIMIZER: L-BFGS-B vs SLSQP, same objective")

    rng = np.random.default_rng(2)

    print(f"  {'p':>3s} {'adjacency':>10s} {'order':>6s} {'max|dP|':>9s} "
          f"{'obj SLSQP':>14s} {'obj L-BFGS-B':>14s} "
          f"{'t SLSQP':>8s} {'t L-BFGS-B':>10s}")

    for p in (6, 10, 16, 24):

        X = generate_dag_data(rng, p, 20000)

        eta = np.full((p, p), 0.5)
        np.fill_diagonal(eta, 0.0)

        slsqp = PEM(eta)
        start = time.perf_counter()
        a = slsqp.fit(X)
        t_a = time.perf_counter() - start

        start = time.perf_counter()
        b = PEM(eta, optimizer="L-BFGS-B",
                max_iter=20000).fit(X)
        t_b = time.perf_counter() - start

        S = slsqp._make_positive_definite(slsqp._sample_covariance(X))
        obj_a = slsqp._map_objective(a.precision_matrix, S, eta, len(X))
        obj_b = slsqp._map_objective(b.precision_matrix, S, eta, len(X))

        same_adjacency = np.array_equal(a.adjacency_matrix, b.adjacency_matrix)
        same_order = a.causal_order == b.causal_order
        d_prob = np.abs(a.inclusion_probabilities - b.inclusion_probabilities).max()

        print(f"  {p:3d} {str(same_adjacency):>10s} {str(same_order):>6s} "
              f"{d_prob:9.1e} {obj_a:14.4f} {obj_b:14.4f} "
              f"{t_a:7.2f}s {t_b:9.2f}s")

        assert same_adjacency and same_order
        assert obj_b <= obj_a + 1e-8 * abs(obj_a)
        assert d_prob < 1e-2

    print("  [PASS] same graphs and orders; L-BFGS-B objective no worse than SLSQP (1e-8 rel.)")


def main():

    print("=" * 70)
    print("PEM SPEED-UP VALIDATION (vs baseline commit "
          f"{BASELINE_COMMIT})")
    print("=" * 70)

    BaselinePEM = load_baseline_pem()

    test_objective_and_precision(BaselinePEM)
    test_gradient()
    test_full_fits(BaselinePEM)
    test_lbfgsb_vs_slsqp()

    print("\n" + "=" * 70)
    print("ALL PEM SPEED-UP TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
