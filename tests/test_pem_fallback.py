"""
Tests for PEM's re-solve order after an L-BFGS-B line-search failure
(causal/pem.py, _estimate_precision; amended 2026-10-26):

    1. a warm L-BFGS-B restart that succeeds is used (no SLSQP);
    2. small problems still fall back to SLSQP after the restarts;
    3. large problems are accepted only if the objective stopped
       changing (within the tolerance); counted as accepted_stalled;
    4. large problems whose objective is still changing fail visibly.

scipy's minimize is replaced by a scripted fake, so each path is forced.

Run from the implementation directory:

    python -m tests.test_pem_fallback
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types

import numpy as np
import scipy.optimize

from causal.pem import PEM


def fake_minimize(script):
    """minimize() replacement returning scripted (method, success, fun)."""

    calls = []

    def minimize(fun, x0, jac=True, method=None, constraints=(), options=None):
        method_expected, success, value = script[len(calls)]
        assert method == method_expected, (len(calls), method, method_expected)
        calls.append(method)
        return types.SimpleNamespace(x=np.asarray(x0), success=success, fun=value,
                                     message="scripted")

    return minimize, calls


def run(script, slsqp_max=None):

    rng = np.random.default_rng(0)
    covariance = np.cov(rng.normal(size=(500, 4)), rowvar=False)
    eta = np.full((4, 4), 0.5)
    np.fill_diagonal(eta, 0.0)

    model = PEM(eta=eta, optimizer="L-BFGS-B", tolerance=1e-6)
    if slsqp_max is not None:
        model.SLSQP_MAX_PARAMETERS = slsqp_max

    minimize, calls = fake_minimize(script)
    original = scipy.optimize.minimize
    scipy.optimize.minimize = minimize
    try:
        try:
            model._estimate_precision(covariance, eta, 500)
            error = None
        except RuntimeError as exc:
            error = str(exc)
    finally:
        scipy.optimize.minimize = original

    return model, calls, error


def main():

    print("=" * 70)
    print("PEM RE-SOLVE ORDER TESTS")
    print("=" * 70)

    L, S = "L-BFGS-B", "SLSQP"

    model, calls, error = run([(L, False, 10.0), (L, True, 9.0)])
    assert calls == [L, L] and error is None
    assert (model.lbfgs_restarts, model.slsqp_retries, model.accepted_stalled) == (1, 0, 0)
    print("[PASS] 1. a successful warm restart is used; no SLSQP")

    model, calls, error = run([(L, False, 10.0)] + [(L, False, 9.0)] * 3 + [(S, True, 8.9)])
    assert calls == [L, L, L, L, S] and error is None
    assert (model.lbfgs_restarts, model.slsqp_retries) == (3, 1)
    print("[PASS] 2. small problem: SLSQP after the restarts")

    model, calls, error = run([(L, False, 10.0), (L, False, 9.0), (L, False, 9.0), (L, False, 9.0)],
                              slsqp_max=0)
    assert calls == [L] * 4 and error is None
    assert (model.slsqp_retries, model.accepted_stalled) == (0, 1)
    print("[PASS] 3. large problem: accepted once the objective stopped changing")

    model, calls, error = run([(L, False, 10.0), (L, False, 9.0), (L, False, 8.0), (L, False, 7.0)],
                              slsqp_max=0)
    assert calls == [L] * 4 and error is not None and model.accepted_stalled == 0
    print("[PASS] 4. large problem still decreasing: fails visibly")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
