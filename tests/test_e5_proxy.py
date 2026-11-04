"""Tests of the E5 tuning proxy (tuning/e5_proxy.py) on tiny synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import scipy.sparse as sp

from causal.graph_builders import horizon_graphs, top_k_sources
from causal.horizon import StructuralIRF
from tuning.e5_proxy import ProxyData, effect_matrices, operator, own_signal_share


def test_operator_equals_a3tgcn_normalisation():
    """A_hat must equal what A3T-GCN builds from the model-oriented graph."""

    from graph.normalization import normalized_adj
    from causal.graph_builders import to_model_orientation

    rng = np.random.default_rng(0)
    g = rng.random((9, 9)) * (rng.random((9, 9)) > 0.6)
    np.fill_diagonal(g, 0.0)

    mine = operator(g).toarray()
    model_input = to_model_orientation(g)
    theirs = normalized_adj(sp.csr_matrix(model_input) + sp.eye(9)).toarray()

    assert np.allclose(mine, theirs, atol=1e-6)


def test_identity_graph_keeps_all_own_signal():
    assert own_signal_share(operator(np.zeros((5, 5)))) == pytest.approx(1.0)


def test_windows_skip_missing_rows_and_stay_in_split():
    traffic = np.ones((60, 3), dtype=np.float32)
    traffic[20, 1] = np.nan

    data = ProxyData(traffic, train_end=40, val_end=60, scale=1.0, lags=4)
    ends = data.window_ends("train", horizon=2, stride=1)

    assert ends.min() == 3
    assert ends.max() == 40 - 2 - 1                    # the target stays inside train
    assert not set(range(20, 24)) & set(ends.tolist())  # windows holding row 20
    assert {19, 24} <= set(ends.tolist())

    val_ends = data.window_ends("val", horizon=2, stride=1)
    assert val_ends.min() == 40 + 3
    assert val_ends.max() == 60 - 2 - 1


def test_test_rows_are_refused():
    with pytest.raises(ValueError):
        ProxyData(np.ones((100, 3), dtype=np.float32), train_end=40, val_end=60, scale=1.0)


def _chain_series(rows=3000, nodes=8, seed=1):
    """x_i(t+1) = 0.2 x_i(t) + 0.7 x_{i-1}(t) + noise: the true source of i is i-1."""

    rng = np.random.default_rng(seed)
    x = np.zeros((rows, nodes))
    for t in range(rows - 1):
        shifted = np.concatenate([[0.0], x[t, :-1]])
        x[t + 1] = 0.2 * x[t] + 0.7 * shifted + rng.normal(size=nodes)
    return x.astype(np.float32)


def test_proxy_prefers_the_true_graph():
    traffic = _chain_series()
    data = ProxyData(traffic[:2500], train_end=2000, val_end=2500, scale=1.0, lags=3)

    n = traffic.shape[1]
    true = np.zeros((n, n)); true[np.arange(1, n), np.arange(0, n - 1)] = 1.0     # G[i, i-1]
    wrong = np.zeros((n, n)); wrong[np.arange(0, n - 1), np.arange(1, n)] = 1.0   # G[i, i+1]
    none = np.zeros((n, n))

    score = {name: data.evaluate(g, [1], stride_train=1, stride_val=1)[1]["rmse"]
             for name, g in (("true", true), ("wrong", wrong), ("none", none))}

    assert score["true"] < score["none"]
    assert score["true"] < score["wrong"]


def test_score_is_deterministic_and_uses_validation_only():
    traffic = _chain_series(rows=1500)
    data = ProxyData(traffic[:1200], train_end=900, val_end=1200, scale=1.0, lags=3)
    g = np.zeros((8, 8)); g[np.arange(1, 8), np.arange(0, 7)] = 1.0

    a = data.evaluate(g, [1])
    b = data.evaluate(g, [1])
    assert a[1]["rmse"] == b[1]["rmse"]

    # Changing a TEST-period row does not exist for the proxy (rows >= val_end are refused),
    # but changing a validation row must change the validation score.
    changed = traffic[:1200].copy()
    changed[1100] += 50.0
    other = ProxyData(changed, train_end=900, val_end=1200, scale=1.0, lags=3)
    assert other.evaluate(g, [1])[1]["rmse"] != a[1]["rmse"]


def _random_var(n=12, seed=3):
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(scale=0.15, size=(3, n, n))
    b0 = np.tril(rng.normal(scale=0.3, size=(n, n)), k=-1)
    p = rng.random((n, n)); p = (p + p.T) / 2
    return coefficients, b0, p


def test_effect_matrices_reproduce_the_e5_builder():
    coefficients, b0, p = _random_var()
    horizons = [1, 3, 6]

    psi = StructuralIRF()._calculate_psi(coefficients, max(horizons))
    effects = effect_matrices(psi, p * b0, horizons)

    expected = horizon_graphs(coefficients, b0, p, horizons, k=4)
    for h in horizons:
        assert np.allclose(top_k_sources(effects[h], 4), expected[h], atol=1e-6)


def test_without_b0_the_effects_are_the_var_responses():
    coefficients, _, _ = _random_var()
    psi = StructuralIRF()._calculate_psi(coefficients, 3)

    effects = effect_matrices(psi, np.zeros((12, 12)), [1])
    expected = np.abs(coefficients[0]) * (1.0 - np.eye(12))

    assert np.allclose(effects[1], expected)


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
