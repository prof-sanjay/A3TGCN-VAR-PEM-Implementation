import sys
from pathlib import Path

import numpy as np
import pytest

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

import build_e9_graphs as b9  # noqa: E402
import run_knockoff_selection as rks  # noqa: E402


def _toy(n=60, seed=1):
    rng = np.random.default_rng(seed)
    effect = np.where(rng.random((n, n)) < 0.3, rng.random((n, n)) * 0.2, 0.0)
    np.fill_diagonal(effect, 0.0)
    selected = rng.random((n, n)) < 0.15
    selected[5] = False                                    # a target with no selected source
    W = rng.random((n, n)) * 0.05
    return effect, selected, W


def test_variant_parser():
    assert b9.parse_variant("e9_g50_q30") == ("g50", 30)
    assert b9.parse_variant("e9_m30_q20") == ("m30", 20)
    for bad in ("e9_q30", "hyb_pc10", "e9_g50", "e9_x50_q30"):
        with pytest.raises(ValueError):
            b9.parse_variant(bad)


def test_weights_exactly_on_the_selection_and_valid():
    effect, selected, W = _toy()
    weights, in_var = b9.weigh_selection(effect, selected, W)
    want = selected & ((effect > 0) | (W > 0))
    np.fill_diagonal(want, False)
    assert np.array_equal(weights > 0, want)
    assert (weights >= 0).all() and not np.diag(weights).any()
    assert not (weights > 0)[5].any()                       # an empty row stays empty
    assert np.array_equal(in_var, selected & (effect > 0))


def test_var_supported_edges_keep_the_standardised_effect():
    effect, selected, W = _toy(seed=2)
    weights, in_var = b9.weigh_selection(effect, selected, W)
    assert np.allclose(weights[in_var], effect[in_var])


def test_added_weights_are_proportional_to_rho_with_one_global_constant():
    effect, selected, W = _toy(seed=3)
    weights, in_var = b9.weigh_selection(effect, selected, W)
    added = (weights > 0) & ~in_var
    ratio = weights[added] / W[added]
    assert np.allclose(ratio, ratio[0])                                     # one kappa for the whole graph
    assert np.isclose(np.median(weights[added]), np.median(weights[in_var]), rtol=1e-9)


def test_no_row_can_blow_up_when_rho_spans_orders_of_magnitude():
    rng = np.random.default_rng(5)
    n = 40
    effect = np.where(rng.random((n, n)) < 0.3, rng.random((n, n)) * 0.2, 0.0)
    np.fill_diagonal(effect, 0.0)
    selected = rng.random((n, n)) < 0.3
    W = 10.0 ** rng.uniform(-6, 0, (n, n))                                  # rho from 1e-6 to 1
    weights, in_var = b9.weigh_selection(effect, selected, W)
    assert weights.max() <= max(effect.max(), np.median(effect[in_var]) / np.median(W[(weights > 0) & ~in_var]))


def test_knockoff_threshold_controls_the_estimated_fdp():
    rng = np.random.default_rng(0)
    W = np.concatenate([rng.normal(2.0, 0.5, 40), rng.normal(0.0, 0.5, 200)])
    for q in (0.1, 0.2, 0.3):
        t = rks.knockoff_threshold(W, q)
        if np.isfinite(t):
            assert (1 + (W <= -t).sum()) / max((W >= t).sum(), 1) <= q + 1e-12
    assert rks.knockoff_threshold(np.zeros(10), 0.2) == np.inf


def test_group_knockoff_gamma_is_one_group_per_source_by_default():
    rng = np.random.default_rng(1)
    sigma = np.corrcoef(rng.standard_normal((500, 8)), rowvar=False)
    assert np.array_equal(rks.make_groups(sigma, None), np.arange(8))
    block = np.kron(np.eye(2), np.full((4, 4), 0.9)) + 0.1 * np.eye(8)
    groups = rks.make_groups(block, 0.5)
    assert len(np.unique(groups)) == 2 and groups[0] == groups[3] and groups[0] != groups[4]
