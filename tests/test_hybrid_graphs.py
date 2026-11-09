import sys
from pathlib import Path

import numpy as np
import pytest

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

import build_hybrid_candidates as bhc  # noqa: E402
import build_hybrid_graphs as bhg  # noqa: E402


def _toy(n=60, seed=1):
    rng = np.random.default_rng(seed)
    pool = rng.random((n, n)) < 0.4
    np.fill_diagonal(pool, False)
    effect = np.where(rng.random((n, n)) < 0.3, rng.random((n, n)) * 0.2, 0.0)
    np.fill_diagonal(effect, 0.0)
    rho = rng.random((n, n)) * pool
    eta = rng.random((n, n))
    return pool, effect, rho, eta


def test_variant_parser():
    assert bhg.parse_variant("vstd") == ("vstd", 0, None)
    assert bhg.parse_variant("hyb_pc10") == ("pc", 10, None)
    assert bhg.parse_variant("hyb_rand5_M60") == ("rand", 5, 60)
    for bad in ("hyb_pc", "hyb_pc0", "hyb_pc25", "pc10", "hyb_xx3"):
        with pytest.raises(ValueError):
            bhg.parse_variant(bad)


@pytest.mark.parametrize("rule", ["pc", "rand", "corr"])
def test_selection_matches_the_proxy_builder_and_weights_are_valid(rule):
    pool, effect, rho, eta = _toy()
    weights, is_core = bhg.weighted_assemble(effect, pool, rho, eta, 10, rule, np.random.default_rng(0))
    extra = rho if rule == "pc" else eta
    # the proxy builder keeps pool sources whose score is 0, the artifact builder drops them (no defined weight)
    support = bhc.assemble(effect, pool & (rho > 0), extra, 10, rule, np.random.default_rng(0))
    assert np.array_equal(weights > 0, support)
    assert (weights >= 0).all() and not np.diag(weights).any()
    assert ((weights > 0).sum(axis=1) <= bhg.K).all()
    assert not (is_core & ~(effect > 0)).any()                 # core edges have a positive VAR effect
    added = (weights > 0) & ~is_core
    assert not (added & ~pool).any()                          # additions come from the candidate pool
    assert np.allclose(weights[is_core], effect[is_core])    # core weights are the standardised effects


def test_added_weights_are_rescaled_to_the_row_core_median():
    pool, effect, rho, eta = _toy(seed=3)
    weights, is_core = bhg.weighted_assemble(effect, pool, rho, eta, 10, "pc", np.random.default_rng(0))
    added = (weights > 0) & ~is_core
    checked = 0
    for i in range(weights.shape[0]):
        if added[i].any() and is_core[i].any():
            assert np.isclose(np.median(weights[i, added[i]] / rho[i, added[i]]) * np.median(rho[i, added[i]]),
                              np.median(weights[i, is_core[i]]), rtol=1e-9)
            checked += 1
    assert checked > 5


def test_vstd_has_no_additions():
    pool, effect, rho, eta = _toy()
    weights, is_core = bhg.weighted_assemble(effect, pool, rho * 0, eta, 0, "vstd", np.random.default_rng(0))
    assert np.array_equal(weights > 0, is_core)
    for i in range(weights.shape[0]):
        nz = np.flatnonzero(effect[i] > 0)
        top = set(nz[np.argsort(-effect[i, nz])[:bhg.K]])
        assert set(np.flatnonzero(weights[i])) == top


def test_pool_cap_keeps_the_most_correlated():
    pool, _, _, eta = _toy()
    capped = bhc.candidate_pool(pool, eta, 8)
    assert capped.sum(axis=1).max() <= 8 and not (capped & ~pool).any()
    for i in range(pool.shape[0]):
        cols = np.flatnonzero(pool[i])
        if len(cols) > 8:
            kept = np.flatnonzero(capped[i])
            assert eta[i, kept].min() >= np.sort(eta[i, cols])[-8] - 1e-12
