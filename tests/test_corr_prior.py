"""
Tests of the correlation-based PEM prior (causal/corr_prior.py).

    python -m pytest tests/test_corr_prior.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.stats import spearmanr

from causal.corr_prior import partial_matrix, pearson_matrix, percentile_eta, spearman_matrix


def _data(rows=3000, n=8, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(rows, n))
    z[:, 1] += 0.8 * z[:, 0]                 # 0 and 1 correlated
    z[:, 2] += 0.5 * z[:, 1]                 # chain 0 -> 1 -> 2
    return z


def test_pearson_equals_numpy_on_complete_data():
    x = _data()
    assert np.allclose(pearson_matrix(x), np.corrcoef(x, rowvar=False), atol=1e-5)


def test_pearson_pairwise_complete_with_missing_values():
    x = _data()
    rng = np.random.default_rng(1)
    x[rng.random(x.shape) < 0.01] = np.nan
    r = pearson_matrix(x)
    ok = np.isfinite(x[:, 0]) & np.isfinite(x[:, 1])
    assert abs(r[0, 1] - np.corrcoef(x[ok, 0], x[ok, 1])[0, 1]) < 0.01
    assert np.all(np.isfinite(r)) and np.allclose(r, r.T, atol=1e-6)


def test_spearman_equals_scipy():
    x = _data(rows=1500)
    assert np.allclose(spearman_matrix(x), spearmanr(x)[0], atol=1e-5)


def test_partial_removes_the_indirect_link_of_a_chain():
    x = _data(rows=20000)
    r, p = pearson_matrix(x), partial_matrix(x, shrinkage=0.0)
    assert abs(r[0, 2]) > 0.3                 # 0 and 2 look related only through 1
    assert abs(p[0, 2]) < 0.05
    assert abs(p[0, 1]) > 0.3 and abs(p[1, 2]) > 0.3


def test_partial_rejects_a_singular_matrix():
    x = _data()
    x[:, 3] = x[:, 4]                         # duplicate sensor
    with pytest.raises(ValueError):
        partial_matrix(x, shrinkage=0.0)


def test_percentile_eta_properties():
    n = 30
    rng = np.random.default_rng(2)
    corr = rng.uniform(-1, 1, size=(n, n)); corr = (corr + corr.T) / 2
    cand = rng.random((n, n)) < 0.3; cand = cand | cand.T
    eta = percentile_eta(corr, cand, 0.05, 0.95)
    assert np.allclose(eta, eta.T) and np.all(np.diag(eta) == 0)
    off = ~np.eye(n, dtype=bool)
    on = cand & off
    assert eta[on].min() >= 0.05 and eta[on].max() <= 0.95
    assert np.all(eta[off & ~cand] == 0.5)    # non-candidates: default, never used by PEM
    # within a row the larger |r| gets the larger prior (before the symmetric max)
    i = int(np.argmax(cand.sum(axis=1)))
    cols = np.flatnonzero(cand[i] & off[i])
    best, worst = cols[np.argmax(np.abs(corr[i, cols]))], cols[np.argmin(np.abs(corr[i, cols]))]
    assert eta[i, best] >= eta[i, worst]


def test_percentile_eta_is_validated():
    with pytest.raises(ValueError):
        percentile_eta(np.eye(3), np.ones((3, 3), bool), 0.9, 0.1)
    with pytest.raises(ValueError):
        percentile_eta(np.eye(3), np.ones((4, 4), bool))
