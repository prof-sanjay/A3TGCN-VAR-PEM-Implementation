"""
Correlation-based PEM prior (E8 prior study).

The PEM prior eta[j, k] is the prior probability that sensors j and k share an
edge. Here it is derived from the correlation of the VAR training residuals
(training rows only), not from the road graph:

    pearson   linear correlation of the residuals
    spearman  Pearson correlation of the ranks of the residuals
    partial   correlation after conditioning on all other sensors
              (inverse of a shrunk correlation matrix)

Missing residuals are never filled: every pair uses the rows where both
sensors are observed (columns are standardised once, so with ~99.5 % coverage
this equals the exact pairwise-complete correlation up to a negligible term).

The correlation becomes a prior probability by a fixed rule, so the choice of
correlation is not mixed up with the choice of mapping:

    w[j, k] = percentile of |r[j, k]| among the candidate pairs of row j,
              mapped to [low, high]
    eta     = max(w[j, k], w[k, j])      (symmetric, as the road prior)

Candidate pairs are the pairs that share a causal unit (the only pairs PEM
can include). The diagonal is 0.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

METHODS = ("pearson", "spearman", "partial")


def _standardise(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    observed = np.isfinite(x)
    count = observed.sum(axis=0)
    if np.any(count < 2):
        raise ValueError("every sensor needs at least two observed residuals.")
    mean = np.where(observed, x, 0.0).sum(axis=0) / count
    centred = np.where(observed, x - mean, 0.0)
    std = np.sqrt((centred ** 2).sum(axis=0) / count)
    if np.any(std <= 0):
        raise ValueError("a sensor has zero residual variance.")
    return (centred / std).astype(np.float32), observed.astype(np.float32)


def pearson_matrix(residuals: np.ndarray, chunk_rows: int = 16384) -> np.ndarray:
    """Pairwise-complete Pearson correlation, shape (n, n), float64."""

    z, m = _standardise(residuals)
    n = z.shape[1]
    numerator = np.zeros((n, n))
    counts = np.zeros((n, n))
    for start in range(0, len(z), chunk_rows):
        zb, mb = z[start:start + chunk_rows], m[start:start + chunk_rows]
        numerator += zb.T @ zb
        counts += mb.T @ mb
    corr = numerator / np.maximum(counts, 1)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def rank_columns(residuals: np.ndarray) -> np.ndarray:
    """Average ranks of the observed values of each column (NaN stays NaN)."""

    residuals = np.asarray(residuals, dtype=np.float64)
    ranks = np.full(residuals.shape, np.nan)
    for j in range(residuals.shape[1]):
        ok = np.isfinite(residuals[:, j])
        ranks[ok, j] = rankdata(residuals[ok, j], method="average")
    return ranks


def spearman_matrix(residuals: np.ndarray) -> np.ndarray:
    return pearson_matrix(rank_columns(residuals))


def partial_matrix(residuals: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    """
    Partial correlation from the inverse of the shrunk correlation matrix
    (1 - lambda) R + lambda I; lambda keeps the inversion stable for highly
    collinear sensors.
    """

    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must be in [0, 1).")
    r = pearson_matrix(residuals)
    shrunk = (1.0 - shrinkage) * r + shrinkage * np.eye(len(r))
    smallest = np.linalg.eigvalsh(shrunk).min()
    if smallest <= 1e-8:
        raise ValueError(f"shrunk correlation matrix is not positive definite (min eigenvalue {smallest:.2e}).")
    precision = np.linalg.inv(shrunk)
    d = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    return partial


def correlation_matrix(residuals, method: str, shrinkage: float = 0.05) -> np.ndarray:
    if method == "pearson":
        return pearson_matrix(residuals)
    if method == "spearman":
        return spearman_matrix(residuals)
    if method == "partial":
        return partial_matrix(residuals, shrinkage)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}.")


def percentile_eta(corr: np.ndarray, candidates: np.ndarray, low: float = 0.05, high: float = 0.95,
                   default: float = 0.5) -> np.ndarray:
    """
    Map |correlation| to a symmetric prior in [low, high] by the percentile of
    |r| among each row's candidate pairs. Non-candidate pairs get ``default``
    (PEM never uses them); the diagonal is 0.
    """

    if not 0.0 < low < high < 1.0:
        raise ValueError("require 0 < low < high < 1.")
    corr = np.asarray(corr, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=bool).copy()
    n = len(corr)
    if corr.shape != (n, n) or candidates.shape != (n, n):
        raise ValueError("corr and candidates must be square and equally sized.")
    np.fill_diagonal(candidates, False)

    w = np.full((n, n), np.nan)
    for i in range(n):
        cols = np.flatnonzero(candidates[i])
        if cols.size == 0:
            continue
        pct = (rankdata(np.abs(corr[i, cols]), method="average") - 0.5) / cols.size   # in (0, 1)
        w[i, cols] = low + (high - low) * pct

    eta = np.fmax(w, w.T)                         # larger of the two directions; NaN only if neither
    eta = np.where(np.isfinite(eta), eta, default)
    np.fill_diagonal(eta, 0.0)
    return eta
