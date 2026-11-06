"""
E8C: correlation-informed spike-and-slab refinement of VAR coefficients.

A proposed coefficient-space MAP adaptation inspired by PEM. It is NOT the
original PEM algorithm: there is no data likelihood and no precision matrix; the
VAR coefficient itself is the observation and is shrunk edge by edge.

For every VAR candidate edge (target i, source j) in the mask M:

    u_ij   = || (A_1[i,j], A_2[i,j], A_3[i,j]) ||_2          aggregated lag magnitude
    eta_ij = prior inclusion probability from traffic correlation (symmetric)
    f_q(w) = exp(-|w| / nu_q) / (2 nu_q),   nu_1 > nu_0 (spike nu_0, slab nu_1)

    w_hat  = argmin_{0 <= w <= u}  (q / 2) (w - u)^2  -  lam * log[(1-eta) f_0(w) + eta f_1(w)]
    P_ij   = eta f_1(w_hat) / (eta f_1(w_hat) + (1-eta) f_0(w_hat))

The objective is separable over edges, so each edge is a one-dimensional
problem; it is solved on a dense grid on [0, u] (the penalty increases with w,
so the minimiser lies in [0, u]) with a parabolic refinement. All three lags of
an edge are scaled by the same factor w_hat / u, so one P_ij exists per edge and
the sign and lag profile of the VAR coefficients are preserved.

Direction comes from VAR only: M is built from the VAR coefficients and the
(symmetric) correlation prior only changes eta.

    logit P = logit(eta) + log(nu0/nu1) + w_hat (1/nu0 - 1/nu1)
"""

from __future__ import annotations

import numpy as np


def edge_magnitude(coefficients: np.ndarray) -> np.ndarray:
    """u[i, j] = L2 norm over lags of A_k[i, j]; diagonal 0."""

    a = np.asarray(coefficients, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"coefficients must have shape (lags, n, n); got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("VAR coefficients must be finite.")
    u = np.sqrt((a ** 2).sum(axis=0))
    np.fill_diagonal(u, 0.0)
    return u


def support_mask(u: np.ndarray) -> np.ndarray:
    """
    M[i, j] = True where the VAR has a non-zero coefficient (any lag), diagonal excluded.
    The VAR of this project is topology-aware (fitted on road-neighbour candidates
    only), so this support is the set of pairs VAR actually estimated.
    """

    mask = np.asarray(u) > 0
    np.fill_diagonal(mask, False)
    return mask


def candidate_mask(u: np.ndarray, per_target: int) -> np.ndarray:
    """M[i, j] = True for the `per_target` sources with the largest u[i, :] (diagonal excluded)."""

    n = len(u)
    if not 1 <= per_target < n:
        raise ValueError("per_target must be between 1 and n-1.")
    score = np.array(u, dtype=np.float64)
    np.fill_diagonal(score, -1.0)
    idx = np.argsort(-score, axis=1, kind="stable")[:, :per_target]
    mask = np.zeros((n, n), dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def correlation_to_eta(corr: np.ndarray, eta_min: float = 0.05, eta_max: float = 0.95) -> np.ndarray:
    """eta = eta_min + (eta_max - eta_min) |R|, symmetric, diagonal 0."""

    if not 0.0 < eta_min < eta_max < 1.0:
        raise ValueError("need 0 < eta_min < eta_max < 1.")
    c = np.abs(np.asarray(corr, dtype=np.float64))
    if not np.allclose(c, c.T, atol=1e-6):
        raise ValueError("correlation matrix must be symmetric.")
    c = np.clip(np.nan_to_num(c, nan=0.0), 0.0, 1.0)
    eta = eta_min + (eta_max - eta_min) * c
    eta = 0.5 * (eta + eta.T)
    np.fill_diagonal(eta, 0.0)
    return eta


def _log_laplace(w, nu):
    return -np.log(2.0 * nu) - w / nu


def log_mixture(w, eta, nu0, nu1):
    """log[(1-eta) f_0(w) + eta f_1(w)], numerically stable."""

    return np.logaddexp(np.log1p(-eta) + _log_laplace(w, nu0), np.log(eta) + _log_laplace(w, nu1))


def posterior_slab_probability(w, eta, nu0, nu1):
    """P = eta f_1(w) / (eta f_1(w) + (1-eta) f_0(w))."""

    w = np.asarray(w, dtype=np.float64)
    log_slab = np.log(eta) + _log_laplace(w, nu1)
    log_spike = np.log1p(-eta) + _log_laplace(w, nu0)
    return np.exp(log_slab - np.logaddexp(log_slab, log_spike))


def objective(w, u, eta, nu0, nu1, lam, q=1.0):
    return 0.5 * q * (w - u) ** 2 - lam * log_mixture(w, eta, nu0, nu1)


def solve_map(u, eta, nu0, nu1, lam, q=1.0, grid=256):
    """Global minimiser of the 1-D objective on [0, u], vectorised over edges."""

    if not (0 < nu0 < nu1):
        raise ValueError("need 0 < nu0 < nu1.")
    if lam < 0 or q <= 0:
        raise ValueError("need lam >= 0 and q > 0.")
    u = np.asarray(u, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    if u.shape != eta.shape:
        raise ValueError("u and eta must have the same shape.")
    if np.any(u < 0) or not np.all(np.isfinite(u)):
        raise ValueError("u must be finite and non-negative.")
    if np.any((eta <= 0) | (eta >= 1)):
        raise ValueError("eta must lie strictly inside (0, 1).")

    flat_u, flat_eta = u.ravel(), eta.ravel()
    best = np.empty_like(flat_u)
    chunk = 20000
    t = np.linspace(0.0, 1.0, grid)
    for s in range(0, len(flat_u), chunk):
        uu = flat_u[s:s + chunk, None]
        ee = flat_eta[s:s + chunk, None]
        w = uu * t[None, :]
        f = objective(w, uu, ee, nu0, nu1, lam, q)
        k = np.argmin(f, axis=1)
        rows = np.arange(len(k))
        k0 = np.clip(k, 1, grid - 2)
        f0, f1, f2 = f[rows, k0 - 1], f[rows, k0], f[rows, k0 + 1]
        denom = f0 - 2 * f1 + f2
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(denom > 1e-300, 0.5 * (f0 - f2) / denom, 0.0)
        shift = np.clip(shift, -1.0, 1.0)
        refined = (k0 + shift) / (grid - 1) * flat_u[s:s + chunk]
        refined = np.clip(refined, 0.0, flat_u[s:s + chunk])
        cand = np.stack([w[rows, k], refined], axis=1)
        fc = np.stack([f[rows, k], objective(refined, flat_u[s:s + chunk], flat_eta[s:s + chunk], nu0, nu1, lam, q)], axis=1)
        pick = np.argmin(fc, axis=1)
        best[s:s + chunk] = cand[rows, pick]
    return best.reshape(u.shape)


def refine(coefficients, mask, eta, nu0, nu1, lam, q=1.0, grid=256):
    """
    Returns (W, P, w_hat): refined coefficients (lags, n, n), posterior slab
    probability P (n, n; 0 outside the mask) and the refined magnitude w_hat.
    """

    a = np.asarray(coefficients, dtype=np.float64)
    u = edge_magnitude(a)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != u.shape or np.any(np.diag(mask)):
        raise ValueError("mask must be (n, n) with an empty diagonal.")

    w_hat = np.zeros_like(u)
    p = np.zeros_like(u)
    w_hat[mask] = solve_map(u[mask], np.asarray(eta)[mask], nu0, nu1, lam, q, grid)
    p[mask] = posterior_slab_probability(w_hat[mask], np.asarray(eta)[mask], nu0, nu1)

    ratio = np.zeros_like(u)
    pos = mask & (u > 0)
    ratio[pos] = w_hat[pos] / u[pos]
    w = a * ratio[None, :, :]
    return w, p, w_hat
