import numpy as np
import pytest

from causal import var_coefficient_pem as vp
from causal.flow_correlation import detrend_time_of_day, flow_correlation


def test_solver_matches_brute_force():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 1, 200)
    eta = rng.uniform(0.05, 0.95, 200)
    w = vp.solve_map(u, eta, 0.05, 0.5, 0.02)
    fine = np.linspace(0, 1, 20001)
    for i in range(0, 200, 10):
        grid = fine * u[i]
        best = grid[np.argmin(vp.objective(grid, u[i], eta[i], 0.05, 0.5, 0.02))]
        assert abs(w[i] - best) < 2e-3 * max(u[i], 1e-9) + 1e-4
        assert 0 <= w[i] <= u[i] + 1e-12


def test_no_penalty_returns_input():
    u = np.array([0.0, 0.2, 0.9])
    w = vp.solve_map(u, np.full(3, 0.5), 0.05, 0.5, 0.0)
    assert np.allclose(w, u, atol=1e-3)


def test_posterior_formula_and_eta_monotone():
    w = np.array([0.0, 0.1, 0.5])
    p_low = vp.posterior_slab_probability(w, 0.2, 0.05, 0.5)
    p_high = vp.posterior_slab_probability(w, 0.8, 0.05, 0.5)
    assert np.all((p_low >= 0) & (p_low <= 1)) and np.all(p_high > p_low)
    assert np.all(np.diff(p_low) > 0)                       # larger coefficient -> more slab
    logit = np.log(0.2 / 0.8) + np.log(0.05 / 0.5) + w * (1 / 0.05 - 1 / 0.5)
    assert np.allclose(np.log(p_low / (1 - p_low)), logit)


def test_candidate_mask_and_refine_keep_structure():
    rng = np.random.default_rng(1)
    n = 30
    a = rng.normal(0, 0.1, (3, n, n))
    mask = vp.candidate_mask(vp.edge_magnitude(a), 5)
    assert mask.sum(axis=1).tolist() == [5] * n and not np.diag(mask).any()
    eta = vp.correlation_to_eta(np.corrcoef(rng.normal(size=(200, n)), rowvar=False))
    w, p, w_hat = vp.refine(a, mask, eta, 0.02, 0.2, 0.01)
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(p))
    assert np.all(p[~mask] == 0) and np.all(w[:, ~mask] == 0)           # non-candidates cannot enter
    assert np.all((p[mask] > 0) & (p[mask] < 1))
    assert np.all(np.abs(w) <= np.abs(a) + 1e-12)                       # shrinkage only
    assert np.all(np.sign(w) * np.sign(a) >= 0)                         # direction / sign preserved
    assert np.all(np.abs(w_hat[mask]) <= vp.edge_magnitude(a)[mask] + 1e-12)


def test_eta_bounds_symmetry_and_validation():
    c = np.array([[1, 0.9, -0.3], [0.9, 1, 0.0], [-0.3, 0.0, 1]])
    eta = vp.correlation_to_eta(c, 0.05, 0.95)
    assert np.allclose(eta, eta.T) and np.all(np.diag(eta) == 0)
    off = eta[~np.eye(3, dtype=bool)]
    assert off.min() >= 0.05 and off.max() <= 0.95
    assert eta[0, 1] > eta[0, 2] > eta[1, 2] - 1e-12
    with pytest.raises(ValueError):
        vp.correlation_to_eta(c, 0.9, 0.1)
    with pytest.raises(ValueError):
        vp.correlation_to_eta(np.array([[1, 0.5], [0.1, 1]]))
    with pytest.raises(ValueError):
        vp.solve_map(np.ones(3), np.full(3, 0.5), 0.5, 0.1, 0.1)
    with pytest.raises(ValueError):
        vp.solve_map(np.ones(3), np.array([0.0, 0.5, 0.5]), 0.1, 0.5, 0.1)


def test_synthetic_true_edges_outrank_noise():
    # j -> i is real (large coefficient, correlated); the others are small noise coefficients.
    rng = np.random.default_rng(2)
    n, t = 12, 4000
    x = np.zeros((t, n))
    for s in range(1, t):
        x[s] = rng.normal(0, 1, n)
        x[s, 1] += 0.8 * x[s - 1, 0]
        x[s, 3] += 0.7 * x[s - 1, 2]
    a = rng.normal(0, 0.02, (1, n, n))
    a[0, 1, 0], a[0, 3, 2] = 0.8, 0.7
    mask = vp.candidate_mask(vp.edge_magnitude(a), 4)
    eta = vp.correlation_to_eta(np.corrcoef(x, rowvar=False))
    _, p, _ = vp.refine(a, mask, eta, 0.02, 0.3, 0.01)
    true_p = [p[1, 0], p[3, 2]]
    noise_p = p[mask & ~np.isin(np.arange(n)[:, None] * n + np.arange(n)[None, :], [1 * n + 0, 3 * n + 2])]
    assert min(true_p) > np.quantile(noise_p, 0.9)


def test_flow_correlation_detrend_and_missing():
    rng = np.random.default_rng(3)
    t = 288 * 10
    daily = np.sin(2 * np.pi * np.arange(t) / 288)[:, None]
    x = daily * np.array([[1.0, 2.0, 3.0]]) + rng.normal(0, 0.1, (t, 3))
    assert flow_correlation(x)[0, 1] > 0.95                            # shared rhythm dominates
    assert abs(flow_correlation(x, detrend=True)[0, 1]) < 0.3          # removed
    x[::7, 0] = np.nan
    r = flow_correlation(x, detrend=True)
    assert np.allclose(r, r.T) and np.all(np.isfinite(r))
    assert np.allclose(detrend_time_of_day(np.zeros((288, 2))), 0)
