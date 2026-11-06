"""
Pearson correlation of the ORIGINAL training traffic flow (no VAR residuals).

Training rows only. Missing values stay missing (pairwise-complete). Optional
detrending subtracts each sensor's average time-of-day profile (period 288 for
5-minute data, estimated on the training rows), because the shared daily
rhythm makes almost every pair of raw series strongly correlated (median |r|
about 0.84 on the VAR candidates) and so uninformative about which pairs are
related.

Correlation is statistical association, not causation.
"""

from __future__ import annotations

import numpy as np

from causal.corr_prior import pearson_matrix

STEPS_PER_DAY = 288


def detrend_time_of_day(flow: np.ndarray, period: int = STEPS_PER_DAY) -> np.ndarray:
    x = np.array(flow, dtype=np.float64)
    phase = np.arange(len(x)) % period
    for p in range(period):
        rows = phase == p
        if rows.any() and np.isfinite(x[rows]).any():
            x[rows] -= np.nanmean(x[rows], axis=0)
    return x


def flow_correlation(flow: np.ndarray, detrend: bool = False, min_overlap: int = 1000) -> np.ndarray:
    """Symmetric correlation matrix (n, n); pairs with < min_overlap shared rows get 0."""

    x = detrend_time_of_day(flow) if detrend else np.asarray(flow, dtype=np.float64)
    corr = pearson_matrix(x)
    observed = np.isfinite(x).astype(np.float32)
    overlap = observed.T @ observed
    corr = np.where(overlap >= min_overlap, corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    return 0.5 * (corr + corr.T)


def cached_flow_correlation(detrend, sensors, train_end, cache_dir):
    """
    Correlation of the original TRAINING flow of `sensors`, cached in
    <cache_dir>/flow_corr_{detrended|raw}.npz. Returns (corr, cache_path).
    """

    from pathlib import Path

    cache_dir = Path(cache_dir)
    path = cache_dir / f"flow_corr_{'detrended' if detrend else 'raw'}.npz"
    if path.exists():
        with np.load(path) as z:
            if np.array_equal(z["sensors"], sensors) and int(z["train_end"]) == train_end:
                return z["corr"].astype(np.float64), str(path)

    from config.config import ExperimentConfig
    from data.loader import load_gba_dataset

    config = ExperimentConfig()
    traffic, _ = load_gba_dataset(config.traffic_path, config.adjacency_path,
                                  num_timesteps=config.num_timesteps, num_nodes=config.num_nodes)
    flow = np.asarray(traffic[:train_end][:, sensors], dtype=np.float64)       # training rows only
    corr = flow_correlation(flow, detrend=detrend)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, corr=corr.astype(np.float32), sensors=sensors, train_end=train_end, detrend=detrend)
    return corr, str(path)
