"""
Candidate links for causal discovery (Phase 6).

Every discovery step (VAR, local PEM, local VAR-LiNGAM) only considers
links in a candidate graph C[target, source] (!= 0 means source may
affect target):

    "road"        the road network (E1-E5, E7)
    "corr_k<K>"   data-driven neighbourhoods (E6): for every sensor, the K
                  sensors whose 5-minute CHANGES (first differences) are
                  most correlated with its own on the TRAINING rows. No
                  road information is used.

First differences are used because raw flows of all sensors share the
daily traffic cycle, which makes almost every pair strongly correlated;
changes measure whether two sensors move together from one 5-minute
step to the next. K = 25 matches the road graph's mean in-degree (25.1),
so E6 and E4 have about the same number of candidate links and differ
only in WHICH sensors are linked.

Correlation with missing values: each sensor's changes are standardised
with its own observed mean / std, missing changes contribute 0, and each
pair is divided by its number of jointly observed changes. (Missing rate
in the training rows is 0.03 %.)

Candidate artifacts: outputs/candidates/<name>.npz with the adjacency,
sensors, train_end, k and method. The run scripts record the candidate
name; PEM and LiNGAM take it from the VAR run they build on, so all
steps of one experiment use the same candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROAD = "road"

CANDIDATE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "candidates"


def unit_options(name: str) -> dict:
    """
    Options for build_causal_units with these candidates. Data-driven
    candidates cap reverse neighbours at K (hub sensors would otherwise
    give units of 1,000+ sensors); road units are unchanged.
    """

    return {} if name == ROAD else {"max_reverse_neighbors": parse_k(name)}


def candidate_tag(name: str) -> str:
    """Suffix for run folder names ('' for the road graph)."""

    return "" if name == ROAD else f"__cand-{name}"


def parse_k(name: str) -> int:

    if not name.startswith("corr_k") or not name[6:].isdigit():
        raise ValueError(f"Unknown candidate graph {name!r}; use 'road' or 'corr_k<K>'.")
    return int(name[6:])


def difference_correlation(train: np.ndarray, chunk: int = 512) -> np.ndarray:
    """
    Pairwise correlation of first differences of the TRAINING rows,
    float64, diagonal 1. Shape (n, n).
    """

    # In place where possible: ~2 copies of the training matrix at peak.
    z = np.diff(np.asarray(train, dtype=np.float32), axis=0)
    observed = np.isfinite(z)
    z[~observed] = 0.0

    count = observed.sum(axis=0).astype(np.float64)
    mean = z.sum(axis=0, dtype=np.float64) / np.maximum(count, 1)
    z -= mean.astype(np.float32)
    z[~observed] = 0.0

    std = np.sqrt(np.einsum("ij,ij->j", z, z, dtype=np.float64) / np.maximum(count, 1))

    if np.any(std <= 0):
        raise ValueError("A sensor has constant training values; cannot correlate.")

    z /= std.astype(np.float32)
    mask = observed.astype(np.float32)
    del observed

    n = z.shape[1]
    corr = np.empty((n, n), dtype=np.float64)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        products = z[:, start:stop].T @ z
        pairs = mask[:, start:stop].T @ mask
        corr[start:stop] = products / np.maximum(pairs, 1)

    np.fill_diagonal(corr, 1.0)
    return corr


def correlation_neighbours(corr: np.ndarray, k: int) -> np.ndarray:
    """
    C[target, source] = |corr| for the k sources with the largest |corr|
    with the target (ties broken by the lower sensor index), 0 elsewhere;
    no self-links. The weights only rank links (unit building); VAR,
    PEM and LiNGAM use the non-zero pattern.
    """

    n = corr.shape[0]

    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, {n - 1}]")

    strength = np.abs(np.asarray(corr, dtype=np.float64)).copy()
    np.fill_diagonal(strength, -np.inf)

    # Stable sort of -strength: equal strengths keep index order.
    top = np.argsort(-strength, axis=1, kind="stable")[:, :k]

    rows = np.repeat(np.arange(n), k)
    adjacency = np.zeros((n, n), dtype=np.float32)
    # Floor at a tiny positive value so a zero correlation stays a link.
    adjacency[rows, top.ravel()] = np.maximum(strength[rows, top.ravel()], 1e-12)
    return adjacency


def save_candidates(name, adjacency, sensors, train_end, k, extra=None):

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    path = CANDIDATE_DIR / f"{name}.npz"
    np.savez(path, adjacency=adjacency, sensors=np.asarray(sensors, dtype=np.int64),
             train_end=int(train_end), k=int(k),
             method="top-k |corr| of first differences, training rows",
             **(extra or {}))
    return path


def load_candidates(name, sensors, train_end, road_adjacency):
    """
    Candidate adjacency for a run on `sensors` / `train_end`.
    `road_adjacency` is the road graph already restricted to `sensors`.
    """

    if name == ROAD:
        return np.asarray(road_adjacency, dtype=np.float32)

    parse_k(name)
    path = CANDIDATE_DIR / f"{name}.npz"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run build_candidates.py --k ...")

    with np.load(path) as artifact:
        if not np.array_equal(artifact["sensors"], sensors) or int(artifact["train_end"]) != int(train_end):
            raise ValueError(f"{path} was built for another sensor set or training split.")
        return np.asarray(artifact["adjacency"], dtype=np.float32)
