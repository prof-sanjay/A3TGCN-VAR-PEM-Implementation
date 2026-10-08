"""
Gap-band filling of missing traffic values for A3T-GCN INPUTS.

Policy (changed by the user on 2026-10-08; see ablation notes):

    gap length L (consecutive missing steps of one sensor)

    TRAINING split (may look ahead inside the training split):
        L <= 6  (<= 30 min)    : linear interpolation in time
        7 <= L <= 24 (<= 2 h)  : neighbour regression, else
                                 linear interpolation
        L > 24                 : neighbour regression

    VALIDATION / TEST splits (no look-ahead; the band is decided
    by the ELAPSED gap length k, because the final length is
    unknown at prediction time):
        k <= 6                 : last observed value
        7 <= k <= 24           : neighbour regression, else
                                 last observed value
        k > 24                 : neighbour regression

    Anything that cannot be filled stays NaN, and windows touching
    it are still excluded.

Neighbour regression:
    For sensor i, the K road neighbours (adjacency in either
    direction) with the highest absolute correlation on OBSERVED
    training rows; ordinary least squares with intercept, fitted on
    observed training rows only. It is applied only where all K
    neighbours are OBSERVED at that timestamp (filled values never
    feed other fills). Predictions are clipped at 0 (flow >= 0).

Rules that keep the ablation valid:
    - Raw files are never modified; filling returns a new array.
    - Filled values are used ONLY as A3T-GCN inputs. Windows whose
      target timestep contains an originally-missing value remain
      excluded, so every scored target is an observed value.
    - VAR and PEM never see filled values.
    - Filling never crosses split boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SHORT_GAP = 6
MEDIUM_GAP = 24

FILL_METHODS = (
    "interpolation",
    "neighbour_regression",
    "last_observed",
)


@dataclass
class NeighbourModel:
    """Same-timestamp regression of one sensor on K neighbours."""

    neighbours: np.ndarray
    coefficients: np.ndarray
    intercept: float
    num_fit_rows: int


# ============================================================
# Gap detection
# ============================================================

def missing_runs(
    missing: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Consecutive True runs of a 1-D boolean array as
    (start, end_exclusive) pairs.
    """

    padded = np.concatenate(
        ([False], missing, [False])
    ).astype(np.int8)

    changes = np.diff(padded)

    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    return list(zip(starts.tolist(), ends.tolist()))


def gap_length_counts(
    data: np.ndarray,
) -> dict:
    """
    Number of gaps and missing values per gap-length band.
    """

    counts = {
        "short_le_6": [0, 0],
        "medium_7_24": [0, 0],
        "long_gt_24": [0, 0],
    }

    for sensor in range(data.shape[1]):

        for start, end in missing_runs(np.isnan(data[:, sensor])):

            length = end - start

            if length <= SHORT_GAP:
                band = "short_le_6"
            elif length <= MEDIUM_GAP:
                band = "medium_7_24"
            else:
                band = "long_gt_24"

            counts[band][0] += 1
            counts[band][1] += length

    return {
        band: {"gaps": gaps, "values": values}
        for band, (gaps, values) in counts.items()
    }


# ============================================================
# Neighbour regression (fitted on TRAINING data only)
# ============================================================

def _candidate_neighbours(
    adjacency: np.ndarray,
    sensor: int,
) -> np.ndarray:

    connected = (adjacency[sensor] != 0) | (adjacency[:, sensor] != 0)
    connected[sensor] = False

    return np.flatnonzero(connected)


def _pairwise_abs_correlation(
    target: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    |Pearson correlation| of target with every candidate column,
    each on the rows where both are observed. Inputs unchanged.
    """

    both = np.isfinite(target)[:, None] & np.isfinite(candidates)

    count = both.sum(axis=0)

    t = np.where(both, target[:, None], 0.0).astype(np.float64)
    c = np.where(both, candidates, 0.0).astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):

        t_mean = t.sum(axis=0) / count
        c_mean = c.sum(axis=0) / count

        t_centered = np.where(both, t - t_mean, 0.0)
        c_centered = np.where(both, c - c_mean, 0.0)

        covariance = (t_centered * c_centered).sum(axis=0)

        correlation = covariance / np.sqrt(
            (t_centered ** 2).sum(axis=0)
            * (c_centered ** 2).sum(axis=0)
        )

    correlation[~np.isfinite(correlation) | (count < 2)] = 0.0

    return np.abs(correlation)


def fit_neighbour_models(
    train_data: np.ndarray,
    adjacency: np.ndarray,
    num_neighbours: int = 5,
    min_fit_rows: int = 1000,
) -> dict[int, NeighbourModel]:
    """
    Fit one neighbour regression per sensor on OBSERVED training
    rows. Sensors without enough neighbours or rows get no model.
    """

    models: dict[int, NeighbourModel] = {}

    for sensor in range(train_data.shape[1]):

        candidates = _candidate_neighbours(adjacency, sensor)

        if len(candidates) == 0:
            continue

        target = train_data[:, sensor]

        correlation = _pairwise_abs_correlation(
            target,
            train_data[:, candidates],
        )

        order = np.argsort(-correlation, kind="stable")

        neighbours = np.sort(
            candidates[order[:num_neighbours]]
        )

        X = train_data[:, neighbours]

        rows = np.isfinite(target) & np.all(np.isfinite(X), axis=1)

        if rows.sum() < min_fit_rows:
            continue

        design = np.column_stack(
            (
                X[rows].astype(np.float64),
                np.ones(rows.sum()),
            )
        )

        solution, *_ = np.linalg.lstsq(
            design,
            target[rows].astype(np.float64),
            rcond=None,
        )

        models[sensor] = NeighbourModel(
            neighbours=neighbours,
            coefficients=solution[:-1],
            intercept=float(solution[-1]),
            num_fit_rows=int(rows.sum()),
        )

    return models


def _neighbour_predictions(
    data: np.ndarray,
    model: NeighbourModel,
    rows: np.ndarray,
) -> np.ndarray:
    """
    Predictions at ``rows``; NaN where any neighbour is missing.
    ``data`` must be the ORIGINAL (unfilled) split.
    """

    X = data[np.ix_(rows, model.neighbours)].astype(np.float64)

    predictions = X @ model.coefficients + model.intercept

    predictions = np.maximum(predictions, 0.0)

    predictions[~np.all(np.isfinite(X), axis=1)] = np.nan

    return predictions


# ============================================================
# Filling
# ============================================================

def fill_split(
    data: np.ndarray,
    models: dict[int, NeighbourModel],
    look_ahead: bool,
) -> tuple[np.ndarray, dict]:
    """
    Fill one split according to the gap-band policy.

    Parameters
    ----------
    data:
        Original split [time, nodes], NaNs preserved. Not modified.

    models:
        Neighbour models fitted on the TRAINING split.

    look_ahead:
        True for the training split (interpolation allowed),
        False for validation / test (past values only).

    Returns
    -------
    filled:
        New float32 array; still NaN where nothing applied.

    counts:
        Number of values filled by each method, and still missing.
    """

    filled = np.array(data, dtype=np.float32, copy=True)

    counts = {method: 0 for method in FILL_METHODS}

    num_timesteps = data.shape[0]

    for sensor in range(data.shape[1]):

        column = data[:, sensor]

        runs = missing_runs(np.isnan(column))

        if not runs:
            continue

        model = models.get(sensor)

        for start, end in runs:

            rows = np.arange(start, end)

            values = np.full(len(rows), np.nan)
            method = np.full(len(rows), -1, dtype=np.int8)

            # Elapsed length k (1-based) at each missing step, and
            # the final gap length (training only).
            elapsed = np.arange(1, len(rows) + 1)
            length = end - start

            has_before = start > 0
            has_after = end < num_timesteps

            if model is not None:
                neighbour = _neighbour_predictions(data, model, rows)
            else:
                neighbour = np.full(len(rows), np.nan)

            if look_ahead:

                if has_before and has_after:
                    interpolation = np.interp(
                        rows,
                        [start - 1, end],
                        [column[start - 1], column[end]],
                    )
                else:
                    interpolation = np.full(len(rows), np.nan)

                if length <= SHORT_GAP:
                    use_interp = np.ones(len(rows), dtype=bool)
                    use_neighbour = np.zeros(len(rows), dtype=bool)
                elif length <= MEDIUM_GAP:
                    use_neighbour = np.isfinite(neighbour)
                    use_interp = ~use_neighbour
                else:
                    use_neighbour = np.isfinite(neighbour)
                    use_interp = np.zeros(len(rows), dtype=bool)

                use_interp &= np.isfinite(interpolation)

                values[use_interp] = interpolation[use_interp]
                method[use_interp] = 0

                values[use_neighbour] = neighbour[use_neighbour]
                method[use_neighbour] = 1

            else:

                last_observed = (
                    column[start - 1] if has_before else np.nan
                )

                short = elapsed <= SHORT_GAP
                medium = (elapsed > SHORT_GAP) & (elapsed <= MEDIUM_GAP)
                long = elapsed > MEDIUM_GAP

                use_neighbour = (medium | long) & np.isfinite(neighbour)

                use_last = (
                    (short | (medium & ~use_neighbour))
                    & np.isfinite(last_observed)
                )

                values[use_neighbour] = neighbour[use_neighbour]
                method[use_neighbour] = 1

                values[use_last] = last_observed
                method[use_last] = 2

            filled[rows, sensor] = values.astype(np.float32)

            for code, name in enumerate(FILL_METHODS):
                counts[name] += int(np.sum(method == code))

    counts["still_missing"] = int(np.isnan(filled).sum())
    counts["originally_missing"] = int(np.isnan(data).sum())

    return filled, counts
