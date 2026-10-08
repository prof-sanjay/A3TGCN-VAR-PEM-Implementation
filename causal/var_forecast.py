"""
One-step VAR predictions and residuals for ANY data split,
using coefficients that were already estimated by
TopologyAwareVAR (causal/var.py) on the training split.

Used for:
    - out-of-sample VAR evaluation on the validation split
      (lag-order / ridge selection without touching the test set)
    - reproducing the in-sample training residuals as a check.

The model is exactly the one in causal/var.py:

    x_i(t) = sum_{k=1}^{p} sum_{j in N(i)} A_k[i, j] x_j(t - k)

with N(i) = {j : adjacency[i, j] != 0} U {i} and no intercept.

Missing values are NOT imputed. A residual e_i(t) exists only when
x_i(t) and every x_j(t - k), j in N(i), k = 1..p, are finite;
otherwise it is NaN. This is the same local-validity rule as the
VAR regression itself. No NaN is replaced, even temporarily.

Pure NumPy: no TensorFlow import, usable in both environments.
"""

from __future__ import annotations

import numpy as np


def var_neighbors(
    adjacency: np.ndarray,
    target_node: int,
) -> np.ndarray:
    """
    Source nodes allowed to predict target_node.

    Identical rule to TopologyAwareVAR._get_neighbors:
    non-zero adjacency row entries, plus the target itself.
    """

    neighbors = np.flatnonzero(
        adjacency[target_node] != 0
    )

    if target_node not in neighbors:
        neighbors = np.append(
            neighbors,
            target_node,
        )

    return np.unique(neighbors)


def var_one_step_residuals(
    data: np.ndarray,
    coefficients: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step-ahead VAR residuals on ``data``.

    Parameters
    ----------
    data:
        Traffic of one split, shape (num_timesteps, num_nodes).
        NaNs preserved.

    coefficients:
        VAR coefficients, shape (lag_order, num_nodes, num_nodes),
        coefficients[k, target, source] = effect of source at lag
        k + 1 on target (VARResult convention).

    adjacency:
        The adjacency used when the VAR was fitted.

    Returns
    -------
    residuals:
        float32 array (num_timesteps, num_nodes). NaN where the
        residual is undefined (first lag_order rows, or any required
        value missing).

    valid_mask:
        bool array (num_timesteps, num_nodes), True where the
        residual is defined.
    """

    data = np.asarray(data, dtype=np.float32)
    coefficients = np.asarray(coefficients)

    if data.ndim != 2:
        raise ValueError(
            "data must have shape (num_timesteps, num_nodes)."
        )

    if coefficients.ndim != 3:
        raise ValueError(
            "coefficients must have shape "
            "(lag_order, num_nodes, num_nodes)."
        )

    lag_order, num_nodes, _ = coefficients.shape

    if data.shape[1] != num_nodes or adjacency.shape != (
        num_nodes,
        num_nodes,
    ):
        raise ValueError(
            "data, coefficients and adjacency sizes do not match."
        )

    num_timesteps = data.shape[0]

    if lag_order >= num_timesteps:
        raise ValueError(
            "lag_order must be smaller than the number of timesteps."
        )

    residuals = np.full(
        (num_timesteps, num_nodes),
        np.nan,
        dtype=np.float32,
    )

    valid_mask = np.zeros(
        (num_timesteps, num_nodes),
        dtype=bool,
    )

    start_time = lag_order

    for target_node in range(num_nodes):

        source_nodes = var_neighbors(
            adjacency,
            target_node,
        )

        # Same column layout as TopologyAwareVAR: lag 1 block,
        # then lag 2 block, ...
        X = np.concatenate(
            [
                data[
                    start_time - lag : num_timesteps - lag,
                    source_nodes,
                ]
                for lag in range(1, lag_order + 1)
            ],
            axis=1,
        )

        y = data[start_time:, target_node]

        beta = np.concatenate(
            [
                coefficients[lag - 1, target_node, source_nodes]
                for lag in range(1, lag_order + 1)
            ]
        ).astype(np.float64)

        valid = (
            np.isfinite(y)
            & np.all(np.isfinite(X), axis=1)
        )

        if not np.any(valid):
            continue

        predictions = X[valid] @ beta

        rows = np.flatnonzero(valid) + start_time

        residuals[rows, target_node] = y[valid] - predictions
        valid_mask[rows, target_node] = True

    return residuals, valid_mask


def residual_summary(
    residuals: np.ndarray,
) -> dict:
    """
    RMSE / MAE of the defined residuals, plus coverage.
    """

    finite = np.isfinite(residuals)

    values = residuals[finite].astype(np.float64)

    return {
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "mae": float(np.mean(np.abs(values))),
        "mean": float(np.mean(values)),
        "num_defined": int(finite.sum()),
        "coverage_percentage": float(100.0 * finite.mean()),
    }
