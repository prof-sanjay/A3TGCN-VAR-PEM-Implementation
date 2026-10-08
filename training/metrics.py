import numpy as np


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Coefficient of determination (R²).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1.0 - (ss_res / ss_tot)


def accuracy(y_true, y_pred, threshold=0.1):
    """
    Prediction accuracy used as a simple relative-error measure.

    A prediction is considered accurate when its relative error
    is within the specified threshold.

    Parameters
    ----------
    threshold : float
        Relative-error tolerance.
        0.1 means 10%.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.maximum(np.abs(y_true), 1e-8)
    relative_error = np.abs(y_true - y_pred) / denominator

    return np.mean(relative_error <= threshold)

def masked_mape(y_true, y_pred, min_true=10.0):
    """
    Mean Absolute Percentage Error (in %) over targets with
    |y_true| >= min_true.

    Low flows (night, zero readings) make percentage errors explode,
    so they are excluded, as is common for traffic flow (LargeST
    reports MAPE on non-zero targets). The threshold is reported
    with the result.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.abs(y_true) >= min_true

    if not np.any(mask):
        return float("nan")

    return float(
        100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))
    )
