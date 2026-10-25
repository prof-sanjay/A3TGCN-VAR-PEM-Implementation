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


def tgcn_accuracy(y_true, y_pred):
    """
    Accuracy as defined in the T-GCN / A3T-GCN papers and their code:

        1 - ||Y - Y_hat||_F / ||Y||_F

    Reported in addition to the project's accuracy() (fraction within
    10 %), which is a different measure.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    norm = np.linalg.norm(y_true)

    if norm == 0:
        return float("nan")

    return float(1.0 - np.linalg.norm(y_true - y_pred) / norm)


# Column names and order used in every metrics file (epoch_metrics.csv,
# run_summary.json, consolidated tables).
METRIC_COLUMNS = ("RMSE", "MAE", "Masked_MAPE", "R2", "Accuracy")


def evaluate_metrics(y_true, y_pred, mape_min_flow=10.0):
    """
    All evaluation metrics on one set of targets / predictions, which
    must be in the same units (vehicles per 5 minutes in this project).

    Aggregation: every metric pools all (window, sensor) entries, i.e.
    over time and sensors together.

        RMSE         sqrt(mean((y - y_hat)^2))
        MAE          mean(|y - y_hat|)
        Masked_MAPE  100 * mean(|y - y_hat| / |y|) over |y| >= mape_min_flow
                     (NaN if no target passes the mask)
        R2           1 - SS_res / SS_tot          (NaN if SS_tot = 0)
        Accuracy     fraction of entries with |y - y_hat| / max(|y|, 1e-8)
                     <= 0.1, i.e. within 10 % of the target. A target of 0
                     only counts as accurate if the prediction is within
                     1e-9 of it. Project-specific; NOT the T-GCN accuracy.
        Accuracy_TGCN  1 - ||Y - Y_hat||_F / ||Y||_F (T-GCN papers)

    Empty input gives NaN for every metric (never 0).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: targets {y_true.shape}, predictions {y_pred.shape}"
        )

    if y_true.size == 0:
        return {name: float("nan") for name in METRIC_COLUMNS + ("Accuracy_TGCN",)}

    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        raise ValueError("targets / predictions contain non-finite values")

    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    return {
        "RMSE": float(rmse(y_true, y_pred)),
        "MAE": float(mae(y_true, y_pred)),
        "Masked_MAPE": masked_mape(y_true, y_pred, mape_min_flow),
        "R2": float(r2_score(y_true, y_pred)) if ss_tot > 0 else float("nan"),
        "Accuracy": float(accuracy(y_true, y_pred)),
        "Accuracy_TGCN": tgcn_accuracy(y_true, y_pred),
    }
