import numpy as np

from ..training.metrics import rmse, mae, r2_score, accuracy


def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation metrics for traffic predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth traffic values.
    y_pred : np.ndarray
        Predicted traffic values.

    Returns
    -------
    dict
        Dictionary containing RMSE, MAE, R2, and accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }