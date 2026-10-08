import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, metric="rmse", save_path=None):
    """
    Plot training and validation metric over epochs.

    Parameters
    ----------
    history : dict
        Dictionary containing lists such as:
        {
            "train_rmse": [...],
            "val_rmse": [...]
        }

    metric : str
        Metric name, for example "rmse" or "mae".

    save_path : str or None
        Optional path for saving the figure.
    """
    train_key = f"train_{metric}"
    val_key = f"val_{metric}"

    if train_key not in history or val_key not in history:
        raise KeyError(
            f"History must contain '{train_key}' and '{val_key}'."
        )

    epochs = np.arange(1, len(history[train_key]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history[train_key], label=f"Training {metric.upper()}")
    plt.plot(epochs, history[val_key], label=f"Validation {metric.upper()}")

    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.title(f"Training vs Validation {metric.upper()}")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_learning_rate(history, save_path=None):
    """
    Plot learning-rate changes over epochs.
    """
    if "learning_rate" not in history:
        raise KeyError("History must contain 'learning_rate'.")

    epochs = np.arange(1, len(history["learning_rate"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["learning_rate"])

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate During Training")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_predictions(y_true, y_pred, num_points=200, save_path=None):
    """
    Plot actual and predicted traffic values.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.

    y_pred : np.ndarray
        Predicted values.

    num_points : int
        Number of points to display.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )

    num_points = min(num_points, len(y_true))

    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:num_points], label="Actual")
    plt.plot(y_pred[:num_points], label="Predicted")

    plt.xlabel("Sample")
    plt.ylabel("Traffic")
    plt.title("Actual vs Predicted Traffic")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()