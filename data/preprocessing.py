import numpy as np


def fit_max_scaler(data: np.ndarray) -> float:
    """
    Fit a simple max-value scaler on training data.

    Parameters
    ----------
    data:
        Traffic data with shape [time, nodes].

    Returns
    -------
    max_value:
        Maximum valid traffic value found in the data.
    """

    if data.ndim != 2:
        raise ValueError(
            f"Expected data with shape [time, nodes], got {data.shape}"
        )

    valid_values = data[np.isfinite(data)]

    if valid_values.size == 0:
        raise ValueError("No valid traffic values found.")

    max_value = np.max(valid_values)

    if max_value <= 0:
        raise ValueError(
            f"Maximum traffic value must be positive, got {max_value}"
        )

    return float(max_value)


def normalize_data(
    data: np.ndarray,
    max_value: float,
) -> np.ndarray:
    """
    Normalize traffic data using a previously fitted maximum value.

    Parameters
    ----------
    data:
        Traffic data with shape [time, nodes].

    max_value:
        Maximum value calculated from training data.

    Returns
    -------
    normalized_data:
        Normalized traffic data.
    """

    if max_value <= 0:
        raise ValueError(
            f"max_value must be positive, got {max_value}"
        )

    return data.astype(np.float32) / max_value


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    pre_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window input and target sequences.

    Parameters
    ----------
    data:
        Traffic data with shape [time, nodes].

    seq_len:
        Number of historical time steps used as input.

    pre_len:
        Number of future time steps to predict.

    Returns
    -------
    X:
        Input sequences with shape
        [samples, seq_len, nodes].

    y:
        Target sequences with shape
        [samples, pre_len, nodes].
    """

    if data.ndim != 2:
        raise ValueError(
            f"Expected data with shape [time, nodes], got {data.shape}"
        )

    if seq_len <= 0:
        raise ValueError("seq_len must be greater than zero.")

    if pre_len <= 0:
        raise ValueError("pre_len must be greater than zero.")

    total_required = seq_len + pre_len

    if len(data) < total_required:
        raise ValueError(
            "Not enough timesteps to create a sequence: "
            f"need at least {total_required}, got {len(data)}"
        )

    X = []
    y = []

    for i in range(len(data) - total_required + 1):
        X.append(
            data[i : i + seq_len]
        )

        y.append(
            data[
                i + seq_len :
                i + seq_len + pre_len
            ]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )