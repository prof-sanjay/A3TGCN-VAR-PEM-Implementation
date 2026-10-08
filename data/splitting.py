import numpy as np


def chronological_split(
    data: np.ndarray,
    train_rate: float = 0.8,
    validation_rate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time-series data chronologically into train,
    validation, and test sets.

    Parameters
    ----------
    data:
        Traffic data with shape [time, nodes].

    train_rate:
        Fraction of the data used for training.

    validation_rate:
        Fraction of the data used for validation.

    Returns
    -------
    train_data:
        Earliest portion of the time series.

    validation_data:
        Middle portion of the time series.

    test_data:
        Latest portion of the time series.
    """

    if data.ndim != 2:
        raise ValueError(
            f"Expected data with shape [time, nodes], got {data.shape}"
        )

    if not 0 < train_rate < 1:
        raise ValueError(
            f"train_rate must be between 0 and 1, got {train_rate}"
        )

    if not 0 <= validation_rate < 1:
        raise ValueError(
            f"validation_rate must be between 0 and 1, got {validation_rate}"
        )

    if train_rate + validation_rate >= 1:
        raise ValueError(
            "train_rate + validation_rate must be less than 1."
        )

    num_timesteps = len(data)

    train_end = int(num_timesteps * train_rate)

    validation_end = int(
        num_timesteps * (train_rate + validation_rate)
    )

    train_data = data[:train_end]
    validation_data = data[train_end:validation_end]
    test_data = data[validation_end:]

    return train_data, validation_data, test_data