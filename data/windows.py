"""
Index-based sliding windows for traffic forecasting.

This module produces exactly the same samples as

    create_sequences(...)  followed by  filter_valid_sequences(...)

but without materialising the window arrays. For LargeST-GBA the
materialised training windows alone would need about

    84,084 x 12 x 2,352 x 4 bytes  ~  9.5 GB

so only the START index of every valid window is stored, and the
windows of one batch are gathered from the traffic array on demand.

Window definition (identical to create_sequences):

    window starting at s:
        X = data[s : s + seq_len]
        y = data[s + seq_len : s + seq_len + pre_len]

A window is valid only when every value in X and y is finite for
every sensor. Invalid windows are excluded.

Missing values are NOT imputed, filled, or modified.

Windows are created separately for each split, so no window can
cross a train / validation / test boundary.
"""

import numpy as np


def _validate_window_arguments(
    data: np.ndarray,
    seq_len: int,
    pre_len: int,
) -> None:
    """Validate arguments exactly as create_sequences does."""

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


def finite_row_mask(
    data: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    Return a boolean vector that is True for timesteps at which
    every sensor value is finite.

    The check is performed in chunks so that no full-size boolean
    matrix is created.

    Parameters
    ----------
    data:
        Traffic data with shape [time, nodes].

    chunk_size:
        Number of timesteps checked at once.

    Returns
    -------
    row_ok:
        Boolean array with shape [time].
    """

    if data.ndim != 2:
        raise ValueError(
            f"Expected data with shape [time, nodes], got {data.shape}"
        )

    row_ok = np.empty(
        len(data),
        dtype=bool,
    )

    for start in range(0, len(data), chunk_size):

        end = min(
            start + chunk_size,
            len(data),
        )

        row_ok[start:end] = np.all(
            np.isfinite(data[start:end]),
            axis=1,
        )

    return row_ok


def valid_window_starts(
    data: np.ndarray,
    seq_len: int,
    pre_len: int,
) -> np.ndarray:
    """
    Find the start index of every valid sliding window.

    A window starting at s uses timesteps

        s, s + 1, ..., s + seq_len + pre_len - 1

    and is valid only when all of those timesteps are finite for
    every sensor.

    Parameters
    ----------
    data:
        Traffic data of ONE split with shape [time, nodes].

    seq_len:
        Number of historical time steps used as input.

    pre_len:
        Number of future time steps to predict.

    Returns
    -------
    starts:
        Ascending int64 array of valid window start indices,
        relative to the beginning of ``data``.
    """

    _validate_window_arguments(
        data,
        seq_len,
        pre_len,
    )

    window_length = seq_len + pre_len

    num_windows = len(data) - window_length + 1

    row_bad = ~finite_row_mask(data)

    # ---------------------------------------------------------
    # Number of non-finite timesteps inside every window,
    # computed with a cumulative sum.
    #
    # bad_in_window[s] = sum(row_bad[s : s + window_length])
    # ---------------------------------------------------------

    cumulative = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(row_bad, dtype=np.int64),
        )
    )

    bad_in_window = (
        cumulative[window_length : window_length + num_windows]
        - cumulative[:num_windows]
    )

    return np.flatnonzero(
        bad_in_window == 0
    ).astype(np.int64)


def valid_window_starts_from_rows(
    input_row_ok: np.ndarray,
    target_row_ok: np.ndarray,
    seq_len: int,
    pre_len: int,
) -> np.ndarray:
    """
    Valid window starts when inputs and targets have separate
    validity rules (used with gap-band filling):

        input timesteps  s .. s + seq_len - 1
            must be True in input_row_ok
            (finite for every sensor AFTER filling)

        target timesteps s + seq_len .. s + seq_len + pre_len - 1
            must be True in target_row_ok
            (OBSERVED for every sensor in the original data)

    With input_row_ok == target_row_ok this is identical to
    valid_window_starts().
    """

    input_row_ok = np.asarray(input_row_ok, dtype=bool)
    target_row_ok = np.asarray(target_row_ok, dtype=bool)

    if input_row_ok.shape != target_row_ok.shape:
        raise ValueError("Row masks must have the same shape.")

    if seq_len <= 0 or pre_len <= 0:
        raise ValueError("seq_len and pre_len must be greater than zero.")

    num_timesteps = len(input_row_ok)
    window_length = seq_len + pre_len

    if num_timesteps < window_length:
        raise ValueError(
            "Not enough timesteps to create a sequence: "
            f"need at least {window_length}, got {num_timesteps}"
        )

    num_windows = num_timesteps - window_length + 1

    def bad_counts(row_ok, offset, length):

        cumulative = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(~row_ok, dtype=np.int64),
            )
        )

        first = np.arange(num_windows) + offset

        return cumulative[first + length] - cumulative[first]

    bad = (
        bad_counts(input_row_ok, 0, seq_len)
        + bad_counts(target_row_ok, seq_len, pre_len)
    )

    return np.flatnonzero(bad == 0).astype(np.int64)


def gather_windows(
    data: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    pre_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gather the windows that begin at the given start indices.

    Parameters
    ----------
    data:
        Traffic data of ONE split with shape [time, nodes].

    starts:
        Window start indices relative to ``data``.

    seq_len:
        Number of historical time steps used as input.

    pre_len:
        Number of future time steps to predict.

    Returns
    -------
    X:
        Input windows with shape [len(starts), seq_len, nodes].

    y:
        Target windows with shape [len(starts), pre_len, nodes].
    """

    starts = np.asarray(
        starts,
        dtype=np.int64,
    )

    window_length = seq_len + pre_len

    if starts.size > 0 and (
        starts.min() < 0
        or starts.max() + window_length > len(data)
    ):
        raise ValueError(
            "Window start index out of range for the supplied data."
        )

    indices = (
        starts[:, None]
        + np.arange(window_length, dtype=np.int64)[None, :]
    )

    windows = data[indices]

    X = np.ascontiguousarray(
        windows[:, :seq_len],
        dtype=np.float32,
    )

    y = np.ascontiguousarray(
        windows[:, seq_len:],
        dtype=np.float32,
    )

    return X, y


def iterate_window_batches(
    data: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    pre_len: int,
    batch_size: int,
):
    """
    Yield (X, y) batches in the order of ``starts``.

    No shuffling is performed, matching the existing
    iterate_batches() behavior in main.py.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for begin in range(0, len(starts), batch_size):

        yield gather_windows(
            data,
            starts[begin : begin + batch_size],
            seq_len,
            pre_len,
        )


def missing_value_report(
    data: np.ndarray,
    seq_len: int,
    pre_len: int,
) -> dict:
    """
    Missing-value and window statistics for ONE split.

    Nothing is modified; this only counts.

    Returns
    -------
    dict
        JSON-serialisable statistics.
    """

    _validate_window_arguments(
        data,
        seq_len,
        pre_len,
    )

    num_timesteps, num_nodes = data.shape

    missing_per_sensor = np.zeros(
        num_nodes,
        dtype=np.int64,
    )

    for start in range(0, num_timesteps, 4096):

        missing_per_sensor += np.count_nonzero(
            ~np.isfinite(data[start : start + 4096]),
            axis=0,
        )

    row_ok = finite_row_mask(data)

    starts = valid_window_starts(
        data,
        seq_len,
        pre_len,
    )

    total_values = num_timesteps * num_nodes
    missing_values = int(missing_per_sensor.sum())

    total_windows = num_timesteps - (seq_len + pre_len) + 1
    valid_windows = int(len(starts))

    return {
        "num_timesteps": int(num_timesteps),
        "num_nodes": int(num_nodes),
        "total_values": int(total_values),
        "missing_values": missing_values,
        "missing_percentage": 100.0 * missing_values / total_values,
        "timesteps_with_missing": int(np.count_nonzero(~row_ok)),
        "sensors_with_missing": int(np.count_nonzero(missing_per_sensor)),
        "max_missing_per_sensor": int(missing_per_sensor.max()),
        "total_windows": int(total_windows),
        "valid_windows": valid_windows,
        "invalid_windows": int(total_windows - valid_windows),
        "valid_window_percentage": 100.0 * valid_windows / total_windows,
    }
