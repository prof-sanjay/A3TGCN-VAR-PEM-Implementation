from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _validate_traffic_and_adjacency(traffic, adjacency):
    """Validate traffic and adjacency compatibility."""

    if traffic.ndim != 2:
        raise ValueError(
            f"Traffic data must be 2D, got shape {traffic.shape}"
        )

    if adjacency.ndim != 2:
        raise ValueError(
            f"Adjacency matrix must be 2D, got shape {adjacency.shape}"
        )

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"Adjacency matrix must be square, got shape {adjacency.shape}"
        )

    if traffic.shape[1] != adjacency.shape[0]:
        raise ValueError(
            "Traffic and adjacency dimensions do not match: "
            f"traffic has {traffic.shape[1]} nodes, "
            f"adjacency has {adjacency.shape[0]} nodes"
        )

    if not np.isfinite(adjacency).all():
        raise ValueError("Adjacency matrix contains NaN or Inf values")


def load_gba_dataset(
    traffic_path,
    adjacency_path,
    num_timesteps=20000,
    num_nodes=1000,
):
    """
    Load the LargeST-GBA traffic dataset.

    Parameters
    ----------
    traffic_path : str or Path
        Path to gba_his_raw_2021.h5.

    adjacency_path : str or Path
        Path to gba_rn_adj.npy.

    num_timesteps : int, default=20000
        Number of chronological timesteps to load.

    num_nodes : int, default=1000
        Number of sensors to load.

    Returns
    -------
    traffic : np.ndarray
        Traffic data with shape:
        (num_timesteps, num_nodes)

    adjacency : np.ndarray
        Road-network adjacency matrix with shape:
        (num_nodes, num_nodes)

    Notes
    -----
    The raw GBA traffic data may contain NaN values.
    NaNs are intentionally preserved. No imputation or
    interpolation is performed here.
    """

    traffic_path = Path(traffic_path)
    adjacency_path = Path(adjacency_path)

    if not traffic_path.exists():
        raise FileNotFoundError(
            f"GBA traffic file not found: {traffic_path}"
        )

    if not adjacency_path.exists():
        raise FileNotFoundError(
            f"GBA adjacency file not found: {adjacency_path}"
        )

    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be greater than 0")

    if num_nodes <= 0:
        raise ValueError("num_nodes must be greater than 0")

    # ---------------------------------------------------------
    # Load traffic from HDF5
    # ---------------------------------------------------------
    with h5py.File(traffic_path, "r") as h5_file:

        if "traffic" not in h5_file:
            raise KeyError(
                f"'traffic' dataset not found in {traffic_path}. "
                f"Available datasets: {list(h5_file.keys())}"
            )

        traffic_dataset = h5_file["traffic"]

        total_timesteps, total_nodes = traffic_dataset.shape

        if num_timesteps > total_timesteps:
            raise ValueError(
                f"Requested {num_timesteps} timesteps, "
                f"but dataset contains only {total_timesteps}"
            )

        if num_nodes > total_nodes:
            raise ValueError(
                f"Requested {num_nodes} nodes, "
                f"but dataset contains only {total_nodes}"
            )

        # Load only the requested subset.
        # NaN values are intentionally preserved.
        #
        # The HDF5 data is float64. It is read in row chunks and
        # cast to float32 chunk by chunk, so the full float64 copy
        # (about 2 GB for the full GBA year) is never held in memory.
        # The result is identical to casting the whole slice at once.
        traffic = np.empty(
            (num_timesteps, num_nodes),
            dtype=np.float32,
        )

        chunk_rows = 8192

        for start in range(0, num_timesteps, chunk_rows):

            end = min(start + chunk_rows, num_timesteps)

            traffic[start:end] = traffic_dataset[
                start:end,
                :num_nodes
            ]

    # ---------------------------------------------------------
    # Load road-network adjacency
    # ---------------------------------------------------------
    full_adjacency = np.load(adjacency_path)

    if full_adjacency.ndim != 2:
        raise ValueError(
            f"Adjacency matrix must be 2D, "
            f"got shape {full_adjacency.shape}"
        )

    if (
        full_adjacency.shape[0] < num_nodes
        or full_adjacency.shape[1] < num_nodes
    ):
        raise ValueError(
            f"Adjacency shape {full_adjacency.shape} is too small "
            f"for requested {num_nodes} nodes"
        )

    # Select the same first num_nodes sensors used in traffic.
    adjacency = full_adjacency[
        :num_nodes,
        :num_nodes
    ]

    adjacency = np.asarray(adjacency, dtype=np.float32)

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------
    _validate_traffic_and_adjacency(
        traffic,
        adjacency
    )

    return traffic, adjacency


def load_pems_bay_dataset(traffic_path, adjacency_path):
    """
    Load the PeMS-BAY dataset.

    Kept unchanged for backward compatibility.
    """

    traffic_path = Path(traffic_path)
    adjacency_path = Path(adjacency_path)

    if not traffic_path.exists():
        raise FileNotFoundError(
            f"Traffic file not found: {traffic_path}"
        )

    if not adjacency_path.exists():
        raise FileNotFoundError(
            f"Adjacency file not found: {adjacency_path}"
        )

    traffic = pd.read_hdf(
        traffic_path,
        key="speed"
    ).values.astype(np.float32)

    adjacency_data = pd.read_pickle(adjacency_path)

    adjacency = _extract_adjacency_matrix(
        adjacency_data
    ).astype(np.float32)

    _validate_traffic_and_adjacency(
        traffic,
        adjacency
    )

    return traffic, adjacency


def _extract_adjacency_matrix(adjacency_data):
    """Extract adjacency matrix from common PeMS-BAY formats."""

    if isinstance(adjacency_data, np.ndarray):
        return adjacency_data

    if isinstance(adjacency_data, dict):

        preferred_keys = [
            "adj_mx",
            "adjacency",
            "adj",
            "matrix",
        ]

        for key in preferred_keys:
            if key in adjacency_data:
                return np.asarray(
                    adjacency_data[key]
                )

        for value in adjacency_data.values():
            array = np.asarray(value)

            if array.ndim == 2 and array.shape[0] == array.shape[1]:
                return array

    if isinstance(adjacency_data, (tuple, list)):

        for value in adjacency_data:
            array = np.asarray(value)

            if array.ndim == 2 and array.shape[0] == array.shape[1]:
                return array

    raise ValueError(
        "Could not extract a square adjacency matrix "
        "from the supplied data"
    )


def load_numpy_dataset(traffic_path, adjacency_path):
    """
    Backward-compatible NumPy dataset loader.
    """

    traffic_path = Path(traffic_path)
    adjacency_path = Path(adjacency_path)

    if not traffic_path.exists():
        raise FileNotFoundError(
            f"Traffic file not found: {traffic_path}"
        )

    if not adjacency_path.exists():
        raise FileNotFoundError(
            f"Adjacency file not found: {adjacency_path}"
        )

    traffic = np.load(traffic_path)
    adjacency = np.load(adjacency_path)

    _validate_traffic_and_adjacency(
        traffic,
        adjacency
    )

    return (
        traffic.astype(np.float32),
        adjacency.astype(np.float32),
    )