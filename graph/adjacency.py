import numpy as np


def validate_adjacency(
    adj: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """
    Validate an adjacency matrix against the expected
    number of graph nodes.

    Parameters
    ----------
    adj:
        Adjacency matrix with shape [nodes, nodes].

    num_nodes:
        Expected number of nodes.

    Returns
    -------
    adj:
        Validated adjacency matrix.
    """

    if adj.ndim != 2:
        raise ValueError(
            f"Adjacency matrix must be 2-dimensional, got {adj.shape}"
        )

    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(
            "Adjacency matrix shape does not match the number "
            f"of nodes: expected {(num_nodes, num_nodes)}, "
            f"got {adj.shape}"
        )

    if not np.all(np.isfinite(adj)):
        raise ValueError(
            "Adjacency matrix contains NaN or infinite values."
        )

    return adj.astype(np.float32)


def add_self_loops(adj: np.ndarray) -> np.ndarray:
    """
    Add self-loops to an adjacency matrix.

    Parameters
    ----------
    adj:
        Adjacency matrix with shape [nodes, nodes].

    Returns
    -------
    adj_with_loops:
        Adjacency matrix with self-loops added.
    """

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"Adjacency matrix must be square, got {adj.shape}"
        )

    adj_with_loops = adj.copy()

    np.fill_diagonal(adj_with_loops, 1.0)

    return adj_with_loops.astype(np.float32)