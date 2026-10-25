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

def graph_diagnostics(adj: np.ndarray) -> dict:
    """
    Statistics of a graph in the [target, source] convention
    (row = target, column = source; diagonal = self-loops).
    """

    adj = np.asarray(adj)
    num_nodes = adj.shape[0]

    off_diagonal = adj.copy()
    np.fill_diagonal(off_diagonal, 0.0)

    edges = off_diagonal != 0
    weights = off_diagonal[edges]

    in_degree = edges.sum(axis=1)    # sources per target
    out_degree = edges.sum(axis=0)   # targets per source

    return {
        "num_nodes": int(num_nodes),
        "num_edges_off_diagonal": int(edges.sum()),
        "density_off_diagonal": float(edges.sum() / (num_nodes * (num_nodes - 1))),
        "num_self_loops": int(np.count_nonzero(np.diag(adj))),
        "num_reciprocal_pairs": int((edges & edges.T).sum() // 2),
        "isolated_nodes": int(np.sum(~edges.any(axis=0) & ~edges.any(axis=1))),
        "is_symmetric": bool(np.array_equal(adj, adj.T)),
        "weight_min": float(weights.min()) if weights.size else 0.0,
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "weight_max": float(weights.max()) if weights.size else 0.0,
        "in_degree_mean": float(in_degree.mean()),
        "in_degree_max": int(in_degree.max()) if num_nodes else 0,
        "nodes_without_incoming": int(np.sum(in_degree == 0)),
        "out_degree_mean": float(out_degree.mean()),
        "out_degree_max": int(out_degree.max()) if num_nodes else 0,
        "nodes_without_outgoing": int(np.sum(out_degree == 0)),
    }


def check_model_graph(
    adj: np.ndarray,
    num_nodes: int,
    allow_self_loops: bool,
) -> dict:
    """
    Checks before a graph is given to A3T-GCN. Raises ValueError on:

        - wrong shape or non-finite values (validate_adjacency)
        - negative weights (A3T-GCN's D^-1/2 normalization needs
          non-negative weights; every graph here is non-negative)
        - self-loops in a causal graph (allow_self_loops=False);
          A3T-GCN adds the identity itself
        - no edges at all

    Directed graphs are NOT symmetrized and weights are NOT changed.
    Returns graph_diagnostics(adj).
    """

    adj = validate_adjacency(np.asarray(adj), num_nodes)

    if np.any(adj < 0):
        raise ValueError(
            f"Graph has {int(np.sum(adj < 0))} negative weights."
        )

    if not allow_self_loops and np.any(np.diag(adj) != 0):
        raise ValueError(
            "Causal graph has self-loops; they must be removed when the "
            "graph is built (A3T-GCN adds the identity)."
        )

    diagnostics = graph_diagnostics(adj)

    if diagnostics["num_edges_off_diagonal"] == 0:
        raise ValueError("Graph has no edges between different sensors.")

    return diagnostics
