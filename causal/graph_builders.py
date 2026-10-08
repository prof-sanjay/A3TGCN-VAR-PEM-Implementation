"""
Graphs given to A3T-GCN in the E0 / E1 / E2 ablation.

Convention for every graph built here:

    G[target, source] != 0   means   source -> target

which is the convention of the VAR coefficients (causal/var.py), of
PEM-LBN (causal/pem.py) and of the VAR neighbour rule, where a road
adjacency entry A[i, j] != 0 lets sensor j predict sensor i.

    E0 : road-network adjacency A                     (unchanged)
    E1 : VAR graph    W = sum_k |A_k|, diagonal 0      (decision D1)
    E2 : PEM graph    P * D, diagonal 0                (formula of the
         original E2 pipeline: symmetric PEM inclusion probability P
         times directed PEM-LBN adjacency D)

Orientation for the model (decision D2):
    graph/normalization.normalized_adj computes D^-1/2 A^T D^-1/2, so
    the T-GCN layer aggregates node i from the nodes j with A[j, i] != 0.
    To make every node aggregate from its sources (parents), each graph
    is transposed exactly once by to_model_orientation() before it is
    given to A3T-GCN. This is applied identically to E0, E1 and E2;
    the A3T-GCN code itself is unchanged.
"""

from __future__ import annotations

import numpy as np


def road_graph(
    adjacency: np.ndarray,
) -> np.ndarray:
    """E0: the road-network adjacency, unchanged."""

    return np.asarray(adjacency, dtype=np.float32).copy()


def var_graph(
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    E1: W[target, source] = sum over lags of |A_k[target, source]|,
    with self-effects removed (A3T-GCN adds self-loops itself).
    """

    coefficients = np.asarray(coefficients, dtype=np.float64)

    if coefficients.ndim != 3:
        raise ValueError(
            "coefficients must have shape (lag_order, nodes, nodes)."
        )

    graph = np.abs(coefficients).sum(axis=0)

    np.fill_diagonal(graph, 0.0)

    return graph.astype(np.float32)


def pem_graph(
    inclusion_probabilities: np.ndarray,
    directed_adjacency: np.ndarray,
) -> np.ndarray:
    """
    E2: P * D with the diagonal removed, exactly as in the original
    E2 pipeline (build_e2_causal_graph in the baseline main.py).

    P: symmetric PEM posterior inclusion probabilities.
    D: directed PEM-LBN adjacency, D[target, source] = 1.
    """

    graph = (
        np.asarray(inclusion_probabilities, dtype=np.float64)
        * np.asarray(directed_adjacency, dtype=np.float64)
    )

    np.fill_diagonal(graph, 0.0)

    return graph.astype(np.float32)


def to_model_orientation(
    graph: np.ndarray,
) -> np.ndarray:
    """
    Transpose a [target, source] graph for A3T-GCN, whose
    normalization transposes it again (decision D2).
    """

    return np.ascontiguousarray(np.asarray(graph, dtype=np.float32).T)


def save_graph_artifact(
    path,
    graph: np.ndarray,
    experiment: str,
    train_end: int,
    sensors: np.ndarray,
    **metadata,
) -> None:
    """
    Save a graph in the artifact format read by main.load_graph().
    ``graph`` is stored in the [target, source] convention.
    """

    np.savez(
        path,
        adjacency=np.asarray(graph, dtype=np.float32),
        experiment=experiment,
        train_end=int(train_end),
        num_nodes=int(graph.shape[0]),
        sensors=np.asarray(sensors, dtype=np.int64),
        orientation="target_source",
        **metadata,
    )
