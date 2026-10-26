"""
Graphs given to A3T-GCN in the E0 / E1 / E2 ablation.

Convention for every graph built here:

    G[target, source] != 0   means   source -> target

which is the convention of the VAR coefficients (causal/var.py), of
PEM-LBN (causal/pem.py) and of the VAR neighbour rule, where a road
adjacency entry A[i, j] != 0 lets sensor j predict sensor i.

    E0 : road-network adjacency A                     (unchanged)
    E1 : VAR graph    W = sum_k |A_k|, diagonal 0      (decision D1)
    E2 : VAR-LiNGAM   sum_k |B_k| + |B_0|, B_k = (I - B_0) A_k
    E3 : PEM graph    P * D, diagonal 0                (the former E2:
         symmetric PEM inclusion probability P times directed PEM-LBN
         adjacency D)
    E4 : VAR-LiNGAM + PEM   sum_k |B_k| + P * |B_0|

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


def structural_lag_matrices(
    coefficients: np.ndarray,
    b0: np.ndarray,
) -> np.ndarray:
    """
    VAR-LiNGAM lagged structural effects B_k = (I - B_0) A_k
    (Hyvarinen et al. 2010), for every lag k.

    coefficients: reduced-form VAR A_k, shape (lags, n, n), [target, source]
    b0:           contemporaneous structural matrix, (n, n), [target, source]

    B_k may contain two-step entries (a lagged effect passed on within
    the same 5 minutes), so its support can exceed the road neighbours
    used by the VAR.
    """

    coefficients = np.asarray(coefficients, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)

    if coefficients.ndim != 3 or b0.shape != coefficients.shape[1:]:
        raise ValueError("coefficients (lags, n, n) and b0 (n, n) do not match.")

    left = np.eye(b0.shape[0]) - b0

    return np.stack([left @ a_k for a_k in coefficients])


def var_lingam_graph(
    coefficients: np.ndarray,
    b0: np.ndarray,
) -> np.ndarray:
    """
    E2: sum_k |B_k| + |B_0|, B_k = (I - B_0) A_k, diagonal 0.
    """

    graph = np.abs(structural_lag_matrices(coefficients, b0)).sum(axis=0)
    graph += np.abs(np.asarray(b0, dtype=np.float64))

    np.fill_diagonal(graph, 0.0)

    return graph.astype(np.float32)


def var_lingam_pem_graph(
    coefficients: np.ndarray,
    b0: np.ndarray,
    inclusion_probabilities: np.ndarray,
) -> np.ndarray:
    """
    E4: sum_k |B_k| + P * |B_0|, diagonal 0.

    PEM's symmetric inclusion probability P weights only the
    contemporaneous effects; the lagged effects B_k use the full B_0
    (the structural VAR-LiNGAM model), so the forecasting-relevant
    lagged information of E2 is kept.
    """

    b0 = np.asarray(b0, dtype=np.float64)
    probabilities = np.asarray(inclusion_probabilities, dtype=np.float64)

    if probabilities.shape != b0.shape:
        raise ValueError("inclusion probabilities and b0 do not match.")

    graph = np.abs(structural_lag_matrices(coefficients, b0)).sum(axis=0)
    graph += probabilities * np.abs(b0)

    np.fill_diagonal(graph, 0.0)

    return graph.astype(np.float32)


def top_k_sources(
    effects: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Keep, for every target (row), the k sources with the largest
    effect; diagonal 0. Ties are broken by the lower source index.
    """

    effects = np.abs(np.asarray(effects, dtype=np.float64)).copy()
    np.fill_diagonal(effects, -np.inf)

    n = effects.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, {n - 1}]")

    top = np.argsort(-effects, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n), k)

    graph = np.zeros((n, n), dtype=np.float64)
    graph[rows, top.ravel()] = effects[rows, top.ravel()]
    graph[graph < 0] = 0.0

    return graph.astype(np.float32)


def horizon_graphs(
    coefficients: np.ndarray,
    b0: np.ndarray,
    inclusion_probabilities: np.ndarray,
    horizons,
    k: int,
) -> dict[int, np.ndarray]:
    """
    E5: one graph per forecast horizon h (steps of 5 minutes).

        Theta_h = Psi_h (I - P * B_0)^{-1}      structural impulse responses
                                               (causal/horizon.py)
        G_h     = |Theta_h| off-diagonal, top-k sources per target

    Psi_h are the reduced-form impulse responses of the VAR. As in E4,
    PEM's inclusion probability P weights the contemporaneous effects
    B_0. |Theta_h[i, j]| is the effect of a shock at sensor j on sensor
    i h steps later; it is dense (effects spread over the network), so
    each target keeps its k strongest sources.
    """

    from .horizon import StructuralIRF

    weighted_b0 = (
        np.asarray(inclusion_probabilities, dtype=np.float64)
        * np.asarray(b0, dtype=np.float64)
    )
    np.fill_diagonal(weighted_b0, 0.0)

    result = StructuralIRF().fit(coefficients, weighted_b0, list(horizons))

    return {
        int(h): top_k_sources(result.effect_matrices[h], k)
        for h in horizons
    }


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
