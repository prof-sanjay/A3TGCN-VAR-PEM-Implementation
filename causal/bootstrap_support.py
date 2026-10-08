from __future__ import annotations

import numpy as np


class BootstrapEdgeSupport:
    """
    Calculate directed edge support from bootstrap causal graphs.

    For an edge j -> k:

        support[j, k] =
            number of bootstrap graphs containing j -> k
            ------------------------------------------------
            number of valid bootstrap graphs

    The adjacency convention follows VAR-LiNGAM:

        B[k, j] != 0  means  j -> k
    """

    def __init__(self, edge_threshold: float = 1e-8) -> None:
        if edge_threshold < 0:
            raise ValueError(
                "edge_threshold must be non-negative."
            )

        self.edge_threshold = edge_threshold

    def _validate_adjacency_matrices(
        self,
        adjacency_matrices: np.ndarray,
    ) -> np.ndarray:
        adjacency_matrices = np.asarray(
            adjacency_matrices,
            dtype=np.float64,
        )

        if adjacency_matrices.ndim != 3:
            raise ValueError(
                "adjacency_matrices must be a 3D array with shape "
                "(num_bootstrap_samples, num_nodes, num_nodes)."
            )

        num_samples, num_nodes, num_nodes_2 = (
            adjacency_matrices.shape
        )

        if num_samples < 1:
            raise ValueError(
                "At least one bootstrap adjacency matrix is required."
            )

        if num_nodes != num_nodes_2:
            raise ValueError(
                "Each adjacency matrix must be square."
            )

        if not np.all(np.isfinite(adjacency_matrices)):
            raise ValueError(
                "adjacency_matrices must contain only finite values."
            )

        return adjacency_matrices

    def calculate_support(
        self,
        adjacency_matrices: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate directed edge support.

        Parameters
        ----------
        adjacency_matrices:
            Array with shape
            (num_bootstrap_samples, num_nodes, num_nodes).

            For each matrix B:

                B[k, j] != 0  =>  j -> k

        Returns
        -------
        np.ndarray
            Support matrix with shape (num_nodes, num_nodes).

            support[k, j] is the proportion of bootstrap
            graphs containing the edge j -> k.
        """

        adjacency_matrices = self._validate_adjacency_matrices(
            adjacency_matrices
        )

        edge_present = (
            np.abs(adjacency_matrices) > self.edge_threshold
        )

        support = np.mean(
            edge_present.astype(np.float64),
            axis=0,
        )

        # A variable cannot be its own causal parent.
        np.fill_diagonal(support, 0.0)

        return support.astype(np.float64)

    def get_stable_edges(
        self,
        support: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Return a boolean matrix identifying stable directed edges.

        An edge is considered stable when its bootstrap support
        is greater than or equal to the specified threshold.
        """

        support = np.asarray(support, dtype=np.float64)

        if support.ndim != 2:
            raise ValueError(
                "support must be a 2D square matrix."
            )

        if support.shape[0] != support.shape[1]:
            raise ValueError(
                "support must be a square matrix."
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        stable_edges = support >= threshold

        np.fill_diagonal(stable_edges, False)

        return stable_edges