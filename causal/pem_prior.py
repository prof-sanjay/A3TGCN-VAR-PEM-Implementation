from __future__ import annotations

import numpy as np


class PEMPrior:
    """
    Construct the symmetric PEM prior-inclusion matrix eta
    from directed bootstrap edge support.

    Bootstrap support convention:
        support[target, source] = support of source -> target

    PEM prior:
        eta[j, k] = max(
            support[j, k],
            support[k, j]
        )

    The resulting eta matrix is symmetric.
    """

    def __init__(self) -> None:
        pass

    def _validate_support(self, support: np.ndarray) -> np.ndarray:
        support = np.asarray(support, dtype=np.float64)

        if support.ndim != 2:
            raise ValueError(
                "Edge-support matrix must be 2-dimensional."
            )

        if support.shape[0] != support.shape[1]:
            raise ValueError(
                "Edge-support matrix must be square."
            )

        if not np.all(np.isfinite(support)):
            raise ValueError(
                "Edge-support matrix must contain only finite values."
            )

        if np.any(support < 0.0) or np.any(support > 1.0):
            raise ValueError(
                "Edge-support values must be in the range [0, 1]."
            )

        return support

    def calculate_eta(self, support: np.ndarray) -> np.ndarray:
        """
        Convert directed bootstrap support into the symmetric
        PEM prior-inclusion matrix eta.

        Parameters
        ----------
        support : np.ndarray
            Directed bootstrap edge-support matrix.

            support[target, source] represents the bootstrap
            support for source -> target.

        Returns
        -------
        np.ndarray
            Symmetric prior-inclusion matrix eta.
        """

        support = self._validate_support(support)

        # PEM uses the maximum support from either direction.
        eta = np.maximum(
            support,
            support.T,
        )

        # Self-edges are not treated as causal edges.
        np.fill_diagonal(eta, 0.0)

        return eta

    def get_prior_edges(
        self,
        eta: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Return a boolean matrix indicating which undirected
        edges have sufficient prior support.

        Parameters
        ----------
        eta : np.ndarray
            Symmetric PEM prior-inclusion matrix.

        threshold : float
            Minimum eta value required for a prior edge.

        Returns
        -------
        np.ndarray
            Symmetric boolean edge matrix.
        """

        eta = self._validate_support(eta)

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be in the range [0, 1]."
            )

        prior_edges = eta >= threshold

        # No self-edges.
        np.fill_diagonal(prior_edges, False)

        return prior_edges


def road_prior_eta(
    adjacency: np.ndarray,
    high: float = 0.7,
    low: float = 0.3,
) -> np.ndarray:
    """
    Symmetric PEM prior-inclusion matrix eta from a directed
    (possibly asymmetric) road adjacency.

    Directional plausibility (PEM paper, Section 3):

        w[target, source] = high   if adjacency[target, source] != 0
                            low    otherwise

    (adjacency[i, j] != 0 means j -> i, the convention of
    causal/graph_builders.py.) The undirected inclusion strength is

        eta[j, k] = max(w[j, k], w[k, j])

    i.e. eta = high when a road link exists in EITHER direction and
    low otherwise. Direction is left to the PEM-LBN ordering step.
    The diagonal is 0 (self-edges are not causal edges).
    """

    if not 0.0 < low < high < 1.0:
        raise ValueError(
            "Require 0 < low < high < 1."
        )

    adjacency = np.asarray(adjacency)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "adjacency must be a square matrix."
        )

    support = np.where(
        adjacency != 0,
        high,
        low,
    ).astype(np.float64)

    return PEMPrior().calculate_eta(support)