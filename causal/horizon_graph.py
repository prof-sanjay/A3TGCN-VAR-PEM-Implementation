from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HorizonGraphResult:
    """
    Horizon-specific causal graphs.

    adjacency_matrices:
        Dictionary mapping each forecasting horizon to its
        directed causal adjacency matrix.

        Example:

            {
                1:  A^(1),
                3:  A^(3),
                6:  A^(6),
                9:  A^(9),
                12: A^(12)
            }

    thresholds:
        Threshold used for each horizon.
    """

    adjacency_matrices: dict[int, np.ndarray]
    thresholds: dict[int, float]


class HorizonGraphBuilder:
    """
    Convert horizon-specific structural causal effects into
    directed horizon-specific graphs.

    Input:

        E^(h) = |Theta_h|

    Output:

        A^(h)

    where:

        A^(h)[i,j] = E^(h)[i,j]
                     if E^(h)[i,j] >= threshold

                     0 otherwise

    The direction is preserved:

        A[i,j] != 0

        means:

            node j -> node i

    This follows the VAR-LiNGAM convention used throughout
    the causal implementation.

    Self-loops are always removed.
    """

    def __init__(
        self,
        threshold: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        threshold:
            Minimum structural effect required for an edge
            to be retained.

            threshold = 0.0
                keeps every non-zero causal effect.

            threshold > 0.0
                removes weak causal effects.
        """

        if threshold < 0.0:
            raise ValueError(
                "threshold must be >= 0."
            )

        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_effect_matrix(
        self,
        effect_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Validate one horizon-specific effect matrix.
        """

        effect_matrix = np.asarray(
            effect_matrix,
            dtype=np.float64,
        )

        if effect_matrix.ndim != 2:
            raise ValueError(
                "Effect matrix must be a 2D array."
            )

        if effect_matrix.shape[0] != effect_matrix.shape[1]:
            raise ValueError(
                "Effect matrix must be square."
            )

        if effect_matrix.shape[0] < 1:
            raise ValueError(
                "Effect matrix must contain at least one node."
            )

        if not np.all(np.isfinite(effect_matrix)):
            raise ValueError(
                "Effect matrix must contain only finite values."
            )

        if np.any(effect_matrix < 0.0):
            raise ValueError(
                "Effect matrix must contain non-negative values."
            )

        return effect_matrix

    def _validate_horizon(
        self,
        horizon: int,
    ) -> int:
        """
        Validate one forecasting horizon.
        """

        horizon = int(horizon)

        if horizon < 1:
            raise ValueError(
                "Forecasting horizon must be >= 1."
            )

        return horizon

    # ------------------------------------------------------------------
    # Build one graph
    # ------------------------------------------------------------------

    def build_graph(
        self,
        effect_matrix: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """
        Convert one horizon-specific effect matrix into
        a directed causal adjacency matrix.

        Parameters
        ----------
        effect_matrix:
            Horizon-specific structural effect matrix:

                E^(h) = |Theta_h|

        threshold:
            Optional threshold overriding the default threshold.

        Returns
        -------
        adjacency:
            Directed weighted causal adjacency matrix.
        """

        effect_matrix = self._validate_effect_matrix(
            effect_matrix
        )

        if threshold is None:
            threshold = self.threshold

        threshold = float(threshold)

        if threshold < 0.0:
            raise ValueError(
                "threshold must be >= 0."
            )

        # Keep effects that satisfy the threshold.
        adjacency = np.where(
            effect_matrix >= threshold,
            effect_matrix,
            0.0,
        )

        # A node is not considered its own causal parent.
        np.fill_diagonal(
            adjacency,
            0.0,
        )

        return adjacency

    # ------------------------------------------------------------------
    # Build graphs for all horizons
    # ------------------------------------------------------------------

    def fit(
        self,
        effect_matrices: dict[int, np.ndarray],
        threshold: float | None = None,
    ) -> HorizonGraphResult:
        """
        Build horizon-specific causal graphs.

        Parameters
        ----------
        effect_matrices:
            Dictionary produced by StructuralIRF:

                {
                    horizon: effect_matrix
                }

            Example:

                {
                    1:  E^(1),
                    3:  E^(3),
                    6:  E^(6),
                    9:  E^(9),
                    12: E^(12)
                }

        threshold:
            Optional common threshold for all horizons.

        Returns
        -------
        HorizonGraphResult
        """

        if not isinstance(effect_matrices, dict):
            raise ValueError(
                "effect_matrices must be a dictionary."
            )

        if len(effect_matrices) == 0:
            raise ValueError(
                "At least one horizon-specific effect matrix "
                "must be provided."
            )

        if threshold is None:
            threshold = self.threshold

        threshold = float(threshold)

        if threshold < 0.0:
            raise ValueError(
                "threshold must be >= 0."
            )

        adjacency_matrices: dict[int, np.ndarray] = {}
        thresholds: dict[int, float] = {}

        reference_shape = None

        for horizon, effect_matrix in effect_matrices.items():

            horizon = self._validate_horizon(
                horizon
            )

            effect_matrix = self._validate_effect_matrix(
                effect_matrix
            )

            # All horizons must describe the same graph.
            if reference_shape is None:
                reference_shape = effect_matrix.shape
            elif effect_matrix.shape != reference_shape:
                raise ValueError(
                    "All horizon effect matrices must have "
                    "the same shape."
                )

            adjacency = self.build_graph(
                effect_matrix,
                threshold=threshold,
            )

            adjacency_matrices[horizon] = adjacency
            thresholds[horizon] = threshold

        return HorizonGraphResult(
            adjacency_matrices=adjacency_matrices,
            thresholds=thresholds,
        )