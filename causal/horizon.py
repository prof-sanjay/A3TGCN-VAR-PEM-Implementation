from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HorizonIRFResult:
    """
    Results of structural impulse-response calculation.

    psi:
        Reduced-form impulse responses.
        Shape: (max_horizon + 1, num_nodes, num_nodes)

    theta:
        Structural impulse responses.
        Shape: (max_horizon + 1, num_nodes, num_nodes)

    effect_matrices:
        Horizon-specific absolute structural effects.

        For each requested horizon h:

            effect[i, j] = |Theta_h[i, j]|, i != j

        Diagonal entries are set to zero.
    """

    psi: np.ndarray
    theta: np.ndarray
    effect_matrices: dict[int, np.ndarray]


class StructuralIRF:
    """
    Calculate structural impulse responses for a VAR model.

    Reduced-form VAR:

        X_t = M_1 X_{t-1} + ... + M_p X_{t-p} + u_t

    Reduced-form impulse responses:

        Psi_0 = I

        Psi_s =
            sum_{k=1}^{min(s,p)}
                M_k Psi_{s-k}

    Structural transformation:

        Theta_s = Psi_s (I - B_0)^(-1)

    where:

        B_0
            is the contemporaneous causal-effect matrix.

    For traffic data sampled every 5 minutes:

        h = 1  -> 5 minutes
        h = 3  -> 15 minutes
        h = 6  -> 30 minutes
        h = 9  -> 45 minutes
        h = 12 -> 60 minutes
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_var_coefficients(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """
        Validate VAR coefficient matrix.

        Expected shape:

            (lag_order, num_nodes, num_nodes)
        """

        coefficients = np.asarray(
            coefficients,
            dtype=np.float64,
        )

        if coefficients.ndim != 3:
            raise ValueError(
                "VAR coefficients must be a 3D array with shape "
                "(lag_order, num_nodes, num_nodes)."
            )

        lag_order = coefficients.shape[0]
        num_nodes = coefficients.shape[1]

        if lag_order < 1:
            raise ValueError(
                "lag_order must be at least 1."
            )

        if coefficients.shape[1] != coefficients.shape[2]:
            raise ValueError(
                "VAR coefficient matrices must be square."
            )

        if num_nodes < 1:
            raise ValueError(
                "Number of nodes must be at least 1."
            )

        if not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "VAR coefficients must contain only finite values."
            )

        return coefficients

    def _validate_b0(
        self,
        b0: np.ndarray,
        num_nodes: int,
    ) -> np.ndarray:
        """
        Validate contemporaneous structural matrix B0.
        """

        b0 = np.asarray(
            b0,
            dtype=np.float64,
        )

        if b0.ndim != 2:
            raise ValueError(
                "B0 must be a 2D square matrix."
            )

        if b0.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"B0 must have shape "
                f"({num_nodes}, {num_nodes}), "
                f"got {b0.shape}."
            )

        if not np.all(np.isfinite(b0)):
            raise ValueError(
                "B0 must contain only finite values."
            )

        return b0

    def _validate_horizons(
        self,
        horizons,
    ) -> list[int]:
        """
        Validate requested positive forecasting horizons.

        Horizon 0 is calculated internally because it is needed
        for structural validation, but it is not treated as a
        forecasting horizon.
        """

        horizons = [int(horizon) for horizon in horizons]

        if len(horizons) == 0:
            raise ValueError(
                "At least one horizon must be provided."
            )

        if any(horizon < 1 for horizon in horizons):
            raise ValueError(
                "Forecasting horizons must be >= 1."
            )

        if len(set(horizons)) != len(horizons):
            raise ValueError(
                "Horizon values must be unique."
            )

        return horizons

    # ------------------------------------------------------------------
    # Reduced-form impulse responses
    # ------------------------------------------------------------------

    def _calculate_psi(
        self,
        coefficients: np.ndarray,
        max_horizon: int,
    ) -> np.ndarray:
        """
        Calculate reduced-form impulse responses.

            Psi_0 = I

            Psi_s =
                sum_{k=1}^{min(s,p)}
                    M_k Psi_{s-k}

        Parameters
        ----------
        coefficients:
            VAR coefficient matrices with shape
            (lag_order, num_nodes, num_nodes).

        max_horizon:
            Maximum horizon to calculate.

        Returns
        -------
        psi:
            Array with shape
            (max_horizon + 1, num_nodes, num_nodes).
        """

        lag_order = coefficients.shape[0]
        num_nodes = coefficients.shape[1]

        psi = np.zeros(
            (
                max_horizon + 1,
                num_nodes,
                num_nodes,
            ),
            dtype=np.float64,
        )

        # Psi_0 = I
        psi[0] = np.eye(
            num_nodes,
            dtype=np.float64,
        )

        # Calculate Psi_1 ... Psi_h
        for s in range(1, max_horizon + 1):

            maximum_lag = min(
                s,
                lag_order,
            )

            for k in range(1, maximum_lag + 1):

                # M_k corresponds to coefficients[k - 1]
                psi[s] += (
                    coefficients[k - 1]
                    @ psi[s - k]
                )

        return psi

    # ------------------------------------------------------------------
    # Structural impulse responses
    # ------------------------------------------------------------------

    def _calculate_theta(
        self,
        psi: np.ndarray,
        b0: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate structural impulse responses.

            Theta_s = Psi_s (I - B0)^(-1)

        The inverse is NOT explicitly calculated.

        Instead, for each horizon s we solve:

            (I - B0)^T X = Psi_s^T

        and then transpose the result:

            X^T = Psi_s (I - B0)^(-1)

        This avoids explicitly forming the matrix inverse.
        """

        num_horizons = psi.shape[0]
        num_nodes = psi.shape[1]

        structural_matrix = (
            np.eye(
                num_nodes,
                dtype=np.float64,
            )
            - b0
        )

        theta = np.zeros_like(psi)

        for s in range(num_horizons):

            theta[s] = np.linalg.solve(
                structural_matrix.T,
                psi[s].T,
            ).T

        return theta

    # ------------------------------------------------------------------
    # Horizon-specific effect matrices
    # ------------------------------------------------------------------

    def _build_effect_matrices(
        self,
        theta: np.ndarray,
        horizons: list[int],
    ) -> dict[int, np.ndarray]:
        """
        Convert structural impulse responses into
        horizon-specific absolute effect matrices.

            E^(h)_{ij}
                = |Theta_h[i,j]|, i != j

        Diagonal entries are set to zero because a node's
        self-response is not treated as a causal graph edge.
        """

        effect_matrices: dict[int, np.ndarray] = {}

        for horizon in horizons:

            effect = np.abs(
                theta[horizon]
            )

            # Remove self-effects.
            np.fill_diagonal(
                effect,
                0.0,
            )

            effect_matrices[horizon] = effect

        return effect_matrices

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def fit(
        self,
        coefficients: np.ndarray,
        b0: np.ndarray,
        horizons,
    ) -> HorizonIRFResult:
        """
        Calculate structural impulse responses and
        horizon-specific causal effect matrices.

        Parameters
        ----------
        coefficients:
            VAR coefficient matrices:

                (lag_order, num_nodes, num_nodes)

        b0:
            Contemporaneous causal matrix:

                (num_nodes, num_nodes)

        horizons:
            Positive forecasting horizons.

            Example:

                [1, 3, 6, 9, 12]

            for 5, 15, 30, 45, and 60 minutes.

        Returns
        -------
        HorizonIRFResult
        """

        # --------------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------------

        coefficients = self._validate_var_coefficients(
            coefficients
        )

        num_nodes = coefficients.shape[1]

        b0 = self._validate_b0(
            b0,
            num_nodes,
        )

        horizons = self._validate_horizons(
            horizons
        )

        max_horizon = max(horizons)

        # --------------------------------------------------------------
        # Reduced-form IRFs
        # --------------------------------------------------------------

        psi = self._calculate_psi(
            coefficients,
            max_horizon,
        )

        # --------------------------------------------------------------
        # Structural IRFs
        # --------------------------------------------------------------

        theta = self._calculate_theta(
            psi,
            b0,
        )

        # --------------------------------------------------------------
        # Horizon-specific causal effects
        # --------------------------------------------------------------

        effect_matrices = self._build_effect_matrices(
            theta,
            horizons,
        )

        return HorizonIRFResult(
            psi=psi,
            theta=theta,
            effect_matrices=effect_matrices,
        )