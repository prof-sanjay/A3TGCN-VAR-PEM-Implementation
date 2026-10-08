from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VARLiNGAMResult:
    """
    Results produced by the VAR-LiNGAM causal discovery stage.

    Attributes
    ----------
    adjacency_matrix:
        Contemporaneous causal matrix B0.

        B0[target, source] represents the estimated causal
        effect of source on target.

    causal_order:
        Estimated causal ordering of the variables.

    residuals:
        The residual data used for causal discovery after
        removing samples that could not be used.

    valid_sample_mask:
        Boolean mask over the original residual samples.
        True means that the sample was used for LiNGAM.

    num_valid_samples:
        Number of samples used for causal discovery.
    """

    adjacency_matrix: np.ndarray
    causal_order: list[int]
    residuals: np.ndarray
    valid_sample_mask: np.ndarray
    num_valid_samples: int


class VARLiNGAM:
    """
    LiNGAM causal discovery applied to VAR residuals.

    The intended research pipeline is:

        Traffic X
          ↓
        VAR
          ↓
        residuals epsilon(t)
          ↓
        DirectLiNGAM
          ↓
        B0 + causal ordering

    No missing-value imputation is performed.

    Samples containing unavailable residuals are excluded from
    the causal-discovery dataset.
    """

    def __init__(
        self,
        random_state: int = 42,
    ) -> None:

        self.random_state = random_state

        try:
            from lingam import DirectLiNGAM
        except ImportError as exc:
            raise ImportError(
                "The 'lingam' package is required for VAR-LiNGAM. "
                "Install it with: pip install lingam"
            ) from exc

        self._DirectLiNGAM = DirectLiNGAM

    def _validate_residuals(
        self,
        residuals: np.ndarray,
    ) -> np.ndarray:
        """
        Validate the VAR residual matrix.

        Expected shape:

            (num_samples, num_nodes)
        """

        residuals = np.asarray(
            residuals,
            dtype=np.float64,
        )

        if residuals.ndim != 2:
            raise ValueError(
                "residuals must have shape "
                "(num_samples, num_nodes)."
            )

        num_samples, num_nodes = residuals.shape

        if num_samples < 2:
            raise ValueError(
                "At least two residual samples are required."
            )

        if num_nodes < 2:
            raise ValueError(
                "At least two variables are required "
                "for causal discovery."
            )

        return residuals

    def _build_valid_sample_mask(
        self,
        residuals: np.ndarray,
    ) -> np.ndarray:
        """
        Identify samples for which every variable is observed.

        No missing values are filled or modified.

        DirectLiNGAM operates on a complete data matrix, so for
        this initial implementation we use complete-case samples.
        """

        return np.all(
            np.isfinite(residuals),
            axis=1,
        )

    def fit(
        self,
        residuals: np.ndarray,
    ) -> VARLiNGAMResult:
        """
        Estimate the contemporaneous causal structure B0.

        Parameters
        ----------
        residuals:
            VAR residual matrix with shape:

                (num_samples, num_nodes)

            NaN values are allowed and are NOT imputed.

        Returns
        -------
        VARLiNGAMResult
        """

        residuals = self._validate_residuals(
            residuals
        )

        valid_sample_mask = self._build_valid_sample_mask(
            residuals
        )

        valid_residuals = residuals[
            valid_sample_mask
        ]

        if valid_residuals.shape[0] < 2:
            raise ValueError(
                "Fewer than two complete residual samples "
                "are available for LiNGAM."
            )

        model = self._DirectLiNGAM(
            random_state=self.random_state
        )

        model.fit(valid_residuals)

        adjacency_matrix = np.asarray(
            model.adjacency_matrix_,
            dtype=np.float64,
        )

        causal_order = [
            int(index)
            for index in model.causal_order_
        ]

        return VARLiNGAMResult(
            adjacency_matrix=adjacency_matrix,
            causal_order=causal_order,
            residuals=valid_residuals,
            valid_sample_mask=valid_sample_mask,
            num_valid_samples=valid_residuals.shape[0],
        )