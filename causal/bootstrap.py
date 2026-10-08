from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapResult:
    samples: np.ndarray
    valid_residuals: np.ndarray
    num_samples: int
    random_state: int


class ResidualBootstrap:
    """
    Bootstrap VAR residual observations with replacement.

    Only complete residual rows are used. Missing values are not imputed.
    """

    def __init__(
        self,
        num_bootstrap_samples: int = 10,
        random_state: int = 42,
    ) -> None:
        if num_bootstrap_samples < 1:
            raise ValueError(
                "num_bootstrap_samples must be at least 1."
            )

        self.num_bootstrap_samples = num_bootstrap_samples
        self.random_state = random_state

    def _validate_residuals(
        self,
        residuals: np.ndarray,
    ) -> np.ndarray:
        residuals = np.asarray(residuals, dtype=np.float64)

        if residuals.ndim != 2:
            raise ValueError(
                "residuals must be a 2D array with shape "
                "(num_timesteps, num_nodes)."
            )

        if residuals.shape[0] < 2:
            raise ValueError(
                "At least two residual observations are required."
            )

        if residuals.shape[1] < 1:
            raise ValueError(
                "Residuals must contain at least one variable."
            )

        return residuals

    def _get_valid_residuals(
        self,
        residuals: np.ndarray,
    ) -> np.ndarray:
        valid_mask = np.all(np.isfinite(residuals), axis=1)
        valid_residuals = residuals[valid_mask]

        if valid_residuals.shape[0] < 2:
            raise ValueError(
                "At least two complete residual observations are "
                "required for bootstrap sampling."
            )

        return valid_residuals

    def fit(
        self,
        residuals: np.ndarray,
    ) -> BootstrapResult:
        residuals = self._validate_residuals(residuals)

        valid_residuals = self._get_valid_residuals(residuals)

        num_valid_samples = valid_residuals.shape[0]
        rng = np.random.default_rng(self.random_state)

        bootstrap_samples = np.empty(
            (
                self.num_bootstrap_samples,
                num_valid_samples,
                valid_residuals.shape[1],
            ),
            dtype=np.float64,
        )

        for bootstrap_index in range(self.num_bootstrap_samples):
            indices = rng.integers(
                low=0,
                high=num_valid_samples,
                size=num_valid_samples,
            )

            bootstrap_samples[bootstrap_index] = (
                valid_residuals[indices]
            )

        return BootstrapResult(
            samples=bootstrap_samples,
            valid_residuals=valid_residuals,
            num_samples=self.num_bootstrap_samples,
            random_state=self.random_state,
        )