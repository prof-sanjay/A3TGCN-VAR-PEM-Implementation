from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import ResidualBootstrap
from .bootstrap_support import BootstrapEdgeSupport
from .var_lingam import VARLiNGAM


@dataclass
class BootstrapLiNGAMResult:
    adjacency_matrices: np.ndarray
    edge_support: np.ndarray
    num_bootstrap_samples: int
    valid_runs: int


class BootstrapLiNGAM:
    """
    Run VAR-LiNGAM across bootstrap residual samples and
    calculate bootstrap edge support.

    The adjacency convention follows VAR-LiNGAM:

        B[k, j] != 0  means  j -> k
    """

    def __init__(
        self,
        num_bootstrap_samples: int = 10,
        random_state: int = 42,
        edge_threshold: float = 1e-8,
    ) -> None:

        if num_bootstrap_samples < 1:
            raise ValueError(
                "num_bootstrap_samples must be at least 1."
            )

        self.num_bootstrap_samples = num_bootstrap_samples
        self.random_state = random_state

        self.bootstrap = ResidualBootstrap(
            num_bootstrap_samples=num_bootstrap_samples,
            random_state=random_state,
        )

        self.support_calculator = BootstrapEdgeSupport(
            edge_threshold=edge_threshold,
        )

    def fit(
        self,
        residuals: np.ndarray,
    ) -> BootstrapLiNGAMResult:
        """
        Run bootstrap VAR-LiNGAM and calculate edge support.

        Parameters
        ----------
        residuals:
            VAR residual matrix with shape:

                (num_timesteps, num_nodes)

        Returns
        -------
        BootstrapLiNGAMResult
            Contains:

            adjacency_matrices:
                VAR-LiNGAM B0 matrices from each bootstrap run.

            edge_support:
                Bootstrap edge-support matrix.

            num_bootstrap_samples:
                Number of requested bootstrap runs.

            valid_runs:
                Number of successfully completed runs.
        """

        bootstrap_result = self.bootstrap.fit(residuals)

        bootstrap_samples = bootstrap_result.samples

        adjacency_matrices = []

        for bootstrap_index in range(
            bootstrap_samples.shape[0]
        ):
            sample = bootstrap_samples[bootstrap_index]

            lingam_model = VARLiNGAM(
                random_state=self.random_state + bootstrap_index
            )

            result = lingam_model.fit(sample)

            adjacency_matrices.append(
                result.adjacency_matrix
            )

        if len(adjacency_matrices) == 0:
            raise RuntimeError(
                "No bootstrap VAR-LiNGAM runs completed successfully."
            )

        adjacency_matrices = np.stack(
            adjacency_matrices,
            axis=0,
        )

        edge_support = self.support_calculator.calculate_support(
            adjacency_matrices
        )

        return BootstrapLiNGAMResult(
            adjacency_matrices=adjacency_matrices,
            edge_support=edge_support,
            num_bootstrap_samples=self.num_bootstrap_samples,
            valid_runs=adjacency_matrices.shape[0],
        )