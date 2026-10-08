import numpy as np
from dataclasses import dataclass

from .causal_units import CausalUnit
from .pem import PEM


@dataclass
class LocalPEMResult:
    """
    Result of applying PEM independently to topology-defined
    causal units.

    global_inclusion_probabilities:
        Global symmetric PEM posterior inclusion probabilities.

        Entry [i, j] represents the posterior probability that
        the pair {i, j} belongs to the slab/edge component.

        This matrix is NOT directional.

    global_adjacency:
        Global directed DAG recovered by PEM-LBN.

        Convention:
            adjacency[target, source] = 1

        therefore:
            adjacency[i, j] = 1

        means:
            j -> i

    unit_inclusion_probabilities:
        Local PEM posterior inclusion matrices, one per
        causal unit.

        These matrices are symmetric and are NOT directional.

    unit_adjacencies:
        Local directed PEM-LBN adjacency matrices, one per
        causal unit.

    unit_causal_orders:
        Causal/topological ordering recovered by PEM for each
        causal unit, expressed using local node indices.

    valid_sample_counts:
        Number of complete observations used for each unit.

    processed_units:
        Number of units successfully processed.
    """

    global_inclusion_probabilities: np.ndarray
    global_adjacency: np.ndarray

    unit_inclusion_probabilities: list[np.ndarray]
    unit_adjacencies: list[np.ndarray]
    unit_causal_orders: list[list[int]]

    valid_sample_counts: np.ndarray
    processed_units: int


class LocalPEM:
    """
    Apply PEM independently to topology-restricted causal units.

    The PEM implementation itself is unchanged. This class handles:

    1. Extracting the residuals belonging to each causal unit.
    2. Removing incomplete observations for that unit.
    3. Running PEM on the resulting local matrix.
    4. Mapping local node indices back to global sensor indices.
    5. Aggregating overlapping local edge probabilities.
    6. Aggregating the directed PEM-LBN adjacency matrices.

    Missing observations are NOT imputed.

    Important
    ---------
    PEM posterior inclusion probabilities are symmetric because
    they are obtained from the symmetric precision matrix.

    Direction is obtained separately through the PEM-LBN
    ordering and parent-selection procedure.
    """

    def __init__(
        self,
        units: list[CausalUnit],
        nu0: float = 0.05,
        nu1: float = 1.0,
        tau: float = 0.01,
        threshold: float = 0.5,
        max_iter: int = 500,
        tolerance: float = 1e-6,
        spectral_norm_bound: float | None = None,
        prior_inclusion: float = 0.5,
        optimizer: str = "SLSQP",
    ) -> None:

        if not 0.0 < prior_inclusion < 1.0:
            raise ValueError(
                "prior_inclusion must be strictly between 0 and 1."
            )

        self.units = units
        self.prior_inclusion = float(prior_inclusion)
        self.optimizer = optimizer

        self.nu0 = nu0
        self.nu1 = nu1
        self.tau = tau
        self.threshold = threshold
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.spectral_norm_bound = spectral_norm_bound

    def _validate_residuals(
        self,
        residuals: np.ndarray,
    ) -> None:

        residuals = np.asarray(
            residuals,
            dtype=np.float64,
        )

        if residuals.ndim != 2:
            raise ValueError(
                "residuals must be a 2D array "
                "with shape (time, nodes)."
            )

        if len(self.units) == 0:
            raise ValueError(
                "At least one causal unit is required."
            )

        num_nodes = residuals.shape[1]

        for unit in self.units:

            if np.any(unit.nodes < 0):
                raise ValueError(
                    "Causal-unit node indices must be non-negative."
                )

            if np.any(unit.nodes >= num_nodes):
                raise ValueError(
                    "Causal-unit node index exceeds "
                    "the number of residual variables."
                )

    def _extract_complete_samples(
        self,
        residuals: np.ndarray,
        nodes: np.ndarray,
    ) -> np.ndarray:
        """
        Extract a local residual matrix and retain only rows
        where every variable in the unit is finite.
        """

        local_residuals = residuals[:, nodes]

        complete_mask = np.all(
            np.isfinite(local_residuals),
            axis=1,
        )

        return local_residuals[complete_mask]

    def _run_pem(
        self,
        local_residuals: np.ndarray,
    ):
        """
        Run the existing PEM implementation on one local unit.

        A uniform prior eta = prior_inclusion (default 0.5) is used
        here because this class is the local PEM execution layer. Construction of informative
        eta values is kept separate.

        Returns
        -------
        PEMResult
            Full PEM result, including:

            - precision matrix
            - covariance matrix
            - symmetric inclusion probabilities
            - directed PEM-LBN adjacency
            - causal ordering
        """

        num_variables = local_residuals.shape[1]

        eta = np.full(
            (num_variables, num_variables),
            self.prior_inclusion,
            dtype=np.float64,
        )

        np.fill_diagonal(
            eta,
            0.0,
        )

        model = PEM(
            eta=eta,
            nu0=self.nu0,
            nu1=self.nu1,
            tau=self.tau,
            threshold=self.threshold,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            spectral_norm_bound=self.spectral_norm_bound,
            optimizer=self.optimizer,
        )

        result = model.fit(local_residuals)

        self.slsqp_retries = getattr(self, "slsqp_retries", 0) + model.slsqp_retries

        return result

    def fit(
        self,
        residuals: np.ndarray,
    ) -> LocalPEMResult:
        """
        Run PEM independently for every causal unit.

        Parameters
        ----------
        residuals:
            Global residual matrix with shape
            (time, num_nodes).

        Returns
        -------
        LocalPEMResult
            Contains both:

            1. Symmetric PEM inclusion probabilities.
            2. Directed PEM-LBN adjacency matrices.
        """

        residuals = np.asarray(
            residuals,
            dtype=np.float64,
        )

        self._validate_residuals(
            residuals
        )

        num_nodes = residuals.shape[1]

        # --------------------------------------------------------------
        # Global symmetric inclusion probabilities
        # --------------------------------------------------------------

        global_probabilities = np.zeros(
            (num_nodes, num_nodes),
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Global directed PEM-LBN adjacency
        #
        # Convention:
        #
        #     adjacency[target, source] = 1
        #
        # therefore:
        #
        #     adjacency[i, j] = 1
        #
        # means:
        #
        #     j -> i
        # --------------------------------------------------------------

        global_adjacency = np.zeros(
            (num_nodes, num_nodes),
            dtype=np.float64,
        )

        unit_probabilities = []
        unit_adjacencies = []
        unit_causal_orders = []

        valid_sample_counts = np.zeros(
            len(self.units),
            dtype=np.int64,
        )

        processed_units = 0

        for unit_index, unit in enumerate(self.units):

            nodes = np.asarray(
                unit.nodes,
                dtype=np.int64,
            )

            # --------------------------------------------------
            # Single-variable units cannot contain a causal edge.
            # --------------------------------------------------

            if len(nodes) < 2:

                unit_probabilities.append(
                    np.zeros(
                        (len(nodes), len(nodes)),
                        dtype=np.float64,
                    )
                )

                unit_adjacencies.append(
                    np.zeros(
                        (len(nodes), len(nodes)),
                        dtype=np.float64,
                    )
                )

                if len(nodes) == 1:
                    unit_causal_orders.append(
                        [0]
                    )
                else:
                    unit_causal_orders.append(
                        []
                    )

                valid_sample_counts[unit_index] = np.count_nonzero(
                    np.isfinite(residuals[:, nodes[0]])
                )

                processed_units += 1

                continue

            # --------------------------------------------------
            # Extract complete observations for this unit.
            # --------------------------------------------------

            local_residuals = self._extract_complete_samples(
                residuals,
                nodes,
            )

            valid_sample_counts[unit_index] = (
                local_residuals.shape[0]
            )

            # --------------------------------------------------
            # Not enough observations for PEM.
            # --------------------------------------------------

            if local_residuals.shape[0] <= len(nodes):

                unit_probabilities.append(
                    np.zeros(
                        (len(nodes), len(nodes)),
                        dtype=np.float64,
                    )
                )

                unit_adjacencies.append(
                    np.zeros(
                        (len(nodes), len(nodes)),
                        dtype=np.float64,
                    )
                )

                unit_causal_orders.append(
                    []
                )

                continue

            # --------------------------------------------------
            # Run PEM.
            # --------------------------------------------------

            try:

                pem_result = self._run_pem(
                    local_residuals
                )

            except Exception as exc:

                # --------------------------------------------------
                # Diagnostic only.
                #
                # No covariance modification, jitter,
                # regularization, skipping, or fallback is applied.
                # The original PEM error is re-raised.
                # --------------------------------------------------

                centered = (
                    local_residuals
                    - np.mean(
                        local_residuals,
                        axis=0,
                        keepdims=True,
                    )
                )

                covariance = (
                    centered.T @ centered
                    / local_residuals.shape[0]
                )

                covariance = (
                    covariance + covariance.T
                ) / 2.0

                rank = np.linalg.matrix_rank(
                    covariance
                )

                eigenvalues = np.linalg.eigvalsh(
                    covariance
                )

                print()
                print("=" * 70)
                print("PEM FAILURE DIAGNOSTIC")
                print("=" * 70)

                print(
                    f"Unit index       : {unit_index}"
                )

                print(
                    f"Target node      : {unit.target}"
                )

                print(
                    f"Unit size        : {len(nodes)}"
                )

                print(
                    f"Complete samples : "
                    f"{local_residuals.shape[0]}"
                )

                print(
                    f"Covariance rank  : {rank}"
                )

                print(
                    f"Expected rank    : "
                    f"{len(nodes)}"
                )

                print(
                    f"Min eigenvalue   : "
                    f"{np.min(eigenvalues):.6e}"
                )

                print(
                    f"Max eigenvalue   : "
                    f"{np.max(eigenvalues):.6e}"
                )

                print(
                    "All residuals finite : "
                    f"{np.all(np.isfinite(local_residuals))}"
                )

                print("-" * 70)

                print("PEM error type:")
                print(type(exc).__name__)

                print("PEM error:")
                print(exc)

                print("=" * 70)

                # Keep the original behavior:
                # do not silently continue after a failed PEM unit.
                raise

            local_probabilities = np.asarray(
                pem_result.inclusion_probabilities,
                dtype=np.float64,
            )

            local_adjacency = np.asarray(
                pem_result.adjacency_matrix,
                dtype=np.float64,
            )

            local_causal_order = list(
                pem_result.causal_order
            )

            unit_probabilities.append(
                local_probabilities
            )

            unit_adjacencies.append(
                local_adjacency
            )

            unit_causal_orders.append(
                local_causal_order
            )

            # --------------------------------------------------
            # Map local inclusion probabilities to global indices.
            #
            # local_probabilities is symmetric.
            #
            # It represents:
            #
            #     P(edge exists)
            #
            # and NOT:
            #
            #     P(source -> target)
            # --------------------------------------------------

            for local_source in range(len(nodes)):

                for local_target in range(len(nodes)):

                    if local_source == local_target:
                        continue

                    source = nodes[local_source]
                    target = nodes[local_target]

                    probability = local_probabilities[
                        local_target,
                        local_source,
                    ]

                    if probability > global_probabilities[
                        target,
                        source,
                    ]:

                        global_probabilities[
                            target,
                            source,
                        ] = probability

            # --------------------------------------------------
            # Map directed PEM-LBN adjacency to global indices.
            #
            # local_adjacency[target, source] = 1
            #
            # means:
            #
            #     source -> target
            # --------------------------------------------------

            for local_source in range(len(nodes)):

                for local_target in range(len(nodes)):

                    if local_source == local_target:
                        continue

                    if local_adjacency[
                        local_target,
                        local_source,
                    ] <= 0.0:
                        continue

                    source = nodes[local_source]
                    target = nodes[local_target]

                    global_adjacency[
                        target,
                        source,
                    ] = 1.0

            processed_units += 1

        # --------------------------------------------------------------
        # No self causal edges.
        # --------------------------------------------------------------

        np.fill_diagonal(
            global_probabilities,
            0.0,
        )

        np.fill_diagonal(
            global_adjacency,
            0.0,
        )

        return LocalPEMResult(
            global_inclusion_probabilities=(
                global_probabilities
            ),
            global_adjacency=(
                global_adjacency
            ),
            unit_inclusion_probabilities=(
                unit_probabilities
            ),
            unit_adjacencies=(
                unit_adjacencies
            ),
            unit_causal_orders=(
                unit_causal_orders
            ),
            valid_sample_counts=(
                valid_sample_counts
            ),
            processed_units=(
                processed_units
            ),
        )