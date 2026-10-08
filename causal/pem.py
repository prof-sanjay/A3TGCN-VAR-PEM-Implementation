from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PEMResult:
    """
    Results returned by PEM-LBN.
    """

    precision_matrix: np.ndarray
    covariance_matrix: np.ndarray
    inclusion_probabilities: np.ndarray
    adjacency_matrix: np.ndarray
    causal_order: list[int]


class PEM:
    """
    Probabilistic Edge Modulation (PEM) for Linear Bayesian Networks.

    PEM uses edge-specific inclusion probabilities eta in a
    spike-and-slab Laplace prior over off-diagonal precision
    matrix entries.

    Model:

        X ~ N(0, Sigma)

        Omega = Sigma^{-1}

    Off-diagonal prior:

        Omega[j,k] | r=0 ~ Laplace(nu0)
        Omega[j,k] | r=1 ~ Laplace(nu1)

        r ~ Bernoulli(eta[j,k])

    where:

        nu1 > nu0 > 0

    The PEM-MAP objective is optimized over positive-definite
    precision matrices.

    IMPORTANT
    ---------
    This implementation is intended for small/medium-sized
    validation experiments.

    The PEM paper reports O(p^4) overall complexity, so applying
    the full PEM procedure directly to thousands of traffic
    sensors is not practical. A block/sparse strategy will be
    required for the LargeST-GBA experiment.
    """

    def __init__(
        self,
        eta: np.ndarray,
        nu0: float = 0.05,
        nu1: float = 1.0,
        tau: float = 0.01,
        threshold: float = 0.5,
        max_iter: int = 500,
        tolerance: float = 1e-6,
        spectral_norm_bound: float | None = None,
        optimizer: str = "SLSQP",
    ) -> None:

        if nu0 <= 0:
            raise ValueError("nu0 must be positive.")

        if nu1 <= nu0:
            raise ValueError(
                "nu1 must be greater than nu0."
            )

        if tau <= 0:
            raise ValueError(
                "tau must be positive."
            )

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "threshold must be strictly between 0 and 1."
            )

        if max_iter < 1:
            raise ValueError(
                "max_iter must be at least 1."
            )

        if tolerance <= 0:
            raise ValueError(
                "tolerance must be positive."
            )

        if (
            spectral_norm_bound is not None
            and spectral_norm_bound <= 0
        ):
            raise ValueError(
                "spectral_norm_bound must be positive "
                "when provided."
            )

        self.eta = self._validate_eta(eta)

        self.nu0 = float(nu0)
        self.nu1 = float(nu1)
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

        # Numerical optimizer for the (unchanged) PEM-MAP objective.
        # "SLSQP" is the original choice. "L-BFGS-B" minimises the
        # same objective with the same exact gradient; it supports no
        # general constraints, so it requires spectral_norm_bound=None.
        if optimizer not in ("SLSQP", "L-BFGS-B"):
            raise ValueError(
                "optimizer must be 'SLSQP' or 'L-BFGS-B'."
            )

        if optimizer == "L-BFGS-B" and spectral_norm_bound is not None:
            raise ValueError(
                "L-BFGS-B cannot enforce spectral_norm_bound."
            )

        self.optimizer = optimizer

        # Number of L-BFGS-B failures re-solved with SLSQP (reported).
        self.slsqp_retries = 0

        self.spectral_norm_bound = (
            None
            if spectral_norm_bound is None
            else float(spectral_norm_bound)
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_eta(
        eta: np.ndarray,
    ) -> np.ndarray:

        eta = np.asarray(
            eta,
            dtype=np.float64,
        )

        if eta.ndim != 2:
            raise ValueError(
                "eta must be a 2-dimensional matrix."
            )

        if eta.shape[0] != eta.shape[1]:
            raise ValueError(
                "eta must be square."
            )

        if not np.all(np.isfinite(eta)):
            raise ValueError(
                "eta must contain only finite values."
            )

        if np.any(eta < 0.0) or np.any(eta > 1.0):
            raise ValueError(
                "eta values must be in [0, 1]."
            )

        if not np.allclose(
            eta,
            eta.T,
            atol=1e-12,
        ):
            raise ValueError(
                "eta must be symmetric."
            )

        return eta

    @staticmethod
    def _validate_samples(
        samples: np.ndarray,
    ) -> np.ndarray:

        samples = np.asarray(
            samples,
            dtype=np.float64,
        )

        if samples.ndim != 2:
            raise ValueError(
                "samples must be a 2-dimensional array."
            )

        if samples.shape[0] < 2:
            raise ValueError(
                "At least two samples are required."
            )

        if samples.shape[1] < 2:
            raise ValueError(
                "At least two variables are required."
            )

        if not np.all(np.isfinite(samples)):
            raise ValueError(
                "samples must contain only finite values."
            )

        return samples

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_covariance(
        samples: np.ndarray,
    ) -> np.ndarray:

        centered = (
            samples
            - np.mean(
                samples,
                axis=0,
                keepdims=True,
            )
        )

        n = centered.shape[0]

        covariance = (
            centered.T @ centered
        ) / n

        covariance = (
            covariance
            + covariance.T
        ) / 2.0

        return covariance

    @staticmethod
    def _make_positive_definite(
        matrix: np.ndarray,
        jitter: float = 1e-8,
    ) -> np.ndarray:

        matrix = (
            matrix
            + matrix.T
        ) / 2.0

        eigenvalues = np.linalg.eigvalsh(
            matrix
        )

        minimum_eigenvalue = np.min(
            eigenvalues
        )

        if minimum_eigenvalue <= 0.0:
            matrix = matrix + (
                -minimum_eigenvalue
                + jitter
            ) * np.eye(
                matrix.shape[0]
            )

        return matrix

    # ------------------------------------------------------------------
    # Spike-and-slab prior
    # ------------------------------------------------------------------

    def _log_prior_density(
        self,
        value: float,
        eta: float,
    ) -> float:
        """
        Log of the spike-and-slab marginal density.

        p(Omega_jk)
            =
            (1-eta) * Laplace(nu0)
            +
            eta * Laplace(nu1)
        """

        abs_value = abs(value)

        # Hard exclusion.
        if eta <= 0.0:
            return float(
                -abs_value / self.nu0
                - np.log(2.0 * self.nu0)
            )

        # Hard inclusion.
        if eta >= 1.0:
            return float(
                -abs_value / self.nu1
                - np.log(2.0 * self.nu1)
            )

        log_spike = (
            np.log1p(-eta)
            - np.log(2.0 * self.nu0)
            - abs_value / self.nu0
        )

        log_slab = (
            np.log(eta)
            - np.log(2.0 * self.nu1)
            - abs_value / self.nu1
        )

        # Stable log-sum-exp.
        maximum = max(
            log_spike,
            log_slab,
        )

        return float(
            maximum
            + np.log(
                np.exp(log_spike - maximum)
                + np.exp(log_slab - maximum)
            )
        )
        """
        Log of the spike-and-slab marginal density.

        p(Omega_jk)
          =
          (1-eta) * Laplace(nu0)
          +
          eta * Laplace(nu1)
        """

        abs_value = abs(value)

        log_spike = (
            np.log1p(-eta)
            - np.log(2.0 * self.nu0)
            - abs_value / self.nu0
        )

        log_slab = (
            np.log(eta)
            - np.log(2.0 * self.nu1)
            - abs_value / self.nu1
        )

        # Handle eta = 0 or eta = 1 safely.
        if eta <= 0.0:
            return log_spike

        if eta >= 1.0:
            return log_slab

        # Stable log-sum-exp.
        maximum = max(
            log_spike,
            log_slab,
        )

        return float(
            maximum
            + np.log(
                np.exp(log_spike - maximum)
                + np.exp(log_slab - maximum)
            )
        )

    def _off_diagonal_penalty(
        self,
        value: float,
        eta: float,
    ) -> float:
        """
        Negative log spike-and-slab prior.
        """

        return -self._log_prior_density(
            value,
            eta,
        )

    # ------------------------------------------------------------------
    # Posterior inclusion probability
    # ------------------------------------------------------------------

    def _posterior_inclusion_probability(
        self,
        omega_value: float,
        eta: float,
    ) -> float:
        """
        Posterior probability that the edge belongs to
        the slab component.

        This corresponds to the posterior inclusion
        probability used by PEM.
        """

        # Hard exclusion:
        # eta = 0 means the edge is assigned zero prior
        # probability of belonging to the slab.
        if eta <= 0.0:
            return 0.0

        # Hard inclusion:
        # eta = 1 means the edge is assigned probability one
        # of belonging to the slab.
        if eta >= 1.0:
            return 1.0

        abs_value = abs(omega_value)

        log_slab = (
            np.log(eta)
            - np.log(2.0 * self.nu1)
            - abs_value / self.nu1
        )

        log_spike = (
            np.log1p(-eta)
            - np.log(2.0 * self.nu0)
            - abs_value / self.nu0
        )

        maximum = max(
            log_slab,
            log_spike,
        )

        slab_weight = np.exp(
            log_slab - maximum
        )

        spike_weight = np.exp(
            log_spike - maximum
        )

        return float(
            slab_weight
            / (
                slab_weight
                + spike_weight
            )
        )
        """
        Posterior probability that the edge belongs to
        the slab component.

        This corresponds to the posterior inclusion
        probability used by PEM.
        """

        abs_value = abs(
            omega_value
        )

        if eta <= 0.0:
            return 0.0

        if eta >= 1.0:
            return 1.0

        log_slab = (
            np.log(eta)
            - np.log(2.0 * self.nu1)
            - abs_value / self.nu1
        )

        log_spike = (
            np.log1p(-eta)
            - np.log(2.0 * self.nu0)
            - abs_value / self.nu0
        )

        maximum = max(
            log_slab,
            log_spike,
        )

        slab_weight = np.exp(
            log_slab - maximum
        )

        spike_weight = np.exp(
            log_spike - maximum
        )

        return float(
            slab_weight
            / (
                slab_weight
                + spike_weight
            )
        )

    # ------------------------------------------------------------------
    # PEM-MAP objective
    # ------------------------------------------------------------------

    def _map_objective(
        self,
        precision: np.ndarray,
        covariance: np.ndarray,
        eta: np.ndarray,
        num_samples: int,
    ) -> float:
        """
        PEM-MAP objective.

        Up to constants independent of Omega:

            n/2 *
            [ tr(S Omega) - log det(Omega) ]

            + prior penalty

        where:

            S = sample covariance
            Omega = precision matrix

        The omitted terms are constants with respect to Omega.
        """

        precision = (
            precision
            + precision.T
        ) / 2.0

        eigenvalues = np.linalg.eigvalsh(
            precision
        )

        if np.any(eigenvalues <= 0.0):
            return 1e100

        sign, logdet = np.linalg.slogdet(
            precision
        )

        if sign <= 0:
            return 1e100

        trace_term = np.trace(
            covariance @ precision
        )

        likelihood = (
            0.5
            * num_samples
            * (
                trace_term
                - logdet
            )
        )

        # Diagonal prior from PEM.
        prior = self.tau * np.trace(precision)

        # Off-diagonal spike-and-slab penalty, summed over j < k.
        # Vectorized form of _off_diagonal_penalty(); same formula.
        upper = np.triu_indices(precision.shape[0], k=1)

        prior += np.sum(
            -self._log_prior_density_array(
                precision[upper],
                eta[upper],
            )
        )

        return float(
            likelihood + prior
        )

    # ------------------------------------------------------------------
    # Vectorized prior and exact gradient (computational only)
    #
    # These evaluate exactly the same PEM-MAP objective as above; they
    # only remove Python loops and supply the analytic gradient to the
    # optimizer instead of a finite-difference approximation.
    # ------------------------------------------------------------------

    def _log_prior_density_array(
        self,
        values: np.ndarray,
        eta: np.ndarray,
    ) -> np.ndarray:
        """
        Element-wise _log_prior_density().
        """

        abs_values = np.abs(values)

        log_spike_only = -abs_values / self.nu0 - np.log(2.0 * self.nu0)
        log_slab_only = -abs_values / self.nu1 - np.log(2.0 * self.nu1)

        mixed = (eta > 0.0) & (eta < 1.0)

        safe_eta = np.where(mixed, eta, 0.5)

        log_mixture = np.logaddexp(
            np.log1p(-safe_eta) + log_spike_only,
            np.log(safe_eta) + log_slab_only,
        )

        return np.where(
            eta <= 0.0,
            log_spike_only,
            np.where(
                eta >= 1.0,
                log_slab_only,
                log_mixture,
            ),
        )

    def _off_diagonal_penalty_derivative(
        self,
        values: np.ndarray,
        eta: np.ndarray,
    ) -> np.ndarray:
        """
        d/dv of the negative log spike-and-slab prior:

            sign(v) * ( w_spike / nu0 + w_slab / nu1 )

        with w_slab the posterior slab weight (the PEM posterior
        inclusion probability) and w_spike = 1 - w_slab.
        """

        abs_values = np.abs(values)

        mixed = (eta > 0.0) & (eta < 1.0)

        safe_eta = np.where(mixed, eta, 0.5)

        log_spike = (
            np.log1p(-safe_eta)
            - np.log(2.0 * self.nu0)
            - abs_values / self.nu0
        )

        log_slab = (
            np.log(safe_eta)
            - np.log(2.0 * self.nu1)
            - abs_values / self.nu1
        )

        slab_weight = np.exp(
            log_slab - np.logaddexp(log_spike, log_slab)
        )

        slab_weight = np.where(
            eta <= 0.0,
            0.0,
            np.where(eta >= 1.0, 1.0, slab_weight),
        )

        slope = (
            (1.0 - slab_weight) / self.nu0
            + slab_weight / self.nu1
        )

        return np.sign(values) * slope

    def _objective_and_gradient(
        self,
        parameters: np.ndarray,
        covariance: np.ndarray,
        eta: np.ndarray,
        num_samples: int,
        num_nodes: int,
    ) -> tuple[float, np.ndarray]:
        """
        PEM-MAP objective and its exact gradient with respect to the
        Cholesky parameters used by _vector_to_precision().

        With Omega = L L^T and f(Omega) the objective,

            dF/dOmega = n/2 (S - Omega^{-1}) + tau I + P,
            P[j,k] = P[k,j] = 0.5 * penalty'(Omega[j,k]),  j != k

            dF/dL = 2 dF/dOmega L          (lower triangle)

            diagonal: L[j,j] = exp(theta_j)
                dF/dtheta_j = dF/dL[j,j] * L[j,j]
                (0 where the exp() argument is clipped)
        """

        lower = self._vector_to_lower(parameters, num_nodes)

        precision = lower @ lower.T
        precision = (precision + precision.T) / 2.0

        value = self._map_objective(
            precision,
            covariance,
            eta,
            num_samples,
        )

        if value >= 1e100:
            return value, np.zeros_like(parameters)

        inverse = np.linalg.inv(precision)
        inverse = (inverse + inverse.T) / 2.0

        gradient_omega = (
            0.5 * num_samples * (covariance - inverse)
            + self.tau * np.eye(num_nodes)
        )

        upper = np.triu_indices(num_nodes, k=1)

        half_slope = 0.5 * self._off_diagonal_penalty_derivative(
            precision[upper],
            eta[upper],
        )

        gradient_omega[upper] += half_slope
        gradient_omega[(upper[1], upper[0])] += half_slope

        gradient_lower = 2.0 * gradient_omega @ lower

        rows, cols = np.tril_indices(num_nodes)

        gradient = gradient_lower[rows, cols].copy()

        diagonal = rows == cols

        clipped = np.abs(parameters[diagonal]) > 30.0

        gradient[diagonal] = np.where(
            clipped,
            0.0,
            gradient[diagonal] * lower[rows[diagonal], cols[diagonal]],
        )

        return value, gradient

    # ------------------------------------------------------------------
    # Cholesky parameterization
    # ------------------------------------------------------------------

    @staticmethod
    def _vector_to_lower(
        parameters: np.ndarray,
        num_nodes: int,
    ) -> np.ndarray:
        """
        Lower-triangular Cholesky factor from unconstrained
        parameters (row-major lower triangle: (0,0), (1,0), (1,1),
        (2,0), ...). Diagonal entries are exp(clip(value, -30, 30)).
        """

        lower = np.zeros(
            (num_nodes, num_nodes),
            dtype=np.float64,
        )

        rows, cols = np.tril_indices(num_nodes)

        values = np.asarray(parameters, dtype=np.float64).copy()

        diagonal = rows == cols

        values[diagonal] = np.exp(
            np.clip(values[diagonal], -30.0, 30.0)
        )

        lower[rows, cols] = values

        return lower

    @staticmethod
    def _vector_to_precision(
        parameters: np.ndarray,
        num_nodes: int,
    ) -> np.ndarray:
        """
        Convert unconstrained parameters into a
        positive-definite precision matrix.

        Omega = L L^T

        with positive diagonal entries of L obtained
        using exp().
        """

        lower = PEM._vector_to_lower(
            parameters,
            num_nodes,
        )

        precision = (
            lower @ lower.T
        )

        precision = (
            precision
            + precision.T
        ) / 2.0

        return precision

    @staticmethod
    def _precision_to_vector(
        precision: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a positive-definite precision matrix
        into Cholesky parameters.
        """

        precision = (
            precision
            + precision.T
        ) / 2.0

        lower = np.linalg.cholesky(
            precision
        )

        num_nodes = precision.shape[0]

        parameters = []

        for j in range(num_nodes):

            for k in range(j + 1):

                if j == k:
                    parameters.append(
                        np.log(
                            max(
                                lower[j, k],
                                1e-12,
                            )
                        )
                    )
                else:
                    parameters.append(
                        lower[j, k]
                    )

        return np.asarray(
            parameters,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Precision optimization
    # ------------------------------------------------------------------

    def _estimate_precision(
        self,
        covariance: np.ndarray,
        eta: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        """
        Numerically optimize the PEM-MAP objective.

        scipy is intentionally used here because this is the
        small/medium-dimensional validation implementation.

        The optimization uses a Cholesky parameterization so
        positive definiteness is guaranteed.
        """

        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ImportError(
                "PEM requires scipy for the MAP optimizer."
            ) from exc

        covariance = self._make_positive_definite(
            covariance
        )

        num_nodes = covariance.shape[0]

        # Initial precision from inverse covariance.
        initial_precision = np.linalg.inv(
            covariance
        )

        initial_precision = (
            initial_precision
            + initial_precision.T
        ) / 2.0

        initial_parameters = (
            self._precision_to_vector(
                initial_precision
            )
        )

        def objective(parameters):

            # Same PEM-MAP objective, returned together with its
            # exact gradient (instead of a finite-difference one).
            return self._objective_and_gradient(
                parameters,
                covariance,
                eta,
                num_samples,
                num_nodes,
            )

        constraints = []

        if self.spectral_norm_bound is not None:

            def spectral_constraint(
                parameters,
            ):

                precision = (
                    self._vector_to_precision(
                        parameters,
                        num_nodes,
                    )
                )

                largest_eigenvalue = np.max(
                    np.linalg.eigvalsh(
                        precision
                    )
                )

                return (
                    self.spectral_norm_bound
                    - largest_eigenvalue
                )

            constraints.append(
                {
                    "type": "ineq",
                    "fun": spectral_constraint,
                }
            )

        if self.optimizer == "SLSQP":

            options = {
                "maxiter": self.max_iter,
                "ftol": self.tolerance,
                "disp": False,
            }

        else:

            # SLSQP's ftol is an accuracy on the objective VALUE,
            # whereas L-BFGS-B's ftol is RELATIVE to |f|. The same
            # value accuracy is obtained by dividing by the initial
            # objective magnitude.
            initial_value = abs(objective(initial_parameters)[0])

            options = {
                "maxiter": self.max_iter,
                "maxfun": 10 * self.max_iter,
                "ftol": self.tolerance / max(1.0, initial_value),
                "gtol": 0.0,
            }

        result = minimize(
            objective,
            initial_parameters,
            jac=True,
            method=self.optimizer,
            constraints=constraints if self.optimizer == "SLSQP" else (),
            options=options,
        )

        if not result.success and self.optimizer == "L-BFGS-B":

            # L-BFGS-B can stop with a line-search failure. The same
            # objective is then minimised with the original SLSQP
            # solver, warm-started at the L-BFGS-B point. The count is
            # kept so that it can be reported.
            self.slsqp_retries += 1

            result = minimize(
                objective,
                result.x,
                jac=True,
                method="SLSQP",
                options={
                    "maxiter": self.max_iter,
                    "ftol": self.tolerance,
                    "disp": False,
                },
            )

        if not result.success:
            raise RuntimeError(
                "PEM-MAP optimization failed: "
                f"{result.message}"
            )

        precision = (
            self._vector_to_precision(
                result.x,
                num_nodes,
            )
        )

        precision = (
            precision
            + precision.T
        ) / 2.0

        return precision

    # ------------------------------------------------------------------
    # Ordering and parent selection
    # ------------------------------------------------------------------

    def _learn_graph(
        self,
        samples: np.ndarray,
    ) -> tuple[
        np.ndarray,
        list[int],
    ]:
        """
        Recover the DAG ordering and parent relationships.

        At each iteration:

        1. Restrict the problem to the active variables.
        2. Estimate the PEM-MAP precision matrix.
        3. Select the variable with minimum diagonal precision.
        4. Calculate posterior inclusion probabilities.
        5. Add parent -> selected edges.
        6. Remove the selected variable.

        Matrix convention:

            adjacency[target, source] = 1

        therefore:

            adjacency[k, j] = 1

        represents:

            j -> k
        """

        num_nodes = samples.shape[1]

        active = list(
            range(num_nodes)
        )

        removal_order = []

        adjacency = np.zeros(
            (num_nodes, num_nodes),
            dtype=np.float64,
        )

        while len(active) > 1:

            active_array = np.asarray(
                active,
                dtype=int,
            )

            active_samples = samples[
                :,
                active_array,
            ]

            active_eta = self.eta[
                np.ix_(
                    active_array,
                    active_array,
                )
            ]

            covariance = (
                self._sample_covariance(
                    active_samples
                )
            )

            precision = (
                self._estimate_precision(
                    covariance,
                    active_eta,
                    samples.shape[0],
                )
            )

            # PEM ordering rule.
            local_selected_index = int(
                np.argmin(
                    np.diag(
                        precision
                    )
                )
            )

            selected_node = active[
                local_selected_index
            ]

            # Parent selection.
            for local_index, candidate_node in enumerate(
                active
            ):

                if candidate_node == selected_node:
                    continue

                omega_value = precision[
                    local_index,
                    local_selected_index,
                ]

                eta_value = self.eta[
                    candidate_node,
                    selected_node,
                ]

                posterior = (
                    self._posterior_inclusion_probability(
                        omega_value,
                        eta_value,
                    )
                )

                if posterior >= self.threshold:

                    # candidate -> selected
                    adjacency[
                        selected_node,
                        candidate_node,
                    ] = 1.0

            removal_order.append(
                selected_node
            )

            active.remove(
                selected_node
            )

        if len(active) == 1:
            removal_order.append(
                active[0]
            )

        # Variables are removed from the end of the
        # causal ordering toward the beginning.
        causal_order = list(
            reversed(
                removal_order
            )
        )

        return (
            adjacency,
            causal_order,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        samples: np.ndarray,
    ) -> PEMResult:
        """
        Fit PEM-LBN to i.i.d. samples.
        """

        samples = self._validate_samples(
            samples
        )

        num_samples = samples.shape[0]
        num_nodes = samples.shape[1]

        if num_nodes != self.eta.shape[0]:
            raise ValueError(
                "Number of variables in samples must "
                "match eta."
            )

        # --------------------------------------------------------------
        # Full-data covariance
        # --------------------------------------------------------------

        covariance = (
            self._sample_covariance(
                samples
            )
        )

        covariance = (
            self._make_positive_definite(
                covariance
            )
        )

        # --------------------------------------------------------------
        # Full-data PEM-MAP precision
        #
        # This is deliberately estimated separately from the
        # ordering procedure. The ordering procedure uses reduced
        # active sets, so its final precision matrix may be smaller
        # than p x p.
        # --------------------------------------------------------------

        precision = (
            self._estimate_precision(
                covariance,
                self.eta,
                num_samples,
            )
        )

        # --------------------------------------------------------------
        # Recover graph and causal ordering
        # --------------------------------------------------------------

        adjacency, causal_order = (
            self._learn_graph(
                samples
            )
        )

        # --------------------------------------------------------------
        # Posterior inclusion probabilities
        # --------------------------------------------------------------

        inclusion_probabilities = np.zeros(
            (num_nodes, num_nodes),
            dtype=np.float64,
        )

        for j in range(num_nodes):

            for k in range(j + 1, num_nodes):

                posterior = (
                    self._posterior_inclusion_probability(
                        precision[j, k],
                        self.eta[j, k],
                    )
                )

                inclusion_probabilities[
                    j,
                    k,
                ] = posterior

                inclusion_probabilities[
                    k,
                    j,
                ] = posterior

        # --------------------------------------------------------------
        # Estimated covariance corresponding to Omega
        # --------------------------------------------------------------

        estimated_covariance = np.linalg.inv(
            precision
        )

        estimated_covariance = (
            estimated_covariance
            + estimated_covariance.T
        ) / 2.0

        return PEMResult(
            precision_matrix=precision,
            covariance_matrix=estimated_covariance,
            inclusion_probabilities=inclusion_probabilities,
            adjacency_matrix=adjacency,
            causal_order=causal_order,
        )