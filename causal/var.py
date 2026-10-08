from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


@dataclass
class VARResult:
    """
    Container for the estimated VAR model.

    Attributes
    ----------
    coefficients:
        Array of shape (lag_order, num_nodes, num_nodes).

        coefficients[k, target, source] represents the effect of
        source node at lag k+1 on target node.

    residuals:
        Array of shape (num_samples, num_nodes).

        NaN is retained wherever a residual could not be estimated
        because the required observations were unavailable.

    valid_mask:
        Boolean array of shape (num_samples, num_nodes).

        True means the corresponding residual was successfully
        estimated.

    lag_order:
        Number of temporal lags used by the VAR model.
    """

    coefficients: np.ndarray
    residuals: np.ndarray
    valid_mask: np.ndarray
    lag_order: int


class TopologyAwareVAR:
    """
    Topology-aware Vector Autoregressive (VAR) model.

    For each target node i:

        x_i(t) =
            sum_{k=1}^{p}
            sum_{j in N(i)}
                A_k[i,j] x_j(t-k)
            + e_i(t)

    where:
        p      = lag_order
        N(i)   = allowed source nodes for target i
        e_i(t) = residual

    Missing observations are NOT imputed.

    A regression sample is used only when all observations required
    for that particular target regression are available.

    TensorFlow is used for the regression linear algebra so that
    matrix operations can execute on the configured GPU.
    """

    def __init__(
        self,
        lag_order: int = 1,
        ridge_alpha: float = 1e-5,
        use_gpu: bool = True,
    ) -> None:

        if lag_order < 1:
            raise ValueError("lag_order must be >= 1.")

        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be >= 0.")

        self.lag_order = lag_order
        self.ridge_alpha = ridge_alpha
        self.use_gpu = use_gpu

        # --------------------------------------------------------
        # TensorFlow ridge solver state.
        #
        # One session and one computation graph are reused for
        # all target-node regressions within a single fit().
        # --------------------------------------------------------

        self._ridge_session = None
        self._ridge_X_placeholder = None
        self._ridge_y_placeholder = None
        self._ridge_beta_operation = None

    def _validate_inputs(
        self,
        data: np.ndarray,
        adjacency: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        data = np.asarray(data, dtype=np.float32)
        adjacency = np.asarray(adjacency, dtype=np.float32)

        if data.ndim != 2:
            raise ValueError(
                "data must have shape (num_timesteps, num_nodes)."
            )

        if adjacency.ndim != 2:
            raise ValueError(
                "adjacency must be a 2D square matrix."
            )

        num_nodes = data.shape[1]

        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(
                "adjacency shape must match the number of nodes. "
                f"Expected {(num_nodes, num_nodes)}, "
                f"got {adjacency.shape}."
            )

        if not np.all(np.isfinite(adjacency)):
            raise ValueError(
                "adjacency contains non-finite values."
            )

        if self.lag_order >= data.shape[0]:
            raise ValueError(
                "lag_order must be smaller than the number of timesteps."
            )

        return data, adjacency

    def _get_neighbors(
        self,
        adjacency: np.ndarray,
        target_node: int,
    ) -> np.ndarray:
        """
        Return source nodes allowed to predict target_node.

        A non-zero adjacency entry means that the corresponding source
        node is allowed as a predictor.

        Self-history is always included.
        """

        neighbors = np.flatnonzero(
            adjacency[target_node] != 0
        )

        if target_node not in neighbors:
            neighbors = np.append(
                neighbors,
                target_node,
            )

        return np.unique(neighbors)

    def _build_regression_data(
        self,
        data: np.ndarray,
        target_node: int,
        source_nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct the local VAR regression for one target node.

        Returns
        -------
        X:
            Predictor matrix.

        y:
            Target vector.

        valid_times:
            Original timestep corresponding to every regression sample.
        """

        num_timesteps = data.shape[0]

        start_time = self.lag_order
        num_samples = num_timesteps - start_time

        num_predictors = (
            len(source_nodes) * self.lag_order
        )

        X = np.empty(
            (num_samples, num_predictors),
            dtype=np.float32,
        )

        y = np.empty(
            num_samples,
            dtype=np.float32,
        )

        valid_times = np.arange(
            start_time,
            num_timesteps,
            dtype=np.int64,
        )

        column = 0

        for lag in range(1, self.lag_order + 1):

            lagged_values = data[
                start_time - lag : num_timesteps - lag,
                source_nodes,
            ]

            width = len(source_nodes)

            X[
                :,
                column : column + width,
            ] = lagged_values

            column += width

        y[:] = data[
            start_time:,
            target_node,
        ]

        valid_mask = (
            np.isfinite(y)
            & np.all(
                np.isfinite(X),
                axis=1,
            )
        )

        return (
            X[valid_mask],
            y[valid_mask],
            valid_times[valid_mask],
        )

    def _initialize_ridge_solver(self) -> None:
        """
        Create the reusable TensorFlow ridge-regression solver.

        The computation graph is created only once and reused for
        every target-node regression during fit().
        """

        if self._ridge_session is not None:
            return

        # --------------------------------------------------------
        # Reusable placeholders.
        #
        # The second dimension is dynamic because different target
        # nodes may have different numbers of topology neighbors.
        # --------------------------------------------------------

        self._ridge_X_placeholder = tf.placeholder(
            dtype=tf.float64,
            shape=[None, None],
            name="var_ridge_X",
        )

        self._ridge_y_placeholder = tf.placeholder(
            dtype=tf.float64,
            shape=[None, 1],
            name="var_ridge_y",
        )

        xtx = tf.matmul(
            self._ridge_X_placeholder,
            self._ridge_X_placeholder,
            transpose_a=True,
        )

        xty = tf.matmul(
            self._ridge_X_placeholder,
            self._ridge_y_placeholder,
            transpose_a=True,
        )

        if self.ridge_alpha > 0:

            identity = tf.eye(
                tf.shape(xtx)[0],
                dtype=tf.float64,
            )

            xtx = (
                xtx
                + self.ridge_alpha * identity
            )

        self._ridge_beta_operation = tf.linalg.solve(
            xtx,
            xty,
        )

        session_config = tf.ConfigProto()

        if self.use_gpu:

            session_config.gpu_options.allow_growth = True

        else:

            session_config.device_count["GPU"] = 0

        self._ridge_session = tf.Session(
            config=session_config
        )

    def _fit_ridge(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Solve:

            beta = (X^T X + alpha I)^(-1) X^T y

        using TensorFlow float64 linear algebra.

        Float64 is used here because the local traffic regression
        matrices can be severely ill-conditioned due to strong
        correlation between neighboring traffic sensors.

        The TensorFlow session and computation graph are reused
        across target-node regressions.
        """

        if X.shape[0] == 0:
            raise ValueError(
                "No valid samples available for regression."
            )

        self._initialize_ridge_solver()

        beta_value = self._ridge_session.run(
            self._ridge_beta_operation,
            feed_dict={
                self._ridge_X_placeholder: X,
                self._ridge_y_placeholder: y.reshape(-1, 1),
            },
        )

        return beta_value[:, 0]

    def _close_ridge_solver(self) -> None:
        """
        Close the TensorFlow session used by the VAR solver.
        """

        if self._ridge_session is not None:
            self._ridge_session.close()

        self._ridge_session = None
        self._ridge_X_placeholder = None
        self._ridge_y_placeholder = None
        self._ridge_beta_operation = None

    def fit(
        self,
        data: np.ndarray,
        adjacency: np.ndarray,
    ) -> VARResult:
        """
        Fit the topology-aware VAR model.

        Parameters
        ----------
        data:
            Traffic data with shape:

                (num_timesteps, num_nodes)

            NaN values are preserved.

        adjacency:
            Spatial adjacency matrix with shape:

                (num_nodes, num_nodes)

        Returns
        -------
        VARResult
        """

        data, adjacency = self._validate_inputs(
            data,
            adjacency,
        )

        num_timesteps, num_nodes = data.shape

        coefficients = np.zeros(
            (
                self.lag_order,
                num_nodes,
                num_nodes,
            ),
            dtype=np.float32,
        )

        residuals = np.full(
            (
                num_timesteps,
                num_nodes,
            ),
            np.nan,
            dtype=np.float32,
        )

        valid_mask = np.zeros(
            (
                num_timesteps,
                num_nodes,
            ),
            dtype=bool,
        )

        try:

            # ----------------------------------------------------
            # One TensorFlow solver is created for the entire VAR
            # fitting process.
            # ----------------------------------------------------

            self._initialize_ridge_solver()

            for target_node in range(num_nodes):

                source_nodes = self._get_neighbors(
                    adjacency,
                    target_node,
                )

                X, y, valid_times = (
                    self._build_regression_data(
                        data,
                        target_node,
                        source_nodes,
                    )
                )

                if len(y) == 0:
                    continue

                beta = self._fit_ridge(
                    X,
                    y,
                )

                num_sources = len(source_nodes)

                for lag_index in range(
                    self.lag_order
                ):

                    start = (
                        lag_index
                        * num_sources
                    )

                    end = (
                        start
                        + num_sources
                    )

                    coefficients[
                        lag_index,
                        target_node,
                        source_nodes,
                    ] = beta[
                        start:end
                    ]

                # ------------------------------------------------
                # Prediction and residual calculation remain
                # unchanged from the previous implementation.
                # ------------------------------------------------

                predictions = X @ beta

                residual_values = (
                    y - predictions
                )

                residuals[
                    valid_times,
                    target_node,
                ] = residual_values

                valid_mask[
                    valid_times,
                    target_node,
                ] = True

            return VARResult(
                coefficients=coefficients,
                residuals=residuals,
                valid_mask=valid_mask,
                lag_order=self.lag_order,
            )

        finally:

            # ----------------------------------------------------
            # Close the session after the complete VAR fit.
            #
            # This prevents the TensorFlow session from remaining
            # alive after fit() finishes.
            # ----------------------------------------------------

            self._close_ridge_solver()