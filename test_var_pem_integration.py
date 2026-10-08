from __future__ import annotations

import numpy as np

from causal.var import TopologyAwareVAR
from causal.pem import PEM


def generate_var_data(
    num_samples: int = 3000,
    random_state: int = 42,
):
    """
    Generate synthetic VAR(1) data with a known contemporaneous
    causal structure.

    Ground-truth contemporaneous graph:

        X1 -> X2 -> X3

    Model:

        X_t = B0 X_t + B1 X_{t-1} + e_t
    """

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # Ground-truth contemporaneous causal matrix B0
    #
    # Convention:
    # B0[target, source] != 0
    # means source -> target.
    # ------------------------------------------------------------
    B0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.0, 0.6, 0.0],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # Ground-truth temporal coefficient matrix B1
    # ------------------------------------------------------------
    B1 = np.array(
        [
            [0.4, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
        ],
        dtype=np.float64,
    )

    num_nodes = B0.shape[0]

    # Independent non-Gaussian noise.
    # Laplace noise is useful for the LiNGAM assumptions.
    noise = rng.laplace(
        loc=0.0,
        scale=1.0,
        size=(num_samples, num_nodes),
    )

    X = np.zeros((num_samples, num_nodes), dtype=np.float64)

    identity = np.eye(num_nodes)

    for t in range(1, num_samples):
        temporal_part = B1 @ X[t - 1]

        # Solve:
        #
        # X_t = B0 X_t + B1 X_{t-1} + e_t
        #
        # therefore:
        #
        # (I - B0) X_t = B1 X_{t-1} + e_t
        rhs = temporal_part + noise[t]

        X[t] = np.linalg.solve(
            identity - B0,
            rhs,
        )

    return X, B0, B1


def create_chain_adjacency(num_nodes: int) -> np.ndarray:
    """
    Create a topology containing the possible contemporaneous
    causal connections:

        X1 -- X2 -- X3

    The VAR implementation uses the adjacency row of a target
    node to determine its allowed source nodes.
    """

    adjacency = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float64,
    )

    # X1 <-> X2
    adjacency[0, 1] = 1.0
    adjacency[1, 0] = 1.0

    # X2 <-> X3
    adjacency[1, 2] = 1.0
    adjacency[2, 1] = 1.0

    return adjacency


def evaluate_graph(
    estimated_adjacency: np.ndarray,
    ground_truth_adjacency: np.ndarray,
):
    """
    Evaluate directed graph recovery.
    """

    estimated = np.asarray(estimated_adjacency) != 0
    truth = np.asarray(ground_truth_adjacency) != 0

    np.fill_diagonal(estimated, False)
    np.fill_diagonal(truth, False)

    true_positive = np.sum(estimated & truth)
    false_positive = np.sum(estimated & ~truth)
    false_negative = np.sum(~estimated & truth)
    true_negative = np.sum(~estimated & ~truth)

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": int(true_positive),
        "FP": int(false_positive),
        "FN": int(false_negative),
        "TN": int(true_negative),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    print("=" * 70)
    print("VAR -> PEM INTEGRATION TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Generate synthetic traffic-like multivariate time series
    # ------------------------------------------------------------
    X, B0_true, B1_true = generate_var_data(
        num_samples=3000,
        random_state=42,
    )

    num_nodes = X.shape[1]

    print("\nSynthetic data:")
    print("  Shape:", X.shape)

    print("\nGround-truth B0:")
    print(B0_true)

    print("\nGround-truth B1:")
    print(B1_true)

    # ------------------------------------------------------------
    # 2. Create topology
    # ------------------------------------------------------------
    adjacency = create_chain_adjacency(num_nodes)

    print("\nTopology adjacency:")
    print(adjacency)

    # ------------------------------------------------------------
    # 3. Estimate VAR
    # ------------------------------------------------------------
    print("\nRunning topology-aware VAR...")

    var_model = TopologyAwareVAR(
        lag_order=1,
        ridge_alpha=1e-5,
        use_gpu=True,
    )

    var_result = var_model.fit(
        data=X,
        adjacency=adjacency,
    )

    print("\nVAR coefficient shape:")
    print(var_result.coefficients.shape)

    print("\nEstimated VAR coefficients:")
    print(var_result.coefficients[0])

    print("\nResidual shape:")
    print(var_result.residuals.shape)

    print("\nValid residual samples:")
    print(
        np.sum(
            np.all(
                np.isfinite(var_result.residuals),
                axis=1,
            )
        )
    )

    # ------------------------------------------------------------
    # 4. Extract complete residual samples
    #
    # PEM requires complete observations.
    # No missing values are imputed.
    # ------------------------------------------------------------
    residuals = var_result.residuals

    valid_mask = np.all(
        np.isfinite(residuals),
        axis=1,
    )

    valid_residuals = residuals[valid_mask]

    print("\nResiduals supplied to PEM:")
    print("  Shape:", valid_residuals.shape)

    if valid_residuals.shape[0] < 100:
        raise RuntimeError(
            "Too few valid residual samples for PEM integration test."
        )

    # ------------------------------------------------------------
    # 5. Use a neutral PEM prior.
    #
    # This is intentionally NOT derived from the ground truth.
    # We want to test whether PEM can recover the structure
    # from the residual data itself.
    # ------------------------------------------------------------
    eta = np.full(
        (num_nodes, num_nodes),
        0.5,
        dtype=np.float64,
    )

    np.fill_diagonal(eta, 0.0)

    print("\nPEM prior eta:")
    print(eta)

    # ------------------------------------------------------------
    # 6. Run PEM
    # ------------------------------------------------------------
    print("\nRunning PEM...")

    pem_model = PEM(
        eta=eta,
        threshold=0.5,
    )

    pem_result = pem_model.fit(
        valid_residuals
    )

    print("\nEstimated PEM precision:")
    print(pem_result.precision_matrix)

    print("\nPEM posterior inclusion probabilities:")
    print(pem_result.inclusion_probabilities)

    print("\nPEM estimated adjacency:")
    print(pem_result.adjacency_matrix)

    print("\nPEM causal order:")
    print(pem_result.causal_order)

    # ------------------------------------------------------------
    # 7. Ground-truth directed graph
    #
    # B0[target, source] != 0
    # means source -> target.
    # ------------------------------------------------------------
    ground_truth_adjacency = (
        np.abs(B0_true) > 1e-8
    ).astype(np.float64)

    np.fill_diagonal(
        ground_truth_adjacency,
        0.0,
    )

    print("\nGround-truth adjacency:")
    print(ground_truth_adjacency)

    # ------------------------------------------------------------
    # 8. Evaluate graph recovery
    # ------------------------------------------------------------
    metrics = evaluate_graph(
        pem_result.adjacency_matrix,
        ground_truth_adjacency,
    )

    print("\nGraph recovery:")
    print("  TP:", metrics["TP"])
    print("  FP:", metrics["FP"])
    print("  FN:", metrics["FN"])
    print("  TN:", metrics["TN"])
    print("  Precision:", metrics["precision"])
    print("  Recall:", metrics["recall"])
    print("  F1:", metrics["f1"])

    # ------------------------------------------------------------
    # 9. Basic integration assertions
    # ------------------------------------------------------------
    assert var_result.coefficients.shape == (
        1,
        num_nodes,
        num_nodes,
    )

    assert var_result.residuals.shape == X.shape

    assert np.all(
        np.isfinite(valid_residuals)
    )

    assert pem_result.precision_matrix.shape == (
        num_nodes,
        num_nodes,
    )

    assert pem_result.covariance_matrix.shape == (
        num_nodes,
        num_nodes,
    )

    assert pem_result.inclusion_probabilities.shape == (
        num_nodes,
        num_nodes,
    )

    assert pem_result.adjacency_matrix.shape == (
        num_nodes,
        num_nodes,
    )

    print("\n" + "=" * 70)
    print("PASS: VAR -> PEM integration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()