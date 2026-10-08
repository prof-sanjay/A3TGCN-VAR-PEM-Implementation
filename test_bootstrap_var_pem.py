from __future__ import annotations

import numpy as np

from causal.var import TopologyAwareVAR
from causal.bootstrap_lingam import BootstrapLiNGAM
from causal.pem_prior import PEMPrior
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

    B0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.0, 0.6, 0.0],
        ],
        dtype=np.float64,
    )

    B1 = np.array(
        [
            [0.4, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
        ],
        dtype=np.float64,
    )

    num_nodes = B0.shape[0]

    noise = rng.laplace(
        loc=0.0,
        scale=1.0,
        size=(num_samples, num_nodes),
    )

    X = np.zeros(
        (num_samples, num_nodes),
        dtype=np.float64,
    )

    identity = np.eye(num_nodes)

    for t in range(1, num_samples):
        rhs = B1 @ X[t - 1] + noise[t]

        X[t] = np.linalg.solve(
            identity - B0,
            rhs,
        )

    return X, B0, B1


def create_chain_adjacency(num_nodes: int) -> np.ndarray:
    """
    Undirected topology:

        X1 -- X2 -- X3

    This defines the candidate spatial neighborhood for VAR.
    """

    adjacency = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float64,
    )

    adjacency[0, 1] = 1.0
    adjacency[1, 0] = 1.0

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

    estimated = np.asarray(
        estimated_adjacency
    ) != 0

    truth = np.asarray(
        ground_truth_adjacency
    ) != 0

    np.fill_diagonal(
        estimated,
        False,
    )

    np.fill_diagonal(
        truth,
        False,
    )

    true_positive = np.sum(
        estimated & truth
    )

    false_positive = np.sum(
        estimated & ~truth
    )

    false_negative = np.sum(
        ~estimated & truth
    )

    true_negative = np.sum(
        ~estimated & ~truth
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
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
    print("BOOTSTRAP VAR-LiNGAM -> PEM INTEGRATION TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Generate synthetic VAR data
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
    # 2. Topology
    # ------------------------------------------------------------
    adjacency = create_chain_adjacency(
        num_nodes
    )

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

    print("\nEstimated VAR coefficients:")
    print(var_result.coefficients[0])

    # ------------------------------------------------------------
    # 4. Extract complete residual observations
    #
    # No imputation is performed.
    # ------------------------------------------------------------
    residuals = var_result.residuals

    valid_mask = np.all(
        np.isfinite(residuals),
        axis=1,
    )

    valid_residuals = residuals[
        valid_mask
    ]

    print("\nResiduals:")
    print("  Original shape:", residuals.shape)
    print(
        "  Valid residual shape:",
        valid_residuals.shape,
    )

    if valid_residuals.shape[0] < 100:
        raise RuntimeError(
            "Too few valid residual samples."
        )

    # ------------------------------------------------------------
    # 5. Bootstrap VAR-LiNGAM
    # ------------------------------------------------------------
    print("\nRunning bootstrap VAR-LiNGAM...")

    bootstrap_lingam = BootstrapLiNGAM(
        num_bootstrap_samples=10,
        random_state=42,
        edge_threshold=1e-8,
    )

    bootstrap_result = bootstrap_lingam.fit(
        valid_residuals
    )

    print("\nBootstrap adjacency shape:")
    print(
        bootstrap_result.adjacency_matrices.shape
    )

    print(
        "\nNumber of bootstrap samples:",
        bootstrap_result.num_bootstrap_samples,
    )

    print(
        "Valid bootstrap runs:",
        bootstrap_result.valid_runs,
    )

    # ------------------------------------------------------------
    # 6. Bootstrap edge support
    # ------------------------------------------------------------
    edge_support = (
        bootstrap_result.edge_support
    )

    print("\nBootstrap directed edge support:")
    print(edge_support)

    # ------------------------------------------------------------
    # 7. Convert directed support into PEM eta
    # ------------------------------------------------------------
    prior_builder = PEMPrior()

    eta = prior_builder.calculate_eta(
        edge_support
    )

    print("\nPEM eta derived from bootstrap support:")
    print(eta)

    # ------------------------------------------------------------
    # 8. Verify eta properties
    # ------------------------------------------------------------
    assert eta.shape == (
        num_nodes,
        num_nodes,
    )

    assert np.all(
        np.isfinite(eta)
    )

    assert np.all(
        eta >= 0.0
    )

    assert np.all(
        eta <= 1.0
    )

    assert np.allclose(
        eta,
        eta.T,
    )

    assert np.allclose(
        np.diag(eta),
        0.0,
    )

    # ------------------------------------------------------------
    # 9. Run PEM using bootstrap-derived eta
    # ------------------------------------------------------------
    print("\nRunning PEM with bootstrap-derived eta...")

    pem_model = PEM(
        eta=eta,
        threshold=0.5,
    )

    pem_result = pem_model.fit(
        valid_residuals
    )

    print("\nPEM precision:")
    print(
        pem_result.precision_matrix
    )

    print(
        "\nPEM posterior inclusion probabilities:"
    )

    print(
        pem_result.inclusion_probabilities
    )

    print("\nPEM estimated adjacency:")
    print(
        pem_result.adjacency_matrix
    )

    print(
        "\nPEM causal order:",
        pem_result.causal_order,
    )

    # ------------------------------------------------------------
    # 10. Ground-truth contemporaneous graph
    # ------------------------------------------------------------
    ground_truth_adjacency = (
        np.abs(B0_true) > 1e-8
    ).astype(np.float64)

    np.fill_diagonal(
        ground_truth_adjacency,
        0.0,
    )

    print(
        "\nGround-truth adjacency:"
    )

    print(
        ground_truth_adjacency
    )

    # ------------------------------------------------------------
    # 11. Evaluate final PEM graph
    # ------------------------------------------------------------
    metrics = evaluate_graph(
        pem_result.adjacency_matrix,
        ground_truth_adjacency,
    )

    print("\nFinal graph recovery:")
    print("  TP:", metrics["TP"])
    print("  FP:", metrics["FP"])
    print("  FN:", metrics["FN"])
    print("  TN:", metrics["TN"])
    print(
        "  Precision:",
        metrics["precision"],
    )
    print(
        "  Recall:",
        metrics["recall"],
    )
    print(
        "  F1:",
        metrics["f1"],
    )

    # ------------------------------------------------------------
    # 12. Final integration assertions
    # ------------------------------------------------------------
    assert (
        bootstrap_result.valid_runs
        == bootstrap_result.num_bootstrap_samples
    )

    assert edge_support.shape == (
        num_nodes,
        num_nodes,
    )

    assert pem_result.precision_matrix.shape == (
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
    print(
        "PASS: Bootstrap VAR-LiNGAM -> PEM "
        "integration completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()