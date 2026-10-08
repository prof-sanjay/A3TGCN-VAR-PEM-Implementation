from __future__ import annotations

import numpy as np

from causal.bootstrap_lingam import BootstrapLiNGAM


def main() -> None:
    print("=" * 60)
    print("BOOTSTRAP VAR-LiNGAM TEST")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Generate synthetic causal data
    # ------------------------------------------------------------
    #
    # True causal structure:
    #
    # X1 -> X2 -> X3
    #
    # VAR-LiNGAM convention:
    #
    # B[i, j] != 0  means  X_j -> X_i
    #
    # Therefore:
    #
    # B[1, 0] = 0.8  => X1 -> X2
    # B[2, 1] = 0.6  => X2 -> X3
    #
    # Structural equation:
    #
    # X = B0 X + e
    #
    # Therefore:
    #
    # X = (I - B0)^(-1) e
    #
    # ------------------------------------------------------------

    np.random.seed(42)

    num_samples = 1000
    num_nodes = 3

    B0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.0, 0.6, 0.0],
        ],
        dtype=np.float64,
    )

    # Generate non-Gaussian Laplace noise.
    noise = np.random.laplace(
        loc=0.0,
        scale=1.0,
        size=(num_samples, num_nodes),
    )

    # Solve:
    #
    # (I - B0) X_t = e_t
    #
    # for every sample.
    identity = np.eye(num_nodes)

    X = np.zeros((num_samples, num_nodes), dtype=np.float64)

    for t in range(num_samples):
        X[t] = np.linalg.solve(
            identity - B0,
            noise[t],
        )

    print(f"\nSynthetic data shape: {X.shape}")

    print("\nTrue causal structure:")
    print("X1 -> X2")
    print("X2 -> X3")

    print("\nTrue B0 matrix:")
    print(B0)

    # ------------------------------------------------------------
    # 2. Run Bootstrap VAR-LiNGAM
    # ------------------------------------------------------------

    model = BootstrapLiNGAM(
        num_bootstrap_samples=5,
        random_state=42,
        edge_threshold=1e-8,
    )

    result = model.fit(X)

    # ------------------------------------------------------------
    # 3. Display results
    # ------------------------------------------------------------

    print(
        f"\nBootstrap adjacency shape: "
        f"{result.adjacency_matrices.shape}"
    )

    print("\nEdge-support matrix:")
    print(result.edge_support)

    print(
        f"\nNumber of requested bootstrap runs: "
        f"{result.num_bootstrap_samples}"
    )

    print(
        f"Number of successful runs: "
        f"{result.valid_runs}"
    )

    # ------------------------------------------------------------
    # 4. Check expected causal directions
    # ------------------------------------------------------------
    #
    # VAR-LiNGAM convention:
    #
    # adjacency_matrix[target, source] != 0
    #
    # Therefore:
    #
    # X1 -> X2 corresponds to [1, 0]
    # X2 -> X3 corresponds to [2, 1]
    #
    # Reverse directions:
    #
    # X2 -> X1 corresponds to [0, 1]
    # X3 -> X2 corresponds to [1, 2]
    # ------------------------------------------------------------

    x1_to_x2 = result.edge_support[1, 0]
    x2_to_x3 = result.edge_support[2, 1]

    x2_to_x1 = result.edge_support[0, 1]
    x3_to_x2 = result.edge_support[1, 2]

    print("\nExpected causal directions:")

    print(
        f"X1 -> X2: {x1_to_x2}"
    )

    print(
        f"X2 -> X3: {x2_to_x3}"
    )

    print("\nReverse directions:")

    print(
        f"X2 -> X1: {x2_to_x1}"
    )

    print(
        f"X3 -> X2: {x3_to_x2}"
    )

    # ------------------------------------------------------------
    # 5. Validation checks
    # ------------------------------------------------------------

    adjacency_shape_correct = (
        result.adjacency_matrices.shape
        == (5, num_nodes, num_nodes)
    )

    support_shape_correct = (
        result.edge_support.shape
        == (num_nodes, num_nodes)
    )

    all_runs_valid = (
        result.valid_runs
        == result.num_bootstrap_samples
    )

    # We use 0.5 as the stability threshold.
    #
    # An edge supported in at least half of the bootstrap runs
    # is considered detected.
    #
    # With 5 bootstrap runs:
    #
    # support >= 0.5
    # means at least 3/5 runs.
    x1_to_x2_detected = x1_to_x2 >= 0.5
    x2_to_x3_detected = x2_to_x3 >= 0.5

    x2_to_x1_not_stable = x2_to_x1 < 0.5
    x3_to_x2_not_stable = x3_to_x2 < 0.5

    support_values_valid = (
        np.all(result.edge_support >= 0.0)
        and np.all(result.edge_support <= 1.0)
    )

    print("\nValidation checks:")

    print(
        f"Adjacency shape correct:    "
        f"{adjacency_shape_correct}"
    )

    print(
        f"Support shape correct:      "
        f"{support_shape_correct}"
    )

    print(
        f"All bootstrap runs valid:    "
        f"{all_runs_valid}"
    )

    print(
        f"X1 -> X2 detected:           "
        f"{x1_to_x2_detected}"
    )

    print(
        f"X2 -> X3 detected:           "
        f"{x2_to_x3_detected}"
    )

    print(
        f"X2 -> X1 not stable:         "
        f"{x2_to_x1_not_stable}"
    )

    print(
        f"X3 -> X2 not stable:         "
        f"{x3_to_x2_not_stable}"
    )

    print(
        f"Support values valid:        "
        f"{support_values_valid}"
    )

    # ------------------------------------------------------------
    # 6. Final result
    # ------------------------------------------------------------

    all_checks_passed = all(
        [
            adjacency_shape_correct,
            support_shape_correct,
            all_runs_valid,
            x1_to_x2_detected,
            x2_to_x3_detected,
            x2_to_x1_not_stable,
            x3_to_x2_not_stable,
            support_values_valid,
        ]
    )

    if all_checks_passed:
        print("\n" + "=" * 60)
        print("PASS: Bootstrap VAR-LiNGAM validation successful.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAIL: One or more checks failed.")
        print("=" * 60)


if __name__ == "__main__":
    main()