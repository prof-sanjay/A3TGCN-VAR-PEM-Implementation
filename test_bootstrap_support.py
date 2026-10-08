import numpy as np

from causal.bootstrap_support import BootstrapEdgeSupport


def main():
    print("=" * 60)
    print("BOOTSTRAP EDGE SUPPORT TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Synthetic VAR-LiNGAM adjacency matrices
    #
    # Convention:
    # B[k, j] != 0  means  j -> k
    # ---------------------------------------------------------

    num_nodes = 3

    run_1 = np.zeros((num_nodes, num_nodes))
    run_1[1, 0] = 0.8   # S1 -> S2
    run_1[2, 1] = 0.6   # S2 -> S3

    run_2 = np.zeros((num_nodes, num_nodes))
    run_2[1, 0] = 0.7   # S1 -> S2
    run_2[2, 1] = 0.5   # S2 -> S3

    run_3 = np.zeros((num_nodes, num_nodes))
    run_3[1, 0] = 0.9   # S1 -> S2

    adjacency_matrices = np.stack(
        [run_1, run_2, run_3],
        axis=0,
    )

    print("\nAdjacency array shape:")
    print(adjacency_matrices.shape)

    # ---------------------------------------------------------
    # Calculate bootstrap support
    # ---------------------------------------------------------

    calculator = BootstrapEdgeSupport()

    support = calculator.calculate_support(
        adjacency_matrices
    )

    print("\nBootstrap support matrix:")
    print(support)

    # ---------------------------------------------------------
    # Check expected support values
    # ---------------------------------------------------------

    print("\nExpected support:")
    print("S1 -> S2:", support[1, 0], " expected: 1.0")
    print("S2 -> S3:", support[2, 1], " expected: 0.6667")
    print("S2 -> S1:", support[0, 1], " expected: 0.0")
    print("S3 -> S2:", support[1, 2], " expected: 0.0")

    # ---------------------------------------------------------
    # Check stable edges at threshold = 0.5
    # ---------------------------------------------------------

    stable_edges = calculator.get_stable_edges(
        support,
        threshold=0.5,
    )

    print("\nStable edges at threshold = 0.5:")
    print(stable_edges)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    checks = [
        np.isclose(support[1, 0], 1.0),
        np.isclose(support[2, 1], 2.0 / 3.0),
        np.isclose(support[0, 1], 0.0),
        np.isclose(support[1, 2], 0.0),
        np.isclose(np.diag(support), 0.0).all(),
        stable_edges[1, 0],
        stable_edges[2, 1],
        not stable_edges[0, 1],
    ]

    print("\nValidation checks:")
    print(f"S1 -> S2 support correct: {checks[0]}")
    print(f"S2 -> S3 support correct: {checks[1]}")
    print(f"S2 -> S1 support correct: {checks[2]}")
    print(f"S3 -> S2 support correct: {checks[3]}")
    print(f"Diagonal is zero:         {checks[4]}")
    print(f"S1 -> S2 is stable:       {checks[5]}")
    print(f"S2 -> S3 is stable:       {checks[6]}")
    print(f"S2 -> S1 is not stable:   {checks[7]}")

    if all(checks):
        print("\nPASS: Bootstrap edge-support implementation is working correctly.")
    else:
        print("\nFAIL: One or more checks failed.")


if __name__ == "__main__":
    main()