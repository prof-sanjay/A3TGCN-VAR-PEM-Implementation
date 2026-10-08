from __future__ import annotations

import numpy as np

from causal.pem_prior import PEMPrior


def main() -> None:
    print("=" * 60)
    print("PEM PRIOR CONSTRUCTION TEST")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Synthetic directed bootstrap support
    # ------------------------------------------------------------
    #
    # Three variables:
    #
    # X1 -> X2
    # X2 -> X3
    #
    # VAR-LiNGAM convention:
    #
    # support[target, source]
    #
    # Therefore:
    #
    # support[1, 0] = X1 -> X2
    # support[2, 1] = X2 -> X3
    #
    # ------------------------------------------------------------

    support = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    print("\nDirected bootstrap support:")
    print(support)

    # ------------------------------------------------------------
    # 2. Construct PEM prior eta
    # ------------------------------------------------------------

    model = PEMPrior()

    eta = model.calculate_eta(support)

    print("\nPEM prior-inclusion matrix eta:")
    print(eta)

    # ------------------------------------------------------------
    # 3. Expected eta
    # ------------------------------------------------------------
    #
    # PEM prior is symmetric:
    #
    # eta[j,k] = max(
    #     support[j,k],
    #     support[k,j]
    # )
    #
    # Therefore:
    #
    # X1 -- X2 = 1.0
    # X2 -- X3 = 1.0
    #
    # ------------------------------------------------------------

    expected_eta = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    eta_correct = np.allclose(
        eta,
        expected_eta,
    )

    # ------------------------------------------------------------
    # 4. Check symmetry
    # ------------------------------------------------------------

    eta_symmetric = np.allclose(
        eta,
        eta.T,
    )

    # ------------------------------------------------------------
    # 5. Check diagonal
    # ------------------------------------------------------------

    diagonal_zero = np.allclose(
        np.diag(eta),
        0.0,
    )

    # ------------------------------------------------------------
    # 6. Check prior edges
    # ------------------------------------------------------------

    prior_edges = model.get_prior_edges(
        eta,
        threshold=0.5,
    )

    expected_prior_edges = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    prior_edges_correct = np.array_equal(
        prior_edges,
        expected_prior_edges,
    )

    print("\nPrior edges at threshold = 0.5:")
    print(prior_edges)

    # ------------------------------------------------------------
    # 7. Check that partial directional support is handled
    # ------------------------------------------------------------
    #
    # Example:
    #
    # X1 -> X3 = 0.3
    # X3 -> X1 = 0.7
    #
    # eta must become:
    #
    # max(0.3, 0.7) = 0.7
    #
    # ------------------------------------------------------------

    partial_support = np.array(
        [
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    partial_eta = model.calculate_eta(
        partial_support
    )

    partial_eta_correct = (
        partial_eta[0, 2] == 0.7
        and partial_eta[2, 0] == 0.7
    )

    print("\nPartial directional support:")
    print(partial_support)

    print("\nResulting eta:")
    print(partial_eta)

    # ------------------------------------------------------------
    # 8. Validation output
    # ------------------------------------------------------------

    print("\nValidation checks:")

    print(
        f"Eta matches expected matrix:    "
        f"{eta_correct}"
    )

    print(
        f"Eta is symmetric:                "
        f"{eta_symmetric}"
    )

    print(
        f"Eta diagonal is zero:            "
        f"{diagonal_zero}"
    )

    print(
        f"Prior edges are correct:         "
        f"{prior_edges_correct}"
    )

    print(
        f"Max-direction rule is correct:  "
        f"{partial_eta_correct}"
    )

    # ------------------------------------------------------------
    # 9. Final result
    # ------------------------------------------------------------

    all_checks_passed = all(
        [
            eta_correct,
            eta_symmetric,
            diagonal_zero,
            prior_edges_correct,
            partial_eta_correct,
        ]
    )

    if all_checks_passed:
        print("\n" + "=" * 60)
        print("PASS: PEM prior construction validation successful.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAIL: One or more checks failed.")
        print("=" * 60)


if __name__ == "__main__":
    main()