from __future__ import annotations

import numpy as np

from causal.pem import PEM


def main() -> None:
    print("=" * 60)
    print("PEM TEST")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------
    #
    # Three variables with a sparse dependency structure:
    #
    # X1 <-> X2
    # X2 <-> X3
    #
    # PEM works with the precision matrix Omega, so we generate
    # a positive-definite covariance matrix first and then sample
    # from the corresponding multivariate Gaussian distribution.
    #
    # ------------------------------------------------------------

    np.random.seed(42)

    num_samples = 1000
    num_nodes = 3

    covariance = np.array(
        [
            [1.0, 0.6, 0.0],
            [0.6, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )

    samples = np.random.multivariate_normal(
        mean=np.zeros(num_nodes),
        cov=covariance,
        size=num_samples,
    )

    print(f"\nSynthetic data shape: {samples.shape}")

    # ------------------------------------------------------------
    # 2. Construct PEM prior
    # ------------------------------------------------------------
    #
    # Prior information:
    #
    # X1 -- X2 : strong prior
    # X2 -- X3 : strong prior
    # X1 -- X3 : no prior
    #
    # ------------------------------------------------------------

    eta = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.9, 0.0, 0.8],
            [0.0, 0.8, 0.0],
        ],
        dtype=np.float64,
    )

    print("\nPEM prior eta:")
    print(eta)

    # ------------------------------------------------------------
    # 3. Run PEM
    # ------------------------------------------------------------

    model = PEM(
        eta=eta,
    )

    result = model.fit(samples)

    # ------------------------------------------------------------
    # 4. Display results
    # ------------------------------------------------------------

    print("\nEstimated precision matrix Omega:")
    print(result.precision_matrix)

    print("\nEstimated covariance matrix:")
    print(result.covariance_matrix)

    print("\nPosterior inclusion probabilities:")
    print(result.inclusion_probabilities)

    print("\nEstimated adjacency matrix:")
    print(result.adjacency_matrix)

    # ------------------------------------------------------------
    # 5. Validation checks
    # ------------------------------------------------------------

    omega_shape_correct = (
        result.precision_matrix.shape
        == (num_nodes, num_nodes)
    )

    covariance_shape_correct = (
        result.covariance_matrix.shape
        == (num_nodes, num_nodes)
    )

    inclusion_shape_correct = (
        result.inclusion_probabilities.shape
        == (num_nodes, num_nodes)
    )

    adjacency_shape_correct = (
        result.adjacency_matrix.shape
        == (num_nodes, num_nodes)
    )

    omega_finite = np.all(
        np.isfinite(result.precision_matrix)
    )

    covariance_finite = np.all(
        np.isfinite(result.covariance_matrix)
    )

    inclusion_values_valid = (
        np.all(result.inclusion_probabilities >= 0.0)
        and
        np.all(result.inclusion_probabilities <= 1.0)
    )

    adjacency_binary = np.all(
        np.isin(
            result.adjacency_matrix,
            [0, 1],
        )
    )

    # ------------------------------------------------------------
    # 6. Check symmetry of Omega
    # ------------------------------------------------------------

    omega_symmetric = np.allclose(
        result.precision_matrix,
        result.precision_matrix.T,
        atol=1e-6,
    )

    # ------------------------------------------------------------
    # 7. Check positive definiteness
    # ------------------------------------------------------------

    try:
        eigenvalues = np.linalg.eigvalsh(
            result.precision_matrix
        )

        omega_positive_definite = np.all(
            eigenvalues > 0
        )

    except np.linalg.LinAlgError:
        omega_positive_definite = False

    # ------------------------------------------------------------
    # 8. Validation output
    # ------------------------------------------------------------

    print("\nValidation checks:")

    print(
        f"Omega shape correct:             "
        f"{omega_shape_correct}"
    )

    print(
        f"Covariance shape correct:         "
        f"{covariance_shape_correct}"
    )

    print(
        f"Inclusion shape correct:         "
        f"{inclusion_shape_correct}"
    )

    print(
        f"Adjacency shape correct:          "
        f"{adjacency_shape_correct}"
    )

    print(
        f"Omega contains finite values:     "
        f"{omega_finite}"
    )

    print(
        f"Covariance contains finite values:"
        f" {covariance_finite}"
    )

    print(
        f"Inclusion probabilities valid:    "
        f"{inclusion_values_valid}"
    )

    print(
        f"Adjacency is binary:              "
        f"{adjacency_binary}"
    )

    print(
        f"Omega is symmetric:               "
        f"{omega_symmetric}"
    )

    print(
        f"Omega is positive definite:       "
        f"{omega_positive_definite}"
    )

    # ------------------------------------------------------------
    # 9. Final result
    # ------------------------------------------------------------

    all_checks_passed = all(
        [
            omega_shape_correct,
            covariance_shape_correct,
            inclusion_shape_correct,
            adjacency_shape_correct,
            omega_finite,
            covariance_finite,
            inclusion_values_valid,
            adjacency_binary,
            omega_symmetric,
            omega_positive_definite,
        ]
    )

    if all_checks_passed:
        print("\n" + "=" * 60)
        print("PASS: PEM validation successful.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAIL: One or more PEM checks failed.")
        print("=" * 60)


if __name__ == "__main__":
    main()