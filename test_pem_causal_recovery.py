from __future__ import annotations

import numpy as np

from causal.pem import PEM


def generate_dag_data(
    num_samples: int = 5000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate data from the known DAG:

        X1 -> X2 -> X3

    Structural equations:

        X1 = e1
        X2 = 0.8 * X1 + e2
        X3 = 0.6 * X2 + e3

    The noise variables are independent Gaussian variables.
    """

    rng = np.random.default_rng(
        random_state
    )

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_samples, 3),
    )

    x1 = noise[:, 0]

    x2 = (
        0.8 * x1
        + noise[:, 1]
    )

    x3 = (
        0.6 * x2
        + noise[:, 2]
    )

    samples = np.column_stack(
        [
            x1,
            x2,
            x3,
        ]
    )

    return samples


def main() -> None:

    print("=" * 70)
    print("PEM CAUSAL RECOVERY TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Generate known causal data
    # --------------------------------------------------------------

    samples = generate_dag_data(
        num_samples=5000,
        random_state=42,
    )

    print(
        f"\nSynthetic data shape: "
        f"{samples.shape}"
    )

    print("\nGround-truth causal graph:")
    print("X1 -> X2")
    print("X2 -> X3")

    # --------------------------------------------------------------
    # Ground-truth adjacency
    #
    # Convention:
    #
    # adjacency[target, source] = 1
    #
    # Therefore:
    #
    # X1 -> X2  => adjacency[1, 0] = 1
    # X2 -> X3  => adjacency[2, 1] = 1
    # --------------------------------------------------------------

    ground_truth = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )

    print("\nGround-truth adjacency:")
    print(ground_truth)

    # --------------------------------------------------------------
    # PEM prior
    #
    # We provide a non-informative / weak prior for this test.
    # The purpose is to test whether PEM can learn the structure
    # from the observed data rather than simply copying the prior.
    # --------------------------------------------------------------

    eta = np.full(
        (3, 3),
        0.5,
        dtype=np.float64,
    )

    np.fill_diagonal(
        eta,
        0.0,
    )

    print("\nPEM prior eta:")
    print(eta)

    # --------------------------------------------------------------
    # Fit PEM
    # --------------------------------------------------------------

    model = PEM(
        eta=eta,
        threshold=0.5,
        max_iter=500,
        tolerance=1e-6,
    )

    result = model.fit(
        samples
    )

    # --------------------------------------------------------------
    # Display results
    # --------------------------------------------------------------

    print("\nEstimated precision matrix:")
    print(result.precision_matrix)

    print("\nPosterior inclusion probabilities:")
    print(
        result.inclusion_probabilities
    )

    print("\nEstimated adjacency:")
    print(
        result.adjacency_matrix
    )

    print("\nEstimated causal order:")
    print(
        result.causal_order
    )

    # --------------------------------------------------------------
    # Compare recovered graph with ground truth
    # --------------------------------------------------------------

    predicted = (
        result.adjacency_matrix.astype(bool)
    )

    actual = (
        ground_truth.astype(bool)
    )

    true_positive = np.sum(
        predicted & actual
    )

    false_positive = np.sum(
        predicted & ~actual
    )

    false_negative = np.sum(
        ~predicted & actual
    )

    true_negative = np.sum(
        ~predicted & ~actual
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print("\nGraph recovery results:")
    print(
        f"True positives:  {true_positive}"
    )
    print(
        f"False positives: {false_positive}"
    )
    print(
        f"False negatives: {false_negative}"
    )
    print(
        f"True negatives:  {true_negative}"
    )

    print(
        f"\nPrecision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 score:  {f1:.4f}"
    )

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    graph_recovered = np.array_equal(
        predicted,
        actual,
    )

    print(
        f"\nExact graph recovery: "
        f"{graph_recovered}"
    )

    if graph_recovered:

        print("\n" + "=" * 70)
        print(
            "PASS: PEM recovered the ground-truth DAG."
        )
        print("=" * 70)

    else:

        print("\n" + "=" * 70)
        print(
            "INFO: PEM did not exactly recover "
            "the ground-truth DAG."
        )
        print(
            "This is not automatically a failure."
        )
        print(
            "We need to inspect the posterior probabilities "
            "and causal ordering."
        )
        print("=" * 70)


if __name__ == "__main__":
    main()