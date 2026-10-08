"""
Robustness test for PEM under different prior-inclusion matrices eta.

Ground-truth causal structure:
    X1 -> X2 -> X3

The experiment compares:
    A. Neutral prior
    B. Correct prior
    C. Moderately wrong prior
    D. Strongly wrong prior
    E. Weak/uninformative prior

This test is intended to evaluate whether PEM remains capable of
recovering the causal graph when the prior information is imperfect.
"""

from __future__ import annotations

import numpy as np

from causal.pem import PEM


def generate_causal_data(
    num_samples: int = 5000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate data from:

        X1 = e1
        X2 = 0.8 X1 + e2
        X3 = 0.6 X2 + e3

    Therefore:

        X1 -> X2 -> X3
    """

    rng = np.random.default_rng(random_state)

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_samples, 3),
    )

    x1 = noise[:, 0]
    x2 = 0.8 * x1 + noise[:, 1]
    x3 = 0.6 * x2 + noise[:, 2]

    return np.column_stack([x1, x2, x3])


def calculate_metrics(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """
    Calculate directed-edge recovery metrics.
    """

    estimated_edges = estimated.astype(bool)
    true_edges = ground_truth.astype(bool)

    np.fill_diagonal(estimated_edges, False)
    np.fill_diagonal(true_edges, False)

    tp = np.sum(estimated_edges & true_edges)
    fp = np.sum(estimated_edges & ~true_edges)
    fn = np.sum(~estimated_edges & true_edges)
    tn = np.sum(~estimated_edges & ~true_edges)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"\n{name}:")
    print(np.round(matrix, 4))


def main() -> None:

    print("=" * 70)
    print("PEM ETA ROBUSTNESS TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Generate synthetic causal data
    # ------------------------------------------------------------

    samples = generate_causal_data(
        num_samples=5000,
        random_state=42,
    )

    print("\nSynthetic data shape:")
    print(samples.shape)

    # ------------------------------------------------------------
    # 2. Ground-truth graph
    #
    # Convention:
    # adjacency[target, source] = 1
    #
    # Therefore:
    # X1 -> X2 : adjacency[1, 0] = 1
    # X2 -> X3 : adjacency[2, 1] = 1
    # ------------------------------------------------------------

    ground_truth = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )

    print_matrix(
        "Ground-truth adjacency",
        ground_truth,
    )

    # ------------------------------------------------------------
    # 3. Define eta priors
    # ------------------------------------------------------------

    eta_cases = {}

    # ------------------------------------------------------------
    # A. Neutral prior
    #
    # Every possible edge gets equal prior support.
    # ------------------------------------------------------------

    eta_cases["A - Neutral"] = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )

    # ------------------------------------------------------------
    # B. Correct prior
    #
    # High eta on true causal relationships.
    # Low eta on non-edges.
    # ------------------------------------------------------------

    eta_cases["B - Correct prior"] = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.9, 0.0, 0.9],
            [0.1, 0.9, 0.0],
        ]
    )

    # ------------------------------------------------------------
    # C. Moderately wrong prior
    #
    # The true X1 -> X2 edge is still supported,
    # but X1-X3 is incorrectly given strong prior support.
    # ------------------------------------------------------------

    eta_cases["C - Moderate misspecification"] = np.array(
        [
            [0.0, 0.9, 0.8],
            [0.9, 0.0, 0.2],
            [0.8, 0.2, 0.0],
        ]
    )

    # ------------------------------------------------------------
    # D. Strongly wrong prior
    #
    # The nonexistent X1-X3 relationship receives very strong
    # prior support, while the true X2-X3 relationship receives
    # weak prior support.
    # ------------------------------------------------------------

    eta_cases["D - Strong misspecification"] = np.array(
        [
            [0.0, 0.2, 0.99],
            [0.2, 0.0, 0.05],
            [0.99, 0.05, 0.0],
        ]
    )

    # ------------------------------------------------------------
    # E. Weak / uninformative prior
    #
    # All possible relationships receive low and approximately
    # equal prior support.
    # ------------------------------------------------------------

    eta_cases["E - Weak prior"] = np.array(
        [
            [0.0, 0.1, 0.1],
            [0.1, 0.0, 0.1],
            [0.1, 0.1, 0.0],
        ]
    )

    # ------------------------------------------------------------
    # 4. Run PEM for every eta configuration
    # ------------------------------------------------------------

    results = {}

    for case_name, eta in eta_cases.items():

        print("\n" + "-" * 70)
        print(case_name)
        print("-" * 70)

        print_matrix(
            "Eta prior",
            eta,
        )

        model = PEM(
            eta=eta,
            threshold=0.5,
            max_iter=100,
        )

        result = model.fit(samples)

        metrics = calculate_metrics(
            result.adjacency_matrix,
            ground_truth,
        )

        results[case_name] = {
            "result": result,
            "metrics": metrics,
        }

        print_matrix(
            "Posterior inclusion probabilities",
            result.inclusion_probabilities,
        )

        print_matrix(
            "Estimated adjacency",
            result.adjacency_matrix,
        )

        print("\nCausal order:")
        print(result.causal_order)

        print("\nMetrics:")
        print(f"TP        : {metrics['TP']}")
        print(f"FP        : {metrics['FP']}")
        print(f"FN        : {metrics['FN']}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1        : {metrics['f1']:.4f}")

    # ------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"{'Case':35s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
    )

    print("-" * 70)

    for case_name, data in results.items():

        metrics = data["metrics"]

        print(
            f"{case_name:35s}"
            f"{metrics['precision']:12.4f}"
            f"{metrics['recall']:12.4f}"
            f"{metrics['f1']:12.4f}"
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()