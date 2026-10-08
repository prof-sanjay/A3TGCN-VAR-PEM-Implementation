"""
Systematic PEM prior-misspecification sweep.

Ground truth:
    X1 -> X2 -> X3

The experiment progressively increases the prior strength assigned
to the incorrect X1-X3 relationship and evaluates graph recovery
across multiple random seeds.
"""

from __future__ import annotations

import numpy as np

from causal.pem import PEM


def generate_causal_data(
    num_samples: int,
    random_state: int,
) -> np.ndarray:
    """
    Generate:

        X1 = e1
        X2 = 0.8 X1 + e2
        X3 = 0.6 X2 + e3

    Ground truth:

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
    """Calculate directed graph-recovery metrics."""

    estimated = estimated.astype(bool)
    ground_truth = ground_truth.astype(bool)

    np.fill_diagonal(estimated, False)
    np.fill_diagonal(ground_truth, False)

    tp = np.sum(estimated & ground_truth)
    fp = np.sum(estimated & ~ground_truth)
    fn = np.sum(~estimated & ground_truth)

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
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_eta(wrong_prior_strength: float) -> np.ndarray:
    """
    Construct a symmetric eta matrix.

    True edges:
        X1-X2
        X2-X3

    Incorrect edge:
        X1-X3

    The incorrect X1-X3 relationship receives progressively
    stronger prior support.

    The true edges retain moderate prior support.
    """

    eta = np.array(
        [
            [0.0, 0.5, wrong_prior_strength],
            [0.5, 0.0, 0.5],
            [wrong_prior_strength, 0.5, 0.0],
        ],
        dtype=float,
    )

    return eta


def main() -> None:

    print("=" * 80)
    print("PEM ETA PRIOR-MISSPECIFICATION SWEEP")
    print("=" * 80)

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    ground_truth = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # Experiment settings
    # ------------------------------------------------------------

    num_samples = 5000

    prior_strengths = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.99,
    ]

    seeds = [
        42,
        123,
        456,
        789,
        1000,
    ]

    # ------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------

    all_results = []

    # ------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------

    for wrong_strength in prior_strengths:

        print("\n" + "-" * 80)
        print(
            f"Wrong prior strength = {wrong_strength:.2f}"
        )
        print("-" * 80)

        eta = build_eta(wrong_strength)

        print("\nEta:")
        print(np.round(eta, 3))

        seed_results = []

        for seed in seeds:

            samples = generate_causal_data(
                num_samples=num_samples,
                random_state=seed,
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

            # ----------------------------------------------------
            # False edge:
            #
            # X1 -> X3 corresponds to
            # adjacency[2, 0]
            # ----------------------------------------------------

            false_edge_probability = (
                result.inclusion_probabilities[2, 0]
            )

            seed_results.append(
                {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fp": metrics["FP"],
                    "false_edge_probability": (
                        false_edge_probability
                    ),
                }
            )

            print(
                f"Seed {seed:4d} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"FP={metrics['FP']} | "
                f"P(false edge)={false_edge_probability:.4f}"
            )

        # --------------------------------------------------------
        # Aggregate over seeds
        # --------------------------------------------------------

        mean_precision = np.mean(
            [r["precision"] for r in seed_results]
        )

        std_precision = np.std(
            [r["precision"] for r in seed_results]
        )

        mean_recall = np.mean(
            [r["recall"] for r in seed_results]
        )

        std_recall = np.std(
            [r["recall"] for r in seed_results]
        )

        mean_f1 = np.mean(
            [r["f1"] for r in seed_results]
        )

        std_f1 = np.std(
            [r["f1"] for r in seed_results]
        )

        mean_fp = np.mean(
            [r["fp"] for r in seed_results]
        )

        mean_false_probability = np.mean(
            [
                r["false_edge_probability"]
                for r in seed_results
            ]
        )

        all_results.append(
            {
                "wrong_strength": wrong_strength,
                "mean_precision": mean_precision,
                "std_precision": std_precision,
                "mean_recall": mean_recall,
                "std_recall": std_recall,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_fp": mean_fp,
                "mean_false_probability": mean_false_probability,
            }
        )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)

    print(
        f"{'Wrong η':>10} "
        f"{'Precision':>20} "
        f"{'Recall':>20} "
        f"{'F1':>20} "
        f"{'Mean FP':>10} "
        f"{'P(false)':>15}"
    )

    print("-" * 100)

    for result in all_results:

        print(
            f"{result['wrong_strength']:10.2f} "
            f"{result['mean_precision']:.4f} ± "
            f"{result['std_precision']:.4f}    "
            f"{result['mean_recall']:.4f} ± "
            f"{result['std_recall']:.4f}    "
            f"{result['mean_f1']:.4f} ± "
            f"{result['std_f1']:.4f}    "
            f"{result['mean_fp']:10.2f} "
            f"{result['mean_false_probability']:15.4f}"
        )

    # ------------------------------------------------------------
    # Interpretation helpers
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        "\nThe important quantities are:"
    )

    print(
        "1. Mean F1        -> overall graph recovery"
    )

    print(
        "2. Mean FP        -> false edges introduced"
    )

    print(
        "3. P(false)       -> posterior inclusion probability "
        "of the incorrect X1-X3 relationship"
    )

    print(
        "\nAs the wrong prior becomes stronger, we expect the "
        "posterior probability of the false edge to increase."
    )

    print(
        "The experiment is specifically checking how strongly "
        "PEM is affected by increasingly incorrect prior information."
    )

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()