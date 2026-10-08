"""
PEM robustness test under different sample sizes and causal strengths.

Ground truth:
    X1 -> X2 -> X3

Experiments:
    - Different sample sizes
    - Strong / medium / weak causal coefficients
    - Neutral prior
    - Strongly misspecified prior

The purpose is to determine whether PEM graph recovery remains
stable when the available data become smaller or causal signals
become weaker.
"""

from __future__ import annotations

import numpy as np

from causal.pem import PEM


def generate_causal_data(
    num_samples: int,
    coefficient_12: float,
    coefficient_23: float,
    random_state: int,
) -> np.ndarray:
    """
    Generate:

        X1 = e1
        X2 = coefficient_12 * X1 + e2
        X3 = coefficient_23 * X2 + e3

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

    x2 = (
        coefficient_12 * x1
        + noise[:, 1]
    )

    x3 = (
        coefficient_23 * x2
        + noise[:, 2]
    )

    return np.column_stack(
        [x1, x2, x3]
    )


def calculate_metrics(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Calculate directed graph recovery metrics."""

    estimated = estimated.astype(bool)
    ground_truth = ground_truth.astype(bool)

    np.fill_diagonal(
        estimated,
        False,
    )

    np.fill_diagonal(
        ground_truth,
        False,
    )

    tp = np.sum(
        estimated & ground_truth
    )

    fp = np.sum(
        estimated & ~ground_truth
    )

    fn = np.sum(
        ~estimated & ground_truth
    )

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
        2.0
        * precision
        * recall
        / (precision + recall)
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


def build_eta(
    misspecified: bool,
) -> np.ndarray:
    """
    Construct the PEM prior.

    Neutral prior:
        eta = 0.5 for every possible relationship.

    Misspecified prior:
        Strongly favors the incorrect X1-X3 relationship.
    """

    if not misspecified:

        return np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )

    return np.array(
        [
            [0.0, 0.5, 0.90],
            [0.5, 0.0, 0.20],
            [0.90, 0.20, 0.0],
        ],
        dtype=float,
    )


def main() -> None:

    print("=" * 90)
    print("PEM DATA ROBUSTNESS TEST")
    print("=" * 90)

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
    # Experiment configurations
    # ------------------------------------------------------------

    causal_strengths = {
        "Strong": (0.8, 0.6),
        "Medium": (0.4, 0.3),
        "Weak": (0.2, 0.15),
    }

    sample_sizes = [
        500,
        1000,
        2500,
        5000,
    ]

    prior_types = {
        "Neutral": False,
        "Misspecified": True,
    }

    seeds = [
        42,
        123,
        456,
        789,
        1000,
    ]

    # ------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------

    all_results = []

    for strength_name, coefficients in causal_strengths.items():

        coefficient_12, coefficient_23 = coefficients

        for num_samples in sample_sizes:

            for prior_name, misspecified in prior_types.items():

                print("\n" + "-" * 90)

                print(
                    f"Strength={strength_name} | "
                    f"Samples={num_samples} | "
                    f"Prior={prior_name}"
                )

                print("-" * 90)

                eta = build_eta(
                    misspecified=misspecified
                )

                seed_results = []

                for seed in seeds:

                    samples = generate_causal_data(
                        num_samples=num_samples,
                        coefficient_12=coefficient_12,
                        coefficient_23=coefficient_23,
                        random_state=seed,
                    )

                    model = PEM(
                        eta=eta,
                        threshold=0.5,
                        max_iter=100,
                    )

                    result = model.fit(
                        samples
                    )

                    metrics = calculate_metrics(
                        result.adjacency_matrix,
                        ground_truth,
                    )

                    false_edge_probability = (
                        result.inclusion_probabilities[2, 0]
                    )

                    seed_results.append(
                        {
                            "precision": metrics[
                                "precision"
                            ],
                            "recall": metrics[
                                "recall"
                            ],
                            "f1": metrics[
                                "f1"
                            ],
                            "fp": metrics[
                                "FP"
                            ],
                            "false_edge_probability": (
                                false_edge_probability
                            ),
                        }
                    )

                    print(
                        f"Seed {seed:4d} | "
                        f"P={metrics['precision']:.4f} | "
                        f"R={metrics['recall']:.4f} | "
                        f"F1={metrics['f1']:.4f} | "
                        f"FP={metrics['FP']} | "
                        f"P(false)="
                        f"{false_edge_probability:.4f}"
                    )

                # ------------------------------------------------
                # Aggregate
                # ------------------------------------------------

                mean_precision = np.mean(
                    [
                        r["precision"]
                        for r in seed_results
                    ]
                )

                std_precision = np.std(
                    [
                        r["precision"]
                        for r in seed_results
                    ]
                )

                mean_recall = np.mean(
                    [
                        r["recall"]
                        for r in seed_results
                    ]
                )

                std_recall = np.std(
                    [
                        r["recall"]
                        for r in seed_results
                    ]
                )

                mean_f1 = np.mean(
                    [
                        r["f1"]
                        for r in seed_results
                    ]
                )

                std_f1 = np.std(
                    [
                        r["f1"]
                        for r in seed_results
                    ]
                )

                mean_fp = np.mean(
                    [
                        r["fp"]
                        for r in seed_results
                    ]
                )

                mean_false_probability = np.mean(
                    [
                        r["false_edge_probability"]
                        for r in seed_results
                    ]
                )

                all_results.append(
                    {
                        "strength": strength_name,
                        "samples": num_samples,
                        "prior": prior_name,
                        "mean_precision": mean_precision,
                        "std_precision": std_precision,
                        "mean_recall": mean_recall,
                        "std_recall": std_recall,
                        "mean_f1": mean_f1,
                        "std_f1": std_f1,
                        "mean_fp": mean_fp,
                        "mean_false_probability": (
                            mean_false_probability
                        ),
                    }
                )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("AGGREGATED RESULTS")
    print("=" * 110)

    print(
        f"{'Strength':>10} "
        f"{'Samples':>8} "
        f"{'Prior':>14} "
        f"{'Precision':>18} "
        f"{'Recall':>18} "
        f"{'F1':>18} "
        f"{'Mean FP':>10} "
        f"{'P(false)':>12}"
    )

    print("-" * 110)

    for result in all_results:

        print(
            f"{result['strength']:>10} "
            f"{result['samples']:8d} "
            f"{result['prior']:>14} "
            f"{result['mean_precision']:.4f} ± "
            f"{result['std_precision']:.4f}    "
            f"{result['mean_recall']:.4f} ± "
            f"{result['std_recall']:.4f}    "
            f"{result['mean_f1']:.4f} ± "
            f"{result['std_f1']:.4f}    "
            f"{result['mean_fp']:10.2f} "
            f"{result['mean_false_probability']:12.4f}"
        )

    print("\n" + "=" * 90)
    print("TEST COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()