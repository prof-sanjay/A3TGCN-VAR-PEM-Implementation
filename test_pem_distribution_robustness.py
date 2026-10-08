"""
PEM robustness under different noise distributions.

Ground truth:
    X1 -> X2 -> X3

The same structural causal model is generated using:

    1. Gaussian noise
    2. Laplace noise
    3. Student-t noise
    4. Skewed non-Gaussian noise

The purpose is to determine whether PEM graph recovery is sensitive
to the distribution of the observed variables/noise.

This is a synthetic validation experiment only. It does not establish
that PEM's theoretical assumptions are satisfied by traffic residuals.
"""

from __future__ import annotations

import numpy as np

from causal.pem import PEM


def generate_noise(
    distribution: str,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate standardized noise for the requested distribution."""

    if distribution == "Gaussian":

        noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(num_samples, 3),
        )

    elif distribution == "Laplace":

        noise = rng.laplace(
            loc=0.0,
            scale=1.0 / np.sqrt(2.0),
            size=(num_samples, 3),
        )

    elif distribution == "Student-t":

        noise = rng.standard_t(
            df=5,
            size=(num_samples, 3),
        )

        # Standardize approximately to unit variance.
        noise = noise / np.sqrt(5.0 / 3.0)

    elif distribution == "Skewed":

        # Exponential noise centered to zero and standardized.
        noise = rng.exponential(
            scale=1.0,
            size=(num_samples, 3),
        )

        noise = noise - 1.0

    else:
        raise ValueError(
            f"Unknown distribution: {distribution}"
        )

    return noise


def generate_causal_data(
    distribution: str,
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

    rng = np.random.default_rng(
        random_state
    )

    noise = generate_noise(
        distribution=distribution,
        num_samples=num_samples,
        rng=rng,
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

    return np.column_stack(
        [x1, x2, x3]
    )


def calculate_metrics(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Calculate directed graph-recovery metrics."""

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


def main() -> None:

    print("=" * 90)
    print("PEM DISTRIBUTION ROBUSTNESS TEST")
    print("=" * 90)

    # ------------------------------------------------------------
    # Ground-truth causal structure
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

    distributions = [
        "Gaussian",
        "Laplace",
        "Student-t",
        "Skewed",
    ]

    sample_sizes = [
        500,
        1000,
        2500,
        5000,
    ]

    seeds = [
        42,
        123,
        456,
        789,
        1000,
    ]

    # Use the neutral prior so that this experiment isolates
    # distributional effects.
    eta = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )

    all_results = []

    # ------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------

    for distribution in distributions:

        for num_samples in sample_sizes:

            print("\n" + "-" * 90)

            print(
                f"Distribution={distribution} | "
                f"Samples={num_samples}"
            )

            print("-" * 90)

            seed_results = []

            for seed in seeds:

                samples = generate_causal_data(
                    distribution=distribution,
                    num_samples=num_samples,
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

                # X1-X3 is a non-edge.
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

            # ----------------------------------------------------
            # Aggregate
            # ----------------------------------------------------

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
                    "distribution": distribution,
                    "samples": num_samples,
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
    # Final summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("AGGREGATED RESULTS")
    print("=" * 110)

    print(
        f"{'Distribution':>12} "
        f"{'Samples':>8} "
        f"{'Precision':>18} "
        f"{'Recall':>18} "
        f"{'F1':>18} "
        f"{'Mean FP':>10} "
        f"{'P(false)':>12}"
    )

    print("-" * 110)

    for result in all_results:

        print(
            f"{result['distribution']:>12} "
            f"{result['samples']:8d} "
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