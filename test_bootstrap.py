import numpy as np

from causal.bootstrap import ResidualBootstrap


def main():
    print("=" * 60)
    print("RESIDUAL BOOTSTRAP TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Create synthetic residual data
    # ---------------------------------------------------------
    rng = np.random.default_rng(123)

    residuals = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(100, 3),
    )

    # Add a few missing observations
    residuals[10, 1] = np.nan
    residuals[25, 2] = np.nan
    residuals[50, 0] = np.nan

    print("\nOriginal residual shape:")
    print(residuals.shape)

    print("Original NaN count:")
    print(np.isnan(residuals).sum())

    # ---------------------------------------------------------
    # 2. Create bootstrap object
    # ---------------------------------------------------------
    bootstrap = ResidualBootstrap(
        num_bootstrap_samples=5,
        random_state=42,
    )

    # ---------------------------------------------------------
    # 3. Generate bootstrap samples
    # ---------------------------------------------------------
    result = bootstrap.fit(residuals)

    print("\nValid residual shape:")
    print(result.valid_residuals.shape)

    print("\nBootstrap samples shape:")
    print(result.samples.shape)

    print("\nNumber of bootstrap samples:")
    print(result.num_samples)

    # ---------------------------------------------------------
    # 4. Check that bootstrap samples contain no NaNs
    # ---------------------------------------------------------
    print("\nNaNs in bootstrap samples:")
    print(np.isnan(result.samples).sum())

    # ---------------------------------------------------------
    # 5. Check reproducibility
    # ---------------------------------------------------------
    bootstrap_again = ResidualBootstrap(
        num_bootstrap_samples=5,
        random_state=42,
    )

    result_again = bootstrap_again.fit(residuals)

    reproducible = np.array_equal(
        result.samples,
        result_again.samples,
    )

    print("\nReproducible with same random seed:")
    print(reproducible)

    # ---------------------------------------------------------
    # 6. Check that different bootstrap samples differ
    # ---------------------------------------------------------
    different_samples = not np.array_equal(
        result.samples[0],
        result.samples[1],
    )

    print("\nDifferent bootstrap samples generated:")
    print(different_samples)

    # ---------------------------------------------------------
    # 7. Final validation
    # ---------------------------------------------------------
    checks = [
        result.samples.shape == (5, 97, 3),
        np.isnan(result.samples).sum() == 0,
        reproducible,
        different_samples,
    ]

    print("\nValidation checks:")
    print(f"Shape correct:       {checks[0]}")
    print(f"No NaNs:             {checks[1]}")
    print(f"Reproducible:        {checks[2]}")
    print(f"Samples differ:      {checks[3]}")

    if all(checks):
        print("\nPASS: Bootstrap implementation is working correctly.")
    else:
        print("\nFAIL: One or more bootstrap checks failed.")


if __name__ == "__main__":
    main()