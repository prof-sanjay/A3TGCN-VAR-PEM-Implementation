"""
Inspect real VAR residuals for Local PEM readiness.

For each topology-defined causal unit:

    U_i = {i} ∪ N_i^in ∪ N_i^out

we calculate the number of complete residual observations:

    N_i = number of rows where every variable in U_i is finite.

No imputation is performed.
No PEM optimization is performed.
"""

from pathlib import Path

import numpy as np

from causal.causal_units import CausalUnitBuilder


# ============================================================
# CONFIGURATION
# ============================================================

VAR_OUTPUT_DIR = Path("outputs") / "var"

RESIDUAL_FILE = VAR_OUTPUT_DIR / "residuals.npy"
VALID_MASK_FILE = VAR_OUTPUT_DIR / "valid_mask.npy"

ADJACENCY_FILE = Path(
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_rn_adj.npy"
)

NUM_NODES = 1_000


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PEM INPUT READINESS INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nCHECKING INPUT FILES")

    for path in [
        RESIDUAL_FILE,
        VALID_MASK_FILE,
        ADJACENCY_FILE,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        print(f"  FOUND: {path}")

    # --------------------------------------------------------
    # Load residuals and mask
    # --------------------------------------------------------

    print("\nLOADING VAR OUTPUTS")

    residuals = np.load(
        RESIDUAL_FILE,
        mmap_mode="r",
    )

    valid_mask = np.load(
        VALID_MASK_FILE,
        mmap_mode="r",
    )

    print(
        f"  Residual shape : {residuals.shape}"
    )

    print(
        f"  Mask shape     : {valid_mask.shape}"
    )

    print(
        f"  Residual dtype : {residuals.dtype}"
    )

    print(
        f"  Mask dtype     : {valid_mask.dtype}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print("\nBASIC VALIDATION")

    if residuals.ndim != 2:
        raise ValueError(
            "Residuals must be a 2-dimensional array."
        )

    if valid_mask.ndim != 2:
        raise ValueError(
            "Valid mask must be a 2-dimensional array."
        )

    if residuals.shape != valid_mask.shape:
        raise ValueError(
            "Residual and valid-mask shapes do not match."
        )

    if residuals.shape[1] != NUM_NODES:
        raise ValueError(
            f"Expected {NUM_NODES} nodes, "
            f"got {residuals.shape[1]}."
        )

    print("  Shape validation: PASSED")

    # --------------------------------------------------------
    # Validate residual / mask consistency
    # --------------------------------------------------------

    print("\nRESIDUAL / MASK CONSISTENCY")

    residual_finite = np.isfinite(
        residuals
    )

    mask_boolean = np.asarray(
        valid_mask,
        dtype=bool,
    )

    consistency = np.array_equal(
        residual_finite,
        mask_boolean,
    )

    print(
        f"  Residual finite == valid mask: "
        f"{'YES' if consistency else 'NO'}"
    )

    if not consistency:
        mismatch_count = np.count_nonzero(
            residual_finite != mask_boolean
        )

        print(
            f"  Mismatched entries: "
            f"{mismatch_count}"
        )

        raise RuntimeError(
            "Residual / valid-mask consistency check failed."
        )

    print("  Validation: PASSED")

    # --------------------------------------------------------
    # Load adjacency
    # --------------------------------------------------------

    print("\nLOADING ROAD-NETWORK ADJACENCY")

    adjacency = np.load(
        ADJACENCY_FILE,
        mmap_mode="r",
    )

    adjacency = np.asarray(
        adjacency[:NUM_NODES, :NUM_NODES],
        dtype=np.float32,
    )

    print(
        f"  Adjacency shape : {adjacency.shape}"
    )

    print(
        f"  Non-zero edges  : "
        f"{np.count_nonzero(adjacency)}"
    )

    # --------------------------------------------------------
    # Build causal units
    # --------------------------------------------------------

    print("\nBUILDING CAUSAL UNITS")

    builder = CausalUnitBuilder()

    units = builder.build(
        adjacency
    )

    print(
        f"  Number of units: {len(units)}"
    )

    # --------------------------------------------------------
    # Calculate complete-case samples
    # --------------------------------------------------------

    print("\nCALCULATING COMPLETE-CASE SAMPLES")

    print(
        "  A sample is usable for a unit only when "
        "all variables in that unit have valid residuals."
    )

    complete_sample_counts = np.zeros(
        len(units),
        dtype=np.int32,
    )

    # Also record the number of variables in each unit.
    unit_sizes = np.zeros(
        len(units),
        dtype=np.int32,
    )

    for unit_index, unit in enumerate(units):

        nodes = unit.nodes

        unit_sizes[unit_index] = len(nodes)

        # ----------------------------------------------------
        # Extract validity for all variables in this unit.
        #
        # Shape:
        #     (timesteps, unit_size)
        # ----------------------------------------------------

        unit_valid = mask_boolean[
            :,
            nodes,
        ]

        # A row is usable only when every variable
        # in the unit is valid.
        complete_rows = np.all(
            unit_valid,
            axis=1,
        )

        complete_count = np.count_nonzero(
            complete_rows
        )

        complete_sample_counts[
            unit_index
        ] = complete_count

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    num_timesteps = residuals.shape[0]

    complete_percentages = (
        100.0
        * complete_sample_counts
        / num_timesteps
    )

    print("\n" + "=" * 70)
    print("COMPLETE-CASE STATISTICS")
    print("=" * 70)

    print(
        f"Minimum complete samples : "
        f"{complete_sample_counts.min()}"
    )

    print(
        f"Maximum complete samples : "
        f"{complete_sample_counts.max()}"
    )

    print(
        f"Mean complete samples    : "
        f"{complete_sample_counts.mean():.3f}"
    )

    print(
        f"Median complete samples  : "
        f"{np.median(complete_sample_counts):.1f}"
    )

    print(
        f"Minimum complete %       : "
        f"{complete_percentages.min():.6f}%"
    )

    print(
        f"Maximum complete %       : "
        f"{complete_percentages.max():.6f}%"
    )

    print(
        f"Mean complete %          : "
        f"{complete_percentages.mean():.6f}%"
    )

    # --------------------------------------------------------
    # Units by size
    # --------------------------------------------------------

    print("\nCOMPLETE SAMPLES BY UNIT SIZE")

    unique_sizes = np.unique(
        unit_sizes
    )

    print(
        f"{'Unit size':>10} "
        f"{'Units':>8} "
        f"{'Min samples':>14} "
        f"{'Mean samples':>14} "
        f"{'Min %':>12}"
    )

    print("-" * 65)

    for size in unique_sizes:

        indices = np.where(
            unit_sizes == size
        )[0]

        counts = complete_sample_counts[
            indices
        ]

        percentages = complete_percentages[
            indices
        ]

        print(
            f"{size:>10} "
            f"{len(indices):>8} "
            f"{counts.min():>14} "
            f"{counts.mean():>14.2f} "
            f"{percentages.min():>12.4f}"
        )

    # --------------------------------------------------------
    # Largest units
    # --------------------------------------------------------

    print("\nLARGEST UNITS")

    largest_indices = np.argsort(
        unit_sizes
    )[::-1][:20]

    print(
        f"{'Rank':>4} "
        f"{'Target':>8} "
        f"{'Size':>6} "
        f"{'Complete':>12} "
        f"{'Complete %':>12}"
    )

    print("-" * 50)

    for rank, index in enumerate(
        largest_indices,
        start=1,
    ):

        unit = units[index]

        print(
            f"{rank:>4} "
            f"{unit.target:>8} "
            f"{unit_sizes[index]:>6} "
            f"{complete_sample_counts[index]:>12} "
            f"{complete_percentages[index]:>12.4f}"
        )

    # --------------------------------------------------------
    # Units with minimum usable data
    # --------------------------------------------------------

    print("\nUNITS WITH FEWEST COMPLETE SAMPLES")

    smallest_indices = np.argsort(
        complete_sample_counts
    )[:20]

    print(
        f"{'Rank':>4} "
        f"{'Target':>8} "
        f"{'Size':>6} "
        f"{'Complete':>12} "
        f"{'Complete %':>12}"
    )

    print("-" * 50)

    for rank, index in enumerate(
        smallest_indices,
        start=1,
    ):

        unit = units[index]

        print(
            f"{rank:>4} "
            f"{unit.target:>8} "
            f"{unit_sizes[index]:>6} "
            f"{complete_sample_counts[index]:>12} "
            f"{complete_percentages[index]:>12.4f}"
        )

    # --------------------------------------------------------
    # Count units by usable-sample threshold
    # --------------------------------------------------------

    print("\nUSABLE-SAMPLE THRESHOLDS")

    thresholds = [
        1_000,
        5_000,
        10_000,
        15_000,
        18_000,
        19_000,
        19_500,
    ]

    for threshold in thresholds:

        count = np.count_nonzero(
            complete_sample_counts >= threshold
        )

        percentage = (
            100.0
            * count
            / len(units)
        )

        print(
            f"  >= {threshold:>6} samples : "
            f"{count:>4} units "
            f"({percentage:>7.3f}%)"
        )

    # --------------------------------------------------------
    # Single-variable units
    # --------------------------------------------------------

    print("\nSINGLE-VARIABLE UNITS")

    single_variable = (
        unit_sizes == 1
    )

    single_count = np.count_nonzero(
        single_variable
    )

    print(
        f"  Number of single-variable units: "
        f"{single_count}"
    )

    if single_count > 0:

        single_samples = complete_sample_counts[
            single_variable
        ]

        print(
            f"  Minimum complete samples: "
            f"{single_samples.min()}"
        )

        print(
            f"  Maximum complete samples: "
            f"{single_samples.max()}"
        )

        print(
            f"  Mean complete samples: "
            f"{single_samples.mean():.3f}"
        )

    # --------------------------------------------------------
    # Potential PEM readiness check
    # --------------------------------------------------------

    print("\nPEM INPUT READINESS")

    # A unit with fewer than 2 variables cannot discover
    # an edge.
    multi_variable = (
        unit_sizes >= 2
    )

    multi_variable_count = np.count_nonzero(
        multi_variable
    )

    print(
        f"  Multi-variable units: "
        f"{multi_variable_count}"
    )

    # Count units with at least 1000 complete samples.
    usable_units = (
        multi_variable
        & (complete_sample_counts >= 1_000)
    )

    usable_count = np.count_nonzero(
        usable_units
    )

    print(
        f"  Multi-variable units with "
        f">= 1,000 complete samples: "
        f"{usable_count}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Total units                  : "
        f"{len(units)}"
    )

    print(
        f"Largest unit                 : "
        f"{unit_sizes.max()} variables"
    )

    print(
        f"Smallest unit                : "
        f"{unit_sizes.min()} variables"
    )

    print(
        f"Mean unit size               : "
        f"{unit_sizes.mean():.3f}"
    )

    print(
        f"Minimum complete observations: "
        f"{complete_sample_counts.min()}"
    )

    print(
        f"Mean complete observations   : "
        f"{complete_sample_counts.mean():.3f}"
    )

    print(
        f"Minimum complete percentage  : "
        f"{complete_percentages.min():.6f}%"
    )

    print(
        f"Mean complete percentage     : "
        f"{complete_percentages.mean():.6f}%"
    )

    print("=" * 70)
    print("PEM INPUT INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()