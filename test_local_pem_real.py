"""
Controlled real-data Local PEM test.

Purpose:
    Run Local PEM on a small representative subset of the
    actual GBA VAR residuals before processing all units.

The test includes:
    - small units
    - medium units
    - large units
    - units with different missing-data patterns

No imputation is performed.

The canonical PEM test uses a neutral prior:

    eta_ij = 0.5

for all off-diagonal pairs.

This script does NOT construct the final global causal graph.
"""

import time
from pathlib import Path

import numpy as np

from causal.causal_units import CausalUnitBuilder
from causal.local_pem import LocalPEM


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

# Number of representative units to test.
NUM_TEST_UNITS = 12

# Minimum number of complete samples required.
MIN_COMPLETE_SAMPLES = 1_000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def select_representative_units(
    units,
    valid_mask,
    num_units,
    min_complete_samples,
):
    """
    Select representative units covering different unit sizes.

    The selection intentionally includes:
        - smallest usable multi-variable units
        - small/medium units
        - large units
        - largest units

    Returns:
        list of unit indices
    """

    candidates = []

    for index, unit in enumerate(units):

        if len(unit.nodes) < 2:
            continue

        complete_rows = np.all(
            valid_mask[:, unit.nodes],
            axis=1,
        )

        complete_count = np.count_nonzero(
            complete_rows
        )

        if complete_count >= min_complete_samples:
            candidates.append(
                (
                    index,
                    len(unit.nodes),
                    complete_count,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No suitable multi-variable units found."
        )

    # Sort by unit size.
    candidates.sort(
        key=lambda x: x[1]
    )

    selected = []

    # --------------------------------------------------------
    # Smallest usable unit
    # --------------------------------------------------------

    selected.append(
        candidates[0][0]
    )

    # --------------------------------------------------------
    # Quantile-based representatives
    # --------------------------------------------------------

    if len(candidates) > 1:

        positions = np.linspace(
            0,
            len(candidates) - 1,
            num=min(
                num_units - 2,
                len(candidates),
            ),
            dtype=int,
        )

        for position in positions:

            index = candidates[position][0]

            if index not in selected:
                selected.append(index)

    # --------------------------------------------------------
    # Largest usable unit
    # --------------------------------------------------------

    selected.append(
        candidates[-1][0]
    )

    # Remove duplicates while preserving order.
    selected = list(
        dict.fromkeys(selected)
    )

    # Limit final count.
    selected = selected[:num_units]

    return selected


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CONTROLLED REAL-DATA LOCAL PEM TEST")
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
    # Load residuals
    # --------------------------------------------------------

    print("\nLOADING VAR RESIDUALS")

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

    if residuals.shape != valid_mask.shape:
        raise ValueError(
            "Residual and valid-mask shapes differ."
        )

    if residuals.shape[1] != NUM_NODES:
        raise ValueError(
            f"Expected {NUM_NODES} nodes, "
            f"got {residuals.shape[1]}."
        )

    # --------------------------------------------------------
    # Validate residual/mask consistency
    # --------------------------------------------------------

    print("\nVALIDATING RESIDUAL/MASK CONSISTENCY")

    finite_residuals = np.isfinite(residuals)

    mask_consistent = np.array_equal(
        finite_residuals,
        valid_mask,
    )

    print(
        f"  Finite residual == valid mask: "
        f"{'YES' if mask_consistent else 'NO'}"
    )

    if not mask_consistent:
        raise RuntimeError(
            "Residual finite-mask and valid_mask do not match."
        )

    # --------------------------------------------------------
    # Load adjacency
    # --------------------------------------------------------

    print("\nLOADING ADJACENCY")

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

    if adjacency.shape != (
        NUM_NODES,
        NUM_NODES,
    ):
        raise ValueError(
            "Adjacency shape does not match "
            "the configured number of nodes."
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
        f"  Total units: {len(units)}"
    )

    # --------------------------------------------------------
    # Select representative units
    # --------------------------------------------------------

    selected_indices = select_representative_units(
        units=units,
        valid_mask=valid_mask,
        num_units=NUM_TEST_UNITS,
        min_complete_samples=MIN_COMPLETE_SAMPLES,
    )

    print("\nSELECTED TEST UNITS")

    print(
        f"{'Test':>5} "
        f"{'Unit index':>12} "
        f"{'Target':>8} "
        f"{'Size':>6} "
        f"{'Complete':>12}"
    )

    print("-" * 50)

    selected_information = []

    for test_number, unit_index in enumerate(
        selected_indices,
        start=1,
    ):

        unit = units[unit_index]

        complete_rows = np.all(
            valid_mask[:, unit.nodes],
            axis=1,
        )

        complete_count = np.count_nonzero(
            complete_rows
        )

        print(
            f"{test_number:>5} "
            f"{unit_index:>12} "
            f"{unit.target:>8} "
            f"{len(unit.nodes):>6} "
            f"{complete_count:>12}"
        )

        selected_information.append(
            (
                test_number,
                unit_index,
                unit,
                complete_count,
            )
        )

    # --------------------------------------------------------
    # Create selected-unit list
    # --------------------------------------------------------

    selected_units = [
        units[index]
        for index in selected_indices
    ]

    print("\nSELECTED PEM UNITS")
    print(
        f"  Total selected units: "
        f"{len(selected_units)}"
    )

    print(
        "  PEM will process ONLY these selected units."
    )

    # --------------------------------------------------------
    # Create Local PEM model
    # --------------------------------------------------------

    print("\nINITIALIZING LOCAL PEM")

    pem = LocalPEM(
        selected_units
    )

    print("  Prior: neutral eta = 0.5")
    print("  Imputation: disabled")
    print("  Data: VAR residuals")

    # --------------------------------------------------------
    # Run PEM
    # --------------------------------------------------------

    print("\nRUNNING PEM")

    start_time = time.perf_counter()

    result = pem.fit(
        residuals
    )

    total_runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        f"\n  Total PEM runtime: "
        f"{total_runtime:.3f} seconds"
    )

    # --------------------------------------------------------
    # Validate overall result structure
    # --------------------------------------------------------

    print("\nVALIDATING PEM RESULT")

    expected_global_shape = (
        NUM_NODES,
        NUM_NODES,
    )

    global_inclusion = (
        result.global_inclusion_probabilities
    )

    global_adjacency = (
        result.global_adjacency
    )

    if global_inclusion.shape != expected_global_shape:

        raise RuntimeError(
            "Global inclusion-probability shape mismatch: "
            f"{global_inclusion.shape}"
        )

    if global_adjacency.shape != expected_global_shape:

        raise RuntimeError(
            "Global adjacency shape mismatch: "
            f"{global_adjacency.shape}"
        )

    if len(result.unit_inclusion_probabilities) != len(
        selected_units
    ):

        raise RuntimeError(
            "Number of unit inclusion-probability "
            "results does not match selected units."
        )

    if len(result.unit_adjacencies) != len(
        selected_units
    ):

        raise RuntimeError(
            "Number of unit adjacency results "
            "does not match selected units."
        )

    if len(result.unit_causal_orders) != len(
        selected_units
    ):

        raise RuntimeError(
            "Number of causal orders does not "
            "match selected units."
        )

    print(
        f"  Global inclusion shape: "
        f"{global_inclusion.shape}"
    )

    print(
        f"  Global adjacency shape: "
        f"{global_adjacency.shape}"
    )

    print(
        f"  Processed units: "
        f"{result.processed_units}"
    )

    # --------------------------------------------------------
    # Global inclusion probability validation
    # --------------------------------------------------------

    print("\nGLOBAL INCLUSION-PROBABILITY VALIDATION")

    finite_inclusion = np.isfinite(
        global_inclusion
    ).all()

    within_range = (
        np.all(global_inclusion >= 0.0)
        and np.all(global_inclusion <= 1.0)
    )

    diagonal_zero = np.allclose(
        np.diag(global_inclusion),
        0.0,
    )

    symmetric = np.allclose(
        global_inclusion,
        global_inclusion.T,
    )

    print(
        f"  Inclusion finite : "
        f"{'YES' if finite_inclusion else 'NO'}"
    )

    print(
        f"  Range [0,1]      : "
        f"{'YES' if within_range else 'NO'}"
    )

    print(
        f"  Diagonal zero    : "
        f"{'YES' if diagonal_zero else 'NO'}"
    )

    print(
        f"  Symmetric        : "
        f"{'YES' if symmetric else 'NO'}"
    )

    if not finite_inclusion:
        raise RuntimeError(
            "PEM produced non-finite inclusion probabilities."
        )

    if not within_range:
        raise RuntimeError(
            "PEM produced inclusion probabilities "
            "outside [0,1]."
        )

    if not diagonal_zero:
        raise RuntimeError(
            "PEM inclusion-probability diagonal "
            "is not zero."
        )

    if not symmetric:
        raise RuntimeError(
            "PEM inclusion probabilities are not symmetric."
        )

    # --------------------------------------------------------
    # Global directed graph validation
    # --------------------------------------------------------

    print("\nGLOBAL DIRECTED GRAPH VALIDATION")

    global_directed_edges = np.count_nonzero(
        global_adjacency
    )

    global_diagonal_zero = np.allclose(
        np.diag(global_adjacency),
        0.0,
    )

    finite_global_adjacency = np.isfinite(
        global_adjacency
    ).all()

    print(
        f"  Directed edges   : "
        f"{global_directed_edges}"
    )

    print(
        f"  Diagonal zero    : "
        f"{'YES' if global_diagonal_zero else 'NO'}"
    )

    print(
        f"  All values finite: "
        f"{'YES' if finite_global_adjacency else 'NO'}"
    )

    if not global_diagonal_zero:
        raise RuntimeError(
            "Global PEM adjacency contains self-edges."
        )

    if not finite_global_adjacency:
        raise RuntimeError(
            "Global PEM adjacency contains non-finite values."
        )

    # --------------------------------------------------------
    # Validate each selected unit
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTED UNIT RESULTS")
    print("=" * 70)

    total_unit_edges = 0

    for result_index, (
        test_number,
        unit_index,
        unit,
        expected_complete_count,
    ) in enumerate(
        selected_information
    ):

        print("\n" + "-" * 70)

        print(
            f"TEST UNIT {test_number}/{len(selected_units)}"
        )

        print(
            f"  Unit index : {unit_index}"
        )

        print(
            f"  Target     : {unit.target}"
        )

        print(
            f"  Unit size  : {len(unit.nodes)}"
        )

        print(
            f"  Nodes      : {unit.nodes.tolist()}"
        )

        # ----------------------------------------------------
        # Sample count returned by Local PEM
        # ----------------------------------------------------

        actual_complete_count = int(
            result.valid_sample_counts[
                result_index
            ]
        )

        print(
            f"  Complete samples: "
            f"{actual_complete_count}"
        )

        if (
            actual_complete_count
            != expected_complete_count
        ):

            raise RuntimeError(
                "Complete-sample count mismatch "
                f"for unit {unit_index}."
            )

        # ----------------------------------------------------
        # Local PEM result
        # ----------------------------------------------------

        inclusion = np.asarray(
            result.unit_inclusion_probabilities[
                result_index
            ],
            dtype=np.float64,
        )

        adjacency_result = np.asarray(
            result.unit_adjacencies[
                result_index
            ],
            dtype=np.float64,
        )

        causal_order = (
            result.unit_causal_orders[
                result_index
            ]
        )

        expected_local_shape = (
            len(unit.nodes),
            len(unit.nodes),
        )

        if inclusion.shape != expected_local_shape:

            raise RuntimeError(
                "Local inclusion shape mismatch "
                f"for unit {unit_index}: "
                f"{inclusion.shape}"
            )

        if adjacency_result.shape != expected_local_shape:

            raise RuntimeError(
                "Local adjacency shape mismatch "
                f"for unit {unit_index}: "
                f"{adjacency_result.shape}"
            )

        print(
            f"  Inclusion shape : "
            f"{inclusion.shape}"
        )

        print(
            f"  Adjacency shape : "
            f"{adjacency_result.shape}"
        )

        print(
            f"  Causal order    : "
            f"{causal_order}"
        )

        # ----------------------------------------------------
        # Local inclusion validation
        # ----------------------------------------------------

        finite_inclusion = np.isfinite(
            inclusion
        ).all()

        within_range = (
            np.all(inclusion >= 0.0)
            and np.all(inclusion <= 1.0)
        )

        diagonal_zero = np.allclose(
            np.diag(inclusion),
            0.0,
        )

        symmetric = np.allclose(
            inclusion,
            inclusion.T,
        )

        print(
            f"  Inclusion finite : "
            f"{'YES' if finite_inclusion else 'NO'}"
        )

        print(
            f"  Range [0,1]      : "
            f"{'YES' if within_range else 'NO'}"
        )

        print(
            f"  Diagonal zero    : "
            f"{'YES' if diagonal_zero else 'NO'}"
        )

        print(
            f"  Symmetric        : "
            f"{'YES' if symmetric else 'NO'}"
        )

        if not finite_inclusion:
            raise RuntimeError(
                f"Non-finite inclusion probabilities "
                f"in unit {unit_index}."
            )

        if not within_range:
            raise RuntimeError(
                f"Inclusion probabilities outside [0,1] "
                f"in unit {unit_index}."
            )

        if not diagonal_zero:
            raise RuntimeError(
                f"Non-zero inclusion diagonal "
                f"in unit {unit_index}."
            )

        if not symmetric:
            raise RuntimeError(
                f"Inclusion probabilities are not symmetric "
                f"in unit {unit_index}."
            )

        # ----------------------------------------------------
        # Local directed graph validation
        # ----------------------------------------------------

        directed_edges = np.count_nonzero(
            adjacency_result
        )

        total_unit_edges += directed_edges

        adjacency_diagonal_zero = np.allclose(
            np.diag(adjacency_result),
            0.0,
        )

        adjacency_finite = np.isfinite(
            adjacency_result
        ).all()

        print(
            f"  Directed edges  : "
            f"{directed_edges}"
        )

        print(
            f"  Adjacency finite: "
            f"{'YES' if adjacency_finite else 'NO'}"
        )

        print(
            f"  Diagonal zero   : "
            f"{'YES' if adjacency_diagonal_zero else 'NO'}"
        )

        if not adjacency_finite:
            raise RuntimeError(
                f"Non-finite adjacency values "
                f"in unit {unit_index}."
            )

        if not adjacency_diagonal_zero:
            raise RuntimeError(
                f"Self-edge detected in unit {unit_index}."
            )

        # ----------------------------------------------------
        # Local edge list
        # ----------------------------------------------------

        print("\n  DISCOVERED LOCAL EDGES")

        edge_count = 0

        for target in range(
            adjacency_result.shape[0]
        ):

            for source in range(
                adjacency_result.shape[1]
            ):

                if adjacency_result[
                    target,
                    source,
                ] <= 0.0:
                    continue

                global_target = (
                    unit.nodes[target]
                )

                global_source = (
                    unit.nodes[source]
                )

                probability = inclusion[
                    target,
                    source,
                ]

                print(
                    f"    {global_source} -> "
                    f"{global_target} "
                    f"(p={probability:.6f})"
                )

                edge_count += 1

        if edge_count == 0:
            print(
                "    No directed edges discovered."
            )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REAL-DATA PEM TEST SUMMARY")
    print("=" * 70)

    print(
        f"Units selected          : "
        f"{len(selected_units)}"
    )

    print(
        f"Units successfully run  : "
        f"{result.processed_units}"
    )

    print(
        f"Total PEM runtime       : "
        f"{total_runtime:.3f} seconds"
    )

    print(
        f"Total discovered edges  : "
        f"{total_unit_edges}"
    )

    if len(selected_units) > 0:

        mean_edges = (
            total_unit_edges
            / len(selected_units)
        )

        print(
            f"Mean edges per unit     : "
            f"{mean_edges:.3f}"
        )

    largest_unit = max(
        len(unit.nodes)
        for unit in selected_units
    )

    smallest_unit = min(
        len(unit.nodes)
        for unit in selected_units
    )

    print(
        f"Smallest tested unit    : "
        f"{smallest_unit} variables"
    )

    print(
        f"Largest tested unit     : "
        f"{largest_unit} variables"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REAL-DATA PEM TEST COMPLETE")
    print("=" * 70)

    print(
        "\nSTATUS: ALL CONTROLLED PEM VALIDATION CHECKS PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()