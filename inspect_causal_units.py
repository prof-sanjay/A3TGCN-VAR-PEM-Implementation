"""
Inspect topology-defined causal units.

For each target node:

    U_i = {i} ∪ N_i^in ∪ N_i^out

This script only inspects the units.
It does not run PEM.
"""

from pathlib import Path

import numpy as np

from causal.causal_units import CausalUnitBuilder


# ============================================================
# CONFIGURATION
# ============================================================

ADJACENCY_FILE = Path(
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_rn_adj.npy"
)

NUM_NODES = 1_000


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CAUSAL UNIT INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load adjacency
    # --------------------------------------------------------

    if not ADJACENCY_FILE.exists():
        raise FileNotFoundError(
            f"Adjacency file not found: {ADJACENCY_FILE}"
        )

    print("\nLoading adjacency...")

    adjacency = np.load(
        ADJACENCY_FILE,
        mmap_mode="r",
    )

    adjacency = np.asarray(
        adjacency[:NUM_NODES, :NUM_NODES],
        dtype=np.float32,
    )

    print(
        f"Adjacency shape : {adjacency.shape}"
    )

    # --------------------------------------------------------
    # Build causal units
    # --------------------------------------------------------

    print("\nBuilding causal units...")

    builder = CausalUnitBuilder()

    units = builder.build(adjacency)

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    unit_sizes = np.array(
        [len(unit.nodes) for unit in units],
        dtype=np.int32,
    )

    incoming_sizes = np.array(
        [
            len(unit.incoming_neighbors)
            for unit in units
        ],
        dtype=np.int32,
    )

    outgoing_sizes = np.array(
        [
            len(unit.outgoing_neighbors)
            for unit in units
        ],
        dtype=np.int32,
    )

    neighbor_sizes = np.array(
        [
            len(unit.neighbors)
            for unit in units
        ],
        dtype=np.int32,
    )

    print("\n" + "=" * 70)
    print("UNIT STATISTICS")
    print("=" * 70)

    print(
        f"Number of units       : {len(units)}"
    )

    print(
        f"Minimum unit size     : {unit_sizes.min()}"
    )

    print(
        f"Maximum unit size     : {unit_sizes.max()}"
    )

    print(
        f"Mean unit size        : {unit_sizes.mean():.6f}"
    )

    print(
        f"Median unit size      : {np.median(unit_sizes):.1f}"
    )

    print(
        f"Minimum neighbors     : {neighbor_sizes.min()}"
    )

    print(
        f"Maximum neighbors     : {neighbor_sizes.max()}"
    )

    print(
        f"Mean neighbors        : {neighbor_sizes.mean():.6f}"
    )

    print(
        f"Median neighbors      : {np.median(neighbor_sizes):.1f}"
    )

    print(
        f"Mean incoming         : {incoming_sizes.mean():.6f}"
    )

    print(
        f"Mean outgoing         : {outgoing_sizes.mean():.6f}"
    )

    # --------------------------------------------------------
    # Unit-size distribution
    # --------------------------------------------------------

    print("\nUNIT SIZE DISTRIBUTION")

    size_values, size_counts = np.unique(
        unit_sizes,
        return_counts=True,
    )

    for size, count in zip(
        size_values,
        size_counts,
    ):

        print(
            f"  Size {size:>3}: "
            f"{count:>4} units"
        )

    # --------------------------------------------------------
    # Largest units
    # --------------------------------------------------------

    print("\nLARGEST CAUSAL UNITS")

    largest_indices = np.argsort(
        unit_sizes
    )[::-1][:20]

    print(
        f"{'Rank':>4} "
        f"{'Target':>8} "
        f"{'Unit':>8} "
        f"{'In':>6} "
        f"{'Out':>6}"
    )

    print("-" * 40)

    for rank, index in enumerate(
        largest_indices,
        start=1,
    ):

        unit = units[index]

        print(
            f"{rank:>4} "
            f"{unit.target:>8} "
            f"{len(unit.nodes):>8} "
            f"{len(unit.incoming_neighbors):>6} "
            f"{len(unit.outgoing_neighbors):>6}"
        )

    # --------------------------------------------------------
    # Validate unit structure
    # --------------------------------------------------------

    print("\nUNIT STRUCTURE VALIDATION")

    structure_passed = True

    for unit in units:

        target = unit.target
        nodes = set(unit.nodes.tolist())

        # Target must belong to its own unit.
        if target not in nodes:
            print(
                f"  FAIL: target {target} "
                "is missing from its unit"
            )
            structure_passed = False

        # Incoming and outgoing neighbors should not
        # contain the target itself.
        if target in set(
            unit.incoming_neighbors.tolist()
        ):
            print(
                f"  FAIL: target {target} "
                "appears in incoming neighbors"
            )
            structure_passed = False

        if target in set(
            unit.outgoing_neighbors.tolist()
        ):
            print(
                f"  FAIL: target {target} "
                "appears in outgoing neighbors"
            )
            structure_passed = False

        # All incoming/outgoing neighbors must belong
        # to the final unit.
        if not set(
            unit.incoming_neighbors.tolist()
        ).issubset(nodes):

            print(
                f"  FAIL: target {target} "
                "has incoming neighbor outside unit"
            )
            structure_passed = False

        if not set(
            unit.outgoing_neighbors.tolist()
        ).issubset(nodes):

            print(
                f"  FAIL: target {target} "
                "has outgoing neighbor outside unit"
            )
            structure_passed = False

    print(
        f"  Validation: "
        f"{'PASSED' if structure_passed else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Check adjacency consistency
    # --------------------------------------------------------

    print("\nADJACENCY CONSISTENCY")

    adjacency_passed = True

    for unit in units:

        target = unit.target

        expected_incoming = set(
            np.flatnonzero(
                adjacency[:, target] != 0
            ).tolist()
        )

        expected_outgoing = set(
            np.flatnonzero(
                adjacency[target, :] != 0
            ).tolist()
        )

        expected_incoming.discard(target)
        expected_outgoing.discard(target)

        actual_incoming = set(
            unit.incoming_neighbors.tolist()
        )

        actual_outgoing = set(
            unit.outgoing_neighbors.tolist()
        )

        if actual_incoming != expected_incoming:
            print(
                f"  FAIL: incoming mismatch "
                f"for target {target}"
            )
            adjacency_passed = False

        if actual_outgoing != expected_outgoing:
            print(
                f"  FAIL: outgoing mismatch "
                f"for target {target}"
            )
            adjacency_passed = False

    print(
        f"  Validation: "
        f"{'PASSED' if adjacency_passed else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    all_passed = (
        structure_passed
        and adjacency_passed
    )

    if all_passed:
        print(
            "ALL CAUSAL UNIT VALIDATION CHECKS PASSED"
        )
    else:
        print(
            "CAUSAL UNIT VALIDATION FAILED"
        )

    print("=" * 70)

    if not all_passed:
        raise RuntimeError(
            "Causal unit validation failed."
        )


if __name__ == "__main__":
    main()