import os
import time
import numpy as np

from causal.causal_units import CausalUnitBuilder
from causal.local_pem import LocalPEM


# ============================================================
# CONFIGURATION
# ============================================================

RESIDUAL_PATH = "outputs/var/residuals.npy"
ADJACENCY_PATH = r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_rn_adj.npy"

NUM_NODES = 1000

# Use a fixed number of residual observations for profiling only.
# This is NOT the final research configuration.
NUM_SAMPLES = 5000

# Representative unit targets from the previous real-data test.
TARGETS = [
    185,
    822,
    922,
    429,
    608,
    902,
    121,
    345,
    526,
    493,
]


# ============================================================
# HELPERS
# ============================================================

def select_unit(units, target):
    for unit in units:
        if unit.target == target:
            return unit

    raise ValueError(f"Target {target} not found in causal units.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOCAL PEM RUNTIME BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nCHECKING INPUT FILES")

    if not os.path.exists(RESIDUAL_PATH):
        raise FileNotFoundError(RESIDUAL_PATH)

    if not os.path.exists(ADJACENCY_PATH):
        raise FileNotFoundError(ADJACENCY_PATH)

    print(f"  FOUND: {RESIDUAL_PATH}")
    print(f"  FOUND: {ADJACENCY_PATH}")

    # --------------------------------------------------------
    # Load residuals
    # --------------------------------------------------------

    print("\nLOADING RESIDUALS")

    residuals = np.load(RESIDUAL_PATH)

    print(f"  Shape: {residuals.shape}")
    print(f"  Dtype: {residuals.dtype}")

    if residuals.ndim != 2:
        raise ValueError("Residuals must be 2-dimensional.")

    if residuals.shape[1] != NUM_NODES:
        raise ValueError(
            f"Expected {NUM_NODES} nodes, "
            f"got {residuals.shape[1]}."
        )

    # Use only the first NUM_SAMPLES observations.
    residuals = residuals[:NUM_SAMPLES]

    print(f"  Benchmark samples: {residuals.shape[0]}")

    # --------------------------------------------------------
    # Load adjacency
    # --------------------------------------------------------

    print("\nLOADING ADJACENCY")

    adjacency = np.load(ADJACENCY_PATH, mmap_mode="r")

    adjacency = np.asarray(
        adjacency[:NUM_NODES, :NUM_NODES],
        dtype=np.float32,
    )

    print(f"  Full adjacency shape : {np.load(ADJACENCY_PATH, mmap_mode='r').shape}")
    print(f"  Benchmark adjacency  : {adjacency.shape}")

    if adjacency.shape != (NUM_NODES, NUM_NODES):
        raise ValueError(
            f"Expected adjacency shape "
            f"({NUM_NODES}, {NUM_NODES}), "
            f"got {adjacency.shape}."
        )

    # --------------------------------------------------------
    # Build causal units
    # --------------------------------------------------------

    print("\nBUILDING CAUSAL UNITS")

    builder = CausalUnitBuilder()
    units = builder.build(adjacency)

    print(f"  Total units: {len(units)}")

    # --------------------------------------------------------
    # Select representative units
    # --------------------------------------------------------

    selected_units = []

    print("\nSELECTED UNITS")
    print("-" * 70)
    print(
        f"{'Test':>4} "
        f"{'Target':>8} "
        f"{'Size':>8} "
        f"{'Complete':>12}"
    )
    print("-" * 70)

    for i, target in enumerate(TARGETS, start=1):

        unit = select_unit(units, target)

        unit_data = residuals[:, unit.nodes]

        complete_mask = np.all(np.isfinite(unit_data), axis=1)
        complete_count = int(np.sum(complete_mask))

        selected_units.append(unit)

        print(
            f"{i:>4} "
            f"{unit.target:>8} "
            f"{len(unit.nodes):>8} "
            f"{complete_count:>12}"
        )

    # --------------------------------------------------------
    # Benchmark each unit separately
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING SINGLE-UNIT BENCHMARK")
    print("=" * 70)

    results = []

    for index, unit in enumerate(selected_units, start=1):

        unit_data = residuals[:, unit.nodes]

        complete_mask = np.all(np.isfinite(unit_data), axis=1)
        complete_count = int(np.sum(complete_mask))

        print("\n" + "-" * 70)
        print(
            f"TEST {index}/{len(selected_units)}"
        )
        print(f"Target node       : {unit.target}")
        print(f"Unit size         : {len(unit.nodes)}")
        print(f"Complete samples  : {complete_count}")
        print("-" * 70)

        # IMPORTANT:
        # LocalPEM.fit() expects the GLOBAL residual matrix.
        # Therefore we create a LocalPEM containing only this unit
        # and still pass the residual matrix.
        pem = LocalPEM([unit])

        start_time = time.perf_counter()

        try:
            result = pem.fit(residuals)

            elapsed = time.perf_counter() - start_time

            local_prob = result.unit_inclusion_probabilities[0]
            local_adj = result.unit_adjacencies[0]

            num_edges = int(np.count_nonzero(local_adj))

            print(f"STATUS            : SUCCESS")
            print(f"Runtime           : {elapsed:.3f} seconds")
            print(f"Iterations/result : completed")
            print(f"Edges             : {num_edges}")

            results.append(
                (
                    unit.target,
                    len(unit.nodes),
                    complete_count,
                    elapsed,
                    "SUCCESS",
                )
            )

        except Exception as exc:

            elapsed = time.perf_counter() - start_time

            print(f"STATUS            : FAILED")
            print(f"Runtime           : {elapsed:.3f} seconds")
            print(f"Error             : {type(exc).__name__}: {exc}")

            results.append(
                (
                    unit.target,
                    len(unit.nodes),
                    complete_count,
                    elapsed,
                    "FAILED",
                )
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"{'Target':>8} "
        f"{'Size':>8} "
        f"{'Samples':>10} "
        f"{'Runtime(s)':>12} "
        f"{'Status':>10}"
    )

    print("-" * 70)

    for target, size, samples, runtime, status in results:
        print(
            f"{target:>8} "
            f"{size:>8} "
            f"{samples:>10} "
            f"{runtime:>12.3f} "
            f"{status:>10}"
        )

    print("\nBENCHMARK COMPLETE")


if __name__ == "__main__":
    main()