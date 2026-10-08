from pathlib import Path

import numpy as np

from causal.causal_units import CausalUnitBuilder


# ============================================================
# Configuration
# ============================================================

ADJACENCY_PATH = Path(
    "datasets",
    "LargeST GBA",
    "gba_rn_adj.npy",
)


# ============================================================
# Load adjacency and build units
# ============================================================

print("=" * 70)
print("GBA CAUSAL UNIT OVERLAP ANALYSIS")
print("=" * 70)

adjacency = np.load(ADJACENCY_PATH)

builder = CausalUnitBuilder(
    include_self=True,
    max_unit_size=None,
    remove_self_loops=True,
)

units = builder.build(adjacency)

print(f"\nNumber of units: {len(units)}")


# ============================================================
# Convert units to sets
# ============================================================

unit_sets = [
    set(unit.nodes.tolist())
    for unit in units
]


# ============================================================
# Basic overlap statistics
# ============================================================

# Every pair of units would require ~2.7 million comparisons.
# We therefore sample pairs for the general overlap statistics.

rng = np.random.default_rng(42)

num_units = len(unit_sets)

num_random_pairs = min(
    100_000,
    num_units * (num_units - 1) // 2,
)

pair_i = rng.integers(
    0,
    num_units,
    size=num_random_pairs,
)

pair_j = rng.integers(
    0,
    num_units,
    size=num_random_pairs,
)

# Remove self-pairs.
mask = pair_i != pair_j

pair_i = pair_i[mask]
pair_j = pair_j[mask]


jaccard_values = []
intersection_sizes = []


for i, j in zip(pair_i, pair_j):

    a = unit_sets[i]
    b = unit_sets[j]

    intersection = len(a & b)
    union = len(a | b)

    intersection_sizes.append(intersection)
    jaccard_values.append(
        intersection / union
    )


jaccard_values = np.asarray(
    jaccard_values,
    dtype=np.float64,
)

intersection_sizes = np.asarray(
    intersection_sizes,
    dtype=np.int64,
)


# ============================================================
# Statistics
# ============================================================

print("\n" + "-" * 70)
print("RANDOM UNIT-PAIR OVERLAP")
print("-" * 70)

print(f"Pairs sampled : {len(jaccard_values)}")

for p in [0, 25, 50, 75, 90, 95, 99, 100]:

    print(
        f"Jaccard P{p:<3}: "
        f"{np.percentile(jaccard_values, p):.4f}"
    )

print(
    f"\nMean Jaccard      : "
    f"{np.mean(jaccard_values):.4f}"
)

print(
    f"Mean intersection : "
    f"{np.mean(intersection_sizes):.2f}"
)

for threshold in [0, 0.1, 0.25, 0.5, 0.75]:

    count = np.sum(
        jaccard_values >= threshold
    )

    percentage = (
        100.0 * count / len(jaccard_values)
    )

    print(
        f"Jaccard >= {threshold:.2f}: "
        f"{count:6d} "
        f"({percentage:7.3f}%)"
    )


# ============================================================
# Topological neighboring-unit overlap
# ============================================================
#
# For each target i, compare U_i with U_j for each topology
# neighbor j.
#
# This is more meaningful than random unit pairs because these
# units are spatially close in the road network.
# ============================================================

neighbor_jaccard = []
neighbor_intersection = []

seen_pairs = set()

for i, unit in enumerate(units):

    candidate_neighbors = unit.neighbors

    for j in candidate_neighbors:

        j = int(j)

        # Avoid evaluating the same pair twice.
        pair = (
            min(i, j),
            max(i, j),
        )

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)

        a = unit_sets[i]
        b = unit_sets[j]

        intersection = len(a & b)
        union = len(a | b)

        neighbor_intersection.append(
            intersection
        )

        neighbor_jaccard.append(
            intersection / union
        )


neighbor_jaccard = np.asarray(
    neighbor_jaccard,
    dtype=np.float64,
)

neighbor_intersection = np.asarray(
    neighbor_intersection,
    dtype=np.int64,
)


print("\n" + "-" * 70)
print("TOPOLOGICALLY NEIGHBORING UNIT OVERLAP")
print("-" * 70)

print(
    f"Unique neighboring unit pairs : "
    f"{len(neighbor_jaccard)}"
)

if len(neighbor_jaccard) > 0:

    for p in [0, 25, 50, 75, 90, 95, 99, 100]:

        print(
            f"Jaccard P{p:<3}: "
            f"{np.percentile(neighbor_jaccard, p):.4f}"
        )

    print(
        f"\nMean Jaccard      : "
        f"{np.mean(neighbor_jaccard):.4f}"
    )

    print(
        f"Mean intersection : "
        f"{np.mean(neighbor_intersection):.2f}"
    )

    for threshold in [0.1, 0.25, 0.5, 0.75]:

        count = np.sum(
            neighbor_jaccard >= threshold
        )

        percentage = (
            100.0 * count / len(neighbor_jaccard)
        )

        print(
            f"Jaccard >= {threshold:.2f}: "
            f"{count:6d} "
            f"({percentage:7.3f}%)"
        )


# ============================================================
# Largest-unit overlap
# ============================================================

unit_sizes = np.array(
    [len(unit.nodes) for unit in units],
    dtype=np.int64,
)

largest_indices = np.argsort(
    unit_sizes
)[-10:][::-1]


print("\n" + "-" * 70)
print("OVERLAP OF THE 10 LARGEST UNITS")
print("-" * 70)

for rank, i in enumerate(
    largest_indices,
    start=1,
):

    unit = units[i]

    # Compare against topology neighbors only.
    values = []

    for j in unit.neighbors:

        j = int(j)

        a = unit_sets[i]
        b = unit_sets[j]

        intersection = len(a & b)
        union = len(a | b)

        values.append(
            intersection / union
        )

    if values:

        print(
            f"target={unit.target:4d}, "
            f"size={len(unit.nodes):3d}, "
            f"neighbor-unit mean Jaccard="
            f"{np.mean(values):.4f}, "
            f"max="
            f"{np.max(values):.4f}"
        )

    else:

        print(
            f"target={unit.target:4d}, "
            f"size={len(unit.nodes):3d}, "
            f"no topology neighbors"
        )


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("OVERLAP ANALYSIS COMPLETE")
print("=" * 70)