"""
Diagnostic analysis of topology-aware causal units for LargeST-GBA.
"""

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
# Load adjacency
# ============================================================

print("=" * 70)
print("GBA CAUSAL UNIT SIZE ANALYSIS")
print("=" * 70)

print(f"\nLoading adjacency:")
print(ADJACENCY_PATH)

adjacency = np.load(ADJACENCY_PATH)

print(f"\nAdjacency shape : {adjacency.shape}")
print(f"Adjacency dtype : {adjacency.dtype}")
print(f"Finite          : {np.all(np.isfinite(adjacency))}")
print(f"Nonzero         : {np.count_nonzero(adjacency)}")


# ============================================================
# Build causal units
# ============================================================

builder = CausalUnitBuilder(
    include_self=True,
    max_unit_size=None,
    remove_self_loops=True,
)

units = builder.build(adjacency)


# ============================================================
# Summary
# ============================================================

summary = builder.summary(units)

print("\n" + "-" * 70)
print("UNIT SUMMARY")
print("-" * 70)

for key, value in summary.items():
    print(f"{key:30s}: {value}")


# ============================================================
# Unit-size statistics
# ============================================================

unit_sizes = np.array(
    [len(unit.nodes) for unit in units],
    dtype=np.int64,
)

neighbor_counts = np.array(
    [len(unit.neighbors) for unit in units],
    dtype=np.int64,
)

incoming_counts = np.array(
    [len(unit.incoming_neighbors) for unit in units],
    dtype=np.int64,
)

outgoing_counts = np.array(
    [len(unit.outgoing_neighbors) for unit in units],
    dtype=np.int64,
)


print("\n" + "-" * 70)
print("UNIT SIZE DISTRIBUTION")
print("-" * 70)

percentiles = [50, 75, 90, 95, 99, 99.5, 100]

for p in percentiles:
    print(
        f"P{p:<5}: "
        f"{np.percentile(unit_sizes, p):.2f}"
    )


# ============================================================
# Threshold counts
# ============================================================

print("\n" + "-" * 70)
print("UNITS ABOVE SIZE THRESHOLDS")
print("-" * 70)

thresholds = [
    15,
    20,
    25,
    30,
    35,
    40,
    50,
    60,
    80,
    100,
]

for threshold in thresholds:
    count = np.sum(unit_sizes > threshold)
    percentage = 100.0 * count / len(unit_sizes)

    print(
        f"> {threshold:3d}: "
        f"{count:4d} units "
        f"({percentage:7.3f}%)"
    )


# ============================================================
# Largest units
# ============================================================

print("\n" + "-" * 70)
print("10 LARGEST CAUSAL UNITS")
print("-" * 70)

largest_indices = np.argsort(
    unit_sizes
)[-10:][::-1]

for rank, idx in enumerate(largest_indices, start=1):

    unit = units[idx]

    print(
        f"{rank:2d}. "
        f"target={unit.target:4d}, "
        f"size={len(unit.nodes):3d}, "
        f"incoming={len(unit.incoming_neighbors):3d}, "
        f"outgoing={len(unit.outgoing_neighbors):3d}, "
        f"union_neighbors={len(unit.neighbors):3d}"
    )


# ============================================================
# Compare incoming/outgoing/union sizes
# ============================================================

print("\n" + "-" * 70)
print("NEIGHBORHOOD STATISTICS")
print("-" * 70)

for name, values in [
    ("Incoming", incoming_counts),
    ("Outgoing", outgoing_counts),
    ("Union", neighbor_counts),
]:

    print(f"\n{name}")

    print(f"  min    : {np.min(values)}")
    print(f"  mean   : {np.mean(values):.3f}")
    print(f"  median : {np.median(values):.3f}")
    print(f"  P90    : {np.percentile(values, 90):.3f}")
    print(f"  P95    : {np.percentile(values, 95):.3f}")
    print(f"  P99    : {np.percentile(values, 99):.3f}")
    print(f"  max    : {np.max(values)}")


# ============================================================
# Sanity checks
# ============================================================

print("\n" + "-" * 70)
print("SANITY CHECKS")
print("-" * 70)

# Every target should occur exactly once in its own unit.
target_in_unit = all(
    unit.target in unit.nodes
    for unit in units
)

# No duplicate nodes inside a unit.
no_duplicates = all(
    len(unit.nodes) == len(np.unique(unit.nodes))
    for unit in units
)

# Target should not appear in neighbor arrays.
target_not_neighbor = all(
    unit.target not in unit.neighbors
    and unit.target not in unit.incoming_neighbors
    and unit.target not in unit.outgoing_neighbors
    for unit in units
)

# Nodes must be valid global indices.
valid_indices = all(
    np.all(
        (unit.nodes >= 0)
        & (unit.nodes < adjacency.shape[0])
    )
    for unit in units
)

print(f"Target included        : {target_in_unit}")
print(f"No duplicate nodes      : {no_duplicates}")
print(f"No self-loop neighbors  : {target_not_neighbor}")
print(f"Valid global indices    : {valid_indices}")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)