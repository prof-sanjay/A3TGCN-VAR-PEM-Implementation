import numpy as np
import h5py

from causal.causal_units import CausalUnitBuilder


TRAFFIC_PATH = (
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_his_raw_2021.h5"
)

ADJACENCY_PATH = (
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_rn_adj.npy"
)


# ------------------------------------------------------------
# Load adjacency
# ------------------------------------------------------------

adjacency = np.load(ADJACENCY_PATH)

builder = CausalUnitBuilder()

units = builder.build(adjacency)


# ------------------------------------------------------------
# Load traffic
#
# The HDF5 file is opened read-only.
# Nothing is modified.
# ------------------------------------------------------------

with h5py.File(TRAFFIC_PATH, "r") as h5_file:

    print("=" * 70)
    print("GBA CAUSAL-UNIT DATA AVAILABILITY")
    print("=" * 70)

    print("HDF5 keys:")
    print(list(h5_file.keys()))

    traffic = h5_file["traffic"][:]


print()
print(f"Traffic shape: {traffic.shape}")


# ------------------------------------------------------------
# Check overall missingness
# ------------------------------------------------------------

finite = np.isfinite(traffic)

total_values = traffic.size
valid_values = np.count_nonzero(finite)
missing_values = total_values - valid_values

print()
print("-" * 70)
print("OVERALL DATA AVAILABILITY")
print("-" * 70)

print(f"Total values:   {total_values}")
print(f"Valid values:   {valid_values}")
print(f"Missing values: {missing_values}")

print(
    f"Missing percentage: "
    f"{100.0 * missing_values / total_values:.4f}%"
)


# ------------------------------------------------------------
# Calculate complete observations for each causal unit
#
# A timestamp is usable for a causal unit only if every sensor
# in that unit has a finite observation at that timestamp.
#
# No values are filled or modified.
# ------------------------------------------------------------

usable_samples = np.zeros(
    len(units),
    dtype=np.int64,
)


unit_sizes = np.zeros(
    len(units),
    dtype=np.int64,
)


for index, unit in enumerate(units):

    node_indices = unit.nodes

    unit_data = traffic[:, node_indices]

    valid_rows = np.all(
        np.isfinite(unit_data),
        axis=1,
    )

    usable_samples[index] = np.count_nonzero(
        valid_rows
    )

    unit_sizes[index] = len(node_indices)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("-" * 70)
print("CAUSAL-UNIT SAMPLE AVAILABILITY")
print("-" * 70)

print(
    f"Minimum usable samples: "
    f"{usable_samples.min()}"
)

print(
    f"Maximum usable samples: "
    f"{usable_samples.max()}"
)

print(
    f"Mean usable samples: "
    f"{usable_samples.mean():.2f}"
)

print(
    f"Median usable samples: "
    f"{np.median(usable_samples):.0f}"
)

print(
    f"Total timestamps: "
    f"{traffic.shape[0]}"
)


# ------------------------------------------------------------
# Percentage of usable observations
# ------------------------------------------------------------

usable_percentage = (
    100.0
    * usable_samples
    / traffic.shape[0]
)

print()
print("-" * 70)
print("USABLE TIMESTAMP PERCENTAGE")
print("-" * 70)

print(
    f"Minimum: "
    f"{usable_percentage.min():.4f}%"
)

print(
    f"Maximum: "
    f"{usable_percentage.max():.4f}%"
)

print(
    f"Mean: "
    f"{usable_percentage.mean():.4f}%"
)

print(
    f"Median: "
    f"{np.median(usable_percentage):.4f}%"
)


# ------------------------------------------------------------
# Units with very low usable sample counts
# ------------------------------------------------------------

print()
print("-" * 70)
print("LOW-DATA CAUSAL UNITS")
print("-" * 70)

thresholds = [
    0,
    1000,
    5000,
    10000,
    25000,
    50000,
    75000,
]

for threshold in thresholds:

    count = np.count_nonzero(
        usable_samples < threshold
    )

    print(
        f"Units with fewer than "
        f"{threshold:,} usable samples: "
        f"{count}"
    )


# ------------------------------------------------------------
# Relationship between unit size and usable samples
# ------------------------------------------------------------

print()
print("-" * 70)
print("UNIT SIZE / DATA AVAILABILITY")
print("-" * 70)

for size in sorted(
    np.unique(unit_sizes)
):

    mask = unit_sizes == size

    print(
        f"Unit size {size:3d}: "
        f"count={np.sum(mask):4d}, "
        f"mean usable="
        f"{usable_samples[mask].mean():.2f}, "
        f"median usable="
        f"{np.median(usable_samples[mask]):.0f}"
    )


# ------------------------------------------------------------
# Example units
# ------------------------------------------------------------

print()
print("-" * 70)
print("EXAMPLE UNITS")
print("-" * 70)

example_indices = [
    0,
    1,
    2,
    3,
    4,
]

for index in example_indices:

    unit = units[index]

    print()
    print(f"Unit index: {index}")
    print(f"Target: {unit.target}")
    print(f"Unit size: {len(unit.nodes)}")
    print(f"Nodes: {unit.nodes}")
    print(
        f"Usable samples: "
        f"{usable_samples[index]}"
    )
    print(
        f"Usable percentage: "
        f"{usable_percentage[index]:.4f}%"
    )


print()
print("=" * 70)
print("DATA AVAILABILITY TEST COMPLETE")
print("=" * 70)