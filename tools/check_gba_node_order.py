from pathlib import Path
import numpy as np
import pandas as pd
import h5py


BASE_DIR = Path(
    r"C:\My Folder\Projects\Horizon-Conditioned Probabilistic "
    r"VAR-LiNGAM Causal Graph for Traffic Forecasting\implementation"
)

DATASET_DIR = BASE_DIR / "datasets" / "LargeST GBA"

TRAFFIC_PATH = DATASET_DIR / "gba_his_raw_2021.h5"
ADJ_PATH = DATASET_DIR / "gba_rn_adj.npy"
META_PATH = DATASET_DIR / "gba_meta.csv"


print("=" * 90)
print("FINAL GBA NODE-ORDER VERIFICATION")
print("=" * 90)

# ------------------------------------------------------------------
# 1. Load HDF5 sensor order
# ------------------------------------------------------------------

with h5py.File(TRAFFIC_PATH, "r") as f:
    hdf5_sensor_ids = np.asarray(f["sensor_ids"][:])
    hdf5_sensor_id2 = np.asarray(f["sensor_id2"][:])

print("\n1. HDF5 NODE ORDER")
print("-" * 90)

print("Number of sensors:", len(hdf5_sensor_ids))
print("First 20 sensor IDs:")
print(hdf5_sensor_ids[:20])

print("\nFirst 20 ID2 values:")
print(hdf5_sensor_id2[:20])


# ------------------------------------------------------------------
# 2. Load metadata
# ------------------------------------------------------------------

meta = pd.read_csv(META_PATH)

metadata_ids = meta["ID"].to_numpy()
metadata_id2 = meta["ID2"].to_numpy()

print("\n2. METADATA NODE ORDER")
print("-" * 90)

print("Number of metadata sensors:", len(metadata_ids))

print("First 20 metadata IDs:")
print(metadata_ids[:20])

print("\nFirst 20 metadata ID2 values:")
print(metadata_id2[:20])


# ------------------------------------------------------------------
# 3. Verify HDF5 ↔ metadata
# ------------------------------------------------------------------

print("\n3. HDF5 ↔ METADATA ORDER")
print("-" * 90)

same_id_order = np.array_equal(
    hdf5_sensor_ids,
    metadata_ids
)

same_id2_order = np.array_equal(
    hdf5_sensor_id2,
    metadata_id2
)

print("sensor_ids == metadata ID :", same_id_order)
print("sensor_id2 == metadata ID2:", same_id2_order)


# ------------------------------------------------------------------
# 4. Examine adjacency
# ------------------------------------------------------------------

adj = np.load(ADJ_PATH)

print("\n4. ADJACENCY")
print("-" * 90)

print("Shape:", adj.shape)
print("dtype:", adj.dtype)
print("Non-zero:", np.count_nonzero(adj))
print("Diagonal non-zero:", np.count_nonzero(np.diag(adj)))


# ------------------------------------------------------------------
# 5. IMPORTANT:
#    Check whether adjacency has an associated node-order file
# ------------------------------------------------------------------

print("\n5. SEARCHING DATASET DIRECTORY FOR NODE-ORDER FILES")
print("-" * 90)

candidate_files = []

for pattern in [
    "*.txt",
    "*.csv",
    "*.json",
    "*.npy",
    "*.npz",
    "*.pkl",
]:
    for p in DATASET_DIR.glob(pattern):
        candidate_files.append(p)

for p in sorted(candidate_files):
    print(" -", p.name)


# ------------------------------------------------------------------
# 6. Compare adjacency-derived structural signatures
#
# If node ordering is correct, each adjacency row corresponds to
# the topology of the sensor at the same metadata index.
#
# We print degree signatures together with sensor IDs so that
# the ordering can be inspected and compared against any official
# node-order source.
# ------------------------------------------------------------------

degree = np.count_nonzero(adj, axis=1)

print("\n6. ADJACENCY NODE SIGNATURES")
print("-" * 90)

print(
    f"{'INDEX':>8} "
    f"{'SENSOR_ID':>12} "
    f"{'ID2':>8} "
    f"{'DEGREE':>8}"
)

print("-" * 45)

for i in range(min(30, len(hdf5_sensor_ids))):
    print(
        f"{i:8d} "
        f"{hdf5_sensor_ids[i]:12d} "
        f"{hdf5_sensor_id2[i]:8d} "
        f"{degree[i]:8d}"
    )


# ------------------------------------------------------------------
# 7. Check adjacency rows/columns for isolated nodes
# ------------------------------------------------------------------

row_degree = np.count_nonzero(adj, axis=1)
col_degree = np.count_nonzero(adj, axis=0)

print("\n7. NODE DEGREE CHECK")
print("-" * 90)

print("Minimum row degree:", row_degree.min())
print("Maximum row degree:", row_degree.max())
print("Mean row degree:", row_degree.mean())

print("Minimum column degree:", col_degree.min())
print("Maximum column degree:", col_degree.max())
print("Mean column degree:", col_degree.mean())

print("Isolated rows:", np.sum(row_degree == 0))
print("Isolated columns:", np.sum(col_degree == 0))


# ------------------------------------------------------------------
# 8. Final interpretation
# ------------------------------------------------------------------

print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)

if same_id_order and same_id2_order:
    print(
        """
HDF5 sensor order and metadata sensor order are confirmed identical.

However, the adjacency matrix itself does not contain sensor IDs.
Therefore, adjacency-index ↔ sensor-ID ordering can only be declared
verified if the dataset construction/source provides the corresponding
node-order mapping.

DO NOT reorder, symmetrize, or modify the adjacency matrix.
"""
    )
else:
    print(
        """
WARNING:
HDF5 and metadata node ordering do not match.
Do not use the adjacency directly until this is resolved.
"""
    )