import numpy as np

from data.loader import load_gba_dataset


# ============================================================
# GBA DATASET PATHS
# ============================================================

TRAFFIC_PATH = r"C:\My Folder\Projects\Horizon-Conditioned Probabilistic VAR-LiNGAM Causal Graph for Traffic Forecasting\implementation\datasets\LargeST GBA\gba_his_raw_2021.h5"
ADJACENCY_PATH = r"C:\My Folder\Projects\Horizon-Conditioned Probabilistic VAR-LiNGAM Causal Graph for Traffic Forecasting\implementation\datasets\LargeST GBA\gba_rn_adj.npy"

# ============================================================
# E0 DEBUG / TEST CONFIGURATION
# ============================================================

NUM_TIMESTEPS = 20000
NUM_NODES = 1000


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LARGEST GBA DATASET LOADER TEST")
print("=" * 70)

print("\nTraffic file:")
print(TRAFFIC_PATH)

print("\nAdjacency file:")
print(ADJACENCY_PATH)

print("\nRequested subset:")
print(f"Timesteps : {NUM_TIMESTEPS}")
print(f"Nodes     : {NUM_NODES}")


traffic, adjacency = load_gba_dataset(
    traffic_path=TRAFFIC_PATH,
    adjacency_path=ADJACENCY_PATH,
    num_timesteps=NUM_TIMESTEPS,
    num_nodes=NUM_NODES,
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("LOADED DATA")
print("=" * 70)

print(f"Traffic shape    : {traffic.shape}")
print(f"Traffic dtype    : {traffic.dtype}")

print(f"Adjacency shape  : {adjacency.shape}")
print(f"Adjacency dtype  : {adjacency.dtype}")


# ============================================================
# TRAFFIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TRAFFIC VALIDATION")
print("=" * 70)

nan_count = np.isnan(traffic).sum()
inf_count = np.isinf(traffic).sum()

finite_count = np.isfinite(traffic).sum()
total_count = traffic.size

print(f"Total values     : {total_count}")
print(f"Finite values    : {finite_count}")
print(f"NaN values       : {nan_count}")
print(f"Inf values       : {inf_count}")

missing_percentage = (
    (nan_count + inf_count) / total_count
) * 100

print(f"Missing/invalid  : {missing_percentage:.6f}%")


finite_values = traffic[np.isfinite(traffic)]

if finite_values.size > 0:
    print(f"Minimum          : {finite_values.min():.4f}")
    print(f"Maximum          : {finite_values.max():.4f}")
    print(f"Mean             : {finite_values.mean():.4f}")
    print(f"Std              : {finite_values.std():.4f}")


# ============================================================
# ADJACENCY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ADJACENCY VALIDATION")
print("=" * 70)

print(f"Shape            : {adjacency.shape}")
print(f"Non-zero edges   : {np.count_nonzero(adjacency)}")
print(f"Density           : {np.count_nonzero(adjacency) / adjacency.size * 100:.6f}%")
print(f"Diagonal non-zero: {np.count_nonzero(np.diag(adjacency))}")
print(f"All finite       : {np.isfinite(adjacency).all()}")

print(f"Minimum          : {adjacency.min():.4f}")
print(f"Maximum          : {adjacency.max():.4f}")


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("CONSISTENCY CHECKS")
print("=" * 70)

traffic_nodes = traffic.shape[1]
adjacency_nodes = adjacency.shape[0]

print(
    f"Traffic nodes == adjacency nodes : "
    f"{traffic_nodes == adjacency_nodes}"
)

print(
    f"Requested timesteps correct       : "
    f"{traffic.shape[0] == NUM_TIMESTEPS}"
)

print(
    f"Requested nodes correct           : "
    f"{traffic.shape[1] == NUM_NODES}"
)

print(
    f"Adjacency square                  : "
    f"{adjacency.shape[0] == adjacency.shape[1]}"
)


# ============================================================
# SAMPLE VALUES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE VALUES")
print("=" * 70)

print("\nFirst 5 timesteps × first 5 sensors:")
print(traffic[:5, :5])

print("\nFirst 5 × 5 adjacency:")
print(adjacency[:5, :5])


# ============================================================
# FINAL STATUS
# ============================================================

all_checks_passed = (
    traffic.shape == (NUM_TIMESTEPS, NUM_NODES)
    and adjacency.shape == (NUM_NODES, NUM_NODES)
    and np.isfinite(adjacency).all()
)

print("\n" + "=" * 70)

if all_checks_passed:
    print("RESULT: GBA DATASET LOADER TEST PASSED")
else:
    print("RESULT: GBA DATASET LOADER TEST FAILED")

print("=" * 70)