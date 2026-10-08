import numpy as np
import h5py

from causal.var import TopologyAwareVAR


# ============================================================
# GBA VAR INTEGRATION TEST
#
# Uses:
#   - actual GBA 2021 traffic data
#   - actual GBA road-network adjacency
#
# Only a small subset is used initially.
# ============================================================


TRAFFIC_FILE = (
    r"C:\My Folder\Projects\Datasets\LargeSTGBA"
    r"\gba_his_raw_2021.h5"
)

ADJACENCY_FILE = (
    r"C:\My Folder\Projects\Datasets\LargeSTGBA"
    r"\gba_rn_adj.npy"
)


# ============================================================
# Test configuration
# ============================================================

NUM_TIMESTEPS = 1000
NUM_NODES = 100

LAG_ORDER = 1
RIDGE_ALPHA = 1e-5


# ============================================================
# Load actual GBA adjacency
# ============================================================

print("=" * 70)
print("GBA TOPOLOGY-AWARE VAR TEST")
print("=" * 70)

print()
print("Loading adjacency...")

adjacency_full = np.load(
    ADJACENCY_FILE,
    mmap_mode="r",
)

print(
    "Full adjacency shape:",
    adjacency_full.shape,
)


# Use the first NUM_NODES sensors.
#
# IMPORTANT:
# This is only a computational test.
# We are not modifying the actual GBA adjacency file.
adjacency = np.asarray(
    adjacency_full[
        :NUM_NODES,
        :NUM_NODES,
    ],
    dtype=np.float32,
)


# ============================================================
# Load actual GBA traffic
# ============================================================

print()
print("Loading traffic...")

with h5py.File(
    TRAFFIC_FILE,
    "r",
) as h5_file:

    traffic_dataset = h5_file["traffic"]

    print(
        "Full traffic shape:",
        traffic_dataset.shape,
    )

    data = np.asarray(
        traffic_dataset[
            :NUM_TIMESTEPS,
            :NUM_NODES,
        ],
        dtype=np.float32,
    )


# ============================================================
# Dataset information
# ============================================================

print()
print("-" * 70)
print("DATA")
print("-" * 70)

print(
    "Test traffic shape:",
    data.shape,
)

print(
    "Test adjacency shape:",
    adjacency.shape,
)

print(
    "NaN values:",
    np.isnan(data).sum(),
)

print(
    "Finite values:",
    np.isfinite(data).sum(),
)

print(
    "Missing percentage:",
    100.0
    * np.isnan(data).sum()
    / data.size,
)


# ============================================================
# Adjacency information
# ============================================================

print()
print("-" * 70)
print("ADJACENCY")
print("-" * 70)

print(
    "Non-zero entries:",
    np.count_nonzero(adjacency),
)

print(
    "Density:",
    np.count_nonzero(adjacency)
    / adjacency.size,
)

print(
    "Self-loops:",
    np.count_nonzero(
        np.diag(adjacency)
    ),
)

print(
    "Exactly symmetric:",
    np.array_equal(
        adjacency,
        adjacency.T,
    ),
)


# ============================================================
# Run topology-aware VAR
# ============================================================

print()
print("-" * 70)
print("RUNNING TOPOLOGY-AWARE VAR")
print("-" * 70)

model = TopologyAwareVAR(
    lag_order=LAG_ORDER,
    ridge_alpha=RIDGE_ALPHA,
    use_gpu=True,
)


result = model.fit(
    data,
    adjacency,
)


# ============================================================
# Inspect results
# ============================================================

print()
print("-" * 70)
print("VAR RESULT")
print("-" * 70)

print(
    "Coefficient shape:",
    result.coefficients.shape,
)

print(
    "Residual shape:",
    result.residuals.shape,
)

print(
    "Valid-mask shape:",
    result.valid_mask.shape,
)

print(
    "Lag order:",
    result.lag_order,
)


# ============================================================
# Residual statistics
# ============================================================

valid_residuals = result.residuals[
    result.valid_mask
]

print()
print("Valid residual values:")
print(
    valid_residuals.size
)

print(
    "Residual NaN count:",
    np.isnan(result.residuals).sum(),
)

print(
    "Residual mean:",
    np.mean(valid_residuals),
)

print(
    "Residual std:",
    np.std(valid_residuals),
)


# ============================================================
# Per-node residual coverage
# ============================================================

valid_counts = np.sum(
    result.valid_mask,
    axis=0,
)

print()
print("-" * 70)
print("RESIDUAL COVERAGE")
print("-" * 70)

print(
    "Minimum valid residuals per node:",
    valid_counts.min(),
)

print(
    "Maximum valid residuals per node:",
    valid_counts.max(),
)

print(
    "Mean valid residuals per node:",
    valid_counts.mean(),
)

print(
    "Minimum coverage (%):",
    100.0
    * valid_counts.min()
    / NUM_TIMESTEPS,
)

print(
    "Maximum coverage (%):",
    100.0
    * valid_counts.max()
    / NUM_TIMESTEPS,
)


# ============================================================
# Coefficient statistics
# ============================================================

print()
print("-" * 70)
print("COEFFICIENTS")
print("-" * 70)

nonzero_coefficients = result.coefficients[
    result.coefficients != 0
]

print(
    "Non-zero coefficients:",
    nonzero_coefficients.size,
)

if nonzero_coefficients.size > 0:

    print(
        "Minimum:",
        nonzero_coefficients.min(),
    )

    print(
        "Maximum:",
        nonzero_coefficients.max(),
    )

    print(
        "Mean:",
        nonzero_coefficients.mean(),
    )


# ============================================================
# Basic correctness checks
# ============================================================

assert result.coefficients.shape == (
    LAG_ORDER,
    NUM_NODES,
    NUM_NODES,
), (
    "Incorrect coefficient shape."
)


assert result.residuals.shape == (
    NUM_TIMESTEPS,
    NUM_NODES,
), (
    "Incorrect residual shape."
)


assert result.valid_mask.shape == (
    NUM_TIMESTEPS,
    NUM_NODES,
), (
    "Incorrect valid-mask shape."
)


assert np.all(
    np.isfinite(
        result.coefficients
    )
), (
    "VAR coefficients contain "
    "non-finite values."
)


assert np.all(
    result.valid_mask[
        result.valid_mask
    ]
), (
    "Valid mask contains an invalid state."
)


# Every valid residual must be finite.
assert np.all(
    np.isfinite(
        result.residuals[
            result.valid_mask
        ]
    )
), (
    "Valid residual entries must be finite."
)


# Invalid residual entries should remain NaN.
assert np.all(
    np.isnan(
        result.residuals[
            ~result.valid_mask
        ]
    )
), (
    "Invalid residual entries must remain NaN."
)


print()
print("=" * 70)
print("GBA TOPOLOGY-AWARE VAR TEST PASSED")
print("=" * 70)