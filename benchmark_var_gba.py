from __future__ import annotations

import time

import h5py
import numpy as np

from causal.var import TopologyAwareVAR


# ============================================================
# CONFIGURATION
# ============================================================

TRAFFIC_FILE = (
    r"C:\My Folder\Projects\Datasets\LargeSTGBA"
    r"\gba_his_raw_2021.h5"
)

ADJACENCY_FILE = (
    r"C:\My Folder\Projects\Datasets\LargeSTGBA"
    r"\gba_rn_adj.npy"
)

NUM_TIMESTEPS = 5000
NUM_NODES = 500

LAG_ORDER = 1
RIDGE_ALPHA = 1e-5


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GBA TOPOLOGY-AWARE VAR SCALE BENCHMARK")
print("=" * 70)

print(
    f"\nBenchmark configuration:"
    f"\n  Timesteps : {NUM_TIMESTEPS}"
    f"\n  Sensors   : {NUM_NODES}"
    f"\n  Lag order : {LAG_ORDER}"
    f"\n  Ridge α   : {RIDGE_ALPHA}"
)


# ============================================================
# LOAD ADJACENCY
# ============================================================

print("\n" + "-" * 70)
print("LOADING ADJACENCY")
print("-" * 70)

adjacency = np.load(ADJACENCY_FILE)

print("Full adjacency shape:", adjacency.shape)

adjacency = np.asarray(
    adjacency[:NUM_NODES, :NUM_NODES],
    dtype=np.float32,
)

print("Benchmark adjacency shape:", adjacency.shape)

nonzero = np.count_nonzero(adjacency)
total = adjacency.size

print("Non-zero entries:", nonzero)
print("Density:", nonzero / total)
print("Self-loops:", np.count_nonzero(np.diag(adjacency)))


# ============================================================
# LOAD TRAFFIC
# ============================================================

print("\n" + "-" * 70)
print("LOADING TRAFFIC")
print("-" * 70)

with h5py.File(TRAFFIC_FILE, "r") as f:
    traffic = np.asarray(
        f["traffic"][:NUM_TIMESTEPS, :NUM_NODES],
        dtype=np.float32,
    )

print("Traffic shape:", traffic.shape)

nan_count = np.count_nonzero(~np.isfinite(traffic))
total_values = traffic.size

print("Missing/non-finite values:", nan_count)
print(
    "Missing percentage:",
    f"{100.0 * nan_count / total_values:.6f}%"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if traffic.shape != (NUM_TIMESTEPS, NUM_NODES):
    raise RuntimeError(
        "Unexpected traffic shape: "
        f"{traffic.shape}"
    )

if adjacency.shape != (NUM_NODES, NUM_NODES):
    raise RuntimeError(
        "Unexpected adjacency shape: "
        f"{adjacency.shape}"
    )


# ============================================================
# RUN VAR
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TOPOLOGY-AWARE VAR")
print("=" * 70)

model = TopologyAwareVAR(
    lag_order=LAG_ORDER,
    ridge_alpha=RIDGE_ALPHA,
    use_gpu=True,
)

start_time = time.perf_counter()

result = model.fit(
    data=traffic,
    adjacency=adjacency,
)

elapsed = time.perf_counter() - start_time


# ============================================================
# RESULT
# ============================================================

print("\n" + "-" * 70)
print("VAR RESULT")
print("-" * 70)

print("Runtime:", f"{elapsed:.3f} seconds")
print("Runtime:", f"{elapsed / 60:.3f} minutes")

print("Coefficient shape:", result.coefficients.shape)
print("Residual shape:", result.residuals.shape)
print("Valid-mask shape:", result.valid_mask.shape)
print("Lag order:", result.lag_order)


# ============================================================
# RESIDUAL STATISTICS
# ============================================================

print("\n" + "-" * 70)
print("RESIDUAL COVERAGE")
print("-" * 70)

valid_counts = np.sum(
    result.valid_mask,
    axis=0,
)

coverage = (
    100.0
    * valid_counts
    / NUM_TIMESTEPS
)

print(
    "Minimum valid residuals per node:",
    int(valid_counts.min()),
)

print(
    "Maximum valid residuals per node:",
    int(valid_counts.max()),
)

print(
    "Mean valid residuals per node:",
    float(valid_counts.mean()),
)

print(
    "Minimum coverage (%):",
    float(coverage.min()),
)

print(
    "Maximum coverage (%):",
    float(coverage.max()),
)

print(
    "Mean coverage (%):",
    float(coverage.mean()),
)


# ============================================================
# RESIDUAL VALUE STATISTICS
# ============================================================

finite_residuals = result.residuals[
    np.isfinite(result.residuals)
]

print("\n" + "-" * 70)
print("RESIDUAL VALUES")
print("-" * 70)

print(
    "Finite residual values:",
    len(finite_residuals),
)

print(
    "NaN residual values:",
    np.count_nonzero(
        ~np.isfinite(result.residuals)
    ),
)

print(
    "Residual mean:",
    float(np.mean(finite_residuals)),
)

print(
    "Residual std:",
    float(np.std(finite_residuals)),
)

print(
    "Residual minimum:",
    float(np.min(finite_residuals)),
)

print(
    "Residual maximum:",
    float(np.max(finite_residuals)),
)


# ============================================================
# COEFFICIENT STATISTICS
# ============================================================

coefficients = result.coefficients

nonzero_coefficients = coefficients[
    coefficients != 0
]

print("\n" + "-" * 70)
print("COEFFICIENTS")
print("-" * 70)

print(
    "Non-zero coefficients:",
    len(nonzero_coefficients),
)

if len(nonzero_coefficients) > 0:

    print(
        "Minimum:",
        float(np.min(nonzero_coefficients)),
    )

    print(
        "Maximum:",
        float(np.max(nonzero_coefficients)),
    )

    print(
        "Mean:",
        float(np.mean(nonzero_coefficients)),
    )

    print(
        "Std:",
        float(np.std(nonzero_coefficients)),
    )


# ============================================================
# TOPOLOGY CONSISTENCY CHECK
# ============================================================

expected_nonzero = np.count_nonzero(
    adjacency
)

actual_nonzero = np.count_nonzero(
    coefficients
)

print("\n" + "-" * 70)
print("TOPOLOGY CONSISTENCY")
print("-" * 70)

print(
    "Non-zero adjacency entries:",
    expected_nonzero,
)

print(
    "Non-zero VAR coefficients:",
    actual_nonzero,
)

if actual_nonzero != expected_nonzero:
    raise RuntimeError(
        "VAR coefficient sparsity does not match "
        "the topology restriction."
    )

print(
    "Topology restriction preserved: YES"
)


# ============================================================
# FINAL CHECKS
# ============================================================

assert result.coefficients.shape == (
    LAG_ORDER,
    NUM_NODES,
    NUM_NODES,
)

assert result.residuals.shape == (
    NUM_TIMESTEPS,
    NUM_NODES,
)

assert result.valid_mask.shape == (
    NUM_TIMESTEPS,
    NUM_NODES,
)

assert np.all(
    np.isfinite(result.coefficients)
)

assert np.all(
    np.isfinite(
        result.residuals[
            result.valid_mask
        ]
    )
)

print("\n" + "=" * 70)
print("GBA VAR SCALE BENCHMARK PASSED")
print("=" * 70)