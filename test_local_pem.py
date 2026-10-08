import numpy as np


from causal.causal_units import CausalUnitBuilder
from causal.local_pem import LocalPEM


# ============================================================
# Synthetic causal system
#
# 6 global variables:
#
#     0 -> 1 -> 2
#     3 -> 4 -> 5
#
# We deliberately create overlapping causal units.
# ============================================================


rng = np.random.default_rng(42)


num_samples = 5000
num_nodes = 6


noise = rng.normal(
    0.0,
    1.0,
    size=(num_samples, num_nodes),
)


X = np.zeros(
    (num_samples, num_nodes),
    dtype=np.float64,
)


# First causal chain
X[:, 0] = noise[:, 0]
X[:, 1] = 0.8 * X[:, 0] + noise[:, 1]
X[:, 2] = 0.7 * X[:, 1] + noise[:, 2]


# Second causal chain
X[:, 3] = noise[:, 3]
X[:, 4] = 0.8 * X[:, 3] + noise[:, 4]
X[:, 5] = 0.7 * X[:, 4] + noise[:, 5]


# ============================================================
# Synthetic topology
#
# The topology contains both causal and non-causal candidate
# relationships.
# ============================================================


adjacency = np.zeros(
    (num_nodes, num_nodes),
    dtype=np.float64,
)


# Chain 0 -> 1 -> 2
adjacency[0, 1] = 1.0
adjacency[1, 2] = 1.0


# Chain 3 -> 4 -> 5
adjacency[3, 4] = 1.0
adjacency[4, 5] = 1.0


# Add reciprocal topology relationships so that
# causal units overlap.
adjacency[1, 0] = 1.0
adjacency[2, 1] = 1.0
adjacency[4, 3] = 1.0
adjacency[5, 4] = 1.0


# Add a non-causal topology connection.
#
# This relationship is present in the road/topology graph,
# but there is deliberately NO causal relationship between
# variables 2 and 3.
adjacency[2, 3] = 1.0
adjacency[3, 2] = 1.0


# ============================================================
# Build causal units
# ============================================================


builder = CausalUnitBuilder()


units = builder.build(adjacency)


print("=" * 70)
print("LOCAL PEM SYNTHETIC TEST")
print("=" * 70)


print()
print(f"Number of causal units: {len(units)}")


for unit in units:
    print(
        f"Target {unit.target}: "
        f"nodes={unit.nodes}"
    )


# ============================================================
# Run Local PEM
# ============================================================


model = LocalPEM(
    units=units,
    nu0=0.05,
    nu1=1.0,
    tau=0.01,
    threshold=0.5,
    max_iter=200,
    tolerance=1e-6,
)


result = model.fit(X)


# ============================================================
# Extract results
# ============================================================


global_probabilities = (
    result.global_inclusion_probabilities
)

global_adjacency = (
    result.global_adjacency
)


# ============================================================
# Basic checks
# ============================================================


print()
print("-" * 70)
print("RESULT")
print("-" * 70)


print(
    "Global probability matrix shape:",
    global_probabilities.shape,
)


print(
    "Global adjacency matrix shape:",
    global_adjacency.shape,
)


print(
    "Processed units:",
    result.processed_units,
)


print(
    "Minimum valid samples:",
    result.valid_sample_counts.min(),
)


print(
    "Maximum valid samples:",
    result.valid_sample_counts.max(),
)


print(
    "Mean valid samples:",
    result.valid_sample_counts.mean(),
)


# ============================================================
# Print global posterior inclusion matrix
#
# IMPORTANT:
#
# This matrix is symmetric.
#
# It represents pair/edge inclusion probability,
# NOT causal direction.
# ============================================================


np.set_printoptions(
    precision=3,
    suppress=True,
)


print()
print("Global inclusion probabilities:")
print(global_probabilities)


# ============================================================
# Print directed PEM-LBN adjacency
#
# Convention:
#
#     adjacency[target, source] = 1
#
# therefore:
#
#     adjacency[i, j] = 1
#
# means:
#
#     j -> i
# ============================================================


print()
print("Global PEM directed adjacency:")
print(global_adjacency)


# ============================================================
# Print local causal orders
#
# These are expressed using LOCAL indices.
# ============================================================


print()
print("Local PEM causal orders:")
print("-" * 70)


for unit_index, (unit, order) in enumerate(
    zip(
        units,
        result.unit_causal_orders,
    )
):

    print(
        f"Unit {unit_index}: "
        f"global nodes={unit.nodes} "
        f"local causal order={order}"
    )


# ============================================================
# Expected causal directions
#
# True causal structure:
#
#     0 -> 1 -> 2
#     3 -> 4 -> 5
#
# Under the [target, source] convention:
#
#     adjacency[1, 0] = 1
#     adjacency[2, 1] = 1
#     adjacency[4, 3] = 1
#     adjacency[5, 4] = 1
# ============================================================


expected_edges = [
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
]


print()
print("-" * 70)
print("EXPECTED CAUSAL DIRECTIONS")
print("-" * 70)


for source, target in expected_edges:

    directed_value = global_adjacency[
        target,
        source,
    ]

    reverse_value = global_adjacency[
        source,
        target,
    ]

    probability = global_probabilities[
        target,
        source,
    ]

    reverse_probability = global_probabilities[
        source,
        target,
    ]

    print(
        f"{source} -> {target}: "
        f"adjacency={directed_value:.1f}, "
        f"probability={probability:.4f}    "
        f"reverse {target} -> {source}: "
        f"adjacency={reverse_value:.1f}, "
        f"probability={reverse_probability:.4f}"
    )


# ============================================================
# Verify expected causal directions
# ============================================================


for source, target in expected_edges:

    assert (
        global_adjacency[target, source] == 1.0
    ), (
        f"Expected causal edge "
        f"{source} -> {target} was not recovered."
    )

    assert (
        global_adjacency[source, target] == 0.0
    ), (
        f"Incorrect reverse causal edge "
        f"{target} -> {source} was recovered."
    )


# ============================================================
# Verify non-causal topology connection
#
# Variables 2 and 3 are connected in the topology but are
# independent in the synthetic causal system.
#
# Therefore PEM should NOT recover:
#
#     2 -> 3
#     3 -> 2
# ============================================================


print()
print("-" * 70)
print("NON-CAUSAL TOPOLOGY CHECK")
print("-" * 70)


print(
    "2 -> 3:",
    global_adjacency[3, 2],
)


print(
    "3 -> 2:",
    global_adjacency[2, 3],
)


assert (
    global_adjacency[3, 2] == 0.0
), (
    "Incorrect causal edge 2 -> 3 was recovered."
)


assert (
    global_adjacency[2, 3] == 0.0
), (
    "Incorrect causal edge 3 -> 2 was recovered."
)


# ============================================================
# Verify inclusion probabilities are symmetric
#
# PEM inclusion probability is pairwise/symmetric.
# Direction comes from the PEM-LBN adjacency.
# ============================================================


assert np.allclose(
    global_probabilities,
    global_probabilities.T,
), (
    "PEM inclusion probabilities must be symmetric."
)


# ============================================================
# Verify diagonal is zero
# ============================================================


assert np.allclose(
    np.diag(global_probabilities),
    0.0,
), (
    "Diagonal probabilities must be zero."
)


assert np.allclose(
    np.diag(global_adjacency),
    0.0,
), (
    "Directed causal adjacency diagonal must be zero."
)


# ============================================================
# Verify dimensions
# ============================================================


assert global_probabilities.shape == (
    num_nodes,
    num_nodes,
), (
    "Incorrect global probability matrix shape."
)


assert global_adjacency.shape == (
    num_nodes,
    num_nodes,
), (
    "Incorrect global adjacency matrix shape."
)


# ============================================================
# Verify valid sample counts
# ============================================================


assert np.all(
    result.valid_sample_counts > 0
), (
    "Every unit should have usable samples."
)


# ============================================================
# Verify all units were processed
# ============================================================


assert (
    result.processed_units == len(units)
), (
    "Not all causal units were processed."
)


# ============================================================
# Final result
# ============================================================


print()
print("=" * 70)
print("LOCAL PEM SYNTHETIC TEST PASSED")
print("=" * 70)