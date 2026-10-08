import numpy as np


ADJACENCY_PATH = (
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_rn_adj.npy"
)


adjacency = np.load(ADJACENCY_PATH)


print("=" * 70)
print("GBA ADJACENCY DIRECTION DIAGNOSTIC")
print("=" * 70)

num_nodes = adjacency.shape[0]

print(f"Shape: {adjacency.shape}")
print(f"Number of nodes: {num_nodes}")

# ------------------------------------------------------------
# OUTGOING DEGREE
#
# Row i:
#     adjacency[i, j] != 0
#
# This is the neighborhood currently used by our VAR code.
# ------------------------------------------------------------

outgoing_degree = np.count_nonzero(adjacency, axis=1)

# ------------------------------------------------------------
# INCOMING DEGREE
#
# Column i:
#     adjacency[j, i] != 0
# ------------------------------------------------------------

incoming_degree = np.count_nonzero(adjacency, axis=0)

# Remove self-loops for the actual topology-degree statistics.
outgoing_degree_no_self = outgoing_degree.copy()
incoming_degree_no_self = incoming_degree.copy()

diagonal_nonzero = np.diag(adjacency) != 0

outgoing_degree_no_self[diagonal_nonzero] -= 1
incoming_degree_no_self[diagonal_nonzero] -= 1


print()
print("-" * 70)
print("OUTGOING DEGREE")
print("-" * 70)

print(
    f"Minimum: {outgoing_degree_no_self.min()}"
)

print(
    f"Maximum: {outgoing_degree_no_self.max()}"
)

print(
    f"Mean: {outgoing_degree_no_self.mean():.6f}"
)

print(
    f"Median: {np.median(outgoing_degree_no_self):.1f}"
)

print(
    f"Zero outgoing neighbors: "
    f"{np.sum(outgoing_degree_no_self == 0)}"
)


print()
print("-" * 70)
print("INCOMING DEGREE")
print("-" * 70)

print(
    f"Minimum: {incoming_degree_no_self.min()}"
)

print(
    f"Maximum: {incoming_degree_no_self.max()}"
)

print(
    f"Mean: {incoming_degree_no_self.mean():.6f}"
)

print(
    f"Median: {np.median(incoming_degree_no_self):.1f}"
)

print(
    f"Zero incoming neighbors: "
    f"{np.sum(incoming_degree_no_self == 0)}"
)


# ------------------------------------------------------------
# COMPLETELY ISOLATED NODES
# ------------------------------------------------------------

no_outgoing = outgoing_degree_no_self == 0
no_incoming = incoming_degree_no_self == 0

completely_isolated = no_outgoing & no_incoming

print()
print("-" * 70)
print("ISOLATION")
print("-" * 70)

print(
    f"Zero outgoing AND zero incoming: "
    f"{np.sum(completely_isolated)}"
)


# ------------------------------------------------------------
# RECIPROCAL CONNECTIONS
# ------------------------------------------------------------

off_diagonal = ~np.eye(num_nodes, dtype=bool)

forward = adjacency != 0
reverse = adjacency.T != 0

forward_off = forward & off_diagonal
reverse_off = reverse & off_diagonal

reciprocal = forward_off & reverse_off

print()
print("-" * 70)
print("DIRECTIONALITY")
print("-" * 70)

print(
    f"Directed non-zero entries "
    f"(excluding self-loops): "
    f"{np.sum(forward_off)}"
)

print(
    f"Reciprocal directed entries: "
    f"{np.sum(reciprocal)}"
)


# ------------------------------------------------------------
# EXAMPLE NODES
# ------------------------------------------------------------

print()
print("-" * 70)
print("EXAMPLE NODES")
print("-" * 70)

zero_outgoing_indices = np.flatnonzero(no_outgoing)

print(
    "First nodes with zero outgoing neighbors:"
)

print(zero_outgoing_indices[:20])


zero_incoming_indices = np.flatnonzero(no_incoming)

print()
print(
    "First nodes with zero incoming neighbors:"
)

print(zero_incoming_indices[:20])


# ------------------------------------------------------------
# COMPARE ROW AND COLUMN NEIGHBORHOODS
# ------------------------------------------------------------

print()
print("-" * 70)
print("FIRST 5 NODE NEIGHBORHOODS")
print("-" * 70)

for node in range(min(5, num_nodes)):

    outgoing = np.flatnonzero(adjacency[node] != 0)
    outgoing = outgoing[outgoing != node]

    incoming = np.flatnonzero(adjacency[:, node] != 0)
    incoming = incoming[incoming != node]

    print()
    print(f"Node {node}")
    print(f"  Outgoing neighbors: {outgoing}")
    print(f"  Incoming neighbors: {incoming}")


print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)