import numpy as np

from causal.causal_units import CausalUnitBuilder


ADJACENCY_PATH = (
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_rn_adj.npy"
)


adjacency = np.load(ADJACENCY_PATH)

builder = CausalUnitBuilder()

units = builder.build(adjacency)

summary = builder.summary(units)


print("=" * 70)
print("CAUSAL UNIT SUMMARY")
print("=" * 70)

for key, value in summary.items():
    print(f"{key}: {value}")


print()
print("First 5 units:")

for unit in units[:5]:

    print()
    print(f"Target: {unit.target}")
    print(f"Nodes: {unit.nodes}")
    print(
        f"Incoming: {unit.incoming_neighbors}"
    )
    print(
        f"Outgoing: {unit.outgoing_neighbors}"
    )
    print(
        f"Combined: {unit.neighbors}"
    )


print()
print("=" * 70)
print("CAUSAL UNIT TEST COMPLETE")
print("=" * 70)