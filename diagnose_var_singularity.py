import numpy as np
import h5py


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

NUM_TIMESTEPS = 1000
NUM_NODES = 100

LAG_ORDER = 1
RIDGE_ALPHA = 1e-5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VAR SINGULARITY DIAGNOSTIC")
print("=" * 70)

print("\nLoading adjacency...")
adjacency = np.load(ADJACENCY_FILE)

print("Full adjacency shape:", adjacency.shape)

print("Loading traffic...")
with h5py.File(TRAFFIC_FILE, "r") as f:
    traffic = np.asarray(
        f["traffic"][:NUM_TIMESTEPS, :NUM_NODES],
        dtype=np.float64,
    )

adjacency = np.asarray(
    adjacency[:NUM_NODES, :NUM_NODES],
    dtype=np.float64,
)

print("Traffic shape:", traffic.shape)
print("Adjacency shape:", adjacency.shape)


# ============================================================
# HELPER: GET NEIGHBORS
# ============================================================

def get_neighbors(target):
    neighbors = np.where(adjacency[target] != 0)[0]

    # Ensure target itself is included.
    if target not in neighbors:
        neighbors = np.concatenate(
            [neighbors, np.array([target], dtype=np.int64)]
        )

    return np.unique(neighbors)


# ============================================================
# DIAGNOSE EACH TARGET
# ============================================================

print("\n" + "-" * 70)
print("ANALYSING LOCAL REGRESSION MATRICES")
print("-" * 70)

results = []

for target in range(NUM_NODES):

    neighbors = get_neighbors(target)

    # --------------------------------------------------------
    # Build lag-1 regression data
    #
    # X[t] = traffic[t-1, neighbors]
    # y[t] = traffic[t, target]
    # --------------------------------------------------------

    X = traffic[LAG_ORDER:, :][:, neighbors]
    y = traffic[LAG_ORDER:, target]

    valid = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(y)
    )

    X = X[valid]
    y = y[valid]

    if X.shape[0] == 0:
        print(
            f"Target {target:3d}: "
            "NO VALID SAMPLES"
        )
        continue

    # --------------------------------------------------------
    # Calculate XtX
    # --------------------------------------------------------

    XtX = X.T @ X

    # Ridge matrix
    ridge_matrix = (
        XtX
        + RIDGE_ALPHA * np.eye(X.shape[1], dtype=np.float64)
    )

    # --------------------------------------------------------
    # Numerical diagnostics
    # --------------------------------------------------------

    rank = np.linalg.matrix_rank(XtX)

    n_features = XtX.shape[0]

    try:
        condition_number = np.linalg.cond(XtX)
    except np.linalg.LinAlgError:
        condition_number = np.inf

    try:
        eigenvalues = np.linalg.eigvalsh(XtX)

        min_eigenvalue = eigenvalues[0]
        max_eigenvalue = eigenvalues[-1]

    except np.linalg.LinAlgError:
        min_eigenvalue = np.nan
        max_eigenvalue = np.nan

    try:
        ridge_condition = np.linalg.cond(ridge_matrix)
    except np.linalg.LinAlgError:
        ridge_condition = np.inf

    try:
        np.linalg.solve(
            ridge_matrix,
            X.T @ y,
        )
        numpy_solve = "SUCCESS"
    except np.linalg.LinAlgError:
        numpy_solve = "FAILED"

    results.append(
        {
            "target": target,
            "neighbors": len(neighbors),
            "samples": X.shape[0],
            "features": n_features,
            "rank": rank,
            "condition": condition_number,
            "ridge_condition": ridge_condition,
            "min_eigenvalue": min_eigenvalue,
            "max_eigenvalue": max_eigenvalue,
            "numpy_solve": numpy_solve,
        }
    )


# ============================================================
# SORT BY CONDITION NUMBER
# ============================================================

results.sort(
    key=lambda x: x["condition"],
    reverse=True,
)


# ============================================================
# PRINT WORST CASES
# ============================================================

print("\n" + "-" * 70)
print("WORST-CONDITIONED TARGETS")
print("-" * 70)

print(
    f"{'Target':>8} "
    f"{'Neighbors':>10} "
    f"{'Samples':>10} "
    f"{'Rank':>8} "
    f"{'Features':>9} "
    f"{'Cond(XtX)':>16} "
    f"{'Cond(Ridge)':>16} "
    f"{'Solve':>10}"
)

print("-" * 100)

for result in results[:15]:

    print(
        f"{result['target']:8d} "
        f"{result['neighbors']:10d} "
        f"{result['samples']:10d} "
        f"{result['rank']:8d} "
        f"{result['features']:9d} "
        f"{result['condition']:16.4e} "
        f"{result['ridge_condition']:16.4e} "
        f"{result['numpy_solve']:>10}"
    )


# ============================================================
# SUMMARY
# ============================================================

failed_numpy = [
    r for r in results
    if r["numpy_solve"] == "FAILED"
]

rank_deficient = [
    r for r in results
    if r["rank"] < r["features"]
]

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Targets analysed:", len(results))
print("Rank-deficient XtX matrices:", len(rank_deficient))
print("Failed NumPy ridge solves:", len(failed_numpy))

if results:

    worst = results[0]

    print("\nWorst-conditioned target:")
    print("  Target:", worst["target"])
    print("  Neighbors:", worst["neighbors"])
    print("  Samples:", worst["samples"])
    print("  Features:", worst["features"])
    print("  Rank:", worst["rank"])
    print("  Condition number:", worst["condition"])
    print("  Ridge condition:", worst["ridge_condition"])
    print("  Minimum eigenvalue:", worst["min_eigenvalue"])
    print("  Maximum eigenvalue:", worst["max_eigenvalue"])
    print("  NumPy solve:", worst["numpy_solve"])

print("\nDiagnostic completed.")