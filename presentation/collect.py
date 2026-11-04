"""Collect test metrics of every finished run into presentation/data.json."""
import json, csv
from pathlib import Path
import numpy as np, scipy.sparse as sp
import sys; sys.path.insert(0, ".")
from graph.normalization import normalized_adj
from causal.graph_builders import to_model_orientation

K = Path("results/kaggle")
RUNS = {  # key: (folder, experiment, horizon, seed)
    "HA_h1": "results/HA", "HA_h6": "results/HA_h6", "HA_h12": "results/HA_h12",
    "NOGRAPH_h1_s42": K/"output_NOGRAPH_final", "NOGRAPH_h6_s42": K/"output_NOGRAPH_h6_final", "NOGRAPH_h12_s42": K/"output_NOGRAPH_h12_final",
    "E7_h1_s42": K/"output_E7_final",
    "E0_h1_s42": K/"output_E0_final", "E0_h1_s7": K/"output_E0_final_seed7", "E0_h12_s42": K/"output_E0_h12_final",
    "E1_h1_s42": K/"output_E1_final", "E1_h1_s7": K/"output_E1_final_seed7", "E1_h6_s42": K/"output_E1_h6_final", "E1_h12_s42": K/"output_E1_h12_final",
    "E2_h1_s42": K/"output_E2_final",
    "E3PxD_h1_s42": K/"output_E3_PxD_final", "E3_h1_s42": K/"output_E3_final",
    "E5_h1_s42": K/"output_E5_final", "E5_h1_s7": K/"output_E5_final_seed7", "E5_h6_s42": K/"output_E5_h6_final", "E5_h12_s42": K/"output_E5_h12_final",
}
out = {}
for key, d in RUNS.items():
    d = Path(d)
    f = d/"results.json" if (d/"results.json").exists() else d/"run_summary.json"
    if not f.exists(): print("missing", key); continue
    r = json.load(open(f))
    t = r.get("test_metrics", {})
    t = {k.lower(): v for k, v in t.items()}
    row = {"rmse": t["rmse"], "mae": t["mae"], "mape": t.get("masked_mape"), "best_epoch": r.get("best_epoch")}
    if (d/"history.csv").exists():      # every training run has it (epoch_metrics.csv is newer)
        row["val_rmse"] = [float(x["val_rmse_original"]) for x in csv.DictReader(open(d/"history.csv"))]
    if "graph" in r and isinstance(r["graph"], dict) and "statistics" in r["graph"]:
        row["edges"] = r["graph"]["statistics"]["num_edges_off_diagonal"]
    out[key] = row

# Own-signal share: median diagonal of the A3T-GCN operator D^-1/2 (A^T + I) D^-1/2.
def share(g):
    op = normalized_adj(sp.csr_matrix(g) + sp.eye(g.shape[0]))
    return float(np.median(op.diagonal()))
G = Path("outputs/graphs")
with np.load(G/"E1__lag3_alpha1e+06.npz") as a: sensors = a["sensors"]
road = np.load("datasets/LargeST GBA/gba_rn_adj.npy", mmap_mode="r")
shares = {"NOGRAPH": 1.0, "E0": share(np.asarray(road[np.ix_(sensors, sensors)], dtype=np.float32))}
for name, pat in [("E1","E1__*.npz"),("E2","E2__*.npz"),("E3","E3__*.npz"),("E4","E4__*.npz"),
                  ("E5_h1","E5__h1__*.npz"),("E5_h6","E5__h6__*.npz"),("E5_h12","E5__h12__*.npz")]:
    with np.load(next(G.glob(pat))) as a: shares[name] = share(to_model_orientation(a["adjacency"]))
with np.load(Path("outputs/graphs_archive")/"E3__lag3_alpha1e+06__nu0_0.0017_nu1_1_tau_0.01_thr_0.5_eta_0.5_L-BFGS-B_std.npz") as a:
    shares["E3PxD"] = share(to_model_orientation(a["adjacency"]))
out["own_signal_share"] = shares
json.dump(out, open("presentation/data.json", "w"), indent=1)
for k, v in out.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "val_rmse"})
