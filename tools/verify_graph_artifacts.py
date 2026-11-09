"""
Verify graph artifacts before they are uploaded to Kaggle: the same conditions main.load_graph enforces, without TensorFlow.

Per file (E8 / E8V style, one file per horizon):
    - method is the one the experiment needs ("var_pem_horizon" for E8, "var_horizon" for E8V)
    - the horizon stored in the file equals the horizon in the file name (__h<h>__)
    - orientation is target_source; sensors equal the sensors of the VAR run; train_end equals the training split
    - adjacency is finite, non-negative, has a zero diagonal and at least one edge (graph.adjacency.check_model_graph)
    - in-degree is at most the stored top_k

    python tools/verify_graph_artifacts.py outputs/hybrid_graphs
"""

import re
import sys
from pathlib import Path

import numpy as np

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))

from graph.adjacency import check_model_graph  # noqa: E402

METHODS = {"E8": "var_pem_horizon", "E8V": "var_horizon"}
TRAIN_END = 84096


def verify(path, sensors):
    problems = []
    name = Path(path).name
    exp = name.split("__")[0]
    h_in_name = re.search(r"__h(\d+)__", name)
    with np.load(path, allow_pickle=False) as z:
        files = {k: z[k] for k in z.files}
    if exp in METHODS and str(files.get("method")) != METHODS[exp]:
        problems.append(f"method {files.get('method')!r} != {METHODS[exp]!r}")
    if h_in_name and int(files.get("horizon", -1)) != int(h_in_name.group(1)):
        problems.append(f"horizon {files.get('horizon')} != file name h{h_in_name.group(1)}")
    if str(files.get("orientation")) != "target_source":
        problems.append(f"orientation {files.get('orientation')!r}")
    if not np.array_equal(np.asarray(files["sensors"]), sensors):
        problems.append("sensor list differs from the VAR run")
    if int(files["train_end"]) != TRAIN_END:
        problems.append(f"train_end {int(files['train_end'])} != {TRAIN_END}")
    adj = np.asarray(files["adjacency"], dtype=np.float32)
    try:
        stats = check_model_graph(adj, len(sensors), allow_self_loops=False)
    except ValueError as error:
        problems.append(str(error))
        stats = {}
    top_k = int(files.get("top_k", 25))
    if ((adj > 0).sum(axis=1) > top_k).any():
        problems.append(f"in-degree above top_k={top_k}")
    return problems, stats


def main():
    folder = Path(sys.argv[1])
    sensors = np.load(IMPLEMENTATION_DIR / "outputs" / "var_train" / "lag3_alpha1e+06" / "sensors.npy")
    files = sorted(folder.glob("E*__*.npz"))
    if not files:
        raise SystemExit(f"no E*__*.npz files in {folder}")
    bad = 0
    for f in files:
        problems, stats = verify(f, sensors)
        mark = "OK  " if not problems else "FAIL"
        bad += bool(problems)
        print(f"{mark} {f.name}  edges {stats.get('num_edges_off_diagonal', '-')}  "
              f"in-degree max {stats.get('in_degree_max', '-')}  weight mean {stats.get('weight_mean', float('nan')):.4f}"
              + ("" if not problems else "   <- " + "; ".join(problems)))
    print(f"\n{len(files) - bad}/{len(files)} artifacts verified")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
