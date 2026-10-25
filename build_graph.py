"""
Build the graph artifacts read by main.py (E1 VAR, E3 VAR + PEM).

    E1:  python build_graph.py E1 [--var-run lag2_alpha1e+06]
    E3:  python build_graph.py E3 [--pem-run <folder under outputs/pem>]

Without --var-run / --pem-run the runs set in config/config.py
(VAR_RUN, PEM_RUN) are used, so the file name matches FINAL_GRAPHS.

Both are built from TRAINING-split results only (run_var.py,
run_local_pem.py). The artifact stores the graph as G[target, source]
together with train_end and the sensor set, which main.load_graph()
checks against the run.

Output: outputs/graphs/<E1|E2>__<source run>.npz (+ .json summary)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from causal.graph_builders import pem_graph, save_graph_artifact, var_graph
from config.config import PEM_RUN, VAR_RUN
from experiments.registry import get_experiment


IMPLEMENTATION_DIR = Path(__file__).resolve().parent

VAR_ROOT = IMPLEMENTATION_DIR / "outputs" / "var_train"
PEM_ROOT = IMPLEMENTATION_DIR / "outputs" / "pem"
GRAPH_ROOT = IMPLEMENTATION_DIR / "outputs" / "graphs"


def summarize(graph):

    off = graph.copy()
    np.fill_diagonal(off, 0.0)

    edges = off != 0
    weights = off[edges]

    return {
        "num_nodes": int(graph.shape[0]),
        "directed_edges": int(edges.sum()),
        "density": float(edges.sum() / (graph.shape[0] * (graph.shape[0] - 1))),
        "two_way_pairs": int(np.count_nonzero(np.triu(edges & edges.T, 1))),
        "nodes_without_incoming": int(np.sum(~edges.any(axis=1))),
        "weight_min": float(weights.min()) if weights.size else 0.0,
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "weight_max": float(weights.max()) if weights.size else 0.0,
    }


def main():

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    # E1 = VAR, E3 = VAR + PEM (the former E2). E2 / E4-E7 builders
    # are added in later phases (experiments/registry.py).
    parser.add_argument("experiment", choices=["E1", "E3"])
    parser.add_argument("--var-run", default=VAR_RUN)
    parser.add_argument("--pem-run", default=PEM_RUN)
    args = parser.parse_args()

    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)

    if args.experiment == "E1":

        if not args.var_run:
            parser.error("E1 requires --var-run")

        source_dir = VAR_ROOT / args.var_run

        with open(source_dir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        graph = var_graph(np.load(source_dir / "coefficients.npy"))
        sensors = np.load(source_dir / "sensors.npy")
        train_end = meta["train_end"]
        source = args.var_run

    else:

        if not args.pem_run:
            parser.error("E3 requires --pem-run")

        source_dir = PEM_ROOT / args.pem_run

        with np.load(source_dir / "pem_result.npz") as result:
            graph = pem_graph(
                result["inclusion_probabilities"],
                result["directed_adjacency"],
            )
            sensors = result["sensors"]
            train_end = int(result["train_end"])

        source = args.pem_run

    path = GRAPH_ROOT / f"{args.experiment}__{source}.npz"

    save_graph_artifact(
        path,
        graph,
        experiment=args.experiment,
        method=get_experiment(args.experiment).graph_method,
        train_end=train_end,
        sensors=sensors,
        source_run=source,
    )

    summary = summarize(graph)
    summary.update({"experiment": args.experiment,
                    "method": get_experiment(args.experiment).graph_method,
                    "source_run": source,
                    "train_end": train_end})

    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
