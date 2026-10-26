"""
Build the graph artifacts read by main.py.

    E1:  python build_graph.py E1 [--var-run RUN]                 VAR
    E2:  python build_graph.py E2 [--var-run RUN] [--lingam-run]  VAR-LiNGAM
    E3:  python build_graph.py E3 [--pem-run RUN]                 VAR + PEM
    E4:  python build_graph.py E4 [--lingam-run] [--pem-run]      VAR-LiNGAM + PEM
    E6:  python build_graph.py E6 --var-run R --lingam-run R --pem-run R
         (E4's formula; every input built on data-driven candidates)
    E7:  python build_graph.py E7 --pem-run R
         (E4's formula; PEM with the road prior)
    E5:  python build_graph.py E5 [--lingam-run] [--pem-run]
         (one graph per horizon in experiments/registry.py: top-K sources
         of the structural impulse responses, K = config.E5_TOP_K)

Without --var-run / --lingam-run / --pem-run the runs set in
config/config.py (VAR_RUN, LINGAM_RUN, PEM_RUN) are used, so the file
name matches FINAL_GRAPHS.

Every input is built from TRAINING-split results only (run_var.py,
run_local_lingam.py, run_local_pem.py); all inputs of one graph must
share the training split and sensor set. The artifact stores the graph
as G[target, source] with its method, train_end and the sensor set,
which main.load_graph() checks against the run.

Output: outputs/graphs/<experiment>__<source run(s)>.npz (+ .json summary)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from causal.graph_builders import (
    horizon_graphs,
    pem_graph,
    save_graph_artifact,
    var_graph,
    var_lingam_graph,
    var_lingam_pem_graph,
)
from config.config import (
    E5_TOP_K,
    LINGAM_RUN,
    LINGAM_RUN_E6,
    PEM_RUN,
    PEM_RUN_E6,
    PEM_RUN_E7,
    VAR_RUN,
    VAR_RUN_E6,
)
from experiments.registry import get_experiment


IMPLEMENTATION_DIR = Path(__file__).resolve().parent

VAR_ROOT = IMPLEMENTATION_DIR / "outputs" / "var_train"
PEM_ROOT = IMPLEMENTATION_DIR / "outputs" / "pem"
LINGAM_ROOT = IMPLEMENTATION_DIR / "outputs" / "lingam"
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


def load_var(run):

    source = VAR_ROOT / run
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    return {
        "coefficients": np.load(source / "coefficients.npy"),
        "sensors": np.load(source / "sensors.npy"),
        "train_end": int(meta["train_end"]),
        "candidates": meta.get("candidates", "road"),
    }


def load_pem(run):

    with np.load(PEM_ROOT / run / "pem_result.npz") as result:
        return {
            "probabilities": result["inclusion_probabilities"],
            "directed": result["directed_adjacency"],
            "sensors": result["sensors"],
            "train_end": int(result["train_end"]),
            # Runs from before these keys existed: road graph, uniform prior.
            "candidates": str(result["candidates"]) if "candidates" in result.files else "road",
            "prior": str(result["prior"]) if "prior" in result.files else "uniform",
            "var_run": str(result["var_run"]) if "var_run" in result.files else None,
        }


def load_lingam(run):

    with np.load(LINGAM_ROOT / run / "lingam_result.npz") as result:
        return {
            "b0": result["b0"],
            "var_run": str(result["var_run"]),
            "sensors": result["sensors"],
            "train_end": int(result["train_end"]),
            "candidates": str(result["candidates"]) if "candidates" in result.files else "road",
        }


def same_split(*parts):
    """All inputs must come from the same training split and sensors."""

    first = parts[0]
    for part in parts[1:]:
        if part["train_end"] != first["train_end"] or not np.array_equal(
            part["sensors"], first["sensors"]
        ):
            raise ValueError("Graph inputs come from different splits / sensor sets.")
    return first["sensors"], first["train_end"]


# Candidates and PEM prior each experiment requires (experiments/registry.py).
REQUIREMENTS = {
    "E4": {"candidates": "road", "prior": "uniform"},
    "E5": {"candidates": "road", "prior": "uniform"},
    "E6": {"candidates": "data-driven", "prior": "uniform"},
    "E7": {"candidates": "road", "prior": "road"},
}


def check_inputs(experiment, var, lingam, pem):
    """All inputs on the experiment's candidates / prior, and one VAR run."""

    need = REQUIREMENTS[experiment]

    for name, part in (("VAR", var), ("LiNGAM", lingam), ("PEM", pem)):
        cand = part["candidates"]
        ok = cand != "road" if need["candidates"] == "data-driven" else cand == "road"
        if not ok:
            raise ValueError(f"{experiment} needs {need['candidates']} candidates; "
                             f"the {name} run used {cand!r}.")

    if pem["prior"] != need["prior"]:
        raise ValueError(f"{experiment} needs a {need['prior']} PEM prior; "
                         f"the PEM run used {pem['prior']!r}.")

    if pem["var_run"] is not None and pem["var_run"] != lingam["var_run"]:
        raise ValueError("PEM and LiNGAM were run on different VAR runs.")


def main():

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("experiment", choices=["E1", "E2", "E3", "E4", "E5", "E6", "E7"])
    parser.add_argument("--var-run", default=None)
    parser.add_argument("--lingam-run", default=None)
    parser.add_argument("--pem-run", default=None)
    args = parser.parse_args()

    # Defaults: the runs set in config.py for this experiment.
    defaults = {
        "E6": (VAR_RUN_E6, LINGAM_RUN_E6, PEM_RUN_E6),
        "E7": (VAR_RUN, LINGAM_RUN, PEM_RUN_E7),
    }.get(args.experiment, (VAR_RUN, LINGAM_RUN, PEM_RUN))

    args.var_run = args.var_run or defaults[0]
    args.lingam_run = args.lingam_run or defaults[1]
    args.pem_run = args.pem_run or defaults[2]

    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)

    if args.experiment == "E1":
        var = load_var(args.var_run)
        graph = var_graph(var["coefficients"])
        sensors, train_end = same_split(var)
        source = args.var_run

    elif args.experiment == "E3":
        pem = load_pem(args.pem_run)
        if pem["candidates"] != "road" or pem["prior"] != "uniform":
            raise ValueError("E3 needs PEM on road candidates with a uniform prior.")
        graph = pem_graph(pem["probabilities"], pem["directed"])
        sensors, train_end = same_split(pem)
        source = args.pem_run

    else:
        lingam = load_lingam(args.lingam_run)

        if lingam["var_run"] != args.var_run:
            raise ValueError(
                f"LiNGAM run used VAR {lingam['var_run']}, not {args.var_run}; "
                "B_k = (I - B_0) A_k needs the same VAR."
            )

        var = load_var(args.var_run)

        if args.experiment == "E2":
            if var["candidates"] != "road" or lingam["candidates"] != "road":
                raise ValueError("E2 uses road candidates.")
            graph = var_lingam_graph(var["coefficients"], lingam["b0"])
            sensors, train_end = same_split(var, lingam)
            source = args.lingam_run
        else:
            pem = load_pem(args.pem_run)
            check_inputs(args.experiment, var, lingam, pem)
            sensors, train_end = same_split(var, lingam, pem)
            source = f"{args.lingam_run}__{args.pem_run}"

            if args.experiment == "E5":
                spec = get_experiment("E5")
                graphs = horizon_graphs(
                    var["coefficients"], lingam["b0"], pem["probabilities"],
                    spec.horizons, E5_TOP_K,
                )
                for h, graph in graphs.items():
                    save(f"E5__h{h}", graph, "E5", sensors, train_end, source,
                         horizon=h, top_k=E5_TOP_K)
                return

            graph = var_lingam_pem_graph(
                var["coefficients"], lingam["b0"], pem["probabilities"]
            )

    save(args.experiment, graph, args.experiment, sensors, train_end, source)


def save(prefix, graph, experiment, sensors, train_end, source, **extra):

    path = GRAPH_ROOT / f"{prefix}__{source}.npz"
    method = get_experiment(experiment).graph_method

    save_graph_artifact(
        path,
        graph,
        experiment=experiment,
        method=method,
        train_end=train_end,
        sensors=sensors,
        source_run=source,
        **extra,
    )

    summary = summarize(graph)
    summary.update({"experiment": experiment, "method": method,
                    "source_run": source, "train_end": train_end, **extra})

    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
