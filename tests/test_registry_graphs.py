"""
Tests for the experiment registry and the graph checks (Phase 2):

    1. Registry: E0-E7 present; experiments whose builder is missing
       are refused with the phase that adds them; E0 refuses a graph
       file; E1 / E3 default to existing final graphs.
    2. Graph method of artifacts: new "method" key and legacy labels
       (old E2 = VAR + PEM = new E3).
    3. check_model_graph rejects wrong shape, NaN, negative weights,
       self-loops in causal graphs and empty graphs; leaves the graph
       unchanged (no symmetrization, same weights).
    4. graph_diagnostics: in / out-degree in the [target, source]
       convention.
    5. Orientation: after to_model_orientation + the original A3T-GCN
       normalization, a node aggregates from its SOURCES only.
    6. main.load_graph on the real E1 / E3 artifacts, the legacy E2
       file for E3, the no-graph baseline, and a method mismatch.

Run from the implementation directory:

    python -m tests.test_registry_graphs
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from causal.graph_builders import to_model_orientation
from config.config import FINAL_GRAPHS, GRAPH_DIR, PEM_RUN, ExperimentConfig
from experiments.registry import EXPERIMENTS, artifact_method, get_experiment
from graph.adjacency import check_model_graph, graph_diagnostics


IMPLEMENTATION_DIR = Path(__file__).resolve().parent.parent


def expect_error(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error as exc:
        return str(exc)
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


def test_registry():

    assert sorted(EXPERIMENTS) == [f"E{i}" for i in range(8)] + ["NOGRAPH"]
    assert {n for n, s in EXPERIMENTS.items() if s.ready} == {"E0", "E1", "E3", "NOGRAPH"}
    assert get_experiment("E3").graph_method == "var_pem"

    expect_error(ValueError, get_experiment, "E9")

    for name, spec in EXPERIMENTS.items():
        if not spec.ready:
            message = expect_error(
                NotImplementedError, ExperimentConfig(experiment=name).validate
            )
            assert "Phase" in message, message

    expect_error(ValueError, ExperimentConfig(experiment="E0", graph_path="x.npz").validate)
    expect_error(ValueError, ExperimentConfig(experiment="NOGRAPH", graph_path="x.npz").validate)

    for name in ("E1", "E3"):
        config = ExperimentConfig(experiment=name)
        config.validate()
        assert Path(config.graph_path) == GRAPH_DIR / FINAL_GRAPHS[name]
        assert Path(config.graph_path).exists()

    print("[PASS] 1. registry E0-E7 + NOGRAPH, unfinished experiments refused, default graphs exist")


def test_artifact_method():

    assert artifact_method({"method": "var_lingam", "experiment": "E2"}) == "var_lingam"
    assert artifact_method({"experiment": "E1"}) == "var"
    assert artifact_method({"experiment": "E2"}) == "var_pem"      # legacy label
    expect_error(ValueError, artifact_method, {"experiment": "E5"})

    print("[PASS] 2. artifact method: new key and legacy labels")


def test_check_model_graph():

    good = np.zeros((4, 4), dtype=np.float32)
    good[1, 0] = 0.5          # 0 -> 1
    good[2, 1] = 2.0          # 1 -> 2
    before = good.copy()

    stats = check_model_graph(good, 4, allow_self_loops=False)
    assert np.array_equal(good, before), "graph was modified"
    assert stats["num_edges_off_diagonal"] == 2 and not stats["is_symmetric"]

    expect_error(ValueError, check_model_graph, good, 5, allow_self_loops=False)

    for bad in (np.nan, -1.0):
        g = good.copy(); g[3, 2] = bad
        expect_error(ValueError, check_model_graph, g, 4, allow_self_loops=False)

    loops = good + np.eye(4, dtype=np.float32)
    expect_error(ValueError, check_model_graph, loops, 4, allow_self_loops=False)
    check_model_graph(loops, 4, allow_self_loops=True)       # road graph

    expect_error(ValueError, check_model_graph, np.eye(4), 4, allow_self_loops=True)

    print("[PASS] 3. graph checks reject bad graphs and never modify the graph")


def test_diagnostics():

    g = np.zeros((3, 3))
    g[1, 0] = g[2, 0] = 1.0   # 0 -> 1, 0 -> 2
    g[2, 1] = 1.0             # 1 -> 2

    d = graph_diagnostics(g)

    assert d["in_degree_max"] == 2                 # node 2 has two sources
    assert d["out_degree_max"] == 2                # node 0 has two targets
    assert d["nodes_without_incoming"] == 1        # node 0
    assert d["nodes_without_outgoing"] == 1        # node 2
    assert d["num_reciprocal_pairs"] == 0

    print("[PASS] 4. in / out-degree in the [target, source] convention")


def test_orientation():

    from graph.normalization import normalized_adj
    import scipy.sparse as sp

    g = np.zeros((3, 3), dtype=np.float32)
    g[1, 0] = 1.0             # source 0 -> target 1

    model_adj = to_model_orientation(g)
    operator = normalized_adj(model_adj + sp.eye(3)).toarray()

    # T-GCN computes output = operator @ x, so row i lists the nodes i reads.
    assert operator[1, 0] > 0, "target 1 does not read its source 0"
    assert operator[0, 1] == 0, "source 0 reads its target 1"
    assert operator[2, 0] == operator[2, 1] == 0

    print("[PASS] 5. A3T-GCN aggregates each node from its sources")


def test_load_graph():

    sys.path.insert(0, str(IMPLEMENTATION_DIR.parent))
    from implementation.main import load_graph

    e3 = GRAPH_DIR / FINAL_GRAPHS["E3"]

    with np.load(e3) as artifact:
        sensors = artifact["sensors"]
        train_end = int(artifact["train_end"])

    road = np.zeros((len(sensors), len(sensors)), dtype=np.float32)

    for name in ("E1", "E3"):
        config = ExperimentConfig(experiment=name)
        config.validate()
        adjacency, info = load_graph(config, road, train_end, sensors)
        assert info["method"] == get_experiment(name).graph_method
        assert info["statistics"]["num_self_loops"] == 0

    legacy = GRAPH_DIR / f"E2__{PEM_RUN}.npz"
    if legacy.exists():
        config = ExperimentConfig(experiment="E3", graph_path=str(legacy))
        config.validate()
        old, info = load_graph(config, road, train_end, sensors)
        assert info["artifact_label"] == "E2" and info["method"] == "var_pem"
        new, _ = load_graph(ExperimentConfig(experiment="E3", graph_path=str(e3)), road, train_end, sensors)
        assert np.array_equal(old, new)

    wrong = ExperimentConfig(experiment="E3", graph_path=str(GRAPH_DIR / FINAL_GRAPHS["E1"]))
    assert "method" in expect_error(ValueError, load_graph, wrong, road, train_end, sensors)

    expect_error(ValueError, load_graph, ExperimentConfig(experiment="E3", graph_path=str(e3)),
                 road, train_end + 1, sensors)

    adjacency, info = load_graph(ExperimentConfig(experiment="NOGRAPH"), road, train_end, sensors)
    assert info["method"] == "none" and not adjacency.any()

    print("[PASS] 6. load_graph: E1 / E3 artifacts, legacy E2 file as E3, "
          "method and split mismatches refused")


def main():

    print("=" * 70)
    print("REGISTRY AND GRAPH CHECK TESTS")
    print("=" * 70)

    test_registry()
    test_artifact_method()
    test_check_model_graph()
    test_diagnostics()
    test_orientation()
    test_load_graph()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
