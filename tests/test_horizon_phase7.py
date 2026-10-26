"""
Tests for multi-horizon forecasting and the E5 graphs (Phase 7):

    1. Windows with target_offset = horizon - 1: inputs unchanged, the
       target h steps after the last input, validity checks only the
       inputs and the target row; offset 0 = previous behaviour.
    2. top_k_sources: k strongest sources per target, diagonal 0.
    3. horizon_graphs: G_h = top-k |Psi_h (I - P * B_0)^-1| off-diagonal,
       checked against a direct calculation.
    4. Config: E5 only at its horizons, one graph file per horizon;
       other experiments at any horizon; horizon > 1 needs pre_len 1.
    5. main.load_graph refuses an E5 graph built for another horizon.

Run from the implementation directory:

    python -m tests.test_horizon_phase7
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

import numpy as np

from causal.graph_builders import horizon_graphs, save_graph_artifact, top_k_sources
from config.config import FINAL_GRAPHS, GRAPH_DIR, ExperimentConfig
from data.windows import gather_windows, valid_window_starts, valid_window_starts_from_rows


def expect_error(error, function, *args):
    try:
        function(*args)
    except error as exc:
        return str(exc)
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


def test_windows():

    data = np.arange(60, dtype=np.float32).reshape(30, 2)

    X, y = gather_windows(data, np.array([0, 5]), 4, 1, target_offset=2)
    assert np.array_equal(X[1, :, 0], data[5:9, 0])
    assert np.array_equal(y[:, 0, 0], data[[6, 11], 0])            # s + 4 + 2

    X0, y0 = gather_windows(data, np.array([3]), 4, 2)              # offset 0
    assert np.array_equal(y0[0], data[7:9])

    ok = np.ones(30, dtype=bool)
    ok[12] = False
    starts = valid_window_starts_from_rows(ok, ok, 4, 1, target_offset=5)
    # window length 10 -> starts 0..20; inputs s..s+3 avoid 12, target s+9 != 12
    expected = [s for s in range(21) if not (s <= 12 <= s + 3) and s + 9 != 12]
    assert starts.tolist() == expected

    rng = np.random.default_rng(0)
    gappy = rng.normal(size=(200, 3)); gappy[rng.random(200) < 0.05, 1] = np.nan
    rows = np.all(np.isfinite(gappy), axis=1)
    assert np.array_equal(valid_window_starts_from_rows(rows, rows, 12, 1),
                          valid_window_starts(gappy, 12, 1))

    print("[PASS] 1. horizon windows: target h steps ahead, only inputs + target checked")


def test_top_k():

    e = np.array([[9.0, 1.0, 3.0, 2.0],
                  [5.0, 0.0, -4.0, 1.0],
                  [1.0, 1.0, 0.0, 1.0],
                  [0.0, 2.0, 3.0, 7.0]])
    g = top_k_sources(e, 2)

    assert np.all(np.diag(g) == 0) and np.all((g != 0).sum(axis=1) == 2)
    assert set(np.flatnonzero(g[0])) == {2, 3} and np.isclose(g[1, 2], 4.0)
    assert set(np.flatnonzero(g[2])) == {0, 1}                      # tie -> lower index
    print("[PASS] 2. top-k sources per target, diagonal 0")


def test_horizon_graphs():

    rng = np.random.default_rng(1)
    n, lags = 5, 2
    A = rng.normal(scale=0.2, size=(lags, n, n))
    B0 = np.zeros((n, n)); B0[1, 0] = 0.6; B0[3, 2] = -0.5
    P = np.full((n, n), 0.8)

    graphs = horizon_graphs(A, B0, P, [1, 3], k=2)

    psi = [np.eye(n)]
    for s in range(1, 4):
        psi.append(sum(A[k - 1] @ psi[s - k] for k in range(1, min(s, lags) + 1)))
    inv = np.linalg.inv(np.eye(n) - P * B0)

    for h in (1, 3):
        expected = top_k_sources(np.abs(psi[h] @ inv), 2)
        assert np.allclose(graphs[h], expected, atol=1e-6), h

    print("[PASS] 3. E5 graphs = top-k |Psi_h (I - P*B_0)^-1|")


def test_config():

    message = expect_error(ValueError, ExperimentConfig(experiment="E5", horizon=2).validate)
    assert "horizons" in message

    config = ExperimentConfig(experiment="E5", horizon=6)
    if (GRAPH_DIR / FINAL_GRAPHS["E5"].format(h=6)).exists():
        config.validate()
    else:
        expect_error(FileNotFoundError, config.validate)
    assert FINAL_GRAPHS["E5"].format(h=6).startswith("E5__h6__")

    ExperimentConfig(experiment="E0", horizon=12).validate()        # comparison runs
    expect_error(ValueError, ExperimentConfig(experiment="E0", horizon=3, pre_len=2).validate)
    expect_error(ValueError, ExperimentConfig(experiment="E0", horizon=0).validate)

    print("[PASS] 4. E5 only at its horizons (one file each); others at any horizon")


def test_load_graph_horizon():

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from implementation.main import load_graph

    n = 4
    sensors = np.arange(n)
    graph = np.zeros((n, n), dtype=np.float32)
    graph[1, 0] = graph[2, 1] = 1.0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "E5__h3__test.npz"
        save_graph_artifact(path, graph, experiment="E5", train_end=100, sensors=sensors,
                            method="var_lingam_pem_horizon", horizon=3)

        right = ExperimentConfig(experiment="E5", graph_path=str(path), horizon=3)
        _, info = load_graph(right, np.zeros((n, n)), 100, sensors)
        assert info["method"] == "var_lingam_pem_horizon"

        wrong = ExperimentConfig(experiment="E5", graph_path=str(path), horizon=6)
        assert "horizon 3" in expect_error(ValueError, load_graph, wrong, np.zeros((n, n)), 100, sensors)

    print("[PASS] 5. load_graph refuses an E5 graph of another horizon")


def main():

    print("=" * 70)
    print("PHASE 7: HORIZON WINDOWS AND E5 GRAPHS")
    print("=" * 70)

    test_windows()
    test_top_k()
    test_horizon_graphs()
    test_config()
    test_load_graph_horizon()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
