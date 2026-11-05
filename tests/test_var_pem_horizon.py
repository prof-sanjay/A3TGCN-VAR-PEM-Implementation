"""
Tests of E8 (horizon-conditioned VAR-PEM, no LiNGAM), causal/var_pem_horizon.py.

Stage 1  impulse responses equal an independent calculation (companion matrix)
Stage 2  P (x) |Psi_h|: element-wise, diagonal 0, no B0
Stage 3  graphs: top-25 rule, orientation, support inside P's support
Stage 4  E8V ablation: |Psi_h| only; equals the existing E5 builder when B0 = 0
Guard    nothing LiNGAM-related is imported

Run from the implementation directory:

    python -m pytest tests/test_var_pem_horizon.py -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from causal.graph_builders import horizon_graphs
from causal.var_pem_horizon import (
    graph_diagnostics,
    impulse_responses,
    lingam_access_report,
    probability_diagnostics,
    var_horizon_graphs,
    var_pem_horizon_graphs,
    weighted_effects,
)


def _var(n=14, p=3, seed=5, scale=0.12):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(p, n, n))


def _probabilities(n=14, seed=6, support=0.4):
    """Symmetric, zero outside a random support, mostly ~1 inside (like PEM's P)."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < support
    mask = np.triu(mask, 1)
    p = np.where(mask, np.where(rng.random((n, n)) < 0.9, 1.0, rng.random((n, n))), 0.0)
    return p + p.T


# ---- Stage 1 ---------------------------------------------------------------

def test_impulse_responses_equal_the_companion_matrix_power():
    """Psi_s is the top-left n x n block of C^s for the VAR(p) companion matrix C."""

    a = _var()
    p, n, _ = a.shape

    companion = np.zeros((n * p, n * p))
    companion[:n, :] = np.concatenate(list(a), axis=1)
    companion[n:, :-n] = np.eye(n * (p - 1))

    psi = impulse_responses(a, [1, 2, 3, 6, 12])

    for h, matrix in psi.items():
        expected = np.linalg.matrix_power(companion, h)[:n, :n]
        assert np.allclose(matrix, expected, atol=1e-10), h


def test_first_horizon_is_the_lag_one_matrix_and_orientation_is_target_source():
    a = _var()
    psi = impulse_responses(a, [1, 2])

    assert np.allclose(psi[1], a[0])                         # Psi_1 = A_1
    assert np.allclose(psi[2], a[0] @ a[0] + a[1])           # Psi_2 = A_1 Psi_1 + A_2 Psi_0

    # A source j that feeds target i at lag 1 only: Psi_1[i, j] != 0, Psi_1[j, i] == 0.
    b = np.zeros((1, 4, 4))
    b[0, 2, 1] = 0.7
    psi_b = impulse_responses(b, [1])
    assert psi_b[1][2, 1] == pytest.approx(0.7) and psi_b[1][1, 2] == 0.0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        impulse_responses(np.zeros((3, 4)), [1])
    with pytest.raises(ValueError):
        impulse_responses(_var(), [0])
    bad = _var(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        impulse_responses(bad, [1])


# ---- Stage 2 ---------------------------------------------------------------

def test_weighted_effects_are_elementwise_abs_and_zero_diagonal():
    a, p = _var(), _probabilities()
    psi = impulse_responses(a, [1, 3, 6])
    effects = weighted_effects(psi, p)

    for h in (1, 3, 6):
        expected = p * np.abs(psi[h])
        np.fill_diagonal(expected, 0.0)
        assert np.allclose(effects[h], expected)
        assert np.all(np.diag(effects[h]) == 0.0)
        assert effects[h].min() >= 0.0


def test_probabilities_are_validated():
    psi = impulse_responses(_var(), [1])
    with pytest.raises(ValueError):
        weighted_effects(psi, np.ones((3, 3)))
    with pytest.raises(ValueError):
        weighted_effects(psi, -np.ones((14, 14)))
    with pytest.raises(ValueError):
        weighted_effects(psi, 2 * np.ones((14, 14)))


# ---- Stage 3 ---------------------------------------------------------------

def test_graphs_keep_top_k_sources_per_target_inside_p_support():
    n, k = 40, 5
    a, p = _var(n=n), _probabilities(n=n, support=0.5)
    psi, effects, graphs = var_pem_horizon_graphs(a, p, [1, 3, 6, 12], k)

    for h, g in graphs.items():
        assert g.shape == (n, n) and g.dtype == np.float32
        assert np.all(np.diag(g) == 0)
        assert ((g != 0).sum(axis=1) <= k).all()
        assert not np.any((g != 0) & (p == 0))                 # never outside P's support

        # each row keeps exactly the k largest effects (ties aside)
        for i in range(n):
            kept = np.sort(g[i][g[i] != 0])[::-1]
            best = np.sort(effects[h][i])[::-1][:len(kept)]
            assert np.allclose(kept, best.astype(np.float32), atol=1e-6)


def test_rows_are_targets_not_sources():
    """G[target, source]: planting one strong source gives a row (target) entry."""

    n = 30
    a = np.zeros((1, n, n)); a[0, 7, 3] = 0.9                  # source 3 -> target 7
    p = np.ones((n, n))
    _, _, graphs = var_pem_horizon_graphs(a, p, [1], 2)

    assert graphs[1][7, 3] > 0 and graphs[1][3, 7] == 0


# ---- Stage 4 ---------------------------------------------------------------

def test_e8v_is_the_existing_e5_builder_with_b0_zero():
    """Independent check against causal.graph_builders.horizon_graphs (E5 code, unchanged)."""

    n, k, horizons = 40, 6, [1, 3, 6, 12]
    a = _var(n=n)

    _, _, mine = var_horizon_graphs(a, horizons, k)
    e5_code = horizon_graphs(a, np.zeros((n, n)), np.ones((n, n)), horizons, k)

    for h in horizons:
        assert np.array_equal(mine[h], e5_code[h]), h


def test_e8_with_p_all_ones_equals_e8v():
    n, k, horizons = 40, 6, [1, 6]
    a = _var(n=n)

    _, _, with_p = var_pem_horizon_graphs(a, np.ones((n, n)), horizons, k)
    _, _, without = var_horizon_graphs(a, horizons, k)

    for h in horizons:
        assert np.array_equal(with_p[h], without[h])


# ---- diagnostics and guard ---------------------------------------------------

def test_diagnostics():
    n = 30
    a, p = _var(n=n), _probabilities(n=n)
    _, _, graphs = var_pem_horizon_graphs(a, p, [1], 4)

    pd = probability_diagnostics(p)
    assert pd["shape"] == [n, n] and pd["symmetric"]
    assert 0 <= pd["percent_zero"] <= 100 and pd["percent_above_0.99"] <= pd["percent_above_0.5"]

    gd = graph_diagnostics(graphs[1])
    assert gd["edges"] == int((graphs[1] != 0).sum())
    assert gd["max_in_degree"] <= 4 and gd["self_loops"] == 0


def test_no_lingam_module_is_imported_by_e8():
    code = ("import sys; import causal.var_pem_horizon as m; "
            "bad=[x for x in sys.modules if 'lingam' in x.lower()]; "
            "assert not bad, bad; print('ok')")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert lingam_access_report(["outputs/var_train/x/coefficients.npy", "pem_result.npz"])["ok"]
    assert not lingam_access_report(["outputs/lingam/lingam_result.npz"])["ok"]
    assert not lingam_access_report(["data/b0.npy"])["ok"]

    # The project folder's own name contains "VAR-LiNGAM": that must not count.
    project = "C:\\My Folder\\Projects\\Horizon-Conditioned Probabilistic VAR-LiNGAM Causal Graph\\implementation"
    assert lingam_access_report([project + "\\outputs\\var_train\\lag3_alpha1e+06\\coefficients.npy"])["ok"]
    assert not lingam_access_report([project + "\\outputs\\lingam\\lag3_alpha1e+06\\lingam_result.npz"])["ok"]
    assert not lingam_access_report(["/kaggle/input/pem-inputs/lingam_result.npz"])["ok"]


def test_artifacts_load_through_main_and_mismatches_are_refused(tmp_path):
    """The saved files are accepted by main.load_graph; wrong horizon / method are refused."""

    sys.path.insert(0, str(ROOT.parent))
    from implementation.main import load_graph
    from causal.var_pem_horizon import save_artifact
    from config.config import ExperimentConfig
    from experiments.registry import get_experiment

    n, k = 30, 4
    a, p = _var(n=n), _probabilities(n=n, support=0.6)
    _, _, g8 = var_pem_horizon_graphs(a, p, [1, 3], k)
    _, _, gv = var_horizon_graphs(a, [1, 3], k)

    sensors = np.arange(n, dtype=np.int64)
    road = np.zeros((n, n), dtype=np.float32)
    paths = {}
    for name, graphs in (("E8", g8), ("E8V", gv)):
        for h in (1, 3):
            path = tmp_path / f"{name}_h{h}.npz"
            save_artifact(path, graphs[h], experiment=name, method=get_experiment(name).graph_method,
                          train_end=500, sensors=sensors, horizon=h, top_k=k, source_run="test")
            paths[(name, h)] = path

    config = ExperimentConfig(experiment="E8", horizon=3, graph_path=str(paths[("E8", 3)]))
    adjacency, info = load_graph(config, road, 500, sensors)
    assert info["source"] == "E8_artifact" and info["method"] == "var_pem_horizon"
    assert np.array_equal(adjacency, g8[3])                                # stored [target, source], unchanged

    config = ExperimentConfig(experiment="E8V", horizon=1, graph_path=str(paths[("E8V", 1)]))
    assert load_graph(config, road, 500, sensors)[1]["method"] == "var_horizon"

    with pytest.raises(ValueError, match="horizon"):                       # an h=1 graph for an h=3 run
        load_graph(ExperimentConfig(experiment="E8", horizon=3, graph_path=str(paths[("E8", 1)])),
                   road, 500, sensors)
    with pytest.raises(ValueError, match="method"):                        # E8V graph used as E8
        load_graph(ExperimentConfig(experiment="E8", horizon=3, graph_path=str(paths[("E8V", 3)])),
                   road, 500, sensors)
    with pytest.raises(ValueError):                                        # wrong training split
        load_graph(ExperimentConfig(experiment="E8", horizon=3, graph_path=str(paths[("E8", 3)])),
                   road, 999, sensors)
    with pytest.raises(ValueError, match="horizon"):                       # no graphs exist for h = 2
        ExperimentConfig(experiment="E8", horizon=2, graph_path=str(paths[("E8", 3)])).validate()


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
