"""
Tests for the data-driven candidate links (Phase 6, E6) and the E4 / E6
/ E7 input checks:

    1. difference_correlation = correlation of first differences
       (exact without gaps; close to pairwise correlation with gaps).
    2. correlation_neighbours: exactly k sources per target, no
       self-links, the strongest |corr| chosen, ties by lower index.
    3. load_candidates: road graph passes through; a saved corr_k
       artifact loads only for its own sensors / training split.
    4. build_graph.check_inputs: E4 / E6 / E7 accept only inputs on
       their candidates and prior, from one VAR run.

Run from the implementation directory:

    python -m tests.test_candidates
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

import numpy as np
import pandas as pd

import causal.candidates as candidates
from build_graph import check_inputs


def expect_error(error, function, *args):
    try:
        function(*args)
    except error as exc:
        return str(exc)
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


def test_correlation():

    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(size=(3000, 1)), axis=0)
    data = base + 0.3 * np.cumsum(rng.normal(size=(3000, 6)), axis=0)

    corr = candidates.difference_correlation(data)
    assert np.allclose(corr, np.corrcoef(np.diff(data, axis=0), rowvar=False), atol=1e-5)

    gappy = data.copy()
    gappy[rng.random(gappy.shape) < 0.01] = np.nan
    pairwise = pd.DataFrame(np.diff(gappy, axis=0)).corr().to_numpy()
    assert np.allclose(candidates.difference_correlation(gappy), pairwise, atol=0.02)

    print("[PASS] 1. correlation of first differences (exact without gaps, close with gaps)")


def test_neighbours():

    corr = np.array([
        [1.0, 0.9, -0.95, 0.1],
        [0.9, 1.0, 0.2, 0.2],
        [-0.95, 0.2, 1.0, 0.5],
        [0.1, 0.2, 0.5, 1.0],
    ])
    c = candidates.correlation_neighbours(corr, 2)

    assert np.all((c != 0).sum(axis=1) == 2) and np.all(np.diag(c) == 0)
    assert np.isclose(c[0, 2], 0.95) and np.isclose(c[0, 1], 0.9)   # weights = |corr|
    assert set(np.flatnonzero(c[0])) == {1, 2}          # |-0.95| counts
    assert set(np.flatnonzero(c[1])) == {0, 2}          # tie 0.2 / 0.2 -> lower index 2
    assert set(np.flatnonzero(c[3])) == {1, 2}

    expect_error(ValueError, candidates.correlation_neighbours, corr, 4)
    print("[PASS] 2. top-k |corr| per target, no self-links, ties by lower index")


def test_unit_cap():

    from causal.causal_units import build_causal_units

    # Sensor 0 is a source ("hub") of every other sensor; sensor 5 has
    # sources 1 and 2.
    n = 8
    c = np.zeros((n, n), dtype=np.float32)
    c[1:, 0] = np.linspace(0.1, 0.8, n - 1)      # t -> uses 0, strength rising
    c[5, 1], c[5, 2] = 0.3, 0.2

    uncapped = build_causal_units(c)
    capped = build_causal_units(c, **candidates.unit_options("corr_k3"))

    assert len(uncapped[0].nodes) == n                       # hub unit: everyone
    assert set(capped[0].nodes) == {0, 5, 6, 7}              # 3 strongest users of 0
    assert set(capped[5].nodes) == set(uncapped[5].nodes) == {0, 1, 2, 5}  # own sources kept
    assert candidates.unit_options("road") == {}
    print("[PASS] 2b. hub cap: reverse neighbours limited to the strongest k, own sources kept")


def test_load():

    road = np.eye(3, dtype=np.float32)
    sensors = np.array([5, 7, 9])

    assert candidates.load_candidates("road", sensors, 100, road) is not None
    assert np.array_equal(candidates.load_candidates("road", sensors, 100, road), road)

    original = candidates.CANDIDATE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        candidates.CANDIDATE_DIR = Path(tmp)
        try:
            adjacency = np.ones((3, 3), dtype=np.float32)
            candidates.save_candidates("corr_k2", adjacency, sensors, 100, 2)

            loaded = candidates.load_candidates("corr_k2", sensors, 100, road)
            assert np.array_equal(loaded, adjacency)

            expect_error(ValueError, candidates.load_candidates, "corr_k2", sensors, 101, road)
            expect_error(ValueError, candidates.load_candidates, "corr_k2", sensors[::-1], 100, road)
            expect_error(FileNotFoundError, candidates.load_candidates, "corr_k9", sensors, 100, road)
            expect_error(ValueError, candidates.load_candidates, "nearest", sensors, 100, road)
        finally:
            candidates.CANDIDATE_DIR = original

    assert candidates.candidate_tag("road") == "" and candidates.candidate_tag("corr_k25") == "__cand-corr_k25"
    print("[PASS] 3. candidates load only for their own sensors / split; road passes through")


def test_input_checks():

    def parts(cand, prior, var_run="v", pem_var_run="v"):
        return ({"candidates": cand}, {"candidates": cand, "var_run": var_run},
                {"candidates": cand, "prior": prior, "var_run": pem_var_run})

    check_inputs("E4", *parts("road", "uniform"))
    check_inputs("E6", *parts("corr_k25", "uniform"))
    check_inputs("E7", *parts("road", "road"))

    expect_error(ValueError, check_inputs, "E4", *parts("road", "road"))
    expect_error(ValueError, check_inputs, "E6", *parts("road", "uniform"))
    expect_error(ValueError, check_inputs, "E7", *parts("road", "uniform"))
    expect_error(ValueError, check_inputs, "E4", *parts("road", "uniform", pem_var_run="other"))

    mixed = ({"candidates": "corr_k25"}, {"candidates": "road", "var_run": "v"},
             {"candidates": "corr_k25", "prior": "uniform", "var_run": "v"})
    expect_error(ValueError, check_inputs, "E6", *mixed)

    print("[PASS] 4. E4 / E6 / E7 accept only their candidates and prior, one VAR run")


def main():

    print("=" * 70)
    print("CANDIDATE LINKS / E6 / E7 TESTS")
    print("=" * 70)

    test_correlation()
    test_neighbours()
    test_unit_cap()
    test_load()
    test_input_checks()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
