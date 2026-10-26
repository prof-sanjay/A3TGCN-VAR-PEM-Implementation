"""
Tests for the Phase 10 reporting scripts:

    1. compare_results: fixed colour per experiment (also with a horizon
       suffix), neutral colour for baselines, never by position.
    2. figure_error_by_horizon: table rows / columns and the figure files.
    3. plot_graphs: summary and Jaccard on a known pair of graphs, and
       all figure / table files written for a real graph artifact.

Run from the implementation directory:

    python -m tests.test_reports
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess
import tempfile

import numpy as np

import compare_results as cr
import plot_graphs as pg
from config.config import FINAL_GRAPHS, GRAPH_DIR


IMPLEMENTATION_DIR = Path(__file__).resolve().parent.parent


def test_colours():

    assert cr.color_of("E5", 0) == cr.color_of("E5@15min", 7) == cr.EXPERIMENT_COLORS["E5"]
    assert cr.color_of("NOGRAPH", 0) == cr.color_of("HA", 3) == cr.BASELINE_COLOR
    assert len(set(cr.EXPERIMENT_COLORS.values())) == 8
    print("[PASS] 1. fixed colour per experiment, baselines neutral")


def test_error_by_horizon():

    runs = [{"label": f"{e}@{5 * h}min", "horizon": h,
             "results": {"test_metrics": {"rmse": 30.0 + h, "mae": 20.0 + h}}}
            for e in ("E0", "E5") for h in (1, 3, 12)]

    with tempfile.TemporaryDirectory() as tmp:
        table = cr.figure_error_by_horizon(runs, Path(tmp), Path(tmp))
        assert sorted(table["horizon_minutes"].unique()) == [5, 15, 60]
        assert set(table["experiment"]) == {"E0", "E5"}
        assert (Path(tmp) / "10_error_by_horizon.png").exists()
        assert (Path(tmp) / "error_by_horizon.csv").exists()

    print("[PASS] 2. error-by-horizon table and figure")


def test_graph_report():

    a = np.zeros((4, 4)); a[1, 0] = a[2, 1] = a[3, 2] = 1.0
    b = np.zeros((4, 4)); b[1, 0] = b[0, 1] = 0.5
    summary = pg.summary_table({"A": a, "B": b})
    assert list(summary["edges"]) == [3, 2]
    assert list(summary["reciprocal_pairs"]) == [0, 1]

    jaccard = pg.jaccard_table({"A": a, "B": b})
    assert np.isclose(jaccard.loc["A", "B"], 1 / 4)     # one shared edge of four

    e1 = GRAPH_DIR / FINAL_GRAPHS["E1"]
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, str(IMPLEMENTATION_DIR / "plot_graphs.py"),
             "--graphs", f"E1={e1}", "--out", tmp],
            capture_output=True, text=True, cwd=IMPLEMENTATION_DIR,
        )
        assert result.returncode == 0, result.stderr[-1500:]
        for name in ("11_graph_map_subset", "12_graph_in_degree", "13_graph_overlap",
                     "14_graph_size", "15_graph_weights"):
            assert (Path(tmp) / "figures" / f"{name}.png").exists(), name
        for name in ("graph_summary.csv", "graph_jaccard.csv", "map_subset_sensors.csv"):
            assert (Path(tmp) / "tables" / name).exists(), name

    print("[PASS] 3. graph summary / overlap values and all graph figures written")


def main():

    print("=" * 70)
    print("REPORTING TESTS")
    print("=" * 70)

    test_colours()
    test_error_by_horizon()
    test_graph_report()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
