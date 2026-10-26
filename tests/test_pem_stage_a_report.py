"""
Tests for pem_stage_a_report.py (Phase 4):

    1. unit_edges maps local [target, source] edges to global
       (source, target) pairs; Jaccard directed vs undirected.
    2. half_dirs finds the split-half runs in row order.
    3. Original selection rule: invalid / too sparse settings excluded;
       the direction check filters; shortlist by directed Jaccard;
       flagged fallback when no setting passes the direction check.
    4. Amended rule: chance-corrected stability; a dense setting with a
       higher raw Jaccard but lower corrected stability loses; road
       prior and thresholds other than 0.5 are not candidates.

Run from the implementation directory:

    python -m tests.test_pem_stage_a_report
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import tempfile

import numpy as np

from selection.pem_stage_a_report import (
    chance_corrected,
    half_dirs,
    jaccard,
    select_shortlist,
    select_shortlist_amended,
    undirected,
    unit_edges,
)


def write_unit(run_dir, index, nodes, local_edges):

    p = len(nodes)
    adjacency = np.zeros((p, p))
    for source, target in local_edges:
        adjacency[target, source] = 1.0          # [target, source]
    (Path(run_dir) / "units").mkdir(parents=True, exist_ok=True)
    np.savez(Path(run_dir) / "units" / f"unit_{index:05d}.npz",
             nodes=np.asarray(nodes), adjacency=adjacency)


def row(name, dirJ, undJ=0.5, low=0.5, edges=50.0, failed=0):
    return {"name": name, "directed_jaccard": dirJ, "undirected_jaccard": undJ,
            "lowvar_to_highvar": low, "edges_per_unit": edges, "failed_units": failed}


def main():

    print("=" * 70)
    print("STAGE A REPORT TESTS")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:

        a = Path(tmp) / "a"
        write_unit(a, 0, [10, 20, 30], [(0, 1), (1, 2)])     # 10->20, 20->30
        write_unit(a, 1, [20, 40], [(1, 0)])                 # 40->20
        edges, counts = unit_edges(a)
        assert edges == {(10, 20), (20, 30), (40, 20)}
        assert list(counts) == [2, 1]

        b = {(20, 10), (20, 30)}                            # one edge reversed
        assert math.isclose(jaccard(edges, b), 1 / 4)
        assert math.isclose(jaccard(undirected(edges), undirected(b)), 2 / 3)
        print("[PASS] 1. local -> global edges; directed vs undirected Jaccard")

        run = Path(tmp) / "tag__sample60_seed42"
        for rows in ("42048-84096", "0-42048"):
            Path(f"{run}__rows{rows}").mkdir(parents=True)
        found = half_dirs({"run_dir": str(run)})
        assert [d.name.rsplit("__rows", 1)[1] for d in found] == ["0-42048", "42048-84096"]
        print("[PASS] 2. split-half runs found in row order")

    rows = [
        row("best_but_scale_driven", 0.60, low=0.72),
        row("good_a", 0.45, low=0.55),
        row("good_b", 0.44, undJ=0.60, low=0.50),
        row("good_c", 0.44, undJ=0.55, low=0.58),
        row("too_sparse", 0.90, edges=5.0, low=0.5),
        row("failed", 0.95, failed=1, low=0.5),
    ]
    shortlist, flagged = select_shortlist(rows)
    assert [r["name"] for r in shortlist] == ["good_a", "good_b"] and not flagged

    shortlist, flagged = select_shortlist([row("x", 0.40, low=0.7), row("y", 0.42, low=0.75)])
    assert [r["name"] for r in shortlist] == ["y", "x"] and flagged
    print("[PASS] 3. selection rule: exclusions, direction filter, ranking, flagged fallback")

    chance, stability = chance_corrected(0.475, 0.380)
    assert math.isclose(chance, 0.380 / 1.620)
    assert math.isclose(stability, (0.475 - chance) / (1 - chance))

    def amended(name, dirJ, density, prior="uniform", thr=0.5, low=0.47):
        r = row(name, dirJ, low=low)
        r.update(prior=prior, threshold=thr, stability=chance_corrected(dirJ, density)[1])
        return r

    rows = [
        amended("dense", 0.48, 0.38),          # stability ~0.32
        amended("sparse", 0.36, 0.06),         # stability ~0.33
        amended("middle", 0.40, 0.27),         # stability ~0.27
        amended("road", 0.60, 0.06, prior="road"),
        amended("threshold", 0.60, 0.06, thr=0.7),
        amended("scale_driven", 0.60, 0.06, low=0.72),
    ]
    shortlist, flagged = select_shortlist_amended(rows)
    assert [r["name"] for r in shortlist] == ["sparse", "dense"] and not flagged
    print("[PASS] 4. amended rule: chance-corrected stability, uniform prior and threshold 0.5 only")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
