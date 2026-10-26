"""
Stage A report for PEM (Phase 4): one table over every sweep file, the
same diagnostics for every setting, and the shortlist for Stage B.

Diagnostics per setting (from the per-unit results on disk; TRAINING
residuals only):

    edges_per_unit       mean directed parent edges per local unit
    directed_jaccard     split-half stability of the directed edges
    undirected_jaccard   split-half stability ignoring direction
    lowvar_to_highvar    share of directed edges pointing from the sensor
                         with the smaller raw residual variance to the
                         larger one (0.5 = unrelated to variance). PEM's
                         ordering assumes comparable error variances; a
                         high share means directions mostly follow
                         sensor noise levels ("varsortability").
    road_link_share      share of directed edges between sensors that
                         are directly linked in the road graph
    chance_jaccard       directed Jaccard expected for two random edge
                         sets of the same local density d: d / (2 - d)
    stability            chance-corrected directed Jaccard:
                         (directed_jaccard - chance) / (1 - chance)

Original selection rule (fixed on 2026-10-25, before the sweeps ran):

    1. no failed units, and at least 10 edges per unit on average
       (excludes trivially sparse graphs, whose overlap is meaningless);
    2. lowvar_to_highvar <= 0.60 (directions not mainly a variance
       artefact); if no setting passes, step 3 uses all settings from
       step 1 and the shortlist is flagged;
    3. shortlist the 2 settings with the highest directed_jaccard
       (ties: higher undirected_jaccard).

AMENDMENT (2026-10-25, after the sweeps, approved by the author; to be
reported as a post-hoc correction). Raw Jaccard rises with graph
density even for random graphs (two random sets of density d overlap
by about d / (2 - d)), so the original rule mechanically favoured the
densest settings. After the correction all standardised settings were
about equally stable (~0.31). Amended rule:

    1-2. as above;
    3.   only settings matching E3's design: uniform prior (the road
         prior belongs to E7) and threshold 0.5 (PEM's median
         probability model; other thresholds are sensitivity checks);
    4.   shortlist the 2 settings with the highest `stability`
         (ties: higher undirected_jaccard).

The final choice between the shortlisted settings is made by
validation training (Stage B), not by stability. Both shortlists are
written to the report.

Run from the implementation directory:

    python selection/pem_stage_a_report.py [--var-run lag3_alpha1e+06]
"""

import sys
from pathlib import Path

# Make the implementation folder importable (run from implementation/:
# python selection/<file>.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from config.config import VAR_RUN, ExperimentConfig
from run_local_pem import residual_scaling


IMPLEMENTATION_DIR = Path(__file__).resolve().parent.parent
PEM_ROOT = IMPLEMENTATION_DIR / "outputs" / "pem"
VAR_ROOT = IMPLEMENTATION_DIR / "outputs" / "var_train"

MIN_EDGES_PER_UNIT = 10
MAX_LOWVAR_SHARE = 0.60
SHORTLIST = 2


def unit_edges(run_dir):
    """Directed global edges (source, target) and per-unit edge counts."""

    edges = set()
    counts = []

    for path in sorted(Path(run_dir, "units").glob("unit_*.npz")):
        with np.load(path) as unit:
            nodes = unit["nodes"]
            targets, sources = np.nonzero(unit["adjacency"] > 0)
            edges.update(zip(nodes[sources].tolist(), nodes[targets].tolist()))
            counts.append(len(targets))

    return edges, np.array(counts)


def undirected(edges):
    return {frozenset(e) for e in edges}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def half_dirs(record):

    if "half_dirs" in record:
        return [Path(d) for d in record["half_dirs"]]

    found = sorted(
        glob.glob(glob.escape(record["run_dir"]) + "__rows*"),
        key=lambda d: int(d.rsplit("__rows", 1)[1].split("-")[0]),
    )
    return [Path(d) for d in found]


def residual_variance(var_run):

    cache = VAR_ROOT / var_run / "residual_std.npy"

    if cache.exists():
        std = np.load(cache)
    else:
        _, std = residual_scaling(np.load(VAR_ROOT / var_run / "residuals.npy", mmap_mode="r"))
        np.save(cache, std)

    return std ** 2


def chance_corrected(directed_jaccard, density):
    """Directed Jaccard corrected for the overlap of random graphs."""

    chance = density / (2.0 - density)
    return chance, (directed_jaccard - chance) / (1.0 - chance)


def select_shortlist_amended(rows):
    """Amended rule; returns (shortlist, direction_rule_unmet)."""

    valid = [r for r in rows
             if r["failed_units"] == 0 and r["edges_per_unit"] >= MIN_EDGES_PER_UNIT
             and np.isfinite(r["stability"])]

    passing = [r for r in valid if r["lowvar_to_highvar"] <= MAX_LOWVAR_SHARE]
    flagged = not passing
    pool = [r for r in (passing if passing else valid)
            if r["prior"] == "uniform" and r["threshold"] == 0.5]

    shortlist = sorted(
        pool, key=lambda r: (-r["stability"], -r["undirected_jaccard"])
    )[:SHORTLIST]

    return shortlist, flagged


def select_shortlist(rows):
    """ORIGINAL rule; returns (shortlist, direction_rule_unmet)."""

    valid = [r for r in rows
             if r["failed_units"] == 0 and r["edges_per_unit"] >= MIN_EDGES_PER_UNIT
             and np.isfinite(r["directed_jaccard"])]

    passing = [r for r in valid if r["lowvar_to_highvar"] <= MAX_LOWVAR_SHARE]
    flagged = not passing
    pool = passing if passing else valid

    shortlist = sorted(
        pool, key=lambda r: (-r["directed_jaccard"], -r["undirected_jaccard"])
    )[:SHORTLIST]

    return shortlist, flagged


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--var-run", default=VAR_RUN)
    args = parser.parse_args()

    variance = residual_variance(args.var_run)
    sensors = np.load(VAR_ROOT / args.var_run / "sensors.npy")
    road = np.load(ExperimentConfig().adjacency_path, mmap_mode="r")
    road = np.asarray(road[np.ix_(sensors, sensors)]) != 0
    np.fill_diagonal(road, False)
    road = road | road.T

    rows = []

    for sweep_file in sorted(PEM_ROOT.glob("sweep_stage_a*.json")):
        for record in json.loads(sweep_file.read_text(encoding="utf-8")):

            if not Path(record["run_dir"]).exists():
                continue

            edges, counts = unit_edges(record["run_dir"])
            halves = half_dirs(record)
            if len(halves) == 2:
                (e1, _), (e2, _) = unit_edges(halves[0]), unit_edges(halves[1])
            else:
                e1 = e2 = set()

            sources = np.array([s for s, _ in edges], dtype=np.int64)
            targets = np.array([t for _, t in edges], dtype=np.int64)

            directed_j = jaccard(e1, e2) if e1 or e2 else float("nan")
            chance, stability = chance_corrected(directed_j, record["local_density"])

            rows.append({
                "sweep": sweep_file.name,
                "standardise": bool(record.get("standardise", False)),
                "prior": record.get("prior", "uniform"),
                "nu0": record["nu0"],
                "nu1": record["nu1"],
                "threshold": record["threshold"],
                "failed_units": record["failed_units"],
                "edges_per_unit": float(counts.mean()) if counts.size else 0.0,
                "local_density": record["local_density"],
                "directed_jaccard": directed_j,
                "chance_jaccard": chance,
                "stability": stability,
                "undirected_jaccard": jaccard(undirected(e1), undirected(e2)) if e1 or e2 else float("nan"),
                "lowvar_to_highvar": float(np.mean(variance[sources] < variance[targets])) if edges else float("nan"),
                "road_link_share": float(np.mean(road[targets, sources])) if edges else float("nan"),
                "cpu_seconds": record.get("cpu_seconds"),
                "run_dir": record["run_dir"],
            })

    if not rows:
        raise SystemExit("No sweep results found.")

    original, original_flagged = select_shortlist(rows)
    shortlist, flagged = select_shortlist_amended(rows)

    for r in rows:
        r["shortlisted"] = r in shortlist
        r["shortlisted_original_rule"] = r in original

    header = ["std", "prior", "nu0", "nu1", "thr", "edges/u", "dirJ", "stab.",
              "undJ", "low->high", "road", ""]
    print(("{:>5} {:>8} {:>8} {:>8} {:>4} {:>7} {:>6} {:>6} {:>6} {:>9} {:>6} {}").format(*header))
    for r in sorted(rows, key=lambda r: (r["standardise"], r["prior"], r["nu0"], r["nu1"], r["threshold"])):
        mark = ("<- shortlist" if r["shortlisted"] else "") + (
            " (original rule)" if r["shortlisted_original_rule"] else "")
        print("{:>5} {:>8} {:>8.2g} {:>8.2g} {:>4} {:>7.1f} {:>6.3f} {:>6.3f} {:>6.3f} {:>9.3f} {:>6.3f} {}".format(
            "yes" if r["standardise"] else "no", r["prior"], r["nu0"], r["nu1"], r["threshold"],
            r["edges_per_unit"], r["directed_jaccard"], r["stability"], r["undirected_jaccard"],
            r["lowvar_to_highvar"], r["road_link_share"], mark))

    if flagged:
        print(f"\nNo setting has low->high share <= {MAX_LOWVAR_SHARE}: shortlist taken from "
              "all valid settings; edge directions remain variance-driven (limitation).")

    out = PEM_ROOT / "stage_a_report.json"
    out.write_text(json.dumps({
        "rule": {"min_edges_per_unit": MIN_EDGES_PER_UNIT,
                 "max_lowvar_to_highvar": MAX_LOWVAR_SHARE, "shortlist": SHORTLIST},
        "rule_amendment": "2026-10-25: rank by chance-corrected stability; uniform prior and "
                          "threshold 0.5 only (see module docstring)",
        "direction_rule_unmet": flagged,
        "rows": rows,
        "shortlist": shortlist,
        "shortlist_original_rule": original,
        "direction_rule_unmet_original_rule": original_flagged,
    }, indent=2), encoding="utf-8")

    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
