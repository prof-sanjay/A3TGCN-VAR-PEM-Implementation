"""
Saturation gate for the PEM prior study.

Compares the per-unit PEM probabilities of PEM runs that use the same units
but different priors (e.g. run_local_pem.py --sample-units 60 for the uniform
prior and for each correlation prior). If P barely changes, the prior has no
effect at these PEM settings and an A3T-GCN comparison would only measure
training noise.

    python tools/prior_gate.py --baseline <run_dir> --runs <run_dir> <run_dir> ...
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_units(run_dir):
    out = {}
    for path in sorted((Path(run_dir) / "units").glob("unit_*.npz")):
        with np.load(path, allow_pickle=True) as u:
            if str(u["status"]) == "ok":
                out[int(u["unit_index"])] = (np.asarray(u["nodes"]), np.asarray(u["inclusion_probabilities"], dtype=np.float64))
    return out


def off_diagonal(p):
    return p[~np.eye(len(p), dtype=bool)]


def compare(base, other):
    shared = sorted(set(base) & set(other))
    a = np.concatenate([off_diagonal(base[i][1]) for i in shared])
    b = np.concatenate([off_diagonal(other[i][1]) for i in shared])
    d = np.abs(a - b)
    return {
        "units_compared": len(shared),
        "entries": int(a.size),
        "share_changed_over_0.01": float((d > 0.01).mean()),
        "share_changed_over_0.10": float((d > 0.10).mean()),
        "share_crossing_threshold_0.5": float(((a >= 0.5) != (b >= 0.5)).mean()),
        "mean_abs_change": float(d.mean()),
        "share_P_between_0.05_and_0.95": float(((b > 0.05) & (b < 0.95)).mean()),
        "share_P_gt_0.99": float((b > 0.99).mean()),
        "share_P_lt_0.01": float((b < 0.01).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()

    base = load_units(args.baseline)
    report = {"baseline": Path(args.baseline).name,
              "baseline_share_P_between_0.05_and_0.95": float(np.mean(
                  [((v > 0.05) & (v < 0.95)).mean() for v in (off_diagonal(p) for _, p in base.values())]))}
    for run in args.runs:
        report[Path(run).name] = compare(base, load_units(run))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
