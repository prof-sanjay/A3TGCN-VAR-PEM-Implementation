"""
Phase 5: can local DirectLiNGAM use an evenly spaced subset of the
training rows without substantially changing B_0?

Compares, unit by unit, runs of run_local_lingam.py on the SAME sampled
units: the full-row run (reference) and runs with --max-rows N.

Per unit:
    rel_diff        ||B_sub - B_full||_F / ||B_full||_F
    support_jaccard Jaccard of the non-zero entries of B_0
    sign_agreement  share of edges present in both with the same sign
    order_tau       Kendall tau between the two causal orders

Acceptance criteria (fixed on 2026-10-25, before the subset runs):

    median rel_diff <= 0.10, median support_jaccard >= 0.80 and
    sign_agreement >= 0.95 (pooled over units). The smallest N that
    passes is used for the full run (config.LINGAM_MAX_ROWS); if none
    passes, the full rows are used.

    python selection/compare_lingam_rows.py <full run dir> <subset run dir> [<subset run dir> ...]
"""

import sys
from pathlib import Path

# Make the implementation folder importable (run from implementation/:
# python selection/<file>.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau


MAX_REL_DIFF = 0.10
MIN_SUPPORT_JACCARD = 0.80
MIN_SIGN_AGREEMENT = 0.95


def load_units(run_dir):

    units = {}
    for path in sorted(Path(run_dir, "units").glob("unit_*.npz")):
        with np.load(path) as u:
            units[int(u["unit_index"])] = {
                "b0": u["b0"], "order": u["causal_order"],
                "status": str(u["status"]), "seconds": float(u["seconds"]),
                "rows": int(u["rows_used"]), "size": len(u["nodes"]),
            }
    return units


def compare(full, sub):

    rows = []
    same_sign = total_common = 0

    for index, ref in full.items():

        if index not in sub or ref["status"] != "ok" or sub[index]["status"] != "ok":
            continue

        a, b = ref["b0"], sub[index]["b0"]
        norm = np.linalg.norm(a)
        support_a, support_b = a != 0, b != 0
        union = np.sum(support_a | support_b)
        common = support_a & support_b

        same_sign += int(np.sum(np.sign(a[common]) == np.sign(b[common])))
        total_common += int(common.sum())

        # Position of every variable in each causal order.
        pos_a = np.argsort(ref["order"])
        pos_b = np.argsort(sub[index]["order"])

        rows.append({
            "unit": index, "size": ref["size"],
            "rel_diff": float(np.linalg.norm(a - b) / norm) if norm > 0 else float("nan"),
            "support_jaccard": float(np.sum(common) / union) if union else float("nan"),
            "order_tau": float(kendalltau(pos_a, pos_b)[0]),
            "seconds_full": ref["seconds"], "seconds_sub": sub[index]["seconds"],
            "rows_sub": sub[index]["rows"],
        })

    sign_agreement = same_sign / total_common if total_common else float("nan")
    return rows, sign_agreement


def main():

    parser = argparse.ArgumentParser(description="Compare local LiNGAM runs unit by unit.")
    parser.add_argument("reference", help="Reference run (e.g. all rows, or half 1).")
    parser.add_argument("others", nargs="+", help="Runs compared with the reference.")
    parser.add_argument("--out-name", default="lingam_row_subset_check.json",
                        help="Output file name (written next to the reference run).")
    args = parser.parse_args()

    full = load_units(args.reference)
    results = []

    for run in args.others:

        rows, sign = compare(full, load_units(run))
        rel = np.array([r["rel_diff"] for r in rows])
        jac = np.array([r["support_jaccard"] for r in rows])
        tau = np.array([r["order_tau"] for r in rows])
        speedup = np.sum([r["seconds_full"] for r in rows]) / np.sum([r["seconds_sub"] for r in rows])

        passed = bool(np.nanmedian(rel) <= MAX_REL_DIFF
                      and np.nanmedian(jac) >= MIN_SUPPORT_JACCARD
                      and sign >= MIN_SIGN_AGREEMENT)

        results.append({
            "run": run, "rows": rows[0]["rows_sub"] if rows else None, "units": len(rows),
            "median_rel_diff": float(np.nanmedian(rel)), "max_rel_diff": float(np.nanmax(rel)),
            "median_support_jaccard": float(np.nanmedian(jac)),
            "min_support_jaccard": float(np.nanmin(jac)),
            "sign_agreement": sign, "median_order_tau": float(np.nanmedian(tau)),
            "speedup": float(speedup), "passes": passed, "per_unit": rows,
        })

        print(f"{Path(run).name}: rows {results[-1]['rows']}, units {len(rows)}, "
              f"rel diff median {np.nanmedian(rel):.3f} (max {np.nanmax(rel):.3f}), "
              f"support Jaccard median {np.nanmedian(jac):.3f} (min {np.nanmin(jac):.3f}), "
              f"sign agreement {sign:.3f}, order tau median {np.nanmedian(tau):.3f}, "
              f"speed-up {speedup:.1f}x -> {'PASS' if passed else 'FAIL'}")

    passing = [r for r in results if r["passes"]]
    chosen = min(passing, key=lambda r: r["rows"]) if passing else None

    print("\nDecision: " + (
        f"use LINGAM_MAX_ROWS = {chosen['rows']}" if chosen
        else "no subset passes; use the full rows (LINGAM_MAX_ROWS = None)"))

    out = Path(args.reference).parent / args.out_name
    out.write_text(json.dumps({
        "criteria": {"max_median_rel_diff": MAX_REL_DIFF,
                     "min_median_support_jaccard": MIN_SUPPORT_JACCARD,
                     "min_sign_agreement": MIN_SIGN_AGREEMENT},
        "reference": args.reference, "results": results,
        "chosen_max_rows": chosen["rows"] if chosen else None,
    }, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
