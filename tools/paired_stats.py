"""
Paired per-sensor statistics for graphs scored by tools/graph_proxy.py --save-persensor.

For a pair (A, B) on the same targets and the same validation rows:
    pooled RMSE ratio A/B with a 95% bootstrap interval over SENSORS,
    mean per-sensor MAE difference (A - B) with its standard error, and the share of sensors where A is better.
A difference is called clear only when the bootstrap interval of the RMSE ratio excludes 1.

    python tools/paired_stats.py outputs/proxy/persensor_full_h1.npz --pairs hyb_pc10:hyb_corr10 hyb_pc10:hyb_rand10 hyb_pc10:V_std
"""

import argparse
import json
from pathlib import Path

import numpy as np


def rmse(sse, cnt, idx):
    return float(np.sqrt(sse[idx].sum() / cnt[idx].sum()))


def compare(z, a, b, n_boot=2000, seed=0):
    sse_a, sae_a, cnt = z[f"{a}__sse"], z[f"{a}__sae"], z[f"{a}__cnt"]
    sse_b, sae_b, cnt_b = z[f"{b}__sse"], z[f"{b}__sae"], z[f"{b}__cnt"]
    both = (cnt > 0) & (cnt_b > 0)
    idx = np.flatnonzero(both)
    ratio = rmse(sse_a, cnt, idx) / rmse(sse_b, cnt_b, idx)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots[k] = rmse(sse_a, cnt, s) / rmse(sse_b, cnt_b, s)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    d = sae_a[idx] / cnt[idx] - sae_b[idx] / cnt_b[idx]
    return {
        "A": a, "B": b, "sensors": int(len(idx)),
        "rmse_A": rmse(sse_a, cnt, idx), "rmse_B": rmse(sse_b, cnt_b, idx),
        "rmse_ratio_A_over_B": ratio, "ratio_ci95": [float(lo), float(hi)],
        "clear_difference": bool(hi < 1.0 or lo > 1.0),
        "mean_sensor_mae_diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "share_sensors_A_better": float((d < 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("npz")
    parser.add_argument("--pairs", nargs="+", required=True, help="A:B pairs, A compared with B")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    z = np.load(args.npz)
    rows = []
    for pair in args.pairs:
        a, b = pair.split(":")
        if f"{a}__sse" not in z.files or f"{b}__sse" not in z.files:
            print(f"skip {pair}: not in the file")
            continue
        r = compare(z, a, b)
        rows.append(r)
        print(f"{a:14s} vs {b:14s}  RMSE {r['rmse_A']:.3f} vs {r['rmse_B']:.3f}  ratio {r['rmse_ratio_A_over_B']:.4f} "
              f"[{r['ratio_ci95'][0]:.4f}, {r['ratio_ci95'][1]:.4f}]  {'CLEAR' if r['clear_difference'] else 'not clear'}  "
              f"| MAE diff {r['mean_sensor_mae_diff']:+.3f} +/- {r['se']:.3f}, A better on {r['share_sensors_A_better']:.0%} of {r['sensors']} sensors")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
