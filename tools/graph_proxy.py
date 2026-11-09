"""
Cheap validation proxy for a graph: "does each target keep the sources it needs?"

For every target sensor i and horizon h, a ridge regression predicts x_i(t + h) from the last LAGS values of the sensor
itself and of the sources the graph gives it (the non-zero entries of row i of G_h, G[target, source]). The regression
is fitted on TRAINING targets only and scored on VALIDATION targets only; the test period is never read.
No filled values are used: rows with any missing value among the target and its features are skipped.

A lower validation RMSE / MAE means the graph kept more of the sources that carry predictive information. This is a
linear stand-in for the A3T-GCN (which takes minutes instead of 5 GPU hours), so it ranks graphs, it does not replace
the final A3T-GCN runs.

Reference graphs built here (no files needed):
    self_only   no sources at all (the graph-free reference)
    road_all    every road candidate of the target (support of the E1 graph, about 25 per target on average)
    random25    25 random road candidates per target (seed fixed): a selection that ignores the data

Usage (from the implementation directory):
    python tools/graph_proxy.py --horizons 1 6 --graph-dir outputs/artifacts_all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))

TRAIN_END, VAL_END = 84096, 94608        # chronological 80 / 10 / 10 split of the 105,120 rows
LAGS = 3                                  # inputs x(t), x(t-1), x(t-2) of each chosen source
LAMBDAS = [1.0, 1e2, 1e3, 1e4, 1e5]       # ridge grid on standardised features; chosen on a training holdout


def load_flow():
    from config.config import ExperimentConfig
    from data.loader import load_gba_dataset

    config = ExperimentConfig()
    traffic, _ = load_gba_dataset(config.traffic_path, config.adjacency_path,
                                  num_timesteps=config.num_timesteps, num_nodes=config.num_nodes)
    sensors = np.load(IMPLEMENTATION_DIR / "outputs" / "var_train" / "lag3_alpha1e+06" / "sensors.npy")
    # column-major: picking the ~26 columns of one target is then a cheap contiguous gather
    return np.asfortranarray(np.asarray(traffic[:VAL_END][:, sensors], dtype=np.float64)), sensors


def load_graph(path):
    with np.load(path, allow_pickle=False) as z:
        if int(z["train_end"]) != TRAIN_END:
            raise ValueError(f"{path}: not built from this training split")
        return np.asarray(z["adjacency"]) != 0, np.asarray(z["sensors"])


def reference_graphs(e1_path, seed=0):
    candidates, sensors = load_graph(e1_path)
    n = candidates.shape[0]
    np.fill_diagonal(candidates, False)
    rng = np.random.default_rng(seed)
    random25 = np.zeros_like(candidates)
    for i in range(n):
        cols = np.flatnonzero(candidates[i])
        if len(cols) > 25:
            cols = rng.choice(cols, size=25, replace=False)
        random25[i, cols] = True
    return {"self_only": np.zeros_like(candidates), "road_all": candidates, "random25": random25}, sensors


def target_design(flow, i, sources, h, first, last):
    """Rows for targets tau in [first, last): features x_j(tau-h-l), l = 0..LAGS-1 for j in {i} + sources; y = x_i(tau)."""
    cols = np.concatenate([[i], np.asarray(sources, dtype=np.int64)])
    start = max(first, h + LAGS - 1)
    sub = flow[:, cols]                                   # columns first: copies (rows x ~26), not whole rows
    # contiguous row ranges instead of fancy row indexing: rows tau - h - lag for tau in [start, last)
    feats = np.concatenate([sub[start - h - lag:last - h - lag] for lag in range(LAGS)], axis=1)
    y = flow[start:last, i]
    ok = np.isfinite(feats).all(axis=1) & np.isfinite(y)
    return feats[ok], y[ok]


def ridge_both(Xtr, ytr, Xva):
    """
    Ridge fits for every lambda in LAMBDAS, from ONE pass over the training rows:
      holdout  fitted on the first 90% of the training targets, scored on the last 10% (chooses lambda)
      full     fitted on all training targets, scored on the validation targets
    Features are standardised with the training statistics (training rows only).
    Returns (holdout errors, validation errors), each shaped (len(LAMBDAS), rows).
    """
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Xtr - mu) / sd
    cut = int(0.9 * len(ytr))
    Za, Zb = Z[:cut], Z[cut:]
    ya, yb = ytr[:cut], ytr[cut:]
    gram_a, gram_b = Za.T @ Za, Zb.T @ Zb
    ym_a, ym_all = ya.mean(), ytr.mean()
    rhs_a = Za.T @ (ya - ym_a)
    rhs_all = rhs_a + Zb.T @ (yb - ym_a) + (ym_a - ym_all) * Z.sum(axis=0)          # = Z.T @ (ytr - ym_all)
    gram_all = gram_a + gram_b
    Zva = (Xva - mu) / sd
    eye = np.eye(Z.shape[1])
    e_hold = np.stack([Zb @ np.linalg.solve(gram_a + lam * eye, rhs_a) + ym_a - yb for lam in LAMBDAS])
    e_val = np.stack([Zva @ np.linalg.solve(gram_all + lam * eye, rhs_all) + ym_all for lam in LAMBDAS])
    return e_hold, e_val


def evaluate(flow, graph, h, targets=None):
    """
    Per-target squared / absolute error sums for every lambda, on a TRAINING holdout and on the VALIDATION targets.
    lambda is chosen later from the holdout only (see summarise), so validation never influences the fit.
    """
    n, L = graph.shape[0], len(LAMBDAS)
    hold_sse, val_sse, val_sae = np.zeros((n, L)), np.zeros((n, L)), np.zeros((n, L))
    hold_cnt, val_cnt = np.zeros(n), np.zeros(n)
    targets = range(n) if targets is None else targets
    for i in targets:
        sources = np.flatnonzero(graph[i])
        sources = sources[sources != i]
        Xtr, ytr = target_design(flow, i, sources, h, 0, TRAIN_END)
        Xva, yva = target_design(flow, i, sources, h, TRAIN_END, VAL_END)
        if len(ytr) < 1000 or len(yva) == 0:
            continue
        e_hold, e_val = ridge_both(Xtr, ytr, Xva)
        e_val = e_val - yva
        hold_sse[i], hold_cnt[i] = (e_hold ** 2).sum(axis=1), e_hold.shape[1]
        val_sse[i], val_sae[i], val_cnt[i] = (e_val ** 2).sum(axis=1), np.abs(e_val).sum(axis=1), len(yva)
    return hold_sse, hold_cnt, val_sse, val_sae, val_cnt


def summarise(hold_sse, hold_cnt, val_sse, val_sae, val_cnt):
    used = val_cnt > 0
    best = int(np.argmin(hold_sse[used].sum(axis=0)))              # lambda chosen on the training holdout only
    return {
        "val_rmse": float(np.sqrt(val_sse[used, best].sum() / val_cnt[used].sum())),
        "val_mae": float(val_sae[used, best].sum() / val_cnt[used].sum()),
        "lambda": LAMBDAS[best],
        "sensors": int(used.sum()),
    }, best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--graph-dir", default=str(IMPLEMENTATION_DIR / "outputs" / "artifacts_all"))
    parser.add_argument("--e1", default=None, help="E1 graph file (road support); default: found in --graph-dir")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "proxy"))
    parser.add_argument("--sample", type=int, default=None, help="evaluate only this many random targets (smoke test)")
    parser.add_argument("--extra", nargs="*", default=[], help="more graphs as name=path (a {h} in path is filled in)")
    parser.add_argument("--only", nargs="*", default=None, help="evaluate only these graph names (E8V is always kept)")
    parser.add_argument("--tag", default=None, help="suffix for the output file name")
    args = parser.parse_args()

    gdir, out = Path(args.graph_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    flow, sensors = load_flow()
    print(f"flow {flow.shape} loaded in {time.time() - t0:.0f}s; train rows < {TRAIN_END}, validation rows {TRAIN_END}-{VAL_END}", flush=True)

    e1 = args.e1 or str(next(gdir.glob("E1__*.npz")))
    refs, ref_sensors = reference_graphs(e1)
    if not np.array_equal(ref_sensors, sensors):
        raise ValueError("graph sensors differ from the VAR run sensors")
    targets = None
    if args.sample:
        targets = np.sort(np.random.default_rng(1).choice(len(sensors), size=args.sample, replace=False))

    results = {}
    for h in args.horizons:
        graphs = dict(refs)
        spec = {
            "E8V": f"E8V__h{h}__lag3_alpha1e+06.npz",
            "E8_flow": f"E8__h{h}__lag3_alpha1e+06__flow_nu0_0.01_nu1_0.2.npz",
            "E8_uniform": f"E8__h{h}__lag3_alpha1e+06__uniform_nu0_0.01_nu1_0.2.npz",
            "E8_shuffle": f"E8__h{h}__lag3_alpha1e+06__flow_shuffle_nu0_0.01_nu1_0.2.npz",
        }
        for name, fname in spec.items():
            g, s = load_graph(gdir / fname)
            assert np.array_equal(s, sensors)
            graphs[name] = g
        for item in args.extra:
            name, path = item.split("=", 1)
            g, s = load_graph(path.format(h=h))
            assert np.array_equal(s, sensors)
            graphs[name] = g

        if args.only:                                   # E8V stays: it is the paired baseline
            graphs = {k: v for k, v in graphs.items() if k in set(args.only) | {"E8V"}}
        per_sensor = {}
        for name, g in graphs.items():
            t1 = time.time()
            parts = evaluate(flow, g, h, targets)
            res, best = summarise(*parts)
            res["avg_sources"] = float((g.sum(axis=1)).mean())
            per_sensor[name] = (parts[3][:, best], parts[4])           # per-sensor validation abs error sums, counts
            results[f"h{h}/{name}"] = res
            print(f"h={h:2d} {name:13s} val RMSE {res['val_rmse']:7.3f}  MAE {res['val_mae']:7.3f}  lambda {res['lambda']:g}  "
                  f"(sources/target {res['avg_sources']:5.1f}, {time.time() - t1:.0f}s)", flush=True)

        # paired per-sensor comparison against E8V: mean MAE difference and the share of sensors that improve
        base_sae, base_cnt = per_sensor["E8V"]
        for name, (sae, cnt) in per_sensor.items():
            if name == "E8V":
                continue
            both = (cnt > 0) & (base_cnt > 0)
            d = sae[both] / cnt[both] - base_sae[both] / base_cnt[both]
            results[f"h{h}/{name}"]["vs_E8V_mean_sensor_mae_diff"] = float(d.mean())
            results[f"h{h}/{name}"]["vs_E8V_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
            results[f"h{h}/{name}"]["vs_E8V_share_sensors_better"] = float((d < 0).mean())

    tag = (f"sample{args.sample}" if args.sample else "full") + (f"_{args.tag}" if args.tag else "")
    (out / f"proxy_{tag}_h{'-'.join(map(str, args.horizons))}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved to {out}  (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
