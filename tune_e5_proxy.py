"""
E5 tuning, Phase A: score graph settings on VALIDATION data with the proxy
of tuning/e5_proxy.py (no A3T-GCN training, no random seed).

Stages (all use only training rows to fit and validation rows to score):

    calibration  Does the proxy rank graphs like A3T-GCN did? The graphs
                 whose A3T-GCN results exist (tuning/reference_a3tgcn.json)
                 are scored by the proxy and compared (Spearman). A low
                 agreement means the proxy cannot be used to tune.
    k            K sweep: top-K sources per sensor of the E5 effect matrix.
    roles        What do LiNGAM's B0 and PEM's P add to the VAR? B0~ = 0
                 (VAR only), B0, P*B0 (E5), hard-thresholded P, B0 cut-offs.
    extras       Extra PEM / LiNGAM results (different nu0 / nu1 / eta /
                 rows) given with --pem-extra / --lingam-extra, scored in the
                 E5 recipe against the base run.

Inputs (all from earlier runs, nothing is recomputed here):
    VAR coefficients + sensors (outputs/var_train/<run>/), LiNGAM B0
    (lingam_result.npz), PEM result of the E4 setting (--pem-base), the GBA
    traffic file and road adjacency.

Run from the implementation directory:

    python tune_e5_proxy.py --pem-base <pem_result.npz> --out results/tuning/phaseA
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from causal.graph_builders import (
    pem_graph,
    top_k_sources,
    var_graph,
    var_lingam_graph,
    var_lingam_pem_graph,
    var_pem_mask_graph,
)
from causal.horizon import StructuralIRF
from config.config import E5_TOP_K, LINGAM_RUN, VAR_RUN, ExperimentConfig
from data.loader import load_gba_dataset
from tuning.e5_proxy import ProxyData, effect_matrices

IMPLEMENTATION_DIR = Path(__file__).resolve().parent
START = time.time()


def log(message):
    print(f"[{(time.time() - START) / 60:6.1f} min] {message}", flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_pem(path, sensors):
    with np.load(path) as r:
        if not np.array_equal(r["sensors"], sensors):
            raise ValueError(f"{path}: sensors differ from the VAR run.")
        return {
            "p": r["inclusion_probabilities"].astype(np.float64),
            "d": r["directed_adjacency"].astype(np.float64),
            "prior": str(r["prior"]) if "prior" in r.files else "uniform",
        }


def load_b0(path, sensors):
    with np.load(path) as r:
        if not np.array_equal(r["sensors"], sensors):
            raise ValueError(f"{path}: sensors differ from the VAR run.")
        return r["b0"].astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def row_of(label, graph_info, horizons, **extra):
    row = {"setting": label, **extra, "edges": graph_info["edges"],
           "own_signal_share": round(graph_info["own_signal_share"], 4)}
    for h in horizons:
        row[f"val_rmse_h{h}"] = round(graph_info[h]["rmse"], 3)
        row[f"val_mae_h{h}"] = round(graph_info[h]["mae"], 3)
    return row


def e5_graphs(effects, k, horizons):
    return {h: top_k_sources(effects[h], k) for h in horizons}


def score_per_horizon(proxy, graphs, horizons, args):
    """E5-style graphs differ per horizon: score graph h at horizon h."""
    scored = {}
    for h in horizons:
        scored[h] = proxy.evaluate(graphs[h], [h], args.stride_train, args.stride_val)
    return {
        "edges": int(np.mean([scored[h]["edges"] for h in horizons])),
        "own_signal_share": float(np.mean([scored[h]["own_signal_share"] for h in horizons])),
        **{h: scored[h][h] for h in horizons},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--var-dir", default=str(IMPLEMENTATION_DIR / "outputs" / "var_train" / VAR_RUN))
    parser.add_argument("--lingam-result",
                        default=str(IMPLEMENTATION_DIR / "outputs" / "lingam" / LINGAM_RUN / "lingam_result.npz"))
    parser.add_argument("--pem-base", required=True, help="pem_result.npz of the E4 setting (uniform prior).")
    parser.add_argument("--pem-extra", nargs="*", default=[], help="more pem_result.npz files to score.")
    parser.add_argument("--lingam-extra", nargs="*", default=[], help="more lingam_result.npz files to score.")
    parser.add_argument("--graphs-dir", default=None, help="optional: stored E5 graphs to verify the reconstruction.")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "results" / "tuning" / "phaseA"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--k-list", type=int, nargs="+", default=[5, 10, 15, 25, 40, 60, 100])
    parser.add_argument("--stages", nargs="+", default=["calibration", "k", "roles", "extras"])
    parser.add_argument("--stride-train", type=int, default=4)
    parser.add_argument("--stride-val", type=int, default=2)
    parser.add_argument("--max-sensors", type=int, default=None, help="smoke test only.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    horizons = sorted(args.horizons)
    results = {"settings": vars(args)}

    def save():
        (out / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")

    # ---- inputs ----------------------------------------------------------
    var_dir = Path(args.var_dir)
    coefficients = np.load(var_dir / "coefficients.npy").astype(np.float64)
    sensors = np.load(var_dir / "sensors.npy")
    b0 = load_b0(args.lingam_result, sensors)
    pem = load_pem(args.pem_base, sensors)
    if pem["prior"] != "uniform":
        raise ValueError("--pem-base must be the uniform-prior (E4) PEM run.")

    config = ExperimentConfig()
    traffic, adjacency = load_gba_dataset(config.traffic_path, config.adjacency_path,
                                          config.num_timesteps, config.num_nodes)
    train_end = int(config.num_timesteps * config.train_rate)
    val_end = int(config.num_timesteps * (config.train_rate + config.val_rate))
    traffic = traffic[:val_end][:, sensors]            # the test rows are never kept
    road = np.asarray(adjacency[np.ix_(sensors, sensors)], dtype=np.float64)
    scale = float(np.nanmax(traffic[:train_end]))

    all_sensors = sensors                       # extras are checked against the full list
    keep = args.max_sensors or len(sensors)     # --max-sensors: smoke tests only

    def fit(m):
        return m[:keep, :keep]

    if args.max_sensors:
        coefficients, b0, road = coefficients[:, :keep, :keep], fit(b0), fit(road)
        pem = {"p": fit(pem["p"]), "d": fit(pem["d"]), "prior": pem["prior"]}
        traffic = traffic[:, :keep]
        sensors = sensors[:keep]

    n = len(sensors)
    log(f"sensors {n} | train rows {train_end} | val rows {val_end - train_end} | scale {scale:g}")
    proxy = ProxyData(traffic, train_end, val_end, scale)
    del traffic

    max_h = max(horizons)
    psi = StructuralIRF()._calculate_psi(coefficients, max_h)
    log("impulse responses Psi computed")

    p_b0 = pem["p"] * b0
    np.fill_diagonal(p_b0, 0.0)
    base_effects = effect_matrices(psi, p_b0, horizons)
    k_default = min(E5_TOP_K, n - 1)
    base_graphs = e5_graphs(base_effects, k_default, horizons)

    # ---- optional: the reconstruction must equal the stored E5 graphs ----
    if args.graphs_dir and not args.max_sensors:
        checks = {}
        for h in horizons:
            files = sorted(Path(args.graphs_dir).glob(f"E5__h{h}__*.npz"))
            if not files:
                continue
            with np.load(files[0]) as a:
                stored = a["adjacency"].astype(np.float64)
            mine = base_graphs[h]
            checks[f"h{h}"] = {
                "same_edge_set": bool(np.array_equal(stored != 0, mine != 0)),
                "max_abs_weight_difference": float(np.abs(stored - mine).max()),
            }
        results["e5_reconstruction_check"] = checks
        log(f"E5 reconstruction vs stored graphs: {checks}")
        save()

    # ---- stage 1: calibration -------------------------------------------
    stages = set(args.stages)

    if "calibration" in stages:
        log("calibration: scoring graphs with known A3T-GCN results")
        graphs = {
            "NOGRAPH": np.zeros((n, n)),
            "E0": road * (1.0 - np.eye(n)),
            "E1": var_graph(coefficients),
            "E2": var_lingam_graph(coefficients, b0),
            "E3a": pem_graph(pem["p"], pem["d"]),
            "E3b": var_pem_mask_graph(coefficients, pem["d"]),
            "E4": var_lingam_pem_graph(coefficients, b0, pem["p"]),
        }
        rows, proxy_val = [], {}
        for name, g in graphs.items():
            info = proxy.evaluate(g, horizons, args.stride_train, args.stride_val)
            rows.append(row_of(name, info, horizons))
            proxy_val[name] = {h: info[h]["rmse"] for h in horizons}
            log(f"  {name:8s} edges {info['edges']:8d} own {info['own_signal_share']:.3f} "
                + " ".join(f"h{h}:{info[h]['rmse']:.2f}" for h in horizons))
        info = score_per_horizon(proxy, base_graphs, horizons, args)
        rows.append(row_of("E5 (per-horizon graph)", info, horizons))
        proxy_val["E5"] = {h: info[h]["rmse"] for h in horizons}

        for extra in args.pem_extra:
            p = load_pem(extra, all_sensors)
            if p["prior"] == "road":
                g = var_lingam_pem_graph(coefficients, b0, fit(p["p"]))
                inf = proxy.evaluate(g, horizons, args.stride_train, args.stride_val)
                rows.append(row_of("E7 (road prior)", inf, horizons))
                proxy_val["E7"] = {h: inf[h]["rmse"] for h in horizons}

        write_csv(out / "calibration_proxy.csv", rows)

        reference = json.loads((IMPLEMENTATION_DIR / "tuning" / "reference_a3tgcn.json").read_text())
        table, correlations = [], {}
        for h in horizons:
            ref = reference.get(f"h{h}", {})
            names = [k for k in ref if k in proxy_val and k != "E7"]   # E7's training run was unstable
            if len(names) < 3:
                continue
            a = [ref[k]["val_rmse"] for k in names]
            b = [proxy_val[k][h] for k in names]
            correlations[f"h{h}"] = {"graphs": names, "spearman": spearman(a, b)}
            for k in names:
                table.append({"horizon": h, "graph": k, "a3tgcn_val_rmse": ref[k]["val_rmse"],
                              "a3tgcn_test_rmse": ref[k]["test_rmse"],
                              "proxy_val_rmse": round(proxy_val[k][h], 3)})
        write_csv(out / "calibration_vs_a3tgcn.csv", table)
        mean_rho = float(np.mean([c["spearman"] for c in correlations.values()])) if correlations else float("nan")
        results["calibration"] = {"per_horizon": correlations, "mean_spearman": mean_rho,
                                  "pass": bool(mean_rho >= 0.7), "threshold": 0.7}
        log(f"calibration: Spearman per horizon "
            f"{ {k: round(v['spearman'], 2) for k, v in correlations.items()} } | mean {mean_rho:.2f} | "
            f"{'PASS' if mean_rho >= 0.7 else 'FAIL: the proxy does not rank graphs like A3T-GCN'}")
        save()

    # ---- stage 2: K sweep -----------------------------------------------
    if "k" in stages:
        log("K sweep (E5 recipe, B0~ = P * B0)")
        rows = []
        for k in args.k_list:
            if k >= n:
                continue
            info = score_per_horizon(proxy, e5_graphs(base_effects, k, horizons), horizons, args)
            rows.append(row_of(f"K={k}", info, horizons, K=k))
            log(f"  K={k:4d} own {info['own_signal_share']:.3f} "
                + " ".join(f"h{h}:{info[h]['rmse']:.2f}" for h in horizons))
        write_csv(out / "k_sweep.csv", rows)
        results["k_sweep"] = rows
        save()

    # ---- stage 3: what do B0 and P add? ----------------------------------
    if "roles" in stages:
        log(f"roles of B0 and P at K={k_default}")
        nz = np.abs(b0[b0 != 0])
        median_cut, high_cut = (np.percentile(nz, 50), np.percentile(nz, 80)) if nz.size else (0, 0)
        variants = {
            "VAR only (B0~ = 0)": np.zeros_like(b0),
            "B0, no P": b0,
            "P * B0 (E5)": pem["p"] * b0,
            "(P > 0.9) * B0": (pem["p"] > 0.9) * b0,
            "(P > 0.99) * B0": (pem["p"] > 0.99) * b0,
            "P * B0, |B0| >= median": pem["p"] * b0 * (np.abs(b0) >= median_cut),
            "P * B0, |B0| >= 80th percentile": pem["p"] * b0 * (np.abs(b0) >= high_cut),
            "0.5 * P * B0": 0.5 * pem["p"] * b0,
        }
        rows = []
        for label, b_eff in variants.items():
            try:
                effects = effect_matrices(psi, b_eff, horizons)
            except np.linalg.LinAlgError:
                log(f"  {label}: (I - B0~) is singular, skipped")
                continue
            info = score_per_horizon(proxy, e5_graphs(effects, k_default, horizons), horizons, args)
            rows.append(row_of(label, info, horizons))
            log(f"  {label:34s} own {info['own_signal_share']:.3f} "
                + " ".join(f"h{h}:{info[h]['rmse']:.2f}" for h in horizons))
        write_csv(out / "roles.csv", rows)
        results["roles"] = rows
        save()

    # ---- stage 4: extra PEM / LiNGAM results ------------------------------
    if "extras" in stages and (args.pem_extra or args.lingam_extra):
        log("extra PEM / LiNGAM results (E5 recipe)")
        rows = []

        def jaccard(g_a, g_b):
            a, b = g_a != 0, g_b != 0
            return float((a & b).sum() / max((a | b).sum(), 1))

        for path in args.pem_extra:
            p = fit(load_pem(path, all_sensors)["p"])
            effects = effect_matrices(psi, p * b0, horizons)
            graphs = e5_graphs(effects, k_default, horizons)
            info = score_per_horizon(proxy, graphs, horizons, args)
            rows.append(row_of(f"PEM {Path(path).parent.name}/{Path(path).stem}", info, horizons, source=str(path),
                               top_k_overlap_with_base=round(float(np.mean(
                                   [jaccard(graphs[h], base_graphs[h]) for h in horizons])), 4)))
        for path in args.lingam_extra:
            b = fit(load_b0(path, all_sensors))
            effects = effect_matrices(psi, pem["p"] * b, horizons)
            graphs = e5_graphs(effects, k_default, horizons)
            info = score_per_horizon(proxy, graphs, horizons, args)
            rows.append(row_of(f"LiNGAM {Path(path).parent.name}/{Path(path).stem}", info, horizons, source=str(path),
                               top_k_overlap_with_base=round(float(np.mean(
                                   [jaccard(graphs[h], base_graphs[h]) for h in horizons])), 4)))
        write_csv(out / "extras.csv", rows)
        results["extras"] = rows
        save()

    # ---- baseline reference row for reading the tables --------------------
    info = score_per_horizon(proxy, base_graphs, horizons, args)
    results["base_E5"] = row_of("E5 base (K=25, P * B0)", info, horizons)
    log("base E5: " + " ".join(f"h{h}:{info[h]['rmse']:.2f}" for h in horizons))
    save()
    log("done")


if __name__ == "__main__":
    main()
