"""
Build the E8 graphs: horizon-conditioned VAR-PEM, WITHOUT DirectLiNGAM / B0.

    E8   S_h = P (x) |Psi_h|     top-25 sources per target, h = 1, 3, 6, 12
    E8V  S_h = |Psi_h|           the ablation that isolates PEM

Inputs (training-split results of earlier runs; nothing is re-estimated):
    VAR coefficients + sensors + metadata   outputs/var_train/<VAR_RUN>/
    PEM probabilities P                      --pem-result (the E4 / E5 PEM run;
                                             PEM reads only VAR residuals, so
                                             that run never depended on LiNGAM)

No LiNGAM module is imported and no B0 / lingam_result file is opened; the
report at the end says so and the build stops if that is not true.

Outputs (never overwrites E0-E7 or E5 files):
    outputs/graphs/E8__h{h}__...npz, E8V__h{h}__...npz (+ .json)   graph artifacts
    outputs/e8/                                                     everything else:
        e8_config.json  e8_diagnostics.json  var_coefficients.npz  var_metadata.json
        pem_probabilities.npz  psi_h{h}.npz  effects_E8_h{h}.npz  effects_E8V_h{h}.npz

Run from the implementation directory:

    python build_e8_graphs.py --pem-result <pem_result.npz>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from causal.var_pem_horizon import (
    graph_diagnostics,
    lingam_access_report,
    probability_diagnostics,
    psi_diagnostics,
    horizon_graphs_from_effects,
    impulse_responses,
    save_artifact,
    weighted_effects,
)
from config import config as C
from experiments.registry import get_experiment

IMPLEMENTATION_DIR = Path(__file__).resolve().parent
START = time.time()
OPENED = []          # every input file this build opens (checked for LiNGAM / B0 at the end)


def log(message):
    print(f"[{(time.time() - START) / 60:6.1f} min] {message}", flush=True)


def load(path, **kwargs):
    OPENED.append(str(path))
    return np.load(path, **kwargs)


def save_npz(path, **arrays):
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})


def jaccard(a, b):
    a, b = a != 0, b != 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--var-dir", default=str(IMPLEMENTATION_DIR / "outputs" / "var_train" / C.VAR_RUN))
    parser.add_argument("--pem-result", required=True, help="pem_result.npz of the E4 / E5 PEM run (uniform prior).")
    parser.add_argument("--graph-dir", default=str(C.GRAPH_DIR), help="where the graph artifacts are written.")
    parser.add_argument("--out-dir", default=str(IMPLEMENTATION_DIR / "outputs" / "e8"))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(get_experiment("E8").horizons))
    parser.add_argument("--k", type=int, default=C.E5_TOP_K)
    parser.add_argument("--e5-graphs-dir", default=None,
                        help="optional, comparison only (never used to build E8): stored E5 graphs.")
    parser.add_argument("--variant", default=None,
                        help="prior-study variant name (e.g. pearson_nu0.01_nu1_0.2). The PEM result may then use "
                             "any prior and PEM setting; only E8 graphs are written, named "
                             "E8__h{h}__<VAR_RUN>__<variant>.npz. E8V is not written (it does not depend on PEM).")
    parser.add_argument("--max-sensors", type=int, default=None, help="smoke test only.")
    args = parser.parse_args()

    horizons = sorted(args.horizons)
    var_dir, graph_dir, out_dir = Path(args.var_dir), Path(args.graph_dir), Path(args.out_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- inputs: VAR (A_k) and PEM (P) only ---------------------------------
    coefficients = load(var_dir / "coefficients.npy").astype(np.float64)
    sensors = load(var_dir / "sensors.npy")
    OPENED.append(str(var_dir / "metadata.json"))
    var_metadata = json.loads((var_dir / "metadata.json").read_text(encoding="utf-8"))
    train_end = int(var_metadata["train_end"])

    with load(args.pem_result) as r:
        if not np.array_equal(r["sensors"], sensors) or int(r["train_end"]) != train_end:
            raise ValueError("PEM result and VAR run differ in sensors or training split.")
        prior = str(r["prior"]) if "prior" in r.files else "uniform"
        candidates = str(r["candidates"]) if "candidates" in r.files else "road"
        p = r["inclusion_probabilities"].astype(np.float64)
    if candidates != "road" or (args.variant is None and prior != "uniform"):
        raise ValueError(f"E8 uses the E5 PEM run (road candidates, uniform prior); got {candidates!r}/{prior!r}. "
                         "A different PEM run needs --variant.")

    if args.max_sensors:                       # smoke test only
        m = args.max_sensors
        coefficients, p, sensors = coefficients[:, :m, :m], p[:m, :m], sensors[:m]
    n = len(sensors)
    log(f"VAR {coefficients.shape} | P {p.shape} | train_end {train_end} | K {args.k} | horizons {horizons}")

    # ---- stages 1-4: impulse responses, S_h, graphs, ablation -----------------
    psi = impulse_responses(coefficients, horizons)            # Psi_h, computed once for both variants
    effects8 = weighted_effects(psi, p)                        # E8:  S_h = P * |Psi_h|
    effectsV = weighted_effects(psi, None)                     # E8V: S_h = |Psi_h|
    graphs8 = horizon_graphs_from_effects(effects8, args.k)
    graphsV = horizon_graphs_from_effects(effectsV, args.k)
    log("impulse responses, S_h and top-K graphs computed for E8 and E8V")

    # ---- checks that stop the build ------------------------------------------
    checks = {
        "psi_1_equals_A_1 (orientation / recursion)": bool(np.allclose(psi[horizons[0]], coefficients[0]))
        if horizons[0] == 1 else None,
        "P_symmetric": bool(np.allclose(p, p.T)),
        "diagonal_zero_E8": all(np.all(np.diag(g) == 0) for g in graphs8.values()),
        "diagonal_zero_E8V": all(np.all(np.diag(g) == 0) for g in graphsV.values()),
        "in_degree_at_most_K": all(((g != 0).sum(axis=1) <= args.k).all() for g in list(graphs8.values()) + list(graphsV.values())),
        "E8_edges_inside_P_support": all(not np.any((g != 0) & (p == 0)) for g in graphs8.values()),
        "E8_weights_below_E8V_weights": all(float(np.max(effects8[h] - effectsV[h])) <= 1e-12 for h in horizons),
    }
    log(f"checks: {checks}")
    if not all(v for v in checks.values() if v is not None):
        raise AssertionError(f"E8 build checks failed: {checks}")

    # ---- diagnostics ------------------------------------------------------------
    diagnostics = {
        "var": psi_diagnostics(coefficients, psi),
        "pem": probability_diagnostics(p),
        "checks": checks,
        "E8": {f"h{h}": graph_diagnostics(graphs8[h]) for h in horizons},
        "E8V": {f"h{h}": graph_diagnostics(graphsV[h]) for h in horizons},
        "E8_vs_E8V_edge_overlap_jaccard": {f"h{h}": jaccard(graphs8[h], graphsV[h]) for h in horizons},
    }

    if args.e5_graphs_dir and not args.max_sensors:
        overlap = {}
        for h in horizons:
            files = sorted(Path(args.e5_graphs_dir).glob(f"E5__h{h}__*.npz"))
            if files:
                with np.load(files[0]) as a:             # a graph artifact, comparison only
                    e5 = a["adjacency"]
                overlap[f"h{h}"] = {"E8_vs_E5": jaccard(graphs8[h], e5), "E8V_vs_E5": jaccard(graphsV[h], e5)}
        diagnostics["edge_overlap_with_stored_E5_graphs_(comparison_only)"] = overlap

    for h in horizons:
        d8, dv = diagnostics["E8"][f"h{h}"], diagnostics["E8V"][f"h{h}"]
        log(f"  h={h:2d}  E8: {d8['edges']:6d} edges, in-degree {d8['min_in_degree']}-{d8['max_in_degree']} "
            f"(mean {d8['average_in_degree']:.1f}), without incoming {d8['percent_without_incoming']:.2f}% | "
            f"E8V: {dv['edges']:6d} edges, mean in-degree {dv['average_in_degree']:.1f}")

    # ---- save: graph artifacts (read by main.load_graph) -------------------------
    method8, methodV = get_experiment("E8").graph_method, get_experiment("E8V").graph_method
    written = []
    targets = [("E8", graphs8, method8, f"{C.VAR_RUN}__{C.PEM_RUN}"),
               ("E8V", graphsV, methodV, C.VAR_RUN)]
    if args.variant:                                # prior study: E8 only, own file names, E8V untouched
        targets = [("E8", graphs8, method8, f"{C.VAR_RUN}__{args.variant}")]
    for name, graphs, method, source in targets:
        for h in horizons:
            file_name = (f"E8__h{h}__{C.VAR_RUN}__{args.variant}.npz" if args.variant
                         else C.FINAL_GRAPHS[name].format(h=h))
            path = graph_dir / file_name
            save_artifact(path, graphs[h], experiment=name, method=method, train_end=train_end,
                          sensors=np.asarray(sensors, dtype=np.int64), horizon=h, top_k=args.k, source_run=source)
            summary = {"experiment": name, "method": method, "horizon": h, "top_k": args.k,
                       "source_run": source, "train_end": train_end, **diagnostics[name][f"h{h}"]}
            path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            written.append(path.name)

    # ---- save: everything else ----------------------------------------------------
    np.savez_compressed(out_dir / "var_coefficients.npz", coefficients=coefficients.astype(np.float32), sensors=sensors)
    (out_dir / "var_metadata.json").write_text(json.dumps(var_metadata, indent=2), encoding="utf-8")
    save_npz(out_dir / "pem_probabilities.npz", probabilities=p)
    for h in horizons:
        save_npz(out_dir / f"psi_h{h}.npz", psi=psi[h])
        save_npz(out_dir / f"effects_E8_h{h}.npz", effects=effects8[h])
        save_npz(out_dir / f"effects_E8V_h{h}.npz", effects=effectsV[h])

    access = lingam_access_report(OPENED)
    diagnostics["lingam_access"] = access
    log(f"LiNGAM access report: {access}")
    if not access["ok"]:
        raise AssertionError("E8 must not touch LiNGAM or B0: " + json.dumps(access))

    config_record = {
        "experiment": "E8 (S_h = P * |Psi_h|) and E8V (S_h = |Psi_h|)",
        "formulas": {"Psi_0": "I", "Psi_s": "sum_{k=1}^{min(s,p)} A_k Psi_{s-k}",
                     "E8": "S_h = P (x) |Psi_h|, diagonal 0", "E8V": "S_h = |Psi_h|, diagonal 0",
                     "graph": "top-K sources per target of S_h, G[target, source]"},
        "horizons": horizons, "top_k": args.k, "var_run": C.VAR_RUN, "pem_run": C.PEM_RUN,
        "var": {"lag": C.VAR_LAG, "ridge_alpha": C.VAR_ALPHA},
        "pem": {"nu0": C.PEM_NU0, "nu1": C.PEM_NU1, "tau": C.PEM_TAU, "threshold": C.PEM_THRESHOLD,
                "prior": prior, "prior_inclusion": C.PEM_PRIOR_INCLUSION, "standardise": C.PEM_STANDARDISE,
                "optimizer": C.PEM_OPTIMIZER, "estimated_by": "the existing E4 / E5 PEM run (not re-estimated)"},
        "variant": args.variant, "pem_result_prior": prior, "pem_result_file": str(args.pem_result),
        "train_end": train_end, "num_sensors": int(n), "inputs_opened": OPENED,
        "graph_files": written, "smoke_test": bool(args.max_sensors),
    }
    (out_dir / "e8_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    (out_dir / "e8_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    log(f"saved {len(written)} graph artifacts to {graph_dir} and the rest to {out_dir}")


if __name__ == "__main__":
    main()
