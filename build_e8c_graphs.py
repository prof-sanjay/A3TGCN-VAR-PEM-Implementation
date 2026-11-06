"""
E8C graphs: correlation-informed spike-and-slab refinement of VAR coefficients
(causal/var_coefficient_pem.py), horizon-conditioned, no VAR residuals, no LiNGAM.

    M      = VAR non-zero support (the pairs the topology-aware VAR estimated)
    R      = Pearson correlation of the ORIGINAL training flow (optionally detrended)
    eta    = eta_min + (eta_max - eta_min) |R|          (symmetric)
    W, P   = coefficient-space MAP + posterior slab probability   (nu0 = a s, nu1 = b s,
             lam = c s nu0, s = median of the candidate coefficient magnitudes)
    S_h    = M P |Psi_h|        Psi from the original A_k (default) or from the refined W
    G_h    = top-25 sources per target of S_h        (G[target, source])

Priors: detrended | raw (traffic correlation), uniform (eta = 0.5), shuffle (detrended
eta shuffled among the candidates; seeded), none (P = 1: the mask alone, the control that
isolates what the PEM-style weighting adds).

Writes outputs/graphs/E8__h{h}__<VAR_RUN>__<variant>.npz (read by ablation-study.ipynb with
GRAPH_VARIANT = "<variant>") and outputs/e8c/<variant>/ (matrices, diagnostics, config).
Training rows only. E0-E7 files are never touched.

    python build_e8c_graphs.py --prior detrended
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from causal import var_coefficient_pem as vp
from causal.flow_correlation import cached_flow_correlation
from causal.graph_builders import top_k_sources
from causal.var_pem_horizon import graph_diagnostics, impulse_responses, save_artifact
from config import config as C
from experiments.registry import get_experiment

HERE = Path(__file__).resolve().parent
START = time.time()
PRIORS = ("detrended", "raw", "uniform", "shuffle", "none")


def log(message):
    print(f"[{(time.time() - START) / 60:6.1f} min] {message}", flush=True)


def jaccard(a, b):
    a, b = a != 0, b != 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    q = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {"min": float(q[0]), "p10": float(q[1]), "median": float(q[2]), "p90": float(q[3]), "max": float(q[4]),
            "mean": float(x.mean())}


load_flow_correlation = cached_flow_correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--prior", choices=PRIORS, default="detrended")
    parser.add_argument("--var-dir", default=str(HERE / "outputs" / "var_train" / C.VAR_RUN))
    parser.add_argument("--nu0-mult", type=float, default=1.0, help="nu0 = nu0_mult * s")
    parser.add_argument("--nu1-mult", type=float, default=5.0, help="nu1 = nu1_mult * s")
    parser.add_argument("--lam-mult", type=float, default=0.5, help="lam = lam_mult * s * nu0")
    parser.add_argument("--q", type=float, default=1.0, help="reliability weight (constant; standard errors are not stored)")
    parser.add_argument("--eta-min", type=float, default=0.05)
    parser.add_argument("--eta-max", type=float, default=0.95)
    parser.add_argument("--psi-from", choices=("V", "W"), default="V", help="impulse responses from the original (V) or refined (W) coefficients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12])
    parser.add_argument("--k", type=int, default=C.E5_TOP_K)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--graph-dir", default=str(C.GRAPH_DIR))
    parser.add_argument("--out-root", default=str(HERE / "outputs" / "e8c"))
    parser.add_argument("--corr-cache-dir", default=str(HERE / "outputs" / "priors"))
    args = parser.parse_args()

    horizons = sorted(args.horizons)
    var_dir = Path(args.var_dir)
    a = np.load(var_dir / "coefficients.npy").astype(np.float64)
    sensors = np.load(var_dir / "sensors.npy")
    meta = json.loads((var_dir / "metadata.json").read_text(encoding="utf-8"))
    train_end = int(meta["train_end"])
    n = len(sensors)

    # ---- Stage A: VAR candidates ------------------------------------------------
    u = vp.edge_magnitude(a)
    mask = vp.support_mask(u)
    s = float(np.median(u[mask]))
    nu0, nu1, lam = args.nu0_mult * s, args.nu1_mult * s, args.lam_mult * s * args.nu0_mult * s
    log(f"VAR {a.shape} | candidates {int(mask.sum())} ({mask.sum() / n:.1f} per target) | s = {s:.5f} | nu0 {nu0:.5f} nu1 {nu1:.5f} lam {lam:.6f}")

    # ---- Stage B: traffic correlation -> eta ------------------------------------
    corr_file = None
    if args.prior in ("detrended", "raw", "shuffle"):
        corr, corr_file = load_flow_correlation(args.prior != "raw", sensors, train_end, Path(args.corr_cache_dir))
        eta = vp.correlation_to_eta(corr, args.eta_min, args.eta_max)
    else:
        corr = None
        eta = np.full((n, n), 0.5)
        np.fill_diagonal(eta, 0.0)
    if args.prior == "shuffle":
        rng = np.random.default_rng(args.seed)
        values = eta[mask].copy()
        rng.shuffle(values)
        eta = np.zeros_like(eta)
        eta[mask] = values                                       # distribution preserved, pairing destroyed
    eta_used = np.where(mask, eta, 0.5)

    # ---- Stage C: coefficient-space MAP -------------------------------------------
    if args.prior == "none":
        w_coef, p, w_hat = a * mask[None], mask.astype(np.float64), u * mask
    else:
        w_coef, p, w_hat = vp.refine(a, mask, eta_used, nu0, nu1, lam, args.q)
    log(f"MAP done: zeroed {np.mean(w_hat[mask] == 0):.3f} | P median {np.median(p[mask]):.3f}")

    # ---- Stage D: horizon graphs -----------------------------------------------------
    psi_source = a * mask[None] if args.psi_from == "V" else w_coef
    psi_V = impulse_responses(a, horizons)
    psi = impulse_responses(psi_source, horizons)
    graphs, effects, graphs_E8V, graphs_mask = {}, {}, {}, {}
    for h in horizons:
        s_h = mask * p * np.abs(psi[h])
        np.fill_diagonal(s_h, 0.0)
        effects[h] = s_h
        graphs[h] = top_k_sources(s_h, args.k)
        v = np.abs(psi_V[h]).copy(); np.fill_diagonal(v, 0.0)
        graphs_E8V[h] = top_k_sources(v, args.k)
        m_h = mask * np.abs(psi_V[h]); np.fill_diagonal(m_h, 0.0)
        graphs_mask[h] = top_k_sources(m_h, args.k)

    # ---- checks (stop the build) -------------------------------------------------------
    offdiag = ~np.eye(n, dtype=bool)
    checks = {
        "var_coefficients_finite": bool(np.isfinite(a).all()),
        "correlation_symmetric": True if corr is None else bool(np.allclose(corr, corr.T, atol=1e-5)),
        "prior_symmetric_before_shuffle": bool(np.allclose(vp.correlation_to_eta(corr, args.eta_min, args.eta_max),
                                                           vp.correlation_to_eta(corr, args.eta_min, args.eta_max).T)) if corr is not None else True,
        "prior_within_bounds": bool(args.prior == "none" or (eta_used[mask].min() >= args.eta_min - 1e-9 and eta_used[mask].max() <= args.eta_max + 1e-9) or args.prior == "uniform"),
        "candidate_direction_preserved": bool(np.all(np.sign(w_coef) * np.sign(a) >= 0)),
        "non_candidates_zero_P": bool(np.all(p[~mask] == 0)),
        "posterior_finite_in_01": bool(np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 + 1e-12),
        "refined_finite": bool(np.isfinite(w_coef).all()),
        "diagonal_zero": all(np.all(np.diag(g) == 0) for g in graphs.values()),
        "in_degree_at_most_K": all(((g != 0).sum(axis=1) <= args.k).all() for g in graphs.values()),
        "edges_inside_candidate_mask": all(not np.any((g != 0) & ~mask) for g in graphs.values()),
        "psi_1_equals_A_1": bool(np.allclose(psi_V[1], a[0])) if 1 in psi_V else True,
        "training_rows_only": train_end == int(meta["train_rows"][1]),
    }
    log(f"checks: {checks}")
    if not all(checks.values()):
        raise AssertionError(f"E8C build checks failed: {checks}")

    variant = args.variant or (f"coef_{args.prior}_a{args.nu0_mult:g}_b{args.nu1_mult:g}_l{args.lam_mult:g}"
                               + ("_psiW" if args.psi_from == "W" else "") + (f"_seed{args.seed}" if args.prior == "shuffle" else ""))
    out_dir = Path(args.out_root) / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = Path(args.graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    # ---- diagnostics --------------------------------------------------------------------
    off_mask = mask & offdiag
    rank = lambda x: np.argsort(np.argsort(x))
    diagnostics = {
        "candidate_edges": int(mask.sum()), "candidates_per_target": quantiles(mask.sum(axis=1)),
        "prior_eta_on_candidates": quantiles(eta_used[mask]),
        "traffic_abs_correlation_on_candidates": quantiles(np.abs(corr)[mask]) if corr is not None else None,
        "coefficient_magnitude_original": quantiles(u[mask]), "coefficient_magnitude_refined": quantiles(w_hat[mask]),
        "share_coefficients_zeroed": float(np.mean(w_hat[mask] == 0)),
        "posterior_P": quantiles(p[mask]), "share_P_between_0.05_and_0.95": float(np.mean((p[mask] > 0.05) & (p[mask] < 0.95))),
        "rank_corr_P_with_original_magnitude": float(np.corrcoef(rank(p[mask]), rank(u[mask]))[0, 1]),
        "rank_corr_eta_with_original_magnitude": float(np.corrcoef(rank(eta_used[mask]), rank(u[mask]))[0, 1]),
        "per_horizon": {f"h{h}": {**graph_diagnostics(graphs[h]),
                                  "retained_edges": int((graphs[h] != 0).sum()),
                                  "jaccard_with_E8V_graph": jaccard(graphs[h], graphs_E8V[h]),
                                  "jaccard_with_mask_only_graph": jaccard(graphs[h], graphs_mask[h])} for h in horizons},
        "checks": checks,
    }
    for h in horizons:
        d = diagnostics["per_horizon"][f"h{h}"]
        log(f"  h={h:2d}: {d['edges']} edges, mean in-degree {d['average_in_degree']:.1f}, density {d['density']:.5f}, "
            f"J(E8V) {d['jaccard_with_E8V_graph']:.3f}, J(mask-only) {d['jaccard_with_mask_only_graph']:.3f}")

    # ---- save -------------------------------------------------------------------------------
    method = get_experiment("E8").graph_method
    written = []
    for h in horizons:
        path = graph_dir / f"E8__h{h}__{C.VAR_RUN}__{variant}.npz"
        save_artifact(path, graphs[h], experiment="E8", method=method, train_end=train_end,
                      sensors=np.asarray(sensors, dtype=np.int64), horizon=h, top_k=args.k,
                      source_run=f"{C.VAR_RUN}__{variant}")
        path.with_suffix(".json").write_text(json.dumps({"experiment": "E8C", "variant": variant, "horizon": h,
                                                         "top_k": args.k, "train_end": train_end,
                                                         **diagnostics["per_horizon"][f"h{h}"]}, indent=2), encoding="utf-8")
        written.append(path.name)
    rows, cols = np.nonzero(mask)
    np.savez_compressed(out_dir / "edges.npz", target=rows.astype(np.int32), source=cols.astype(np.int32),
                        original_coefficients=a[:, rows, cols].astype(np.float32), refined_coefficients=w_coef[:, rows, cols].astype(np.float32),
                        eta=eta_used[rows, cols].astype(np.float32), posterior_P=p[rows, cols].astype(np.float32),
                        traffic_corr=(corr[rows, cols].astype(np.float32) if corr is not None else np.zeros(len(rows), np.float32)))
    config_record = {"experiment": "E8C", "variant": variant, "prior": args.prior, "args": vars(args),
                     "scale_s": s, "nu0": nu0, "nu1": nu1, "lam": lam, "q": args.q,
                     "formulas": {"u": "||(A_1,A_2,A_3)[i,j]||_2", "eta": "eta_min + (eta_max-eta_min)|R_ij|",
                                  "map": "argmin_w q/2 (w-u)^2 - lam log[(1-eta) f0(w) + eta f1(w)], 0<=w<=u",
                                  "posterior": "eta f1(w)/(eta f1(w) + (1-eta) f0(w))",
                                  "S_h": "M P |Psi_h|", "graph": "top-K sources per target, G[target, source]"},
                     "corr_cache": corr_file, "train_end": train_end, "num_sensors": int(n),
                     "no_var_residuals": True, "no_lingam": True, "graph_files": written}
    (out_dir / "config.json").write_text(json.dumps(config_record, indent=2, default=str), encoding="utf-8")
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    log(f"saved {len(written)} graphs to {graph_dir} and details to {out_dir}; variant = {variant}")


if __name__ == "__main__":
    main()
