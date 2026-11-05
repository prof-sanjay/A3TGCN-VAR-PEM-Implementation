"""
E8: horizon-conditioned VAR-PEM graphs, WITHOUT LiNGAM.

E5 builds   Theta_h = Psi_h (I - P * B0)^-1   and keeps the top-K sources of |Theta_h|.
E8 removes DirectLiNGAM and B0 entirely:

    S_h = P (x) |Psi_h|            (element-wise product, diagonal 0)
    G_h = top-K sources per target of S_h        (K = 25, as E5)

    Psi_0 = I ,   Psi_s = sum_{k=1}^{min(s, p)} A_k Psi_{s-k}

E8V is the ablation that isolates PEM: S_h = |Psi_h| (no P).

This is a NEW formulation, not an algebraic variant of E5: E5 passes the
same-time effects through (I - P B0)^-1; E8 has no same-time effects and uses
P only as a weight on the VAR responses. Because P is non-zero only for pairs
that share a local PEM unit, and is almost exactly 0 or 1 where it is
non-zero, P (x) |Psi_h| mostly acts as a mask that keeps the sources inside
the PEM-supported neighbourhood; E8V shows what the mask changes.

Conventions (identical to E5):
    A_k[target, source], Psi_h[target, source], P symmetric,
    G_h[target, source], stored as such; the model orientation (one
    transposition) is applied later by main.py, as for every experiment.

This module imports no LiNGAM code and reads no B0.
"""

from __future__ import annotations

import sys

import numpy as np

from causal.graph_builders import top_k_sources
from causal.horizon import StructuralIRF


def impulse_responses(coefficients: np.ndarray, horizons) -> dict[int, np.ndarray]:
    """Psi_h for each requested horizon (reduced-form VAR responses), float64."""

    horizons = sorted({int(h) for h in horizons})
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must be positive integers.")

    irf = StructuralIRF()
    coefficients = irf._validate_var_coefficients(coefficients)
    psi = irf._calculate_psi(coefficients, max(horizons))

    return {h: psi[h] for h in horizons}


def weighted_effects(psi: dict[int, np.ndarray], probabilities=None) -> dict[int, np.ndarray]:
    """
    S_h = P (x) |Psi_h| with the diagonal set to 0; with probabilities=None,
    S_h = |Psi_h| (the E8V ablation).
    """

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        n = next(iter(psi.values())).shape[0]

        if probabilities.shape != (n, n):
            raise ValueError(f"P has shape {probabilities.shape}, expected {(n, n)}.")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("P must be finite.")
        if probabilities.min() < 0.0 or probabilities.max() > 1.0 + 1e-9:
            raise ValueError("P must lie in [0, 1].")

    effects = {}
    for h, matrix in psi.items():
        s = np.abs(matrix)
        if probabilities is not None:
            s = probabilities * s
        np.fill_diagonal(s, 0.0)
        effects[h] = s

    return effects


def horizon_graphs_from_effects(effects: dict[int, np.ndarray], k: int) -> dict[int, np.ndarray]:
    """Top-k sources per target of each effect matrix (the E5 selection rule)."""

    return {h: top_k_sources(s, k) for h, s in effects.items()}


def var_pem_horizon_graphs(coefficients, probabilities, horizons, k):
    """E8: returns (psi, effects, graphs), each a dict keyed by horizon."""

    psi = impulse_responses(coefficients, horizons)
    effects = weighted_effects(psi, probabilities)
    return psi, effects, horizon_graphs_from_effects(effects, k)


def var_horizon_graphs(coefficients, horizons, k):
    """E8V (ablation): S_h = |Psi_h|. Returns (psi, effects, graphs)."""

    psi = impulse_responses(coefficients, horizons)
    effects = weighted_effects(psi, None)
    return psi, effects, horizon_graphs_from_effects(effects, k)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def psi_diagnostics(coefficients: np.ndarray, psi: dict[int, np.ndarray]) -> dict:
    """VAR coefficient / impulse response shapes and value ranges per horizon."""

    return {
        "var_coefficient_shape": list(np.asarray(coefficients).shape),
        "impulse_response_shape": list(next(iter(psi.values())).shape),
        "per_horizon": {
            int(h): {"min": float(m.min()), "max": float(m.max()),
                     "min_abs_off_diagonal": float(np.abs(m[~np.eye(len(m), dtype=bool)]).min()),
                     "max_abs_off_diagonal": float(np.abs(m[~np.eye(len(m), dtype=bool)]).max())}
            for h, m in psi.items()
        },
    }


def probability_diagnostics(p: np.ndarray) -> dict:
    """Shape, share of zero / high probabilities, mean and median of the non-zero ones."""

    p = np.asarray(p, dtype=np.float64)
    off = ~np.eye(len(p), dtype=bool)
    values = p[off]
    nonzero = values[values > 0]

    return {
        "shape": list(p.shape),
        "percent_zero": float(100 * np.mean(values == 0)),
        "percent_above_0.5": float(100 * np.mean(values > 0.5)),
        "percent_above_0.99": float(100 * np.mean(values > 0.99)),
        "mean_nonzero": float(nonzero.mean()) if nonzero.size else 0.0,
        "median_nonzero": float(np.median(nonzero)) if nonzero.size else 0.0,
        "symmetric": bool(np.allclose(p, p.T)),
    }


def graph_diagnostics(graph: np.ndarray) -> dict:
    """Edge count, in-degree (row) statistics, isolated sensors, density, weights."""

    graph = np.asarray(graph)
    n = len(graph)
    edges = graph != 0
    np.fill_diagonal(edges, False)
    in_degree = edges.sum(axis=1)
    out_degree = edges.sum(axis=0)
    weights = graph[edges]

    return {
        "edges": int(edges.sum()),
        "average_in_degree": float(in_degree.mean()),
        "min_in_degree": int(in_degree.min()),
        "max_in_degree": int(in_degree.max()),
        "max_out_degree": int(out_degree.max()),
        "percent_isolated_sensors": float(100 * np.mean((in_degree == 0) & (out_degree == 0))),
        "percent_without_incoming": float(100 * np.mean(in_degree == 0)),
        "density": float(edges.sum() / (n * (n - 1))),
        "weight_min": float(weights.min()) if weights.size else 0.0,
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "weight_max": float(weights.max()) if weights.size else 0.0,
        "self_loops": int(np.count_nonzero(np.diag(graph))),
    }


def lingam_access_report(opened_files=()) -> dict:
    """
    Evidence that nothing LiNGAM-related was used: no lingam module is
    imported in this process, and none of the files the E8 build opened
    looks like a LiNGAM result or a B0 file.
    """

    import re
    from pathlib import PurePath

    modules = sorted(name for name in sys.modules if "lingam" in name.lower())

    def looks_like_lingam(path) -> bool:
        # Only the file name and its two parent folders: the project folder's own
        # name ("... VAR-LiNGAM Causal Graph ...") must not count.
        parts = [p.lower() for p in PurePath(str(path).replace("\\", "/")).parts[-3:]]
        name_tokens = re.split(r"[^a-z0-9]+", parts[-1])
        return any("lingam" in p for p in parts) or "b0" in name_tokens

    files = [str(f) for f in opened_files if looks_like_lingam(f)]

    return {
        "lingam_modules_imported": modules,
        "lingam_or_b0_files_opened": files,
        "b0_used": False,
        "ok": not modules and not files,
    }


def save_artifact(path, graph: np.ndarray, **fields) -> None:
    """
    Graph artifact in the format main.load_graph reads (same fields as
    causal.graph_builders.save_graph_artifact), compressed: the graphs are
    mostly zeros, so a 17 MB file becomes well under 1 MB.
    """

    np.savez_compressed(
        path,
        adjacency=np.asarray(graph, dtype=np.float32),
        num_nodes=int(graph.shape[0]),
        orientation="target_source",
        **fields,
    )
