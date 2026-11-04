"""
Validation proxy for graph hyperparameters (E5 tuning, Phase A).

Why a proxy: one A3T-GCN run takes 5-6 hours and its seed noise
(0.8-1.4 test RMSE) is larger than the differences between graph settings
of the same family, so graphs cannot be tuned by training A3T-GCN on each.

What the proxy imitates: A3T-GCN never uses a graph edge by edge. It mixes
every sensor's signal with its sources' signals through the FIXED operator

    A_hat = D^-1/2 (G + I) D^-1/2 ,   D_jj = 1 + sum_i G[i, j]
    (G[target, source]; this is graph/normalization.py after the single
    transposition of decision D2)

and then applies weights that are SHARED by all sensors. The proxy keeps
exactly these two properties and drops everything else:

    u(t)        = A_hat x(t)                    (graph mixing, per sensor)
    y_i(t + h)  ~ b + sum_{k=0}^{L-1} w_k u_i(t - k)

with ONE set of coefficients (w, b) fitted by least squares over all
sensors on the training split (stride-subsampled windows) and scored on
the validation split. A graph that dilutes the own signal with
uninformative sources is penalised, a graph whose mixed signal is
informative is rewarded, and NOGRAPH is A_hat = I (a shared AR model).

Rules that keep it valid:
    - only training rows are used to fit, only validation rows to score;
      test rows are never read (the caller passes traffic[:val_end]);
    - observed values only: a window is used when all of its L input rows
      are fully observed, and a target only when it was observed;
    - windows never cross a split boundary;
    - no randomness: a score is a deterministic function of the graph.

The proxy is a linear stand-in. tune_e5_proxy.py first checks that it ranks
graphs whose A3T-GCN results are known in the same order (calibration).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def operator(graph) -> sp.csr_matrix:
    """A_hat = D^-1/2 (G + I) D^-1/2 for G[target, source] (dense or sparse)."""

    g = sp.csr_matrix(graph, dtype=np.float64)
    n = g.shape[0]

    degree = np.asarray(g.sum(axis=0)).ravel() + 1.0
    scale = 1.0 / np.sqrt(degree)

    a = (g + sp.eye(n, format="csr")).tocsr()

    return (sp.diags(scale) @ a @ sp.diags(scale)).tocsr()


def own_signal_share(op: sp.csr_matrix) -> float:
    """Median diagonal of A_hat: the share of a sensor's own signal."""

    return float(np.median(op.diagonal()))


class ProxyData:
    """
    Traffic rows [0, val_end) scaled to ~[0, 1], with observed-value masks.

    traffic:   (rows, N) flow with NaN for missing; rows >= val_end must not
               be passed (the test split is never touched).
    train_end: first validation row.
    val_end:   first test row.
    """

    def __init__(self, traffic: np.ndarray, train_end: int, val_end: int,
                 scale: float, lags: int = 12):

        if traffic.shape[0] != val_end:
            raise ValueError("pass traffic[:val_end]: test rows are not allowed here.")
        if not 0 < train_end < val_end:
            raise ValueError("need 0 < train_end < val_end.")

        self.train_end = int(train_end)
        self.val_end = int(val_end)
        self.scale = float(scale)
        self.lags = int(lags)

        x = np.asarray(traffic, dtype=np.float32) / np.float32(self.scale)
        self.observed = np.isfinite(x)
        self.x = np.where(self.observed, x, np.float32(0.0)).astype(np.float32)
        self.num_nodes = x.shape[1]

        # Row r can feed a window only if every sensor is observed at r.
        self.row_ok = self.observed.all(axis=1)

    # -- windows -----------------------------------------------------------

    def window_ends(self, split: str, horizon: int, stride: int) -> np.ndarray:
        """
        Last input row t of every usable window of a split: inputs are rows
        t-L+1..t (all fully observed, all inside the split), target row t+h
        (inside the split). Every `stride`-th usable window is returned.
        """

        lo, hi = (0, self.train_end) if split == "train" else (self.train_end, self.val_end)

        first = lo + self.lags - 1
        last = hi - horizon - 1
        if last < first:
            return np.empty(0, dtype=np.int64)

        ends = np.arange(first, last + 1)

        bad = np.concatenate([[0], np.cumsum(~self.row_ok)])
        bad_in_window = bad[ends + 1] - bad[ends - self.lags + 1]

        return ends[bad_in_window == 0][::stride]

    # -- features ----------------------------------------------------------

    def mix(self, op: sp.csr_matrix) -> np.ndarray:
        """u(t) = A_hat x(t) for every row, shape (rows, N), float32."""

        product = op.astype(np.float32).dot(self.x.T)

        return np.ascontiguousarray(np.asarray(product).T)

    def _samples(self, mixed: np.ndarray, ends: np.ndarray, horizon: int):
        """Yield (Z, y) blocks: Z (n, L+1) with intercept last, y (n,)."""

        chunk = 512

        for c in range(0, len(ends), chunk):
            tt = ends[c:c + chunk]

            target = self.x[tt + horizon]
            keep = self.observed[tt + horizon]

            z = np.empty((len(tt), self.num_nodes, self.lags + 1), dtype=np.float32)
            for k in range(self.lags):
                z[:, :, k] = mixed[tt - k]
            z[:, :, self.lags] = 1.0

            yield z[keep].astype(np.float64), target[keep].astype(np.float64)

    # -- fit and score -----------------------------------------------------

    def fit(self, mixed: np.ndarray, horizon: int, stride: int = 4):
        """Shared least-squares coefficients on training windows."""

        ends = self.window_ends("train", horizon, stride)
        if len(ends) == 0:
            raise ValueError("no usable training window.")

        size = self.lags + 1
        gram = np.zeros((size, size))
        moment = np.zeros(size)

        for z, y in self._samples(mixed, ends, horizon):
            gram += z.T @ z
            moment += z.T @ y

        ridge = 1e-9 * np.trace(gram) / size
        return np.linalg.solve(gram + ridge * np.eye(size), moment)

    def score(self, mixed: np.ndarray, weights: np.ndarray, horizon: int,
              split: str = "val", stride: int = 2) -> dict:
        """RMSE / MAE of the shared linear forecaster, in vehicles per 5 min."""

        ends = self.window_ends(split, horizon, stride)
        if len(ends) == 0:
            raise ValueError(f"no usable {split} window.")

        sse = sae = 0.0
        count = 0

        for z, y in self._samples(mixed, ends, horizon):
            error = z @ weights - y
            sse += float(error @ error)
            sae += float(np.abs(error).sum())
            count += len(y)

        return {
            "rmse": float(np.sqrt(sse / count)) * self.scale,
            "mae": float(sae / count) * self.scale,
            "windows": int(len(ends)),
            "samples": int(count),
        }

    def evaluate(self, graph, horizons, stride_train: int = 4, stride_val: int = 2) -> dict:
        """
        Proxy validation scores of one graph at several horizons.
        The graph is mixed once; each horizon has its own fitted weights.
        """

        op = operator(graph)
        mixed = self.mix(op)

        out = {"own_signal_share": own_signal_share(op),
               "edges": int((sp.csr_matrix(graph) != 0).sum())}

        for h in horizons:
            weights = self.fit(mixed, h, stride_train)
            out[int(h)] = self.score(mixed, weights, h, "val", stride_val)

        return out


# ---------------------------------------------------------------------------
# E5 graph recipe with replaceable ingredients
# ---------------------------------------------------------------------------

def effect_matrices(psi: np.ndarray, b0_effective: np.ndarray, horizons) -> dict:
    """
    |Theta_h| off-diagonal for each horizon, with Theta_h = Psi_h (I - B0~)^-1
    (the same algebra as causal/horizon.py). `psi` holds Psi_0..Psi_max.
    """

    from causal.horizon import StructuralIRF

    horizons = [int(h) for h in horizons]
    irf = StructuralIRF()

    b0_effective = np.array(b0_effective, dtype=np.float64)
    np.fill_diagonal(b0_effective, 0.0)

    theta = irf._calculate_theta(psi[horizons], b0_effective)

    return {
        h: np.abs(theta[i]) * (1.0 - np.eye(theta.shape[1]))
        for i, h in enumerate(horizons)
    }
