"""Figures for the ablation report (presentation/figures/*.png)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import compare_results as cr          # palette + rcParams shared with the reports

C = cr.EXPERIMENT_COLORS
GRAY = cr.BASELINE_COLOR
INK = cr.TEXT_PRIMARY
INK2 = cr.TEXT_SECONDARY
D = json.load(open("presentation/data.json"))
OUT = Path("presentation/figures")
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def col(name):
    """Experiment colour; baselines (HA, NOGRAPH) in muted gray."""
    return C.get(name[:2], GRAY) if name.startswith("E") else GRAY


# ---------- block diagrams ----------

def box(ax, x, y, w, h, text, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=INK, lw=1))


def blank(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, xlim); ax.set_ylim(0, ylim); ax.axis("off"); ax.grid(False)
    return fig, ax


def framework():
    fig, ax = blank((9, 6.2), 10, 7.4)
    w, h = 2.6, 0.8
    box(ax, 0.2, 6.1, w, h, "LargeST-GBA 2021\n2,352 sensors, 5-min flow")
    box(ax, 0.2, 4.9, w, h, "Sensor filter (copied series removed)\n2,056 sensors")
    box(ax, 0.2, 3.7, w, h, "Chronological split\ntrain 80% / val 10% / test 10%")
    arrow(ax, 1.5, 6.1, 1.5, 5.7); arrow(ax, 1.5, 4.9, 1.5, 4.5)

    ax.text(5.0, 7.15, "Causal graph discovery (training split only)", ha="center",
            fontsize=9, weight="bold", color=INK)
    box(ax, 3.7, 6.1, w, h, "1. Road-restricted VAR (lag 3)\nlag matrices A_k, residuals", "#f4f4f2")
    box(ax, 3.7, 4.9, w, h, "2. Local DirectLiNGAM on\nresiduals -> B0 (same-time effects)", "#f4f4f2")
    box(ax, 3.7, 3.7, w, h, "3. Local PEM (spike-and-slab)\ninclusion prob. P, edges D", "#f4f4f2")
    box(ax, 3.7, 2.5, w, h, "4. Impulse responses\nTheta_h for each horizon h", "#f4f4f2")
    arrow(ax, 2.8, 4.1, 3.7, 6.5)
    for y in (6.1, 4.9, 3.7):
        arrow(ax, 5.0, y, 5.0, y - 0.4)

    box(ax, 7.1, 4.3, 2.8, 1.7, "Graph builder (one graph per run)\nE0 road  |  E1 VAR\nE2 VAR-LiNGAM  |  E3 VAR on PEM\n"
                                "E4 E2 + P  |  E5 horizon top-25\nE6 data-driven  |  E7 road prior")
    for y in (6.5, 5.3, 4.1, 2.9):
        arrow(ax, 6.3, y, 7.1, 5.1)

    box(ax, 7.1, 1.6, 2.8, 1.2, "A3T-GCN (unchanged)\nGCN + GRU + attention\n12 inputs -> 1 output", "#e9eef7")
    arrow(ax, 8.5, 4.3, 8.5, 2.8)
    ax.plot([1.5, 1.5], [3.7, 2.0], color=INK, lw=1)
    arrow(ax, 1.5, 2.0, 7.1, 2.0)
    ax.text(1.7, 2.12, "windows: 12 past steps -> value h steps ahead", fontsize=7, color=INK2)
    box(ax, 7.1, 0.1, 2.8, 0.9, "Forecast at 5 / 15 / 30 / 60 min\nRMSE, MAE, MAPE on test split")
    arrow(ax, 8.5, 1.6, 8.5, 1.0)
    save(fig, "fig01_framework.png")


def a3tgcn():
    fig, ax = blank((9, 3.4), 10, 3.6)
    box(ax, 0.1, 2.3, 1.8, 0.9, "Graph G\n[target, source]")
    box(ax, 0.1, 0.6, 1.8, 0.9, "Input window\n12 steps x 2,056")
    box(ax, 2.4, 2.3, 2.0, 0.9, "Normalise\nD^-1/2 (G^T + I) D^-1/2", "#f4f4f2")
    box(ax, 2.4, 0.6, 2.0, 0.9, "T-GCN cell (x 12)\ngraph conv inside GRU", "#e9eef7")
    box(ax, 5.0, 0.6, 1.8, 0.9, "Attention over\n12 hidden states", "#e9eef7")
    box(ax, 7.3, 0.6, 1.2, 0.9, "Output\nlayer")
    box(ax, 8.9, 0.6, 1.0, 0.9, "Flow at\nt + h")
    arrow(ax, 1.9, 2.75, 2.4, 2.75); arrow(ax, 3.4, 2.3, 3.4, 1.5); arrow(ax, 1.9, 1.05, 2.4, 1.05)
    arrow(ax, 4.4, 1.05, 5.0, 1.05); arrow(ax, 6.8, 1.05, 7.3, 1.05); arrow(ax, 8.5, 1.05, 8.9, 1.05)
    ax.text(4.6, 2.75, "the only place the graph enters the model;\nevery experiment changes G only",
            fontsize=7.5, color=INK2, va="center")
    save(fig, "fig02_a3tgcn.png")


# ---------- results charts ----------

FIVE_MIN = [("HA", "HA_h1"), ("NOGRAPH", "NOGRAPH_h1_s42"), ("E0", "E0_h1_s42"), ("E1", "E1_h1_s42"),
            ("E2", "E2_h1_s42"), ("E3a (P*D)", "E3PxD_h1_s42"), ("E3b", "E3_h1_s42"), ("E5", "E5_h1_s42"), ("E7", "E7_h1_s42")]


def bars_5min():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, m in zip(axes, ["rmse", "mae"]):
        names = [n for n, _ in FIVE_MIN]
        vals = [D[k][m] for _, k in FIVE_MIN]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=[col(n) for n in names], height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.6, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y, names); ax.set_xlim(0, max(vals) * 1.15); ax.grid(axis="y", visible=False)
        ax.set_title(f"Test {m.upper()} at 5 min (seed 42), lower is better")
    fig.tight_layout(); save(fig, "fig03_bars_5min.png")


def horizon():
    mins = {1: 5, 3: 15, 6: 30, 12: 60}
    series = {"HA": "HA_h{h}", "NOGRAPH": "NOGRAPH_h{h}_s42", "E0": "E0_h{h}_s42",
              "E1": "E1_h{h}_s42", "E5": "E5_h{h}_s42"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, m in zip(axes, ["rmse", "mae"]):
        for name, pat in series.items():
            pts = [(mins[h], D[pat.format(h=h)][m]) for h in mins if pat.format(h=h) in D]
            x, v = zip(*pts)
            ls = "--" if name in ("HA", "NOGRAPH") else "-"
            ax.plot(x, v, ls, marker="s" if name == "NOGRAPH" else "o", ms=6, color=col(name), label=name)
        ax.set_xticks([5, 15, 30, 60], ["5", "15\n(pending)", "30", "60"]); ax.set_xlim(0, 65)
        ax.set_xlabel("forecast horizon (minutes)"); ax.set_title(f"Test {m.upper()} vs horizon (seed 42)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.25), ncol=5)
    fig.tight_layout(); save(fig, "fig04_horizon.png")


def seeds():
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    exps = ["E0", "E1", "E5"]
    x = np.arange(3)
    for i, (s, alpha) in enumerate([("s42", 0.5), ("s7", 1.0)]):
        xs = x + (i - 0.5) * 0.36
        vals = [D[f"{e}_h1_{s}"]["rmse"] for e in exps]
        ax.bar(xs, vals, 0.34, color=[col(e) for e in exps], alpha=alpha)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.15, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [f"{e}\nseed 42  |  seed 7" for e in exps]); ax.set_ylim(24, 31.5)
    ng = D["NOGRAPH_h1_s42"]["rmse"]
    ax.axhline(ng, ls="--", color=GRAY, lw=1)
    ax.text(2.6, ng - 0.4, "NOGRAPH, seed 42", fontsize=7, color=INK2, ha="right")
    ax.set_title("Seed effect: test RMSE at 5 min (y-axis starts at 24)"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "fig05_seeds.png")


def own_signal():
    s = D["own_signal_share"]
    pts = [("NOGRAPH", "NOGRAPH", "NOGRAPH_h1_s42"), ("E0", "E0", "E0_h1_s42"), ("E1", "E1", "E1_h1_s42"),
           ("E2", "E2", "E2_h1_s42"), ("E3a (P*D)", "E3PxD", "E3PxD_h1_s42"), ("E3b", "E3", "E3_h1_s42"),
           ("E5", "E5_h1", "E5_h1_s42")]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, sk, rk in pts:
        ax.scatter(s[sk] * 100, D[rk]["rmse"], s=64, color=col(name), edgecolor="white", linewidth=1.5, zorder=3)
        off = {"E1": (-16, -12), "E5": (6, 6)}.get(name, (6, 4))
        ax.annotate(name, (s[sk] * 100, D[rk]["rmse"]), xytext=off, textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("own-signal share after normalisation (median over sensors, %)")
    ax.set_ylabel("test RMSE, 5 min, seed 42"); ax.set_xlim(-5, 112)
    ax.set_title("Share of its own signal each sensor keeps vs test error")
    fig.tight_layout(); save(fig, "fig06_own_signal.png")


def val_curves():
    runs = [("NOGRAPH", "NOGRAPH_h1_s42"), ("E0", "E0_h1_s42"), ("E1", "E1_h1_s42"),
            ("E2", "E2_h1_s42"), ("E3b", "E3_h1_s42"), ("E5", "E5_h1_s42"), ("E7", "E7_h1_s42")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.8), sharex=True, sharey=True)
    axes.flat[-1].axis("off")
    for ax, (name, k) in zip(axes.flat, runs):
        v = D[k]["val_rmse"]
        ax.plot(range(1, len(v) + 1), v, marker="o", ms=3, color=col(name))
        ax.set_title(name); ax.set_yscale("log"); ax.set_ylim(20, 300)
    for ax in axes[1, :3]:
        ax.set_xlabel("epoch")
    axes[0, 3].xaxis.set_tick_params(labelbottom=True); axes[0, 3].set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation RMSE (log)")
    fig.suptitle("Validation RMSE per epoch, 5 min, seed 42 (spikes = training collapses)",
                 fontsize=10, weight="bold")
    fig.tight_layout(); save(fig, "fig07_validation_curves.png")


def time_of_day():
    def hourly(path):
        z = np.load(path)
        hours = pd.to_datetime(z["timestamps_ns"]).hour.values
        err = np.abs(z["predictions"] - z["targets"])
        return pd.Series(np.nanmean(err, axis=1)).groupby(hours).mean()

    K = "results/kaggle/"
    series = {"HA": "results/HA/test_predictions.npz",
              "NOGRAPH": K + "output_NOGRAPH_final/test_predictions.npz",
              "E0": K + "output_E0_final/test_predictions.npz",
              "E1": K + "output_E1_final/test_predictions.npz"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, p in series.items():
        h = hourly(p)
        ax.plot(h.index, h.values, "--" if name in ("HA", "NOGRAPH") else "-",
                marker="s" if name == "NOGRAPH" else None, ms=4, color=col(name), label=name)
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("hour of day"); ax.set_ylabel("test MAE")
    ax.set_title("Error by hour of day, 5 min, seed 42"); ax.legend(ncol=4, loc="upper left")
    fig.tight_layout(); save(fig, "fig08_time_of_day.png")


# ---------- one block diagram per experiment ----------

STAGES = ["road", "var", "lingam", "pem", "irf"]
STAGE_TEXT = {
    "road": "Candidate links\nroad network",
    "var": "VAR (lag 3)\nlag matrices A_k",
    "lingam": "DirectLiNGAM\nsame-time effects B0",
    "pem": "PEM\nedge prob. P, edges D",
    "irf": "Impulse responses\nTheta_h, top-25",
}
EXPERIMENT_DIAGRAMS = {
    "NOGRAPH": dict(used=[], graph="Empty graph\nA = 0"),
    "E0": dict(used=["road"], text={"road": "Road network\n(used as the graph)"}, graph="G = road adjacency"),
    "E1": dict(used=["road", "var"], graph="G = sum_k |A_k|"),
    "E2": dict(used=["road", "var", "lingam"], graph="G = sum_k |B_k| + |B0|\nB_k = (I - B0) A_k"),
    "E3a": dict(used=["road", "var", "pem"], graph="G = P x D"),
    "E3b": dict(used=["road", "var", "pem"], graph="G = W x D\nW = sum_k |A_k|"),
    "E4": dict(used=["road", "var", "lingam", "pem"], graph="G = sum_k |B_k| + P x |B0|"),
    "E5": dict(used=["road", "var", "lingam", "pem", "irf"], graph="G_h = top-25 sources\nof Theta_h (per horizon)"),
    "E6": dict(used=["road", "var", "lingam", "pem"], text={"road": "Candidate links: top-25\ncorrelated sensors"},
               graph="G = sum_k |B_k| + P x |B0|\n(data-driven candidates)"),
    "E7": dict(used=["road", "var", "lingam", "pem"], text={"pem": "PEM, road prior\n0.7 road / 0.3 other"},
               graph="G = sum_k |B_k| + P x |B0|\n(road prior in PEM)"),
}
UNUSED = "#b4b2ac"


def stage_box(ax, x, y, w, h, text, used):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#ffffff" if used else SURFACE_FADED, ec=INK if used else UNUSED,
                                lw=1.4 if used else 1, ls="-" if used else "--"))
    ax.text(x + w / 2, y + h / 2 + (0 if used else 0.08), text, ha="center", va="center", fontsize=7.5,
            color=INK if used else UNUSED)
    if not used:
        ax.text(x + w / 2, y + 0.12, "not used", ha="center", fontsize=6.5, color=UNUSED, style="italic")


SURFACE_FADED = "#f7f7f5"


def experiment_diagram(name, used, graph, text=None):
    text = {**STAGE_TEXT, **(text or {})}
    fig, ax = blank((9, 3.7), 10, 4.3)
    ax.text(0.1, 4.15, f"{name}: models used (solid) and not used (dashed)", fontsize=9.5, weight="bold", color=INK)
    w, h, gap = 1.84, 0.85, 0.13
    xs = [0.1 + i * (w + gap) for i in range(len(STAGES))]
    y_top = 2.3
    for x, s in zip(xs, STAGES):
        stage_box(ax, x, y_top, w, h, text[s], s in used)
    # pipeline order between used stages (each step works on the previous one's output)
    chain = [s for s in STAGES if s in used]
    for a, b in zip(chain, chain[1:]):
        ia, ib = STAGES.index(a), STAGES.index(b)
        if ib == ia + 1:
            arrow(ax, xs[ia] + w, y_top + h / 2, xs[ib], y_top + h / 2)
        else:                                   # skips an unused stage: route above it
            ax.annotate("", (xs[ib] + w / 2, y_top + h), (xs[ia] + w / 2, y_top + h),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1, connectionstyle="arc3,rad=-0.22"))
    # graph builder, model, forecast
    gx, gy, gw, gh = 0.1, 0.45, 3.0, 1.05
    stage_box(ax, gx, gy, gw, gh, "Graph builder\n" + graph, True)
    if chain:                                   # last model's output -> graph builder
        x = xs[STAGES.index(chain[-1])] + w / 2
        ax.annotate("", (min(x, gx + gw - 0.3), gy + gh), (x, y_top),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    stage_box(ax, 4.0, 0.45, 2.3, 1.05, "A3T-GCN (unchanged)\nGCN + GRU + attention", True)
    stage_box(ax, 7.2, 0.45, 2.7, 1.05, "Forecast at t + h\n(5 / 15 / 30 / 60 min)", True)
    arrow(ax, gx + gw, gy + gh / 2, 4.0, 0.975); arrow(ax, 6.3, 0.975, 7.2, 0.975)
    ax.text(4.0, 0.15, "input: last 12 steps of all 2,056 sensors", fontsize=7, color=INK2)
    save(fig, f"exp_{name}.png")


def ha_diagram():
    fig, ax = blank((9, 1.9), 10, 2.0)
    ax.text(0.1, 1.8, "HA: no learned model and no graph", fontsize=9.5, weight="bold", color=INK)
    stage_box(ax, 0.1, 0.3, 2.6, 1.05, "Training data\n(80% split)", True)
    stage_box(ax, 3.6, 0.3, 3.0, 1.05, "Average flow per\nweekday, 5-min slot, sensor", True)
    stage_box(ax, 7.5, 0.3, 2.4, 1.05, "Forecast = average\nfor the target's slot", True)
    arrow(ax, 2.7, 0.825, 3.6, 0.825); arrow(ax, 6.6, 0.825, 7.5, 0.825)
    save(fig, "exp_HA.png")


def experiment_diagrams():
    ha_diagram()
    for name, spec in EXPERIMENT_DIAGRAMS.items():
        experiment_diagram(name, **spec)


for f in (framework, a3tgcn, bars_5min, horizon, seeds, own_signal, val_curves, time_of_day, experiment_diagrams):
    f()
print("ok")
