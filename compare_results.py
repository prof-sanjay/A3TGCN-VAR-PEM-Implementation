"""
Compare runs of E0-E7 / NOGRAPH: tables, significance tests and figures.

    python compare_results.py --runs E0=results/E0 E1=results/E1 E5=results/E5
                              [--pairs E5:E0 E5:E4] [--out results/comparison]

The FIRST run is the reference: by default every other run is compared
with it (deltas, significance tests, per-sensor maps); --pairs overrides.
Runs at different forecast horizons (main.py --horizon) are compared
only within the same horizon (they score different targets); with
several horizons each gets its own sub-folder (h1/, h3/, ...) plus
figures/10_error_by_horizon and tables/error_by_horizon.csv.

Every run folder must come from main.py WITHOUT --selection-only
(it needs results.json, test_predictions.npz, per_sensor_metrics.csv,
time_of_day_metrics.csv, history.csv). The runs must have scored the
SAME test targets (same rows and sensors); this is checked.

Outputs (in --out):
    tables/   comparison.csv/.md, deltas.csv, breakdown_inputs.csv,
              significance.csv, graph_comparison.csv,
              var_grid.csv, pem_stage_a.csv (if those inputs exist)
    figures/  *.png (200 dpi) and *.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy import stats  # noqa: E402


IMPLEMENTATION_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Visual style (validated reference palette, light mode)
#   E0 blue, E1 orange, E2 aqua - fixed per experiment, never by rank.
#   Diverging blue (better) <-> gray <-> red (worse) for differences.
# ------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Validated 8-slot categorical palette (dataviz reference, light mode),
# one FIXED slot per experiment, never by rank. Baselines are neutral
# reference marks, not series. Slots 3-5 are below 3:1 contrast on the
# surface, so every figure carries visible labels and a CSV table.
EXPERIMENT_COLORS = {
    "E0": "#2a78d6",   # blue
    "E1": "#eb6834",   # orange
    "E2": "#1baf7a",   # aqua
    "E3": "#eda100",   # yellow
    "E4": "#e87ba4",   # magenta
    "E5": "#008300",   # green
    "E6": "#6250d6",   # violet
    "E7": "#e34948",   # red
}
BASELINE_COLOR = "#898781"   # NOGRAPH, HA: muted ink, dashed
FALLBACK_COLORS = list(EXPERIMENT_COLORS.values())

DIVERGING = LinearSegmentedColormap.from_list(
    "better_worse",
    ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a8201f"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "font.size": 9,
    "axes.axisbelow": True,
})


def experiment_of(label):
    """'E5@15min' -> 'E5'; labels may carry a horizon suffix."""

    return str(label).split("@")[0]


def color_of(label, index):

    experiment = experiment_of(label)

    if experiment in ("NOGRAPH", "HA"):
        return BASELINE_COLOR

    return EXPERIMENT_COLORS.get(experiment, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def save_figure(fig, out_dir, name):

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def direct_label(ax, x, y, text, color):
    """Label a line at its end (identity is never colour-alone)."""

    ax.annotate(
        text, (x, y), xytext=(4, 0), textcoords="offset points",
        va="center", fontsize=8, color=TEXT_PRIMARY,
        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": color, "lw": 1},
    )


# ============================================================
# Loading and consistency
# ============================================================

def load_run(label, run_dir):

    run_dir = Path(run_dir)

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    if results.get("selection_only"):
        raise ValueError(f"{label}: selection run, no test outputs.")

    predictions = np.load(run_dir / "test_predictions.npz")

    return {
        "label": label,
        "dir": run_dir,
        "horizon": int(results.get("config", {}).get("horizon", 1)),
        "results": results,
        "pred": {key: predictions[key] for key in predictions.files},
        "per_sensor": pd.read_csv(run_dir / "per_sensor_metrics.csv"),
        "time_of_day": pd.read_csv(run_dir / "time_of_day_metrics.csv"),
        "history": pd.read_csv(run_dir / "history.csv"),
    }


def check_same_targets(runs):

    reference = runs[0]["pred"]

    for run in runs[1:]:

        current = run["pred"]

        for key in ("target_rows", "sensors"):
            if not np.array_equal(reference[key], current[key]):
                raise ValueError(
                    f"{run['label']} scored different {key} than "
                    f"{runs[0]['label']}; runs are not comparable."
                )

        if not np.allclose(reference["targets"], current["targets"]):
            raise ValueError(f"{run['label']}: different target values.")


# ============================================================
# Tables
# ============================================================

def comparison_table(runs):

    rows = []

    for run in runs:

        r = run["results"]
        m = r["test_metrics"]
        epochs = len(r["history"])

        rows.append({
            "run": run["label"],
            "MAE": m["mae"],
            "RMSE": m["rmse"],
            "masked_MAPE_%": m.get("masked_mape"),
            "R2": m["r2"],
            "Accuracy": m["accuracy"],
            "best_val_RMSE": r["best_val_rmse_original"],
            "best_epoch": r["best_epoch"],
            "epochs": epochs,
            "training_hours": r["training_seconds"] / 3600.0,
            "seconds_per_epoch": r["training_seconds"] / max(1, epochs),
            "train_windows": r["valid_sequences"]["train"],
            "val_windows": r["valid_sequences"]["val"],
            "test_windows": r["valid_sequences"]["test"],
            "sensors": len(r["sensors"]),
            "graph_edges": r["graph"]["statistics"]["num_edges_off_diagonal"],
            "graph_density": r["graph"]["statistics"]["density_off_diagonal"],
        })

    return pd.DataFrame(rows)


def markdown_table(table):
    """Markdown table without the optional 'tabulate' dependency."""

    def cell(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(table.columns) + " |",
        "|" + "|".join("---" for _ in table.columns) + "|",
    ]

    for _, row in table.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")

    return "\n".join(lines) + "\n"


def deltas_table(table, pairs):

    rows = []

    for new, base in pairs:

        a = table.set_index("run").loc[new]
        b = table.set_index("run").loc[base]

        for metric in ("MAE", "RMSE", "masked_MAPE_%", "R2", "Accuracy"):

            rows.append({
                "comparison": f"{new} - {base}",
                "metric": metric,
                "base": b[metric],
                "new": a[metric],
                "difference": a[metric] - b[metric],
                "relative_%": 100.0 * (a[metric] - b[metric]) / abs(b[metric]),
            })

    return pd.DataFrame(rows)


def breakdown_table(runs):

    rows = []

    for run in runs:
        for part, values in run["results"]["test_breakdown"].items():
            rows.append({"run": run["label"], "inputs": part, **values})

    return pd.DataFrame(rows)


def diebold_mariano(loss_a, loss_b):
    """
    Diebold-Mariano test on the loss differential d = loss_a - loss_b
    (one-step forecasts) with a Newey-West variance (Bartlett kernel,
    lag = floor(T^(1/3))). Negative statistic: loss_a is smaller.
    """

    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T = len(d)
    mean = d.mean()
    centered = d - mean
    lag = int(np.floor(T ** (1.0 / 3.0)))

    variance = np.dot(centered, centered) / T

    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        variance += 2.0 * weight * np.dot(centered[k:], centered[:-k]) / T

    statistic = mean / np.sqrt(variance / T)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(statistic)))

    return statistic, p_value, lag


def significance_table(runs_by_label, pairs):

    rows = []

    for new, base in pairs:

        a = runs_by_label[new]
        b = runs_by_label[base]

        mae_a = a["per_sensor"]["mae"].to_numpy()
        mae_b = b["per_sensor"]["mae"].to_numpy()

        wilcoxon = stats.wilcoxon(mae_a, mae_b, alternative="two-sided")

        error_a = np.abs(a["pred"]["predictions"] - a["pred"]["targets"]).mean(axis=1)
        error_b = np.abs(b["pred"]["predictions"] - b["pred"]["targets"]).mean(axis=1)

        dm, dm_p, lag = diebold_mariano(error_a, error_b)

        rows.append({
            "comparison": f"{new} vs {base}",
            "sensors": len(mae_a),
            "sensors_improved_%": 100.0 * np.mean(mae_a < mae_b),
            "median_sensor_MAE_change": float(np.median(mae_a - mae_b)),
            "wilcoxon_statistic": float(wilcoxon.statistic),
            "wilcoxon_p": float(wilcoxon.pvalue),
            "timesteps": len(error_a),
            "DM_statistic_MAE": float(dm),
            "DM_p": float(dm_p),
            "DM_newey_west_lag": lag,
        })

    return pd.DataFrame(rows)


LOCAL_DATA_DIR = IMPLEMENTATION_DIR / "datasets" / "LargeST GBA"


def local_path(recorded, name):
    """
    Use the path recorded in results.json if it exists here; otherwise
    (e.g. a run downloaded from Kaggle) the local dataset folder.
    """

    recorded = Path(recorded)

    return recorded if recorded.exists() else LOCAL_DATA_DIR / name


def load_graph_for(run, sensors):

    graph = run["results"]["graph"]

    if graph["source"] == "road_adjacency":
        full = np.load(local_path(graph["path"], "gba_rn_adj.npy"), mmap_mode="r")
        return np.asarray(full[np.ix_(sensors, sensors)], dtype=np.float32)

    path = Path(graph["path"])

    if not path.exists():
        path = IMPLEMENTATION_DIR / "outputs" / "graphs" / path.name

    if not path.exists():
        return None

    with np.load(path) as artifact:
        return np.asarray(artifact["adjacency"], dtype=np.float32)


def graph_table(runs):

    sensors = runs[0]["pred"]["sensors"]

    graphs = {run["label"]: load_graph_for(run, sensors) for run in runs}

    rows = []
    in_degrees = {}
    reference = None

    for label, graph in graphs.items():

        if graph is None:
            continue

        edges = graph != 0
        np.fill_diagonal(edges, False)

        if reference is None:
            reference = (label, edges)

        overlap = (
            (edges & reference[1]).sum() / max(1, (edges | reference[1]).sum())
        )

        in_degree = edges.sum(axis=1)       # G[target, source]
        in_degrees[label] = in_degree

        rows.append({
            "graph": label,
            "edges": int(edges.sum()),
            "density": float(edges.sum() / (len(edges) * (len(edges) - 1))),
            "two_way_pairs": int(np.triu(edges & edges.T, 1).sum()),
            "nodes_without_incoming": int((in_degree == 0).sum()),
            "in_degree_mean": float(in_degree.mean()),
            "in_degree_max": int(in_degree.max()),
            f"jaccard_vs_{reference[0]}": float(overlap),
        })

    return pd.DataFrame(rows), in_degrees


# ============================================================
# Figures
# ============================================================

def figure_validation_curves(runs, out):

    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    for i, run in enumerate(runs):

        h = run["history"]
        color = color_of(run["label"], i)

        ax.plot(h["epoch"], h["val_rmse_original"], color=color,
                marker="o", markersize=4, label=run["label"])

        best = h["val_rmse_original"].idxmin()
        ax.plot(h["epoch"][best], h["val_rmse_original"][best], "o",
                markersize=8, mfc=SURFACE, mec=color, mew=2)

        direct_label(ax, h["epoch"].iloc[-1], h["val_rmse_original"].iloc[-1],
                     run["label"], color)

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (vehicles / 5 min)")
    ax.set_title("Validation RMSE per epoch (ring = selected checkpoint)")
    ax.legend(loc="upper right")
    save_figure(fig, out, "01_validation_curves")


def figure_test_metrics(table, out):

    metrics = [("MAE", "MAE (veh / 5 min)"), ("RMSE", "RMSE (veh / 5 min)"),
               ("masked_MAPE_%", "MAPE (%, flow >= 10)")]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))

    for ax, (metric, title) in zip(axes, metrics):

        values = table[metric].to_numpy(dtype=float)
        colors = [color_of(run, i) for i, run in enumerate(table["run"])]

        bars = ax.bar(table["run"], values, color=colors, width=0.6,
                      edgecolor=SURFACE, linewidth=2)

        # Bars start at zero (bar length must not exaggerate
        # differences); exact values are printed on the bars and
        # the differences are in deltas.csv / 04_delta_histograms.
        ax.set_ylim(0, values.max() * 1.12)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)

        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT_PRIMARY)

    fig.suptitle("Test metrics (lower is better)",
                 color=TEXT_PRIMARY, fontsize=11, fontweight="bold")
    save_figure(fig, out, "02_test_metrics")


def figure_delta_map(runs_by_label, new, base, meta, out):

    a = runs_by_label[new]["per_sensor"]
    b = runs_by_label[base]["per_sensor"]

    delta = (a["mae"] - b["mae"]).to_numpy()
    sensors = a["sensor_index"].to_numpy()

    lat = meta["Lat"].to_numpy()[sensors]
    lng = meta["Lng"].to_numpy()[sensors]

    limit = np.percentile(np.abs(delta), 95)
    limit = max(limit, 1e-6)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))

    order = np.argsort(np.abs(delta))           # largest changes on top
    points = ax.scatter(lng[order], lat[order], c=delta[order], s=9,
                        cmap=DIVERGING,
                        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
                        linewidths=0.3, edgecolors=SURFACE)

    bar = fig.colorbar(points, ax=ax, shrink=0.8, extend="both")
    bar.set_label(f"MAE change {new} - {base} (veh / 5 min)\n"
                  "blue = better, red = worse", color=TEXT_SECONDARY)

    improved = 100.0 * np.mean(delta < 0)
    ax.set_title(f"Per-sensor MAE change, {new} vs {base}\n"
                 f"{improved:.0f}% of sensors improved")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat.mean())), adjustable="datalim")
    save_figure(fig, out, f"03_delta_map_{new}_vs_{base}")


def figure_delta_histogram(runs_by_label, pairs, out):

    fig, axes = plt.subplots(1, len(pairs), figsize=(3.2 * len(pairs), 2.8),
                             squeeze=False)

    for ax, (new, base) in zip(axes[0], pairs):

        delta = (runs_by_label[new]["per_sensor"]["mae"]
                 - runs_by_label[base]["per_sensor"]["mae"]).to_numpy()

        limit = np.percentile(np.abs(delta), 99)
        bins = np.linspace(-limit, limit, 41)

        ax.hist(delta[delta < 0], bins=bins, color="#3987e5",
                edgecolor=SURFACE, linewidth=0.5, label="better")
        ax.hist(delta[delta >= 0], bins=bins, color="#e34948",
                edgecolor=SURFACE, linewidth=0.5, label="worse")
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.axvline(np.median(delta), color=TEXT_PRIMARY, linewidth=1,
                   linestyle="--", label=f"median {np.median(delta):+.2f}")

        ax.set_title(f"{new} - {base}")
        ax.set_xlabel("Per-sensor MAE change")
        ax.legend()

    axes[0][0].set_ylabel("Sensors")
    save_figure(fig, out, "04_delta_histograms")


def figure_time_of_day(runs, out):

    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    for i, run in enumerate(runs):

        t = run["time_of_day"]
        hours = t["slot"] * 5 / 60.0
        color = color_of(run["label"], i)

        smoothed = t["mae"].rolling(3, center=True, min_periods=1).mean()

        ax.plot(hours, smoothed, color=color, label=run["label"])
        direct_label(ax, hours.iloc[-1], smoothed.iloc[-1], run["label"], color)

    ax.set_xlim(0, 24.8)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Time of day (h)")
    ax.set_ylabel("Test MAE (veh / 5 min)")
    ax.set_title("Test MAE by time of day (15-min smoothing)")
    ax.legend(loc="upper left")
    save_figure(fig, out, "05_time_of_day")


def figure_example_series(runs, meta, out, days=7):

    reference = runs[0]["pred"]

    targets = reference["targets"]
    times = pd.to_datetime(reference["timestamps_ns"])

    mean_flow = targets.mean(axis=0)
    picks = [int(np.argsort(mean_flow)[int(q * (len(mean_flow) - 1))])
             for q in (0.5, 0.9)]

    end = times[0] + pd.Timedelta(days=days)
    window = times < end

    fig, axes = plt.subplots(len(picks), 1, figsize=(8.4, 2.6 * len(picks)),
                             sharex=True)

    for ax, j in zip(np.atleast_1d(axes), picks):

        ax.plot(times[window], targets[window, j], color=TEXT_PRIMARY,
                linewidth=1.2, label="Observed")

        for i, run in enumerate(runs):
            ax.plot(times[window], run["pred"]["predictions"][window, j],
                    color=color_of(run["label"], i), linewidth=1.0,
                    alpha=0.9, label=run["label"])

        sensor = int(reference["sensors"][j])
        ax.set_title(f"Sensor {int(reference['sensor_ids'][j])} "
                     f"({meta['Fwy'].iloc[sensor]}), first {days} test days")
        ax.set_ylabel("Flow (veh / 5 min)")
        ax.legend(ncol=4, loc="upper right")

    save_figure(fig, out, "06_example_series")


def figure_in_degree(in_degrees, out):

    if not in_degrees:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.2))

    largest = max(int(d.max()) for d in in_degrees.values())
    bins = np.arange(0, largest + 2)

    for i, (label, degree) in enumerate(in_degrees.items()):
        ax.hist(degree, bins=bins, histtype="step", linewidth=1.6,
                color=color_of(label, i), label=label)

    ax.set_xlabel("Incoming edges per sensor")
    ax.set_ylabel("Sensors")
    ax.set_title("In-degree distribution of the graphs given to A3T-GCN")
    ax.legend()
    save_figure(fig, out, "07_in_degree")


def var_grid(out_tables, out_figures):

    rows = []

    for folder in ("var_train", "var_train_2122"):

        for meta_file in (IMPLEMENTATION_DIR / "outputs" / folder).glob("*/metadata.json"):

            meta = json.loads(meta_file.read_text(encoding="utf-8"))

            rows.append({
                "sensor_set": meta["num_nodes"],
                "lag": meta["lag_order"],
                "alpha": meta["ridge_alpha"],
                "val_rmse": meta["val_residuals"]["rmse"],
                "val_mae": meta["val_residuals"]["mae"],
                "max_abs_coefficient": meta.get("max_abs_coefficient"),
                "runtime_s": meta["runtime_seconds"],
            })

    if not rows:
        return

    table = pd.DataFrame(rows).sort_values(["sensor_set", "lag", "alpha"])
    table.to_csv(out_tables / "var_grid.csv", index=False)

    final = table[table["sensor_set"] == table["sensor_set"].min()]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    for i, (alpha, group) in enumerate(final.groupby("alpha")):
        ax.plot(group["lag"], group["val_rmse"], marker="o",
                color=FALLBACK_COLORS[i % 3], label=f"alpha = {alpha:g}")

    ax.set_xticks(sorted(final["lag"].unique()))
    ax.set_xlabel("VAR lag order")
    ax.set_ylabel("Validation one-step RMSE")
    ax.set_title(f"VAR selection ({int(final['sensor_set'].iloc[0])} sensors)")
    ax.legend()
    save_figure(fig, out_figures, "08_var_selection")


def pem_stage_a(out_tables, out_figures):

    path = IMPLEMENTATION_DIR / "outputs" / "pem" / "sweep_stage_a.json"

    if not path.exists():
        return

    table = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    table.drop(columns=["run_dir"], errors="ignore").to_csv(
        out_tables / "pem_stage_a.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))

    for i, (ratio, group) in enumerate(table.assign(
            ratio=table["nu1"] / table["nu0"]).groupby("ratio")):

        color = FALLBACK_COLORS[i % 3]
        axes[0].plot(group["nu0"], group["local_density"], marker="o",
                     color=color, label=f"nu1 = {ratio:g} x nu0")
        axes[1].plot(group["nu0"], group["split_half_jaccard"], marker="o",
                     color=color, label=f"nu1 = {ratio:g} x nu0")

    for ax, title in zip(axes, ["Local directed edge density",
                                "Split-half stability (Jaccard)"]):
        ax.set_xscale("log")
        ticks = sorted(table["nu0"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("nu0")
        ax.set_title(title)
        ax.legend()

    axes[1].set_ylim(0, 1)
    save_figure(fig, out_figures, "09_pem_stage_a")


def figure_error_by_horizon(runs, out_tables, out_figures):
    """Test RMSE and MAE against the forecast horizon, one line per experiment."""

    rows = [{
        "experiment": experiment_of(run["label"]),
        "label": run["label"],
        "horizon_steps": run["horizon"],
        "horizon_minutes": 5 * run["horizon"],
        "RMSE": run["results"]["test_metrics"]["rmse"],
        "MAE": run["results"]["test_metrics"]["mae"],
    } for run in runs]

    table = pd.DataFrame(rows).sort_values(["experiment", "horizon_minutes"])
    table.to_csv(out_tables / "error_by_horizon.csv", index=False)

    # Two panels, one measure each (never two scales on one axis).
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharex=True)

    groups = list(table.groupby("experiment", sort=False))
    minutes = sorted(table["horizon_minutes"].unique())

    for ax, metric in zip(axes, ("RMSE", "MAE")):

        ends = []

        for i, (experiment, group) in enumerate(groups):

            color = color_of(experiment, i)
            dashed = experiment in ("NOGRAPH", "HA")
            ax.plot(group["horizon_minutes"], group[metric], color=color,
                    linewidth=2, linestyle="--" if dashed else "-",
                    marker="o", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=experiment)
            last = group.iloc[-1]
            ends.append([last["horizon_minutes"], last[metric], experiment, color])

        ax.set_title(f"Test {metric} (veh / 5 min)")
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_xticks(minutes)
        ax.set_ylim(bottom=0)
        # Room on the right for the end labels.
        ax.set_xlim(minutes[0] - 3, minutes[-1] + 0.22 * (minutes[-1] - minutes[0] + 1))

        # Direct labels for up to 4 lines, nudged apart so they never overlap.
        if len(ends) <= 4:
            low, high = ax.get_ylim()
            gap = 0.075 * (high - low)
            ends.sort(key=lambda e: e[1])
            for k in range(1, len(ends)):
                ends[k][1] = max(ends[k][1], ends[k - 1][1] + gap)
            for x, y, text, color in ends:
                direct_label(ax, x, y, text, color)

    axes[0].legend(loc="upper left")
    fig.suptitle("Error by forecast horizon (lower is better)",
                 color=TEXT_PRIMARY, fontsize=11, fontweight="bold")
    save_figure(fig, out_figures, "10_error_by_horizon")

    return table


# ============================================================
# Main
# ============================================================

def compare_group(runs, out, pair_args):
    """Tables and figures for runs that scored the same test targets."""

    check_same_targets(runs)

    labels = [run["label"] for run in runs]
    runs_by_label = {run["label"]: run for run in runs}

    if pair_args:
        pairs = [tuple(item.split(":", 1)) for item in pair_args]
        pairs = [(a, b) for a, b in pairs if a in runs_by_label and b in runs_by_label]
    else:
        # Every run against the reference (the first run, normally E0).
        pairs = [(label, labels[0]) for label in labels[1:]]

    out_tables = out / "tables"
    out_figures = out / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(local_path(
        Path(runs[0]["results"]["config"]["traffic_path"]).with_name("gba_meta.csv"),
        "gba_meta.csv",
    ))

    # Tables
    table = comparison_table(runs)
    table.to_csv(out_tables / "comparison.csv", index=False)
    (out_tables / "comparison.md").write_text(
        markdown_table(table), encoding="utf-8")

    deltas = deltas_table(table, pairs)
    deltas.to_csv(out_tables / "deltas.csv", index=False)

    breakdown_table(runs).to_csv(out_tables / "breakdown_inputs.csv", index=False)

    significance = significance_table(runs_by_label, pairs)
    significance.to_csv(out_tables / "significance.csv", index=False)

    graphs, in_degrees = graph_table(runs)
    graphs.to_csv(out_tables / "graph_comparison.csv", index=False)

    # Figures
    figure_validation_curves(runs, out_figures)
    figure_test_metrics(table, out_figures)

    for new, base in pairs:
        figure_delta_map(runs_by_label, new, base, meta, out_figures)

    if pairs:
        figure_delta_histogram(runs_by_label, pairs, out_figures)
    figure_time_of_day(runs, out_figures)
    figure_example_series(runs, meta, out_figures)
    figure_in_degree(in_degrees, out_figures)

    print(table.to_string(index=False))
    print()
    print(deltas.to_string(index=False))
    print()
    print(significance.to_string(index=False))
    print()
    print(graphs.to_string(index=False))

    return out_tables, out_figures


def main_grouped():

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="LABEL=run_dir; the first run is the reference.")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="NEW:BASE pairs to compare (default: each vs the first run).")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "results" / "comparison"))
    args = parser.parse_args()

    runs = []
    for item in args.runs:
        label, directory = item.split("=", 1)
        runs.append(load_run(label, directory))

    out = Path(args.out)
    horizons = sorted({run["horizon"] for run in runs})

    for horizon in horizons:
        group = [run for run in runs if run["horizon"] == horizon]
        target = out if len(horizons) == 1 else out / f"h{horizon}"
        print(f"\n=== horizon {horizon} ({5 * horizon} min): {[r['label'] for r in group]}")
        compare_group(group, target, args.pairs)

    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    var_grid(out / "tables", out / "figures")
    pem_stage_a(out / "tables", out / "figures")

    if len(horizons) > 1:
        print(figure_error_by_horizon(runs, out / "tables", out / "figures").to_string(index=False))

    print(f"\nSaved tables and figures to {out}")


# ============================================================
# Main (original single-horizon flow, kept for reference)
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="LABEL=run_dir, e.g. E0=results/E0")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "results" / "comparison"))
    args = parser.parse_args()

    runs = []

    for item in args.runs:
        label, directory = item.split("=", 1)
        runs.append(load_run(label, directory))

    check_same_targets(runs)

    labels = [run["label"] for run in runs]
    runs_by_label = {run["label"]: run for run in runs}

    pairs = [(labels[i], labels[i - 1]) for i in range(1, len(labels))]
    if len(labels) >= 3:
        pairs.append((labels[-1], labels[0]))

    out = Path(args.out)
    out_tables = out / "tables"
    out_figures = out / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(local_path(
        Path(runs[0]["results"]["config"]["traffic_path"]).with_name("gba_meta.csv"),
        "gba_meta.csv",
    ))

    # Tables
    table = comparison_table(runs)
    table.to_csv(out_tables / "comparison.csv", index=False)
    (out_tables / "comparison.md").write_text(
        markdown_table(table), encoding="utf-8")

    deltas = deltas_table(table, pairs)
    deltas.to_csv(out_tables / "deltas.csv", index=False)

    breakdown_table(runs).to_csv(out_tables / "breakdown_inputs.csv", index=False)

    significance = significance_table(runs_by_label, pairs)
    significance.to_csv(out_tables / "significance.csv", index=False)

    graphs, in_degrees = graph_table(runs)
    graphs.to_csv(out_tables / "graph_comparison.csv", index=False)

    var_grid(out_tables, out_figures)
    pem_stage_a(out_tables, out_figures)

    # Figures
    figure_validation_curves(runs, out_figures)
    figure_test_metrics(table, out_figures)

    for new, base in pairs:
        figure_delta_map(runs_by_label, new, base, meta, out_figures)

    figure_delta_histogram(runs_by_label, pairs, out_figures)
    figure_time_of_day(runs, out_figures)
    figure_example_series(runs, meta, out_figures)
    figure_in_degree(in_degrees, out_figures)

    print(table.to_string(index=False))
    print()
    print(deltas.to_string(index=False))
    print()
    print(significance.to_string(index=False))
    print()
    print(graphs.to_string(index=False))
    print(f"\nSaved tables and figures to {out}")


if __name__ == "__main__":
    main_grouped()
