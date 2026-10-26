"""
Graph figures and tables (Phase 10): what each method's graph looks like.

Graphs: the road graph (E0) and every final graph in config.FINAL_GRAPHS
that exists on disk (E5 at the --e5-horizon), or explicit
--graphs LABEL=path.npz. All graphs are in the [target, source]
convention (row = target, column = source).

Outputs (--out, default results/reports/graphs):
    figures/11_graph_map_subset   the same 40 sensors at their map
                                  positions, one panel per graph; each
                                  sensor's 3 strongest sources inside the
                                  subset, arrows source -> target, width /
                                  opacity by weight (within the graph)
    figures/12_graph_in_degree    distribution of sources per sensor
    figures/13_graph_overlap      Jaccard overlap of the edge sets
    figures/14_graph_size         number of edges and density
    figures/15_graph_weights      distribution of edge weights
    tables/graph_summary.csv, graph_jaccard.csv, map_subset_sensors.csv

Subset rule (fixed, not chosen by looking at results): the 40 sensors
closest to the sensor at the network's median latitude / longitude.

Note: an edge weight's meaning differs by method (road distance weight,
|VAR coefficient|, PEM inclusion probability x direction, |structural
effect|); weights are compared within a graph, never across graphs.

    python plot_graphs.py [--e5-horizon 1] [--graphs E9=path.npz ...]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare_results import (  # noqa: E402  (shared style and palette)
    BASELINE_COLOR,
    GRID,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    color_of,
    save_figure,
)
from config.config import FINAL_GRAPHS, GRAPH_DIR, ExperimentConfig  # noqa: E402


IMPLEMENTATION_DIR = Path(__file__).resolve().parent
SUBSET_SIZE = 40
MAP_SOURCES_PER_SENSOR = 3   # strongest sources drawn per sensor (readability)

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_graphs(args):
    """Ordered {label: (graph [target, source], sensors)}; road first."""

    graphs = {}
    sensors = None

    if args.graphs:
        items = [item.split("=", 1) for item in args.graphs]
    else:
        items = []
        for name, file in FINAL_GRAPHS.items():
            path = GRAPH_DIR / file.format(h=args.e5_horizon)
            if path.exists():
                items.append((name, str(path)))

    for label, path in items:
        with np.load(path) as artifact:
            graphs[label] = np.asarray(artifact["adjacency"], dtype=np.float64)
            if sensors is None:
                sensors = np.asarray(artifact["sensors"], dtype=np.int64)
            elif not np.array_equal(sensors, artifact["sensors"]):
                raise ValueError(f"{label}: different sensor set.")

    if sensors is None:
        raise SystemExit("No graph artifacts found.")

    road = np.load(ExperimentConfig().adjacency_path, mmap_mode="r")
    road = np.asarray(road[np.ix_(sensors, sensors)], dtype=np.float64)

    ordered = {"E0": road}
    ordered.update(graphs)

    for graph in ordered.values():
        np.fill_diagonal(graph, 0.0)

    return ordered, sensors


def summary_table(graphs):

    rows = []
    for label, g in graphs.items():
        edges = g != 0
        weights = g[edges]
        in_degree = edges.sum(axis=1)
        rows.append({
            "graph": label,
            "edges": int(edges.sum()),
            "density": float(edges.sum() / (g.shape[0] * (g.shape[0] - 1))),
            "in_degree_mean": float(in_degree.mean()),
            "in_degree_median": float(np.median(in_degree)),
            "in_degree_max": int(in_degree.max()),
            "sensors_without_sources": int(np.sum(in_degree == 0)),
            "reciprocal_pairs": int(np.count_nonzero(np.triu(edges & edges.T, 1))),
            "weight_median": float(np.median(weights)) if weights.size else 0.0,
            "weight_p90": float(np.quantile(weights, 0.9)) if weights.size else 0.0,
        })
    return pd.DataFrame(rows)


def jaccard_table(graphs):

    labels = list(graphs)
    edges = {k: g != 0 for k, g in graphs.items()}
    matrix = np.zeros((len(labels), len(labels)))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            union = np.sum(edges[a] | edges[b])
            matrix[i, j] = np.sum(edges[a] & edges[b]) / union if union else np.nan
    return pd.DataFrame(matrix, index=labels, columns=labels)


def map_subset(sensors, meta):

    lat = meta["Lat"].to_numpy()[sensors]
    lng = meta["Lng"].to_numpy()[sensors]
    centre = np.argmin((lat - np.median(lat)) ** 2 + (lng - np.median(lng)) ** 2)
    distance = (lat - lat[centre]) ** 2 + ((lng - lng[centre]) * np.cos(np.radians(lat[centre]))) ** 2
    subset = np.sort(np.argsort(distance)[:SUBSET_SIZE])
    return subset, lat, lng


def figure_map(graphs, sensors, meta, out):

    subset, lat, lng = map_subset(sensors, meta)
    labels = list(graphs)
    columns = min(4, len(labels))
    rows = int(np.ceil(len(labels) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.6 * rows + 0.4),
                             squeeze=False)

    x, y = lng[subset], lat[subset]

    for ax, label in zip(axes.ravel(), labels):

        g = graphs[label][np.ix_(subset, subset)]
        color = color_of(label, labels.index(label))
        total = int(np.count_nonzero(g))

        # Each sensor's strongest sources only (the full subgraph is a tangle).
        drawn = np.zeros_like(g)
        for t in range(len(subset)):
            row = np.abs(g[t])
            top = np.argsort(-row, kind="stable")[:MAP_SOURCES_PER_SENSOR]
            top = top[row[top] > 0]
            drawn[t, top] = row[top]

        targets, sources = np.nonzero(drawn)
        weights = drawn[targets, sources]
        scale = weights / weights.max() if weights.size else weights

        for t, s, w in zip(targets, sources, scale):
            ax.annotate("", xy=(x[t], y[t]), xytext=(x[s], y[s]),
                        arrowprops={"arrowstyle": "-|>", "color": color,
                                    "lw": 0.5 + 1.0 * w, "alpha": 0.35 + 0.65 * w,
                                    "shrinkA": 3, "shrinkB": 3, "mutation_scale": 5})

        ax.scatter(x, y, s=10, color=TEXT_SECONDARY, zorder=3,
                   edgecolors=SURFACE, linewidths=0.8)
        ax.set_title(f"{label}  ({total} edges in subset)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color(GRID)
        ax.set_aspect(1 / np.cos(np.radians(np.median(y))))

    for ax in axes.ravel()[len(labels):]:
        ax.set_visible(False)

    fig.suptitle(f"The same {SUBSET_SIZE} sensors in every graph (map positions)\n"
                 f"each sensor's {MAP_SOURCES_PER_SENSOR} strongest sources; "
                 "arrow = source -> target, darker = stronger within that graph",
                 color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
    save_figure(fig, out, "11_graph_map_subset")

    return pd.DataFrame({"sensor_index": sensors[subset], "lat": y, "lng": x})


def figure_in_degree(graphs, out):

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = list(graphs)

    seen = {}

    for i, label in enumerate(labels):
        edges = graphs[label] != 0
        degree = np.sort(edges.sum(axis=1))
        share = np.arange(1, len(degree) + 1) / len(degree)

        # A graph with the same edges as an earlier one would hide it:
        # draw it dashed on top and say so in the legend.
        same = next((other for other, e in seen.items() if np.array_equal(e, edges)), None)
        ax.step(degree, share, where="post", color=color_of(label, i), linewidth=2,
                linestyle=(0, (4, 3)) if same else "-",
                label=label if same is None else f"{label} (same edges as {same})")
        seen.setdefault(label, edges)

    ax.set_xlabel("Sources per sensor (in-degree)")
    ax.set_ylabel("Share of sensors (cumulative)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("How many sources each sensor gets")
    save_figure(fig, out, "12_graph_in_degree")


def figure_overlap(jaccard, out):

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    n = len(jaccard)
    fig, ax = plt.subplots(figsize=(max(4.8, 1.6 + 0.62 * n), max(3.6, 1.0 + 0.55 * n)))

    image = ax.imshow(jaccard.to_numpy(), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(jaccard.columns)
    ax.set_yticks(range(n)); ax.set_yticklabels(jaccard.index)
    ax.grid(False)

    for i in range(n):
        for j in range(n):
            value = jaccard.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if value > 0.55 else TEXT_PRIMARY)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Jaccard overlap of edges")
    ax.set_title("Shared edges between graphs")
    save_figure(fig, out, "13_graph_overlap")


def figure_size(summary, out):

    fig, ax = plt.subplots(figsize=(1.6 + 0.7 * len(summary), 3.2))
    colors = [color_of(g, i) for i, g in enumerate(summary["graph"])]
    bars = ax.bar(summary["graph"], summary["edges"], color=colors, width=0.6,
                  edgecolor=SURFACE, linewidth=2)
    for bar, edges, density in zip(bars, summary["edges"], summary["density"]):
        ax.annotate(f"{edges:,}\n{100 * density:.2f} %",
                    (bar.get_x() + bar.get_width() / 2, edges), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=7, color=TEXT_PRIMARY)
    ax.set_ylim(0, summary["edges"].max() * 1.22)
    ax.set_ylabel("Directed edges")
    ax.grid(axis="x", visible=False)
    ax.set_title("Graph size (edges, density)")
    save_figure(fig, out, "14_graph_size")


def figure_weights(graphs, out):

    labels = list(graphs)
    columns = min(4, len(labels))
    rows = int(np.ceil(len(labels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.0 * rows + 0.4),
                             squeeze=False)

    for i, (ax, label) in enumerate(zip(axes.ravel(), labels)):
        w = graphs[label][graphs[label] != 0]
        w = w[w > 0]
        if w.size:
            ax.hist(w, bins=np.logspace(np.log10(w.min()), np.log10(w.max()), 30),
                    color=color_of(label, i), edgecolor=SURFACE, linewidth=0.5)
            ax.set_xscale("log")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(labels):]:
        ax.set_visible(False)

    fig.suptitle("Edge weights within each graph (log scale; units differ by method)",
                 color=TEXT_PRIMARY, fontsize=10, fontweight="bold")
    save_figure(fig, out, "15_graph_weights")


def main():

    parser = argparse.ArgumentParser(description="Graph figures and tables.")
    parser.add_argument("--graphs", nargs="*", default=None, help="LABEL=path.npz (default: FINAL_GRAPHS)")
    parser.add_argument("--e5-horizon", type=int, default=1)
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "results" / "reports" / "graphs"))
    args = parser.parse_args()

    graphs, sensors = load_graphs(args)

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(Path(ExperimentConfig().adjacency_path).with_name("gba_meta.csv"))

    summary = summary_table(graphs)
    jaccard = jaccard_table(graphs)
    summary.to_csv(out / "tables" / "graph_summary.csv", index=False)
    jaccard.to_csv(out / "tables" / "graph_jaccard.csv")

    figure_map(graphs, sensors, meta, out / "figures").to_csv(
        out / "tables" / "map_subset_sensors.csv", index=False)
    figure_in_degree(graphs, out / "figures")
    figure_overlap(jaccard, out / "figures")
    figure_size(summary, out / "figures")
    figure_weights(graphs, out / "figures")

    print(summary.to_string(index=False))
    print()
    print(jaccard.round(3).to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
