// Builds presentation/ablation_study_report.docx (plain Word report of the ablation study).
// Run from the implementation folder:  node presentation/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ImageRun, AlignmentType, BorderStyle, LevelFormat, PageBreak,
  TableOfContents,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "presentation", "figures", f);
const CONTENT_WIDTH = 9026;            // A4 minus 1" margins, DXA
const FONT = "Arial";

// ---------- helpers ----------
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [new TextRun(t)] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });

// Text with **bold** segments.
function runs(text) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) =>
    s.startsWith("**") ? new TextRun({ text: s.slice(2, -2), bold: true }) : new TextRun(s));
}
const p = (t) => new Paragraph({ spacing: { after: 120 }, children: runs(t) });
const label = (t) => new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 }, children: [new TextRun({ text: t, bold: true })] });
const bullets = (items) => items.map((t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: runs(t) }));
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, width, bold = false, italic = false) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold, italics: italic, size: 19 })] })],
  });
}

// rows[0] is the header. Cells starting with "[paste" are written in italics.
function table(rows, weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((w / total) * CONTENT_WIDTH));
  widths[widths.length - 1] += CONTENT_WIDTH - widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => cell(c, widths[j], i === 0, String(c).startsWith("[paste"))),
    })),
  });
}

// A bordered box that says what must be pasted there.
function placeholder(text, height = 1) {
  const lines = [new Paragraph({ children: [new TextRun({ text: "TO BE PASTED: " + text, italics: true, size: 19 })] })];
  for (let i = 0; i < height; i++) lines.push(new Paragraph({ children: [] }));
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: [CONTENT_WIDTH],
    rows: [new TableRow({ children: [new TableCell({
      borders: { top: { style: BorderStyle.DASHED, size: 6, color: "000000" }, bottom: { style: BorderStyle.DASHED, size: 6, color: "000000" },
                 left: { style: BorderStyle.DASHED, size: 6, color: "000000" }, right: { style: BorderStyle.DASHED, size: 6, color: "000000" } },
      width: { size: CONTENT_WIDTH, type: WidthType.DXA },
      margins: { top: 100, bottom: 100, left: 120, right: 120 },
      children: lines,
    })] })],
  });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

let figNo = 0;
function figure(file, caption, maxWidth = 600) {
  const { w, h } = pngSize(file);
  const width = Math.min(maxWidth, w);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 }, children: [
      new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width, height: Math.round((h / w) * width) },
                     altText: { title: caption, description: caption, name: path.basename(file) } }) ] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: `Figure ${figNo}. ${caption}`, italics: true, size: 19 })] }),
  ];
}

let tabNo = 0;
const tcap = (t) => { tabNo += 1; return new Paragraph({ keepNext: true, spacing: { before: 160, after: 60 }, children: [new TextRun({ text: `Table ${tabNo}. ${t}`, italics: true, size: 19 })] }); };

const RES_HEAD = ["Run", "Test RMSE", "Test MAE", "Masked MAPE (%)", "Best epoch"];
const RES_W = [3, 2, 2, 2, 1.5];

// Standard four-part experiment section.
function experiment({ title, diagram, change, graph, results, figures = [], inference, extra = [] }) {
  const name = path.basename(diagram, ".png").replace("exp_", "");
  return [
    h2(title),
    label("What this experiment changes"), ...[].concat(change).map(p),
    label("Block diagram: models used"),
    ...figure(diagram, name === "HA" ? "Steps of the HA baseline." : `Models used in ${name}. Solid boxes are used; dashed grey boxes are not used.`),
    label("Graph facts"), ...bullets(graph),
    label("Results"), tcap(`${title.replace(/^[\d.]+\s*/, "").split(":")[0]} results (test split).`), table([RES_HEAD, ...results], RES_W),
    ...extra,
    ...figures.flatMap(([f, c, w]) => figure(f, c, w)),
    label("Inference"), ...bullets(inference),
  ];
}

// ---------- content ----------
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 2400, after: 240 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Horizon-Conditioned Probabilistic VAR-LiNGAM Causal Graph for Traffic Forecasting", bold: true, size: 36 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 }, children: [new TextRun({ text: "Ablation study: architecture, results and inferences", size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Student: [paste your name]", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Guide: [paste guide's name]", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Institute: [paste institute name]", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240 }, children: [new TextRun({ text: "Status of results: 3 November 2026", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480 }, children: [new TextRun({ text: "Boxes marked \"TO BE PASTED\" and table cells marked \"[paste ...]\" are results of runs that are not finished yet.", italics: true, size: 20 })] }),
  pageBreak(),
  new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 28 })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new TextRun({ text: "(Right-click the contents and choose Update Field if it is empty.)", italics: true, size: 18 })] }),
  pageBreak(),
);

// 1. Introduction
children.push(
  h1("1. Introduction"),
  h2("1.1 Problem and goal"),
  p("Traffic forecasting predicts the flow at every road sensor a few minutes to an hour ahead. Graph neural networks such as A3T-GCN mix each sensor's history with that of its neighbours through a graph. The usual graph is the road network: two sensors are linked when they are close on the road."),
  p("This project asks whether a graph learned from the data with causal-discovery methods gives better forecasts than the road graph. The causal graph is built in steps (VAR, then VAR-LiNGAM, then PEM edge probabilities, then horizon-specific impulse responses). The ablation study adds one step at a time and keeps the forecasting model (A3T-GCN) unchanged, so any change in accuracy comes from the graph."),
  h2("1.2 Dataset and experimental setup"),
  tcap("Setup shared by every experiment."),
  table([
    ["Item", "Value"],
    ["Dataset", "LargeST-GBA 2021 (Greater Bay Area), traffic flow every 5 minutes, 105,120 time steps"],
    ["Sensors", "2,352 in the file; 296 removed as copies of other sensors (same value on at least 90% of training rows, or correlation at least 0.999); 2,056 used"],
    ["Split", "Chronological: train 80%, validation 10%, test 10%"],
    ["Missing values", "Inputs: gaps filled by gap length (training data only). Targets: must be originally observed, never filled"],
    ["Input / output", "12 past steps (1 hour) -> flow h steps ahead; h = 1, 3, 6, 12 (5, 15, 30, 60 min)"],
    ["Model", "A3T-GCN, 100 GRU units, unchanged for every experiment"],
    ["Training", "20 epochs, batch 8, learning rate 0.001 (halved after 5 epochs without improvement), gradient clipping 5, L2 0.0015"],
    ["Checkpoint", "Epoch with the lowest validation RMSE"],
    ["Seeds", "42 for every experiment; 7 for E0, E1, E5 at 5 min"],
    ["Hardware", "Kaggle T4 GPU (training); laptop CPU (VAR, LiNGAM, PEM, graphs)"],
  ], [1, 3.2]),
  h2("1.3 Metrics and statistical tests"),
  ...bullets([
    "**RMSE** and **MAE** in vehicles per 5 minutes, over all test sensors and windows. **Masked MAPE** only counts targets with flow of at least 10.",
    "**% sensors better**: share of the 2,056 sensors whose test MAE is lower for the first model of a pair. Wilcoxon signed-rank test on the per-sensor MAE.",
    "**Diebold-Mariano (DM) test**: compares the two models' MAE at each test time step (Newey-West lag 17). p < 0.05 means the difference is unlikely to be chance.",
    "**Test windows**: a window counts only if its target is observed at all 2,056 sensors, so 5,183 of 10,500 test windows are scored at 5 min (5,178 at 30 min, 5,172 at 60 min). Every model is scored on exactly the same targets.",
  ]),
  pageBreak(),
);

// 2. Architecture
children.push(
  h1("2. Architecture"),
  h2("2.1 Framework block diagram"),
  ...figure(FIG("fig01_framework.png"), "Proposed framework. Causal discovery runs on the training split only; the graph builder makes one graph per experiment; A3T-GCN is unchanged."),
  ...bullets([
    "**Step 1, VAR**: a ridge vector autoregression (lag 3, alpha 1e6) where each sensor is predicted from its own past and its road neighbours' past. It gives lag matrices A_1, A_2, A_3 and residuals.",
    "**Step 2, VAR-LiNGAM**: DirectLiNGAM on the VAR residuals (local units: a sensor and its road neighbours) gives B0, the same-time causal effects. Lagged structural effects are B_k = (I - B0) A_k.",
    "**Step 3, PEM**: a spike-and-slab model gives each candidate edge an inclusion probability P and a directed edge set D.",
    "**Step 4, horizon conditioning**: the structural impulse responses Theta_h measure the effect of each sensor on every other sensor h steps later; E5 keeps the 25 strongest sources per sensor for each horizon.",
  ]),
  h2("2.2 Forecasting model: A3T-GCN"),
  ...figure(FIG("fig02_a3tgcn.png"), "A3T-GCN. The graph enters only through the normalised operator used by the graph convolution inside the GRU."),
  p("The graph G is stored as G[target, source] and transposed once before normalisation, so each sensor reads from its sources. After normalisation, each sensor's own value keeps a weight (its **own-signal share**) and the rest goes to its neighbours. This share is used in Section 4 to explain the results."),
  h2("2.3 Ablation map"),
  tcap("Experiments of the ablation study. Each experiment changes only the graph."),
  table([
    ["ID", "Graph", "Question it answers", "Status"],
    ["HA", "No model: average of training values per weekday, 5-min slot and sensor", "Lower bound every model must beat", "Done (5, 30, 60 min)"],
    ["NOGRAPH", "A3T-GCN with an empty graph (each sensor sees only itself)", "Does any graph help?", "5, 30, 60 min done; seed 7 pending"],
    ["E0", "Road network (original A3T-GCN)", "Baseline graph", "5 min (2 seeds), 60 min done; 30 min pending"],
    ["E1", "VAR: sum over lags of |A_k|", "Do learned weights beat road weights?", "5 min (2 seeds), 30, 60 min done"],
    ["E2", "VAR-LiNGAM: sum |B_k| + |B0|", "Do same-time causal effects help?", "5 min done"],
    ["E3", "E3a: P x D (original); E3b: VAR weights on PEM edges (amended)", "Does PEM edge selection help?", "Both done (5 min)"],
    ["E4", "VAR-LiNGAM + PEM: sum |B_k| + P x |B0|", "Does PEM weighting of B0 help?", "Graph built; not trained (almost equal to E2)"],
    ["E5", "Top-25 sources from impulse responses, one graph per horizon (full framework)", "Do horizon-specific graphs help?", "5 min (2 seeds), 30, 60 min done"],
    ["E6", "E4 on data-driven candidate links (top-25 correlated sensors)", "Does the road restriction limit discovery?", "Pending"],
    ["E7", "E4 with a road prior in PEM (0.7 on road links, 0.3 elsewhere)", "Does a road prior help PEM?", "Done (5 min)"],
  ], [1, 3, 2.4, 2]),
  pageBreak(),
);

// 3. Results per experiment
children.push(h1("3. Results of each experiment"));

children.push(...experiment({
  title: "3.1 Baseline: Historical Average (HA)",
  diagram: FIG("exp_HA.png"),
  change: "No learning. The forecast for a sensor is the average of its training values at the same weekday and 5-minute slot. It does not use the last hour of data, so its error is the same at every horizon.",
  graph: ["No graph."],
  results: [
    ["HA, 5 min", "54.92", "33.24", "19.90", "-"],
    ["HA, 15 min", "[paste: results/HA_h3]", "[paste]", "[paste]", "-"],
    ["HA, 30 min", "54.91", "33.23", "19.90", "-"],
    ["HA, 60 min", "54.89", "33.21", "19.91", "-"],
  ],
  inference: [
    "Every trained model beats HA clearly at 5 and 30 min (about half the RMSE at 5 min).",
    "At 60 min HA is a strong baseline: only E1 has a significantly lower MAE (DM p = 0.016); E0 and E5 tie with it on MAE.",
    "HA is better than the models at night at 60 min because a one-hour input window cannot see the weekly pattern HA uses.",
  ],
}));

children.push(...experiment({
  title: "3.2 NOGRAPH: A3T-GCN without a graph",
  diagram: FIG("exp_NOGRAPH.png"),
  change: "The graph is empty (A = 0), so the normalised operator is the identity. Each sensor uses only its own history; all sensors share the same GRU and attention weights.",
  graph: ["0 edges; own-signal share 100%."],
  results: [
    ["NOGRAPH, 5 min, seed 42", "27.58", "16.87", "11.19", "20"],
    ["NOGRAPH, 5 min, seed 7", "[paste: output_NOGRAPH_final_seed7]", "[paste]", "[paste]", "[paste]"],
    ["NOGRAPH, 30 min, seed 42", "39.46", "25.67", "17.49", "19"],
    ["NOGRAPH, 60 min, seed 42", "43.87", "28.29", "16.76", "20"],
  ],
  inference: [
    "At 5 min (seed 42) NOGRAPH beats every graph model: it is better than E1 on 82% of sensors, E0 on 88% and E5 on 74% (DM p < 0.001 for all three).",
    "The gain is largest in daytime and at busy sensors (MAE 21.8 vs 24.4 for E1 at the busiest quarter of sensors). Five minutes ahead, a sensor's next value mostly depends on its own recent flow; mixing in neighbours adds noise.",
    "At 30 min the VAR graph helps: E1 (37.47) beats NOGRAPH (39.46) on 92% of sensors (DM p < 0.001). E5 ties NOGRAPH (DM p = 0.54).",
    "At 60 min NOGRAPH (43.87) beats every graph model by a wide margin: E1 47.41, E5 48.97, E0 49.87. Only 4-7% of sensors are better with a graph (DM p < 0.001). NOGRAPH is also far below HA (54.89).",
    "So, on seed 42, a graph helps only at 30 min. The 30-min gain (2.0 RMSE) is only a little larger than the seed effect (up to 1.4), so it needs a second seed; the 60-min loss (3.5 RMSE) is clearer.",
    "Caution: NOGRAPH is not free of collapses either. At 30 min its validation RMSE jumped to 389 at epoch 6, and at 60 min it was still improving at epoch 20.",
  ],
}));

children.push(...experiment({
  title: "3.3 E0: A3T-GCN with the road-network graph",
  diagram: FIG("exp_E0.png"),
  change: "The original A3T-GCN setting: the road adjacency of the dataset, including self-loops.",
  graph: ["51,686 directed edges between the 2,056 sensors, mean in-degree 25.1.", "Own-signal share 20% (median)."],
  results: [
    ["E0, 5 min, seed 42", "29.76", "19.34", "11.89", "15"],
    ["E0, 5 min, seed 7", "30.13", "19.33", "11.80", "16"],
    ["E0, 15 min, seed 42", "[paste: output_E0_h3_final]", "[paste]", "[paste]", "[paste]"],
    ["E0, 30 min, seed 42", "[paste: output_E0_h6_final]", "[paste]", "[paste]", "[paste]"],
    ["E0, 60 min, seed 42", "49.87", "33.19", "20.41", "15"],
  ],
  inference: [
    "E0 is stable across seeds (29.76 vs 30.13), unlike E1 and E5.",
    "It is the weakest of the main graphs at every horizon tested: worse than E1 on both seeds and at 60 min, and worse than NOGRAPH at 5 min.",
    "At 60 min its MAE ties HA (DM p = 0.98).",
  ],
}));

children.push(...experiment({
  title: "3.4 E1: VAR-learned graph",
  diagram: FIG("exp_E1.png"),
  change: "G = |A_1| + |A_2| + |A_3| from the road-restricted VAR (lag 3, alpha 1e6), diagonal removed. Lag and alpha were chosen on validation one-step RMSE.",
  graph: ["Same 51,686 edges as the road graph (the VAR is restricted to road neighbours); only the weights differ.", "Weight mean 0.075; own-signal share 39%."],
  results: [
    ["E1, 5 min, seed 42", "28.10", "18.36", "10.49", "15"],
    ["E1, 5 min, seed 7", "26.71", "16.90", "9.97", "20"],
    ["E1, 15 min, seed 42", "[paste: output_E1_h3_final]", "[paste]", "[paste]", "[paste]"],
    ["E1, 30 min, seed 42", "37.47", "23.58", "13.90", "20"],
    ["E1, 60 min, seed 42", "47.41", "31.32", "19.67", "19"],
  ],
  inference: [
    "E1 is the best graph at every horizon and on both seeds. It beats E0 on 59% (seed 42) and 93% (seed 7) of sensors at 5 min and on 70% at 60 min (DM p < 0.001).",
    "E1 is the only model that beats NOGRAPH anywhere: at 30 min, on 92% of sensors. At 5 and 60 min NOGRAPH is better (Section 3.2).",
    "The edges are the same as the road graph, so the gain comes only from the learned weights: VAR weights describe how strongly one sensor's past predicts another's.",
    "E1 is the only model whose MAE beats HA at 60 min (DM p = 0.016).",
    "Seed effect is large: 28.10 (seed 42) vs 26.71 (seed 7).",
  ],
}));

children.push(...experiment({
  title: "3.5 E2: VAR-LiNGAM graph",
  diagram: FIG("exp_E2.png"),
  change: "G = sum over k of |B_k| + |B0|, with B_k = (I - B0) A_k and B0 from DirectLiNGAM on 10,000 rows of VAR residuals per local unit. Adds same-time (contemporaneous) causal effects.",
  graph: ["101,685 edges (about twice E1), mean in-degree 49, maximum 160.", "Own-signal share 24%.", "B0 is unstable across the two halves of the training data (split-half Jaccard 0.31)."],
  results: [["E2, 5 min, seed 42", "32.38", "21.88", "12.83", "14"]],
  inference: [
    "E2 is worse than both E0 and E1: E1 is better on 85% of sensors and E0 on 75% (DM p < 0.001).",
    "Adding B0 doubles the graph and lowers the own-signal share from 39% to 24%; each sensor's own signal is diluted by many weak, unstable links.",
    "Training was unstable (validation spikes at epochs 6, 8 and 15-19).",
  ],
}));

children.push(...experiment({
  title: "3.6 E3a: VAR + PEM, original form (P x D)",
  diagram: FIG("exp_E3a.png"),
  change: "G = P x D: PEM inclusion probabilities on the PEM directed edges (the original design of E3).",
  graph: ["75,885 edges.", "P is almost binary: 85% of probabilities are above 0.99 and 87.5% of candidate links are kept, so the graph is effectively unweighted and dense.", "Own-signal share 3%."],
  results: [["E3a, 5 min, seed 42", "57.17", "39.66", "26.60", "18"]],
  inference: [
    "E3a is worse than HA. After normalisation each sensor keeps only 3% of its own signal, so the model cannot use a sensor's own history.",
    "PEM probabilities are not usable directly as edge weights. This led to the amended E3 (E3b), approved on 28 October 2026 and reported as a post-hoc change.",
  ],
}));

children.push(...experiment({
  title: "3.7 E3b: VAR weights on PEM edges (amended E3)",
  diagram: FIG("exp_E3b.png"),
  change: "G = W x D: E1's VAR weights kept only where PEM keeps a directed edge. Tests PEM as an edge selector while keeping the weights of E1.",
  graph: ["32,873 edges (64% of E1's).", "44 sensors have no incoming edge.", "Own-signal share 50%."],
  results: [["E3b, 5 min, seed 42", "28.30", "19.62", "15.89", "7"]],
  inference: [
    "E3b is worse than E1 (E1 better on 84% of sensors, DM p = 1e-8) and ties E0 (DM p = 0.17).",
    "Removing a third of E1's edges with PEM does not help, even though the own-signal share rises to 50%.",
    "Validation was good up to epoch 7 and then collapsed; the high masked MAPE (15.9) shows larger errors at low flow.",
  ],
}));

children.push(...experiment({
  title: "3.8 E4: VAR-LiNGAM + PEM",
  diagram: FIG("exp_E4.png"),
  change: "G = sum over k of |B_k| + P x |B0|: PEM probabilities weight the same-time effects of E2.",
  graph: ["Graph built: 101,685 edges, own-signal share 24%.", "It differs from E2 in only 523 entries, because P is almost binary."],
  results: [["E4, 5 min, seed 42", "not trained", "-", "-", "-"]],
  inference: [
    "Not trained: the graph is almost identical to E2.",
    "E7 (Section 3.12) differs from E4 by only 0.025% of the total weight, so the E7 run also stands in for E4. Its result (36.22) differs from E2 (32.38) by 3.8 RMSE although the graphs are nearly the same, which shows that the difference is training noise, not the graph.",
  ],
}));

children.push(...experiment({
  title: "3.9 E5: horizon-specific causal graph at 5 min (full framework)",
  diagram: FIG("exp_E5.png"),
  change: "For each horizon h, the structural impulse responses Theta_h (from A_k, B0 and P) give the effect of every sensor on every other sensor h steps later. Each sensor keeps its 25 strongest sources (25 = mean road in-degree). One graph and one model per horizon.",
  graph: ["About 50,600 edges; in-degree exactly 25 for most sensors.", "Own-signal share 39%.", "Overlap with E1's edges (Jaccard) 0.43."],
  results: [
    ["E5, 5 min, seed 42", "28.17", "18.47", "11.41", "4"],
    ["E5, 5 min, seed 7", "27.40", "17.81", "10.90", "20"],
  ],
  inference: [
    "E5 beats the road graph (E5 better than E0 on 62% of sensors, DM p < 0.001).",
    "It does not beat E1: a tie at seed 42 (DM p = 0.15) and worse at seed 7 (E1 better on 78% of sensors, DM p < 0.001). Two-seed means: E1 27.41, E5 27.78.",
    "The seed-42 run collapsed after epoch 4 (best epoch 4).",
  ],
}));

children.push(...experiment({
  title: "3.10 E5 at longer horizons (15, 30, 60 min)",
  diagram: FIG("exp_E5.png"),
  change: "Same method with the graph for h = 3, 6 and 12.",
  graph: ["About 50,900 edges per horizon; own-signal share 40% (30 min) and 41% (60 min).", "Overlap with E1's edges 0.34 at 60 min: the horizon graphs pick different sources."],
  results: [
    ["E5, 15 min, seed 42", "[paste: output_E5_h3_final]", "[paste]", "[paste]", "[paste]"],
    ["E5, 30 min, seed 42", "39.98", "25.82", "14.90", "15"],
    ["E5, 60 min, seed 42", "48.97", "32.88", "20.00", "20"],
  ],
  inference: [
    "E1 is better than E5 at 30 min (on 86% of sensors) and at 60 min (on 66%), DM p < 0.001 for both.",
    "E5 does not beat NOGRAPH at any horizon: tie at 30 min (DM p = 0.54), worse at 60 min (NOGRAPH better on 96% of sensors).",
    "At 60 min E5 ties HA on MAE (DM p = 0.67).",
    "So horizon-specific graphs do not improve on one VAR graph at any horizon tested.",
    "The 30-min run was very unstable (validation RMSE 108-179 at several epochs).",
  ],
}));

children.push(...experiment({
  title: "3.11 E6: data-driven candidate links",
  diagram: FIG("exp_E6.png"),
  change: "VAR, LiNGAM and PEM are run on candidate links from data (each sensor's 25 most correlated training sensors) instead of the road network; the graph uses E4's formula.",
  graph: ["[paste: outputs/graphs/E6__...json: edges, nodes without incoming, own-signal share]"],
  results: [["E6, 5 min, seed 42", "[paste: output_E6_final]", "[paste]", "[paste]", "[paste]"]],
  inference: ["[paste: compare with E4/E2 (same formula, road candidates) and with E1]"],
}));

children.push(...experiment({
  title: "3.12 E7: road prior in PEM",
  diagram: FIG("exp_E7.png"),
  change: "Same as E4, but PEM uses a prior inclusion probability of 0.7 on road links and 0.3 elsewhere instead of 0.5 everywhere.",
  graph: [
    "PEM was re-run on all 2,056 units with the road prior (Kaggle CPU, 5.7 hours, no failures); the graph was then built with E4's formula.",
    "101,685 edges, exactly the same edges as E4 (none added or removed). Only 940 weights changed (48 by more than 10%); the total weight changed by 0.025%.",
    "All the change is on road-linked pairs (weights slightly up); non-road pairs did not change at all, because B0 is almost zero there and P cannot act on it.",
    "Own-signal share 24.3%, the same as E4.",
  ],
  results: [["E7, 5 min, seed 42", "36.22", "24.42", "14.48", "20"]],
  inference: [
    "The road prior does not change the graph in any meaningful way: PEM's probabilities are already close to 0 or 1, so a prior of 0.7 or 0.3 cannot move them.",
    "E7 is worse than E2 (32.38; E2 better on 63% of sensors, DM p < 0.001), E1 (28.10) and NOGRAPH (27.58). Because the E7 and E2 graphs are almost identical, this gap is not caused by the prior.",
    "The validation curve shows why: validation RMSE stayed between 66 and 96 from epoch 2 to 18 and dropped to 36 only in the last two epochs. The run was still recovering when training stopped.",
    "Conclusion: neither PEM nor an informative PEM prior improves the graph. Two nearly identical dense graphs giving 3.8 RMSE different results also shows that single-seed differences of a few RMSE between dense graphs are not reliable.",
  ],
}));
children.push(pageBreak());

// 4. Comparisons
children.push(
  h1("4. Comparison of all experiments"),
  h2("4.1 Summary table"),
  tcap("Test RMSE / MAE of every run. Seed 42 unless stated."),
  table([
    ["Experiment", "5 min, seed 42", "5 min, seed 7", "5 min, mean", "15 min", "30 min", "60 min"],
    ["HA", "54.92 / 33.24", "-", "-", "[paste]", "54.91 / 33.23", "54.89 / 33.21"],
    ["NOGRAPH", "27.58 / 16.87", "[paste]", "[paste]", "[paste]", "39.46 / 25.67", "43.87 / 28.29"],
    ["E0 road", "29.76 / 19.34", "30.13 / 19.33", "29.94 / 19.34", "[paste]", "[paste]", "49.87 / 33.19"],
    ["E1 VAR", "28.10 / 18.36", "26.71 / 16.90", "27.41 / 17.63", "[paste]", "37.47 / 23.58", "47.41 / 31.32"],
    ["E2 VAR-LiNGAM", "32.38 / 21.88", "-", "-", "-", "-", "-"],
    ["E3a P x D", "57.17 / 39.66", "-", "-", "-", "-", "-"],
    ["E3b VAR on PEM edges", "28.30 / 19.62", "-", "-", "-", "-", "-"],
    ["E4", "not trained (= E7 graph)", "-", "-", "-", "-", "-"],
    ["E5 horizon graphs", "28.17 / 18.47", "27.40 / 17.81", "27.78 / 18.14", "[paste]", "39.98 / 25.82", "48.97 / 32.88"],
    ["E6", "[paste]", "-", "-", "-", "-", "-"],
    ["E7 road prior", "36.22 / 24.42", "-", "-", "-", "-", "-"],
  ], [2.2, 1.6, 1.6, 1.6, 1, 1.6, 1.6]),
  p("Inference: E1 has the lowest error among the graph models wherever it was run. NOGRAPH has the lowest error at 5 and 60 min (seed 42); E1 is lowest at 30 min."),

  h2("4.2 Accuracy at 5 minutes"),
  ...figure(FIG("fig03_bars_5min.png"), "Test RMSE and MAE of every experiment at 5 min, seed 42."),
  ...bullets([
    "Order (RMSE): NOGRAPH 27.58 < E1 28.10 < E5 28.17 < E3b 28.30 < E0 29.76 < E2 32.38 < E7 36.22 < HA 54.92 < E3a 57.17.",
    "E1, E5 and E3b are within 0.2 RMSE of each other; the seed effect (up to 1.4 RMSE) is larger than that gap, so their order on one seed is not reliable. The significance tests in Section 4.6 settle it.",
    "The dense graphs (E2, E7, E3a) are clearly worse than the road graph.",
  ]),

  h2("4.3 Error vs forecast horizon"),
  ...figure(FIG("fig04_horizon.png"), "Test RMSE and MAE vs horizon. 15-min points and E0 at 30 min are pending (the E0 line joins 5 and 60 min directly)."),
  ...bullets([
    "Error grows with the horizon for every trained model; HA stays flat.",
    "E1 stays the lowest graph model at 5, 30 and 60 min; the gap to E0 and E5 grows with the horizon in MAE.",
    "NOGRAPH is lowest at 5 min, above E1 at 30 min, and clearly lowest at 60 min (43.87 vs E1 47.41). Its error grows more slowly from 30 to 60 min than any graph model's.",
    "At 60 min the graph models' MAE approaches HA's (only E1 is significantly below it), while NOGRAPH stays far below HA (28.29 vs 33.21).",
  ]),

  h2("4.4 Effect of the random seed"),
  ...figure(FIG("fig05_seeds.png"), "Test RMSE at 5 min for seed 42 and seed 7. The dashed line is NOGRAPH (seed 42)."),
  ...bullets([
    "E1 improves by 1.39 RMSE and E5 by 0.76 from seed 42 to seed 7; E0 hardly changes (+0.37).",
    "The seed effect is as large as the differences between experiments, so a single-seed comparison can be misleading. E1 > E5 > E0 holds on both seeds.",
    "[paste: NOGRAPH seed 7, to check whether its 5-min win holds on a second seed]",
  ]),

  h2("4.5 Own-signal share explains the extremes"),
  ...figure(FIG("fig06_own_signal.png"), "Median share of its own signal each sensor keeps after normalisation vs test RMSE (5 min, seed 42)."),
  tcap("Graph statistics."),
  table([
    ["Graph", "Directed edges", "Own-signal share (median)", "Test RMSE 5 min"],
    ["NOGRAPH", "0", "100%", "27.58"],
    ["E0 road", "51,686", "20%", "29.76"],
    ["E1 VAR", "51,686", "39%", "28.10"],
    ["E2 VAR-LiNGAM", "101,685", "24%", "32.38"],
    ["E3a P x D", "75,885", "3%", "57.17"],
    ["E3b VAR on PEM edges", "32,873", "50%", "28.30"],
    ["E4", "101,685", "24%", "not trained"],
    ["E7 road prior", "101,685", "24%", "36.22"],
    ["E5 (5 / 30 / 60 min graphs)", "50,555 / 50,927 / 50,927", "39% / 40% / 41%", "28.17"],
  ], [2.4, 2, 2, 1.6]),
  ...bullets([
    "The worst graphs keep the least own signal (E3a 3%, E0 20%, E2 24%); the best model keeps all of it (NOGRAPH 100%).",
    "The relation is not strict in the middle: E3b keeps 50% but is worse than E1 (39%), because it also removes useful edges and leaves 44 sensors with no neighbours.",
    "Dense graphs (E2, E3a, about 76,000-102,000 edges) train worse than graphs of about 50,000 edges.",
  ]),

  h2("4.6 Statistical significance"),
  tcap("Pairwise tests. \"% sensors better\" is for the first model of the pair; DM p from the per-timestep MAE."),
  table([
    ["Pair", "Horizon, seed", "% sensors better", "DM p", "Verdict"],
    ["E1 vs E0", "5 min, 42", "59%", "< 0.001", "E1 better"],
    ["E1 vs E0", "5 min, 7", "93%", "< 0.001", "E1 better"],
    ["E1 vs E0", "60 min, 42", "70%", "< 0.001", "E1 better"],
    ["E5 vs E0", "5 min, 42", "62%", "< 0.001", "E5 better"],
    ["E5 vs E1", "5 min, 42", "56%", "0.15", "Tie"],
    ["E5 vs E1", "5 min, 7", "22%", "< 0.001", "E1 better"],
    ["E5 vs E1", "30 min, 42", "14%", "< 0.001", "E1 better"],
    ["E5 vs E1", "60 min, 42", "34%", "< 0.001", "E1 better"],
    ["E2 vs E1", "5 min, 42", "15%", "< 0.001", "E1 better"],
    ["E2 vs E0", "5 min, 42", "25%", "< 0.001", "E0 better"],
    ["E3b vs E1", "5 min, 42", "16%", "1e-8", "E1 better"],
    ["E3b vs E0", "5 min, 42", "30%", "0.17", "Tie"],
    ["NOGRAPH vs E1", "5 min, 42", "82%", "< 0.001", "NOGRAPH better"],
    ["NOGRAPH vs E0", "5 min, 42", "88%", "< 0.001", "NOGRAPH better"],
    ["NOGRAPH vs E5", "5 min, 42", "74%", "< 0.001", "NOGRAPH better"],
    ["E1 vs HA (MAE)", "60 min", "-", "0.016", "E1 better"],
    ["E5 vs HA (MAE)", "60 min", "-", "0.67", "Tie"],
    ["E0 vs HA (MAE)", "60 min", "-", "0.98", "Tie"],
    ["NOGRAPH vs E1", "30 min, 42", "8%", "< 0.001", "E1 better"],
    ["NOGRAPH vs E5", "30 min, 42", "39%", "0.54", "Tie"],
    ["NOGRAPH vs E1", "60 min, 42", "94%", "< 0.001", "NOGRAPH better"],
    ["NOGRAPH vs E5", "60 min, 42", "96%", "< 0.001", "NOGRAPH better"],
    ["NOGRAPH vs E0", "60 min, 42", "96%", "< 0.001", "NOGRAPH better"],
    ["E7 vs E2", "5 min, 42", "37%", "< 0.001", "E2 better"],
    ["E7 vs E1", "5 min, 42", "14%", "< 0.001", "E1 better"],
  ], [2, 1.6, 1.6, 1.2, 1.8]),
  ...bullets([
    "Every claim \"E1 beats E0\" and \"E1 beats E5\" is significant except E5 vs E1 at 5 min seed 42 (tie).",
    "No causal refinement (E2, E3b, E5, E7) is significantly better than E1 anywhere.",
    "A graph beats NOGRAPH significantly only once: E1 at 30 min.",
  ]),

  h2("4.7 Error by time of day"),
  ...figure(FIG("fig08_time_of_day.png"), "Test MAE per hour of day at 5 min, seed 42."),
  ...bullets([
    "All models are worst in the morning (7-8 h) and evening (16-18 h) peaks; HA is much worse than every model at all hours at 5 min.",
    "NOGRAPH has the lowest error from about 7 h to 22 h. E1 is the best between about 0 h and 4 h, when traffic is light and neighbours give useful information.",
    "E1 is lower than E0 at most hours, mainly in the evening (17-22 h).",
  ]),

  h2("4.8 Where on the network the VAR graph helps"),
  ...figure(path.join(ROOT, "results", "reports", "seed7_E0_E1_E5", "figures", "03_delta_map_E0_vs_E1.png"),
            "Per-sensor MAE change of E0 relative to E1 (5 min, seed 7). Red: E0 worse than E1; blue: E0 better.", 430),
  ...bullets([
    "E1 is better than E0 almost everywhere (93% of sensors at seed 7), not only in one region.",
    "Many of the largest gains (dark red) are at freeway interchanges, where the road graph links many sensors and the VAR weights can favour the ones that matter.",
  ]),

  h2("4.9 Training stability"),
  ...figure(FIG("fig07_validation_curves.png"), "Validation RMSE per epoch (log scale), 5 min, seed 42."),
  ...bullets([
    "Every graph model had validation spikes (collapses) at the fixed learning rate 0.001: E0 at epochs 2-6, E1 at 5-8 and 16-19, E2 at 6, 8 and 15-19, E3b from epoch 9, E5 from epoch 5. E7 stayed at 66-96 from epoch 2 to 18 and recovered only in the last two epochs.",
    "NOGRAPH had the smoothest curve at 5 min, but at 30 min it also collapsed once (389 at epoch 6). Graph mixing makes collapses more frequent, but they are not caused by the graph alone.",
    "Because the best-validation checkpoint is used, the collapses mostly change the best epoch (E5: epoch 4, E3b: epoch 7) rather than the reported test error, but they add noise to single-seed comparisons.",
  ]),
  pageBreak(),
);

// 5-7. Conclusions, limitations, pending
children.push(
  h1("5. Conclusions"),
  ...bullets([
    "**Learned VAR weights beat road weights.** E1 uses exactly the road graph's edges with VAR weights and beats E0 on both seeds and at 5 and 60 min.",
    "**The causal refinements do not add accuracy.** Adding same-time LiNGAM effects (E2), PEM edge selection or weighting (E3a, E3b, E4), horizon-specific impulse-response graphs (E5) and a road prior in PEM (E7) never beat E1 significantly; several are worse than the road graph.",
    "**PEM barely changes the graph.** Its probabilities are close to 0 or 1, so weighting by P (E4) or adding a road prior (E7, 0.025% weight change) leaves the graph almost unchanged.",
    "**Dense graphs hurt.** Graphs that dilute each sensor's own signal (E3a 3%, E0 20%, E2 and E7 24%) give the highest errors.",
    "**A graph helps A3T-GCN only at 30 minutes.** NOGRAPH is the best model at 5 min and, clearly, at 60 min (43.87 vs E1 47.41). E1 beats NOGRAPH only at 30 min (37.47 vs 39.46), by a margin close to the seed effect.",
    "**At 60 minutes only E1 among the graph models beats the historical average** on MAE; NOGRAPH beats it by a wide margin.",
  ]),
  h1("6. Limitations"),
  ...bullets([
    "Most experiments have one seed; the seed effect (up to 1.4 RMSE) is as large as the differences between experiments.",
    "Training collapses occur at the fixed learning rate 0.001 in every graph model; the protocol was kept identical for all experiments rather than tuned per graph.",
    "20 epochs for every run; E1 at 60 min reached its best at epoch 19 and may be under-trained.",
    "Only 49% of test windows (5,183 of 10,500) are scored, because a target must be observed at all 2,056 sensors.",
    "A3T-GCN sees one hour of input, so it cannot use the weekly pattern that makes HA strong at night.",
    "LiNGAM uses 10,000 residual rows per unit and runs on local units, not on all 2,056 sensors jointly; PEM settings were taken from the paper without a separate validation stage; E3 was amended after its first result (approved, reported as post-hoc).",
  ]),
  h1("7. Pending work"),
  tcap("Runs still to be done. Each fills the matching placeholders in this report."),
  table([
    ["Run", "Where it is used", "Status"],
    ["NOGRAPH and E1, 30 min, seed 7 (does the only graph win hold?)", "3.2, 3.4, 4.6, 5", "[paste status]"],
    ["NOGRAPH, 5 min, seed 7", "3.2, 4.1, 4.4", "[paste status]"],
    ["E0, 30 min, seed 42", "3.3, 4.1, 4.3", "[paste status]"],
    ["HA, NOGRAPH, E0, E1, E5 at 15 min", "3.1-3.10, 4.1, 4.3", "[paste status]"],
    ["E6 VAR, LiNGAM, PEM on data-driven candidates, then training", "3.11, 4.1", "Optional (PEM unlikely to change the result)"],
    ["Third seed (123) for E0, E1, E5, NOGRAPH at 5 min", "4.4", "Optional"],
  ], [3.5, 2.2, 1.6]),
);

// ---------- document ----------
const doc = new Document({
  creator: "Ablation study",
  title: "Ablation study report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

const out = process.env.REPORT_OUT || path.join(ROOT, "presentation", "ablation_study_report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("Saved", out); });
