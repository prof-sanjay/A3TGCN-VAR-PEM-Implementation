"""
Collect the outputs of the local queue (tools/run_local_queue.py) into one readable file: outputs/local_queue/RESULTS.md.
Reads only files the queue wrote; prints the file as well.
"""

import json
from pathlib import Path

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
OUT = IMPLEMENTATION_DIR / "outputs" / "local_queue"
PROXY = IMPLEMENTATION_DIR / "outputs" / "proxy"


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table(title, rows, header):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    md = ["# Local queue results (validation proxy, horizon 1)", "",
          "All numbers use TRAINING rows for fitting and VALIDATION rows for scoring; the test period is never read.",
          "Lower RMSE / MAE is better.", ""]

    status = load(OUT / "status.json") or {}
    md.append(table("Job status", [(k, v["state"], f"{v['minutes']} min") for k, v in status.items()], ["job", "state", "time"]))

    # ---- real PEM vs controls (500 targets)
    d = load(PROXY / "proxy_sample500_pemreal_h1.json")
    if d:
        order = ["E8V", "V_std", "hyb_pc5", "hyb_pc10", "hyb_pc15", "hyb_corr10", "hyb_rand10",
                 "hyb_pemflow5", "hyb_pemflow10", "hyb_pemflow15", "hyb_pemuni5", "hyb_pemuni10", "hyb_pemuni15"]
        label = {"hyb_pc": "stand-in score", "hyb_corr": "correlation only", "hyb_rand": "random", "hyb_pemflow": "REAL PEM, flow prior",
                 "hyb_pemuni": "REAL PEM, uniform prior", "V_std": "standardised core, no additions", "E8V": "E8V (raw |Psi|)"}
        rows = []
        for n in order:
            v = d.get(f"h1/{n}")
            if v:
                key = next((k for k in label if n.startswith(k)), n)
                rows.append((n, label.get(key, ""), f"{v['val_rmse']:.3f}", f"{v['val_mae']:.3f}", f"{v['avg_sources']:.1f}"))
        md.append(table("Real PEM on the forecasting relation, 500 targets", rows, ["graph", "additions ranked by", "val RMSE", "val MAE", "sources/target"]))
    p = load(OUT / "paired_real_pem_500.json")
    if p:
        md.append(table("Paired comparison on the same 500 targets (A vs B)", [
            (r["A"], r["B"], f"{r['rmse_A']:.3f} vs {r['rmse_B']:.3f}", f"{r['rmse_ratio_A_over_B']:.4f} [{r['ratio_ci95'][0]:.4f}, {r['ratio_ci95'][1]:.4f}]",
             "CLEAR" if r["clear_difference"] else "not clear", f"{r['share_sensors_A_better']:.0%}") for r in p],
            ["A", "B", "RMSE", "ratio A/B [95% CI]", "difference", "sensors where A better"]))

    # ---- full-target paired proxy
    f = load(OUT / "paired_full_2056.json")
    if f:
        md.append(table("Paired comparison on all 2,056 targets (A vs B)", [
            (r["A"], r["B"], f"{r['rmse_A']:.3f} vs {r['rmse_B']:.3f}", f"{r['rmse_ratio_A_over_B']:.4f} [{r['ratio_ci95'][0]:.4f}, {r['ratio_ci95'][1]:.4f}]",
             "CLEAR" if r["clear_difference"] else "not clear", f"{r['share_sensors_A_better']:.0%}") for r in f],
            ["A", "B", "RMSE", "ratio A/B [95% CI]", "difference", "sensors where A better"]))

    # ---- pool-size sweep
    rows = []
    base = load(PROXY / "proxy_sample500_hybrid_h1.json") or {}
    if base:
        rows.append(("whole pool (mean 67)",) + tuple(f"{base[f'h1/hyb_{r}10']['val_rmse']:.3f}" if f"h1/hyb_{r}10" in base else "-" for r in ("pc", "corr", "rand")))
    for m in (60, 40, 30, 20):
        dm = load(PROXY / f"proxy_sample500_poolM{m}_h1.json")
        if dm:
            rows.append((f"M = {m}",) + tuple(f"{dm[f'h1/hyb_{r}10_M{m}']['val_rmse']:.3f}" if f"h1/hyb_{r}10_M{m}" in dm else "-" for r in ("pc", "corr", "rand")))
    if rows:
        md.append(table("Pool-size sweep (hybrid with 10 reserved additions; validation RMSE, 500 targets)", rows,
                        ["pool", "stand-in score (pc)", "correlation only", "random"]))

    text = "\n".join(md)
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
