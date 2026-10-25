"""
Phase 3: choose the shared training settings on VALIDATION data only.

Rule (fixed on 2026-10-25, before any Phase 3 result existed):

    candidates   every selection run (its learning rate) x every epoch
                 budget B in --budgets (default 20 and 30)
    score        lowest validation RMSE among epochs <= B
                 (epoch_metrics.csv; the first B epochs of a longer run
                 follow the same schedule as a run of exactly B epochs)
    reference    the current setting: learning rate 0.001, 20 epochs
    decision     keep the reference unless a candidate's score is at
                 least --margin (default 2 %) lower; otherwise take the
                 candidate with the lowest score. The margin guards
                 against single-seed noise (one run per setting).

Instability is reported, not used: epochs whose validation RMSE is more
than 1.5x the run's best.

The test set is not read. The output names the one command that
evaluates the chosen run's test set (main.py --evaluate-only).

    python select_shared_settings.py results/E0_lr0.001_30ep results/E0_lr0.0005_30ep
"""

import argparse
import json
from pathlib import Path

from training.run_logger import read_epoch_metrics


REFERENCE = {"learning_rate": 0.001, "epochs": 20}


def candidates_of(run_dir: Path, budgets):

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    if isinstance(results.get("test_metrics"), dict):
        raise ValueError(f"{run_dir} already has test metrics; use selection runs only.")

    rows = read_epoch_metrics(run_dir / "epoch_metrics.csv")
    trained = len(rows)
    lr = float(results["config"]["learning_rate"])
    run_best = min(r["RMSE"] for r in rows)

    for budget in budgets:

        if budget > trained:
            continue

        within = [r for r in rows if r["epoch"] <= budget]
        best = min(within, key=lambda r: (r["RMSE"], r["epoch"]))

        is_full = budget == trained
        milestone_ok = str(budget) in {str(k) for k in results.get("milestones", {})}

        if not (is_full or milestone_ok):
            # No checkpoint exists for this budget: it cannot be evaluated.
            continue

        yield {
            "run": str(run_dir),
            "learning_rate": lr,
            "epochs": budget,
            "val_rmse": best["RMSE"],
            "best_epoch": best["epoch"],
            "spike_epochs": sum(r["RMSE"] > 1.5 * run_best for r in within),
            "stopped_early": results.get("stopped_early"),
            "checkpoint": "best_model" if is_full else f"best_until_epoch_{budget}",
            "evaluate_command": (
                f"python -m implementation.main --evaluate-only {run_dir}"
                + ("" if is_full else f" --milestone {budget}")
            ),
        }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    candidates = [c for run in args.runs for c in candidates_of(Path(run), args.budgets)]

    if not candidates:
        raise SystemExit("No evaluable candidates.")

    reference = [
        c for c in candidates
        if abs(c["learning_rate"] - REFERENCE["learning_rate"]) < 1e-12
        and c["epochs"] == REFERENCE["epochs"]
    ]

    best = min(candidates, key=lambda c: (c["val_rmse"], c["epochs"]))

    if reference:
        ref = reference[0]
        improvement = 1.0 - best["val_rmse"] / ref["val_rmse"]
        chosen = best if improvement >= args.margin else ref
        reason = (
            f"best candidate improves validation RMSE by {improvement:.2%} over the "
            f"reference (margin {args.margin:.0%}): "
            + ("switch" if chosen is best and chosen is not ref else "keep the reference")
        )
    else:
        chosen = best
        reason = "reference (lr 0.001, 20 epochs) not among the candidates: lowest score"

    print(f"{'learning rate':>13} {'epochs':>6} {'val RMSE':>9} {'best ep':>7} {'spikes':>6}  run")
    for c in sorted(candidates, key=lambda c: (c["learning_rate"], c["epochs"])):
        mark = "  <- chosen" if c is chosen else ""
        print(f"{c['learning_rate']:>13g} {c['epochs']:>6} {c['val_rmse']:>9.3f} "
              f"{c['best_epoch']:>7} {c['spike_epochs']:>6}  {c['run']}{mark}")

    print(f"\nDecision: {reason}")
    print(f"Chosen  : learning rate {chosen['learning_rate']:g}, {chosen['epochs']} epochs")
    print(f"Test set (once): {chosen['evaluate_command']}")

    decision = {
        "rule": __doc__.split("Rule")[1].split("The test set")[0].strip(),
        "margin": args.margin,
        "reference": REFERENCE,
        "candidates": candidates,
        "chosen": chosen,
        "reason": reason,
    }

    out = Path(args.out) if args.out else Path(args.runs[0]).parent / "phase3_decision.json"
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
