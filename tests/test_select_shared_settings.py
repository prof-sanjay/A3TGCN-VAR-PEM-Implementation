"""
Tests for select_shared_settings.py (Phase 3 decision rule):

    1. A candidate less than 2 % better than the reference (lr 0.001,
       20 epochs) does not replace it.
    2. A candidate at least 2 % better is chosen, with the command that
       evaluates exactly its checkpoint.
    3. Budgets without a checkpoint (no milestone, not the full run)
       are not candidates; runs with test metrics are refused.

Run from the implementation directory:

    python -m tests.test_select_shared_settings
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import subprocess
import tempfile

from training.run_logger import EpochMetricsLogger


IMPLEMENTATION_DIR = Path(__file__).resolve().parent.parent


def make_run(root, name, lr, values, milestones=(20,), tested=False):

    run = Path(root) / name
    logger = EpochMetricsLogger(run / "epoch_metrics.csv")
    for epoch, v in enumerate(values, start=1):
        logger.log_epoch(epoch, {"RMSE": v, "MAE": v, "Masked_MAPE": 1.0, "R2": 0.9,
                                 "Accuracy": 0.5}, lr, 1.0, 1.0)

    best_until = {
        str(m): min(range(1, m + 1), key=lambda e: values[e - 1]) for m in milestones
    }
    (run / "results.json").write_text(json.dumps({
        "config": {"learning_rate": lr, "training_epoch": len(values)},
        "milestones": best_until,
        "test_metrics": {"RMSE": 1.0} if tested else "not evaluated (selection run)",
    }), encoding="utf-8")
    return run


def select(*runs, extra=()):

    out = Path(runs[0]).parent / "decision.json"
    result = subprocess.run(
        [sys.executable, str(IMPLEMENTATION_DIR / "selection" / "select_shared_settings.py"),
         *map(str, runs), "--out", str(out), *extra],
        capture_output=True, text=True, cwd=IMPLEMENTATION_DIR,
    )
    if result.returncode != 0:
        return None, result.stdout + result.stderr
    return json.loads(out.read_text(encoding="utf-8")), result.stdout


def main():

    print("=" * 70)
    print("PHASE 3 SELECTION RULE TESTS")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:

        # lr 0.001: best 30.0 within 20 epochs, 29.7 by epoch 30 (1 % better)
        a = make_run(Path(tmp) / "one", "lr001", 0.001,
                     [40] * 10 + [30.0] + [35] * 9 + [29.7] + [33] * 9)
        # lr 0.0005: best 30.2 within 20, 29.9 within 30 (< 2 % better)
        b = make_run(Path(tmp) / "one", "lr0005", 0.0005,
                     [40] * 19 + [30.2] + [29.9] + [33] * 9)

        decision, out = select(a, b)
        assert decision["chosen"]["learning_rate"] == 0.001
        assert decision["chosen"]["epochs"] == 20
        assert decision["chosen"]["checkpoint"] == "best_until_epoch_20"
        assert decision["chosen"]["evaluate_command"].endswith("--milestone 20")
        print("[PASS] 1. less than 2 % better: the reference is kept")

        # lr 0.0005 reaches 29.0 by epoch 30: 3.3 % better -> switch
        c = make_run(Path(tmp) / "two", "lr001", 0.001,
                     [40] * 10 + [30.0] + [35] * 19)
        d = make_run(Path(tmp) / "two", "lr0005", 0.0005,
                     [40] * 25 + [29.0] + [33] * 4)

        decision, out = select(c, d)
        chosen = decision["chosen"]
        assert (chosen["learning_rate"], chosen["epochs"]) == (0.0005, 30)
        assert chosen["checkpoint"] == "best_model"
        assert "--milestone" not in chosen["evaluate_command"]
        print("[PASS] 2. at least 2 % better: switch, command evaluates that checkpoint")

        # Budget 20 without a milestone checkpoint is not a candidate.
        e = make_run(Path(tmp) / "three", "nomilestone", 0.001,
                     [30.0] * 30, milestones=())
        decision, out = select(e)
        assert [c["epochs"] for c in decision["candidates"]] == [30]

        tested = make_run(Path(tmp) / "four", "tested", 0.001, [30.0] * 30, tested=True)
        decision, out = select(tested)
        assert decision is None and "already has test metrics" in out
        print("[PASS] 3. budgets without a checkpoint skipped; tested runs refused")

        # --reference-run: a completed (tested) lr 0.001 / 20-epoch run.
        ref = make_run(Path(tmp) / "five", "E0_final", 0.001,
                       [40] * 14 + [30.0] + [31] * 5, milestones=(), tested=True)
        slight = make_run(Path(tmp) / "five", "lr0005_slight", 0.0005,
                          [40] * 20 + [29.7] + [33] * 9)          # 1 % better
        decision, out = select(slight, extra=["--reference-run", str(ref)])
        assert decision["chosen"]["run"] == str(ref)
        assert decision["chosen"]["evaluate_command"].startswith("already evaluated")

        clear = make_run(Path(tmp) / "six", "lr0005_clear", 0.0005,
                         [40] * 20 + [29.0] + [33] * 9)           # 3.3 % better
        ref6 = make_run(Path(tmp) / "six", "E0_final", 0.001,
                        [40] * 14 + [30.0] + [31] * 5, milestones=(), tested=True)
        decision, out = select(clear, extra=["--reference-run", str(ref6)])
        assert (decision["chosen"]["learning_rate"], decision["chosen"]["epochs"]) == (0.0005, 30)

        wrong = make_run(Path(tmp) / "seven", "not_ref", 0.001, [30.0] * 30, tested=True)
        decision, out = select(slight, extra=["--reference-run", str(wrong)])
        assert decision is None and "not a completed" in out
        print("[PASS] 4. --reference-run: tested reference accepted, 2 % rule applied, wrong runs refused")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
