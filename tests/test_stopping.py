"""
Tests for training/stopping.py (Phase 3):

    1. Early stopping: off by default; stops after `patience` epochs
       without improvement; an improvement resets the count.
    2. Time budget: no epoch is started that would end after the
       budget (expected length = longest epoch so far).
    3. Milestones: best epoch among epochs <= M, unaffected by later
       epochs; which checkpoints must be saved at each epoch.

Run from the implementation directory:

    python -m tests.test_stopping
"""

import sys
from pathlib import Path

# Make the implementation folder importable when this file is run
# directly (python tests/<file>.py or the editor's Run button).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.stopping import MilestoneTracker, TrainingStopper


def test_early_stopping():

    off = TrainingStopper()
    for value in [5, 6, 7, 8, 9, 10, 11]:
        off.after_epoch(value, 10.0)
        assert not off.should_stop(0.0)

    stopper = TrainingStopper(early_stopping_patience=3)
    stops = []
    for value in [5, 6, 4, 4.5, 4.2, 4.1, 3.9, 5, 5, 5]:
        stopper.after_epoch(value, 10.0)
        stops.append(stopper.should_stop(0.0))

    # Improvements at epochs 1, 3, 7; three non-improving epochs follow
    # epoch 3 (4, 5, 6) -> stop before epoch 7 would be signalled after 6.
    assert stops.index(True) == 5, stops
    assert "early stopping" in stopper.reason

    try:
        TrainingStopper(early_stopping_patience=0)
    except ValueError:
        pass
    else:
        raise AssertionError("patience 0 accepted")

    print("[PASS] 1. early stopping off by default, stops after patience, resets on improvement")


def test_time_budget():

    stopper = TrainingStopper(max_seconds=100.0)

    assert not stopper.should_stop(10.0)          # no epoch yet: start it
    stopper.after_epoch(1.0, 30.0)
    assert not stopper.should_stop(40.0)          # 40 + 30 <= 100
    stopper.after_epoch(0.9, 20.0)                # longest stays 30
    assert not stopper.should_stop(70.0)          # 70 + 30 = 100, still fits
    assert stopper.should_stop(71.0)              # 71 + 30 > 100
    assert "time budget" in stopper.reason

    print("[PASS] 2. time budget: an epoch is only started if it fits")


def test_milestones():

    tracker = MilestoneTracker([20, 5])
    assert tracker.milestones == [5, 20]

    saves = {}
    values = [10, 9, 9.5, 8, 8.5, 7, 7.5, 6.9]     # epochs 1..8
    for epoch, value in enumerate(values, start=1):
        saves[epoch] = tracker.update(epoch, value)

    assert tracker.best_epoch == {5: 4, 20: 8}
    assert saves[4] == [5, 20] and saves[6] == [20] and saves[8] == [20]
    assert saves[3] == [] and saves[5] == []

    assert MilestoneTracker(None).update(1, 1.0) == []

    print("[PASS] 3. milestones: best epoch <= M kept; later epochs never change it")


def main():

    print("=" * 70)
    print("STOPPING / MILESTONE TESTS")
    print("=" * 70)

    test_early_stopping()
    test_time_budget()
    test_milestones()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
