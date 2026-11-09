"""
Stage graph artifacts for upload to the Kaggle artifacts dataset: verify, then copy into ONE folder.
Only files that pass tools/verify_graph_artifacts.py are copied; the folder holds nothing else, so it can be uploaded as is.

    python tools/stage_upload.py --src outputs/hybrid_graphs --variants hyb_pc10 vstd --horizons 1 6 12 --dest outputs/upload_hybrid
    python tools/stage_upload.py --src outputs/hybrid_graphs --all --dest outputs/upload_hybrid
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

import verify_graph_artifacts as vga  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--variants", nargs="*", default=[], help="graph variants, e.g. hyb_pc10 vstd")
    parser.add_argument("--horizons", type=int, nargs="*", default=[1, 6, 12])
    parser.add_argument("--all", action="store_true", help="every E*__*.npz of --src")
    args = parser.parse_args()

    src, dest = Path(args.src), Path(args.dest)
    if args.all:
        files = sorted(src.glob("E*__*.npz"))
    else:
        files = [src / f"E8__h{h}__lag3_alpha1e+06__{v}.npz" for v in args.variants for h in args.horizons]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit("missing graph files (build them first with build_hybrid_graphs.py):\n  " + "\n  ".join(missing))

    sensors = np.load(IMPLEMENTATION_DIR / "outputs" / "var_train" / "lag3_alpha1e+06" / "sensors.npy")
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in files:
        problems, _ = vga.verify(f, sensors)
        if problems:
            print(f"SKIPPED {f.name}: {'; '.join(problems)}")
            continue
        shutil.copy2(f, dest / f.name)
        copied += 1
    print(f"{copied}/{len(files)} verified graphs copied to {dest}")
    print("contents:", sorted(p.name for p in dest.iterdir()))
    sys.exit(0 if copied == len(files) else 1)


if __name__ == "__main__":
    main()
