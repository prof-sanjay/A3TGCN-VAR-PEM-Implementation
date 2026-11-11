"""
Run the local follow-up jobs one after another, unattended. No internet is needed: every job reads local files only.

    python tools/run_local_queue.py            # run everything not yet finished
    python tools/run_local_queue.py --list     # show the jobs and their status

Each job writes its own log to outputs/local_queue/<job>.log; status.json records ok / failed and the time taken.
A failed job does not stop the queue. Jobs already marked ok are skipped, so the queue can be started again after an
interruption (the PEM jobs also resume from their checkpoints). At the end RESULTS.md is written (tools/summarise_local_queue.py).
While it runs, the process asks Windows not to put the computer to sleep (the display may still turn off).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = IMPLEMENTATION_DIR / "outputs" / "local_queue"
PG = "outputs/proxy_graphs"

POOL_SIZES = [20, 30, 40, 60]
PEM_NAMES = ["hyb_pemflow5", "hyb_pemflow10", "hyb_pemflow15", "hyb_pemuni5", "hyb_pemuni10", "hyb_pemuni15"]
CONTROL_NAMES = ["hyb_pc10", "hyb_corr10", "hyb_rand10", "V_std"]


def extra(names, h=1):
    return [f"{n}={PG}/{n}__h{{h}}.npz" for n in names]


def jobs():
    j = []
    j.append(("1_pem_real_flow", [PY, "-u", "tools/run_pem_forecast_relation.py", "--horizon", "1", "--prior", "flow",
                                  "--sample", "500", "--workers", "8"]))
    j.append(("2_pem_real_uniform", [PY, "-u", "tools/run_pem_forecast_relation.py", "--horizon", "1", "--prior", "uniform",
                                     "--sample", "500", "--workers", "8"]))
    j.append(("3_build_pem_hybrids", [PY, "-u", "tools/build_pem_hybrids.py", "--horizon", "1", "--k-add", "5", "10", "15"]))
    names4 = PEM_NAMES + CONTROL_NAMES
    j.append(("4_proxy_real_pem_500", [PY, "-u", "tools/graph_proxy.py", "--horizons", "1", "--sample", "500", "--only", *names4,
                                       "--extra", *extra(names4), "--tag", "pemreal", "--save-persensor", "--out", "outputs/proxy"]))
    j.append(("5_paired_real_pem", [PY, "tools/paired_stats.py", "outputs/proxy/persensor_sample500_pemreal_h1.npz", "--pairs",
                                    "hyb_pemflow10:hyb_corr10", "hyb_pemflow10:hyb_rand10", "hyb_pemflow10:hyb_pc10",
                                    "hyb_pemflow10:hyb_pemuni10", "hyb_pemflow10:V_std", "hyb_pemflow10:E8V",
                                    "hyb_pc10:hyb_corr10", "hyb_pc10:hyb_rand10", "hyb_pc10:V_std",
                                    "--json", "outputs/local_queue/paired_real_pem_500.json"]))
    names6 = CONTROL_NAMES
    j.append(("6_proxy_full_paired", [PY, "-u", "tools/graph_proxy.py", "--horizons", "1", "--only", *names6,
                                      "--extra", *extra(names6), "--tag", "fullpaired", "--save-persensor", "--out", "outputs/proxy"]))
    j.append(("7_paired_full", [PY, "tools/paired_stats.py", "outputs/proxy/persensor_full_fullpaired_h1.npz", "--pairs",
                                "hyb_pc10:hyb_corr10", "hyb_pc10:hyb_rand10", "hyb_pc10:V_std", "hyb_pc10:E8V",
                                "hyb_corr10:hyb_rand10", "V_std:E8V",
                                "--json", "outputs/local_queue/paired_full_2056.json"]))
    for m in POOL_SIZES:
        j.append((f"8_pool_M{m}_build", [PY, "-u", "tools/build_hybrid_candidates.py", "--horizons", "1", "--k-add", "10",
                                         "--pool-size", str(m)]))
        names = [f"hyb_{r}10_M{m}" for r in ("pc", "rand", "corr")]
        j.append((f"8_pool_M{m}_proxy", [PY, "-u", "tools/graph_proxy.py", "--horizons", "1", "--sample", "500", "--only", *names,
                                         "--extra", *extra(names), "--tag", f"poolM{m}", "--out", "outputs/proxy"]))
    j.append(("9_summary", [PY, "tools/summarise_local_queue.py"]))
    return j


def keep_awake(on):
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--redo", nargs="*", default=[], help="job names to run again even if marked ok")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    queue = jobs()
    if args.list:
        for name, _ in queue:
            print(f"{name:26s} {status.get(name, {}).get('state', 'pending')}")
        return

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    keep_awake(True)
    try:
        for name, cmd in queue:
            if status.get(name, {}).get("state") == "ok" and name not in args.redo:
                print(f"[skip] {name} (already ok)", flush=True)
                continue
            print(f"[{time.strftime('%H:%M:%S')}] start {name}", flush=True)
            t0 = time.time()
            with open(OUT / f"{name}.log", "w", encoding="utf-8") as log:
                log.write(" ".join(cmd) + "\n\n")
                log.flush()
                code = subprocess.run(cmd, cwd=IMPLEMENTATION_DIR, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            status[name] = {"state": "ok" if code == 0 else "failed", "exit_code": code,
                            "minutes": round((time.time() - t0) / 60, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            status_path.write_text(json.dumps(status, indent=2))
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {status[name]['state']} in {status[name]['minutes']} min", flush=True)
    finally:
        keep_awake(False)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
