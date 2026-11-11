#!/bin/bash
# E9 power round: group knockoffs (|r|>=0.5, 0.7) and a smaller pool (M=30); proxy + paired stats on the same 500 targets.
PY="C:/Users/sanja/gpu-envs/mamba/envs/traffic-causal/python.exe"
cd "$(dirname "$0")/.."
set -e
"$PY" -u tools/run_knockoff_selection.py --horizon 1 --sample 500 --q 0.2 0.3 --k-add 10 --group-corr 0.5 --tag _g50
"$PY" -u tools/run_knockoff_selection.py --horizon 1 --sample 500 --q 0.2 0.3 --k-add 10 --group-corr 0.7 --tag _g70
"$PY" -u tools/run_knockoff_selection.py --horizon 1 --sample 500 --q 0.2 0.3 --k-add 10 --pool-size 30 --tag _m30
NAMES="e9_g50_q20 e9_g50_q30 e9_g50h10 e9_g70_q20 e9_g70_q30 e9_g70h10 e9_m30_q20 e9_m30_q30 e9_m30h10 hyb_pc10 hyb_corr10"
EXTRA=""
for n in $NAMES; do EXTRA="$EXTRA $n=outputs/proxy_graphs/${n}__h{h}.npz"; done
"$PY" -u tools/graph_proxy.py --horizons 1 --sample 500 --only $NAMES --extra $EXTRA --tag e9power --save-persensor
PAIRS=""
for n in e9_g50_q20 e9_g50_q30 e9_g50h10 e9_g70_q20 e9_g70_q30 e9_g70h10 e9_m30_q20 e9_m30_q30 e9_m30h10; do PAIRS="$PAIRS $n:hyb_pc10 $n:E8V"; done
"$PY" tools/paired_stats.py outputs/proxy/persensor_sample500_e9power_h1.npz --pairs $PAIRS --json outputs/local_queue/paired_e9_power_500.json
echo CHAIN_DONE
