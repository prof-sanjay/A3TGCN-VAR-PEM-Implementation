#!/bin/bash
# E9 confirmation on ALL 2,056 targets for the group-knockoff variant (|r|>=0.5): selection, proxy, paired stats.
PY="C:/Users/sanja/gpu-envs/mamba/envs/traffic-causal/python.exe"
cd "$(dirname "$0")/.."
set -e
"$PY" -u tools/run_knockoff_selection.py --horizon 1 --q 0.2 0.3 --k-add 10 --group-corr 0.5 --tag _g50 --out outputs/proxy_graphs_full
NAMES="e9_g50_q20 e9_g50_q30 e9_g50h10 hyb_pc10"
EXTRA="e9_g50_q20=outputs/proxy_graphs_full/e9_g50_q20__h{h}.npz e9_g50_q30=outputs/proxy_graphs_full/e9_g50_q30__h{h}.npz e9_g50h10=outputs/proxy_graphs_full/e9_g50h10__h{h}.npz hyb_pc10=outputs/proxy_graphs/hyb_pc10__h{h}.npz"
"$PY" -u tools/graph_proxy.py --horizons 1 --only $NAMES --extra $EXTRA --tag e9full --save-persensor
"$PY" tools/paired_stats.py outputs/proxy/persensor_full_e9full_h1.npz --pairs e9_g50_q30:hyb_pc10 e9_g50_q20:hyb_pc10 e9_g50h10:hyb_pc10 e9_g50_q30:E8V e9_g50_q20:E8V e9_g50h10:E8V --json outputs/local_queue/paired_e9_full_2056.json
echo CHAIN_DONE
