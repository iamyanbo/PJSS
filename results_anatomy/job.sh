#!/bin/bash
# Cost anatomy + scaling vs DEDi (2026-10-28 overnight). Cores 0-4 belong to the fast-baselines job; not used here.
#  ours (core 15): op counts on all 80 head-to-head runs, then the scaling runs (D = 4-10 with joint_nd, degree-4 lib)
#  DEDi (GPU, 10 slots, one fresh process per run so worker memory cannot grow across runs; 6 h cap per run):
#    1. anatomy: 10 head-to-head runs (2 per condition), full 35-pair grid, with op counts
#    2. scaling: 10 runs at D = 4, 5, 6, 6, 10 (2 each) and 6 runs with the degree-4 library (35 terms),
#       lambda = 1e-3 row (5 pairs; contains the single pair (1e-3, 1))
cd ~/eqdisc_v3
O=results_anatomy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 XLA_PYTHON_CLIENT_PREALLOCATE=false
OURS=~/eqdisc_v2/.venv/bin/python; GPU=~/eqdisc_dedi_gpu/.venv/bin/python
B=benchmarks/cost_anatomy.py
echo "$(date "+%F %T") start" >> $O/progress.log
( taskset -c 15 $OURS $B ours --out $O/ours_h2h.jsonl > $O/ours_h2h.log 2>&1
  taskset -c 15 $OURS $B ours --runs $O/runs_nd.json --lib nd --out $O/ours_nd.jsonl > $O/ours_nd.log 2>&1
  EQ_DEG=4 taskset -c 15 $OURS $B ours --runs $O/runs_poly4.json --lib poly4 --out $O/ours_poly4.jsonl > $O/ours_poly4.log 2>&1
  echo "$(date "+%F %T") ours done" >> $O/progress.log ) &
# task list: one DEDi process per line
python3 - > $O/tasks.txt <<'EOF'
import json
O = "results_anatomy"
for r in json.load(open(f"{O}/runs_dedi_anatomy.json")):
    print(f"--run {r} --pairs full --out {O}/dedi_anatomy.jsonl")
for p, i in json.load(open(f"{O}/runs_nd.json")):
    print(f"--data {p} --index {i} --lib nd --pairs row --out {O}/dedi_nd.jsonl")
for p, i in json.load(open(f"{O}/runs_poly4.json")):
    print(f"EQ_DEG=4 --data {p} --index {i} --lib poly4 --pairs row --out {O}/dedi_poly4.jsonl")
EOF
CORES=(5 6 7 8 9 16 17 18 19 10)
for s in $(seq 0 9); do
  ( n=0
    while read -r line; do
      if [ $((n % 10)) -eq $s ]; then
        env=""; case "$line" in EQ_DEG=4*) env="EQ_DEG=4"; line=${line#EQ_DEG=4 };; esac
        echo "$(date "+%F %T") slot $s start: $line" >> $O/progress.log
        env $env timeout -k 60 21600 taskset -c ${CORES[$s]} $GPU $B dedi --ext ~/eqdisc_dedi --device gpu $line >> $O/dedi_slot$s.log 2>&1
        rc=$?  # (the launched version printed $? inside "$(date ...)", which reset it: every end line said exit 0)
        echo "$(date "+%F %T") slot $s end (exit $rc): $line" >> $O/progress.log
      fi
      n=$((n + 1))
    done < $O/tasks.txt ) &
done
wait
echo "$(date "+%F %T") DONE" >> $O/progress.log
