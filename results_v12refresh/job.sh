#!/bin/bash
# v1.2 refresh of the paper's main tables: hard-120 and dysts with the current default (v1.2), one core (14).
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
O=results_v12refresh; P=~/eqdisc_v2/.venv/bin/python
echo "$(date "+%F %T") start" >> $O/progress.log
[ -e $O/hard120_v12.json ] || taskset -c 14 $P benchmarks/final_bench.py --kind poly3 --data data/hard120.npz \
  --arms final_v12 --workers 1 --output $O/hard120_v12.json > $O/hard120.log 2>&1
echo "$(date "+%F %T") hard120 done" >> $O/progress.log
[ -e $O/dysts_v12.json ] || taskset -c 14 $P benchmarks/dysts_bench.py --method final --data data/dysts_test.npz \
  --workers 1 --output $O/dysts_v12.json > $O/dysts.log 2>&1
echo "$(date "+%F %T") DONE" >> $O/progress.log
