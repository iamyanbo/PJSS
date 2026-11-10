#!/bin/bash
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
P=.venv/bin/python
I=results_hard120
echo "$(date +%T) start ours" > $I/ours_progress.log
$P benchmarks/final_bench.py --kind poly3 --data data/hard120.npz --arms final C --workers 11 --output $I/ours_hard120.json > $I/ours_hard120.log 2>&1
echo "$(date +%T) end hard120 rc=$?" >> $I/ours_progress.log
$P benchmarks/final_bench.py --kind poly3 --data data/hybrid_confirm.npz --index-file $I/hard40_idx.json --arms final C --workers 11 --output $I/ours_hard40_repro.json > $I/ours_hard40_repro.log 2>&1
echo "$(date +%T) end hard40 rc=$?" >> $I/ours_progress.log
echo "$(date +%T) OURS_DONE" >> $I/ours_progress.log
