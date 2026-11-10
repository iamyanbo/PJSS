#!/bin/bash
# Ablation job list as run on the 12-core Windows PC (8 workers, BLAS threads 1), 2026-10-25.
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
I=results_ablation
run() { name=$1; shift; echo "$(date +%T) start $name" >> $I/progress.log
        python benchmarks/ablation.py "$@" --workers 8 --output $I/$name.json > $I/$name.log 2>&1
        rc=$?; echo "$(date +%T) end $name rc=$rc" >> $I/progress.log; [ $rc -eq 0 ] || echo "$(date +%T) FAILED $name" >> $I/progress.log; }
run hard40  --set hard40
run sealed  --set sealed  --every 5
run colored --set colored --every 5
run dysts   --set dysts   --every 11
echo "$(date +%T) ALL_DONE" >> $I/progress.log
