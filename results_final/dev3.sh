#!/bin/bash
# Dev extension: the other half of nd_dev (half 1 gave prior 5/0, p = 0.06 < pre-set bar not met).
cd ~/eqdisc_v2
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
I=results_final
echo "$(date +%T) start dev_nd2" >> $I/dev_progress.log
.venv/bin/python final_bench.py --kind nd --data data/nd_dev.npz --index-file $I/idx_ndhalf2.json --arms off p15 --workers 18 --output $I/dev_nd2.json > $I/dev_nd2.log 2>&1
echo "$(date +%T) end dev_nd2 rc=$? DEV3_DONE" >> $I/dev_progress.log
