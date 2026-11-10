#!/bin/bash
# Resume of the dysts stage after the Spark hard stop at ~21:02 (205/287 runs in dysts.partial.json).
# 6 workers (baselines job finished). --resume skips the indices already in the partial checkpoint.
cd ~/eqdisc_v3
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=.venv/bin/python
I=results_ablation
echo "$(date +%T) start dysts (resume after reboot)" >> $I/progress.log
$P benchmarks/ablation.py --set dysts --every 17 --resume --workers 6 --output $I/dysts.json >> $I/dysts.log 2>&1
rc=$?; echo "$(date +%T) end dysts rc=$rc" >> $I/progress.log; [ $rc -eq 0 ] || echo "$(date +%T) FAILED dysts" >> $I/progress.log
echo "$(date +%T) ALL_DONE" >> $I/progress.log
