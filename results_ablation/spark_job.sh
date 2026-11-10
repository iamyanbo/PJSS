#!/bin/bash
# Remaining ablation stages, run on the Spark (~/eqdisc_v3) at 4 workers because the PC had to be freed.
# Subsampled further than planned for the PC (colored every 9th = 89 runs, dysts D=3 every 17th = 287 runs).
cd ~/eqdisc_v3
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=.venv/bin/python
I=results_ablation
mkdir -p $I
run() { name=$1; shift; echo "$(date +%T) start $name" >> $I/progress.log
        $P benchmarks/ablation.py "$@" --resume --workers 4 --output $I/$name.json >> $I/$name.log 2>&1
        rc=$?; echo "$(date +%T) end $name rc=$rc" >> $I/progress.log; [ $rc -eq 0 ] || echo "$(date +%T) FAILED $name" >> $I/progress.log; }
run colored --set colored --every 9
run dysts   --set dysts   --every 17
echo "$(date +%T) ALL_DONE" >> $I/progress.log
