#!/bin/bash
# Rerun of the native-CHOLMOD parts of results_timing/job.sh. The first attempt (2026-10-28 11:53-13:24) failed on
# every native call: the adapter (benchmarks/dedi_native_shim) skipped its own folder by name and the job used it via
# a symlink, so it imported itself recursively. Fixed by comparing resolved paths. Shim rows and ours are unaffected.
cd ~/eqdisc_v3
O=results_timing
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
NAT=~/mm/cholmod/bin/python
PAIRS="1e-3,1 1,1e-2 0.1,1e-2 1,1"
echo "$(date "+%F %T") native rerun start; load $(cat /proc/loadavg)" >> $O/progress.log
for core in 0 15; do
  taskset -c $core $NAT benchmarks/dedi_gpu_bench.py --device cpu --ext ~/eqdisc_dedi_native --index 43 --pairs $PAIRS --out $O/solver.jsonl --tag native_c$core >> $O/solver_native.log 2>&1 &
done
( for i in 0 2 4 6 8; do
  taskset -c 16 $NAT benchmarks/dedi_gpu_bench.py --device cpu --ext ~/eqdisc_dedi_native --data data/timing_scaling.npz --index $i --pairs 1e-3,1 --out $O/scaling_dedi.jsonl --tag native_c16 >> $O/scaling_native.log 2>&1
done ) &
wait
echo "$(date "+%F %T") native rerun DONE" >> $O/progress.log
