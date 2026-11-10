#!/bin/bash
# Timing/scaling study, run on an otherwise idle Spark AFTER the head-to-head (waits for results_h2h/COMPLETE, written by results_h2h/finish.sh).
#  1. DEDi solver: run 43 pairs with our exact scipy stand-in vs native CHOLMOD (scikit-sparse 0.5 behind an
#     adapter, ~/eqdisc_dedi/shim_native), on a slow core (0) and a fast core (15).
#  2. Scaling on core 15: DEDi native, pair (1e-3, 1), N = 500..8000; ours v1.2 at every N; poly4/poly5 at N = 2000.
#  3. Ours on hard-120 runs 43 and 109, cores 0 and 15, 3 repeats.
cd ~/eqdisc_v3
O=results_timing
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
until [ -e results_h2h/COMPLETE ]; do sleep 600; done
echo "$(date "+%F %T") start; load $(cat /proc/loadavg)" >> $O/progress.log
SHIM=~/eqdisc_dedi/.venv/bin/python; NAT=~/mm/cholmod/bin/python; OURS=~/eqdisc_v2/.venv/bin/python
PAIRS="1e-3,1 1,1e-2 0.1,1e-2 1,1"
for core in 0 15; do
  taskset -c $core $SHIM benchmarks/dedi_gpu_bench.py --device cpu --ext ~/eqdisc_dedi --index 43 --pairs $PAIRS --out $O/solver.jsonl --tag shim_c$core >> $O/solver.log 2>&1
  taskset -c $core $NAT benchmarks/dedi_gpu_bench.py --device cpu --ext ~/eqdisc_dedi_native --index 43 --pairs $PAIRS --out $O/solver.jsonl --tag native_c$core >> $O/solver.log 2>&1
done
echo "$(date "+%F %T") solver done" >> $O/progress.log
for i in 0 2 4 6 8; do
  taskset -c 15 $NAT benchmarks/dedi_gpu_bench.py --device cpu --ext ~/eqdisc_dedi_native --data data/timing_scaling.npz --index $i --pairs 1e-3,1 --out $O/scaling_dedi.jsonl --tag native_c15 >> $O/scaling.log 2>&1
done
taskset -c 15 $OURS benchmarks/final_bench.py --kind poly3 --data data/timing_scaling.npz --arms final_v12 --index-file $O/idx_all.json --workers 1 --output $O/scaling_ours_poly3.json >> $O/scaling.log 2>&1
for k in poly4 poly5; do
  EQ_DEG=${k#poly} taskset -c 15 $OURS benchmarks/final_bench.py --kind $k --data data/timing_scaling.npz --arms final_v12 --index-file $O/idx_n2000.json --workers 1 --output $O/scaling_ours_$k.json >> $O/scaling.log 2>&1
done
echo "$(date "+%F %T") scaling done" >> $O/progress.log
for core in 0 15; do for rep in 1 2 3; do
  taskset -c $core $OURS benchmarks/final_bench.py --kind poly3 --data data/hard120.npz --arms final_v12 --index-file $O/idx_hard.json --workers 1 --output $O/ours_hard_c${core}_r$rep.json >> $O/ours.log 2>&1
done; done
echo "$(date "+%F %T") DONE" >> $O/progress.log
