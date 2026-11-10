#!/bin/bash
# Re-time ours v1.2 on the same cores/conditions as the fast baselines (Cortex-A725 cores 0-4, one process per
# core, threads = 1), after results_fast/job.sh has finished. Same command as results_h2h/job.sh (which ran on
# core 13, also an A725, while 18 DEDi workers loaded the machine).
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
O=results_fast; L=$O/progress.log
until grep -q "ALL DONE" $L; do sleep 60; done
echo "$(date +%T) ours retime start" >> $L
k=0
for c in hard white coloured t3 short; do
  case $c in hard) D=data/hard120.npz;; white|short) D=data/joint_sealed.npz;; coloured) D=data/sealed_colored2.npz;; t3) D=data/sealed_student.npz;; esac
  taskset -c $k ~/eqdisc_v2/.venv/bin/python benchmarks/final_bench.py --kind poly3 --data $D --arms final_v12 \
    --index-file results_h2h/idx_$c.json --workers 1 --output $O/ours_retime_$c.json > $O/ours_retime_$c.log 2>&1 &
  k=$((k+1))
done
wait
echo "$(date +%T) ours retime done" >> $L
