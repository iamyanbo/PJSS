#!/bin/bash
# Head-to-head: full-grid DEDi (10 GPU workers on fast cores, 8 CPU workers on slow cores) + ours v1.2 on core 13.
# usage: bash results_h2h/job.sh DEADLINE_SECONDS   (resumable: stale claims are removed first)
cd ~/eqdisc_v3
DL=$1; OUT=results_h2h
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p $OUT/claims $OUT/pairs $OUT/runs
for c in $OUT/claims/*; do [ -e "$c" ] && [ ! -e "$OUT/pairs/$(basename $c).json" ] && rm -f "$c"; done
rm -f $OUT/pairs/*.tmp
echo "$(date "+%F %T") start deadline=${DL}s" >> $OUT/progress.log
sha256sum data/hard120.npz data/joint_sealed.npz data/sealed_colored2.npz data/sealed_student.npz \
  benchmarks/dedi_h2h.py benchmarks/final_bench.py $OUT/order.json >> $OUT/progress.log
for c in hard white coloured t3 short; do
  case $c in hard) D=data/hard120.npz;; white|short) D=data/joint_sealed.npz;; coloured) D=data/sealed_colored2.npz;; t3) D=data/sealed_student.npz;; esac
  [ -e $OUT/ours_$c.json ] || taskset -c 13 ~/eqdisc_v2/.venv/bin/python benchmarks/final_bench.py --kind poly3 --data $D \
    --arms final_v12 --index-file $OUT/idx_$c.json --workers 1 --output $OUT/ours_$c.json >> $OUT/ours.log 2>&1
done &
k=0; for core in 5 6 7 8 9 15 16 17 18 19; do
  timeout -k 60 $DL taskset -c $core ~/eqdisc_dedi_gpu/.venv/bin/python benchmarks/dedi_h2h.py --device gpu \
    --ext ~/eqdisc_dedi --order $OUT/order.json --outdir $OUT --worker $k > $OUT/gpu$k.err 2>&1 & k=$((k+1))
done
k=0; for core in 0 1 2 3 4 10 11 12; do
  timeout -k 60 $DL taskset -c $core ~/eqdisc_dedi/.venv/bin/python benchmarks/dedi_h2h.py --device cpu \
    --ext ~/eqdisc_dedi --order $OUT/order.json --outdir $OUT --worker $k > $OUT/cpu$k.err 2>&1 & k=$((k+1))
done
wait
echo "$(date "+%F %T") $(ls $OUT/pairs | wc -l) pairs saved" >> $OUT/progress.log
echo "$(date "+%F %T") DONE" >> $OUT/progress.log
