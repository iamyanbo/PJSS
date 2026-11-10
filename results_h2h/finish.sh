#!/bin/bash
# Finisher: after job.sh's workers have all exited, clear stale claims (left by killed workers) and run every
# missing pair with fresh GPU workers (a run already bound to GPU stays on GPU; CPU-bound runs get CPU workers),
# repeating until every pair of every run in order.json has a result or the deadline passes. Then writes COMPLETE.
# Added 2026-10-28 after GPU workers were stopped/killed for memory (GPU worker RSS grows to ~6.5 GB).
cd ~/eqdisc_v3
OUT=results_h2h
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 XLA_PYTHON_CLIENT_PREALLOCATE=false
END=$(( $(date -d "2026-10-27 22:32:46" +%s) + 144000 ))   # job.sh's 40 h deadline
alive() { ps -eo args | grep -c "^[^ ]*python benchmarks/dedi_h2h.py"; }
until grep -q DONE $OUT/progress.log && [ "$(alive)" = 0 ]; do sleep 120; done
for pass in 1 2 3; do
  for c in $OUT/claims/*; do [ -e "$OUT/pairs/$(basename $c).json" ] || rm -f "$c"; done
  missing=$(( $(python3 -c "import json;print(len(json.load(open('$OUT/order.json')))*35)") - $(ls $OUT/pairs | grep -c json) ))
  echo "$(date "+%F %T") finish pass $pass: $missing pairs missing" >> $OUT/progress.log
  [ $missing = 0 ] && break
  left=$(( END - $(date +%s) )); [ $left -le 0 ] && break
  k=20; for core in 5 6 7 8 9 15 16 17 18 19; do
    timeout -k 60 $left taskset -c $core ~/eqdisc_dedi_gpu/.venv/bin/python benchmarks/dedi_h2h.py --device gpu \
      --ext ~/eqdisc_dedi --order $OUT/order.json --outdir $OUT --worker $k > $OUT/gpu$k.err 2>&1 & k=$((k+1))
  done
  k=20; for core in 0 1 2 3 4 10 11 12; do
    timeout -k 60 $left taskset -c $core ~/eqdisc_dedi/.venv/bin/python benchmarks/dedi_h2h.py --device cpu \
      --ext ~/eqdisc_dedi --order $OUT/order.json --outdir $OUT --worker $k > $OUT/cpu$k.err 2>&1 & k=$((k+1))
  done
  wait
done
echo "$(date "+%F %T") COMPLETE $(ls $OUT/pairs | grep -c json) pairs" >> $OUT/progress.log
touch $OUT/COMPLETE
