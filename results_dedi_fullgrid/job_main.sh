#!/bin/bash
# Main full-grid DEDi job (pre-registered: results_dedi_fullgrid/PREREGISTRATION.json, order.json).
# 10 single-core workers on cores 10-19; each starts once the pilot worker on its core has exited.
# Hard deadline 2026-10-27 13:00 local. Resumable: rerun this script (stale claims removed, finished pairs skipped).
cd ~/eqdisc_v3
OUT=results_dedi_fullgrid/main; END=$(date -d "2026-10-27 13:00" +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
mkdir -p $OUT/claims $OUT/pairs
for c in $OUT/claims/*; do [ -e "$c" ] && [ ! -e "$OUT/pairs/$(basename $c).json" ] && rm -f "$c"; done
rm -f $OUT/pairs/*.tmp
echo "$(date "+%F %T") start main" >> $OUT/progress.log
sha256sum data/hard120.npz benchmarks/dedi_fullgrid.py results_dedi_fullgrid/order.json >> $OUT/progress.log
for k in $(seq 0 9); do
  (
    while pgrep -f "[o]utdir results_dedi_fullgrid/pilot --worker $k\$" > /dev/null; do sleep 30; done
    DL=$(( END - $(date +%s) )); [ $DL -le 0 ] && exit 0
    echo "$(date "+%F %T") worker $k start (${DL}s left)" >> $OUT/progress.log
    timeout -k 60 $DL taskset -c $((10 + k)) ~/eqdisc_dedi/.venv/bin/python benchmarks/dedi_fullgrid.py \
      --data data/hard120.npz --ext ~/eqdisc_dedi --order results_dedi_fullgrid/order.json --outdir $OUT --worker $k \
      >> $OUT/worker$k.err 2>&1
    echo "$(date "+%F %T") worker $k exit rc=$?" >> $OUT/progress.log
  ) &
done
wait
echo "$(date "+%F %T") $(ls $OUT/pairs | wc -l) pairs saved" >> $OUT/progress.log
echo "$(date "+%F %T") DONE" >> $OUT/progress.log
