#!/bin/bash
# Full-grid DEDi: 10 single-core workers pinned to cores 10-19, shared task list, per-pair results.
# usage: bash job.sh ORDER_JSON OUTDIR DEADLINE_SECONDS   (resumable: stale claims are removed first)
cd ~/eqdisc_v3
ORDER=$1; OUT=$2; DL=$3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
mkdir -p $OUT/claims $OUT/pairs
for c in $OUT/claims/*; do [ -e "$c" ] && [ ! -e "$OUT/pairs/$(basename $c).json" ] && rm -f "$c"; done
rm -f $OUT/pairs/*.tmp
echo "$(date "+%F %T") start order=$ORDER deadline=${DL}s" >> $OUT/progress.log
sha256sum data/hard120.npz benchmarks/dedi_fullgrid.py >> $OUT/progress.log
for k in $(seq 0 9); do
  timeout -k 60 $DL taskset -c $((10 + k)) ~/eqdisc_dedi/.venv/bin/python benchmarks/dedi_fullgrid.py \
    --data data/hard120.npz --ext ~/eqdisc_dedi --order $ORDER --outdir $OUT --worker $k > $OUT/worker$k.err 2>&1 &
done
wait
echo "$(date "+%F %T") $(ls $OUT/pairs | wc -l) pairs saved" >> $OUT/progress.log
echo "$(date "+%F %T") DONE" >> $OUT/progress.log
