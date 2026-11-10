#!/bin/bash
# DEDi on hard120 in the pre-registered order (results_hard120/dedi_order.json), same settings as the hard-40 run
# (--grid small). Hard cap: 12 cores via taskset; 14 h wall deadline. Results saved after every finished run.
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
I=results_hard120
echo "$(date "+%F %T") start dedi" > $I/dedi_progress.log
sha256sum data/hard120.npz >> $I/dedi_progress.log
timeout -k 120 50400 taskset -c 3-9,15-19 ~/eqdisc_dedi/.venv/bin/python benchmarks/dedi_bench.py --data data/hard120.npz \
  --ext ~/eqdisc_dedi --index-file $I/dedi_order.json --grid small --workers 12 --output $I/dedi_hard120.json > $I/dedi_hard120.log 2>&1
echo "$(date "+%F %T") end dedi rc=$?" >> $I/dedi_progress.log
echo "$(date "+%F %T") DEDI_DONE" >> $I/dedi_progress.log
kill -TERM -- -$$
