#!/bin/bash
# Resume after the Spark crash (~21:02): same pre-registered order minus the 18 finished runs, same settings
# (--grid small). Hard cap 14 cores via taskset; 11 h wall deadline. Results saved after every finished run.
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
I=results_hard120
echo "$(date "+%F %T") start dedi resume" >> $I/dedi_progress.log
sha256sum data/hard120.npz >> $I/dedi_progress.log
timeout -k 120 39600 taskset -c 3-9,13-19 ~/eqdisc_dedi/.venv/bin/python benchmarks/dedi_bench.py --data data/hard120.npz \
  --ext ~/eqdisc_dedi --index-file $I/dedi_order_resume.json --grid small --workers 14 --output $I/dedi_hard120_part2.json > $I/dedi_hard120_part2.log 2>&1
echo "$(date "+%F %T") end dedi resume rc=$?" >> $I/dedi_progress.log
echo "$(date "+%F %T") DEDI_DONE" >> $I/dedi_progress.log
kill -TERM -- -$$
