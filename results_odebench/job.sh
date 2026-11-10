#!/bin/bash
# Queue: wait for Exp 1 primary (HUBER_DONE), run ODEBench (spare cores 0-2,10,11), then Exp 1 extensions.
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 TF_CPP_MIN_LOG_LEVEL=3
O=results_odebench
P=$O/progress.log
echo "$(date '+%F %T') queued; waiting for HUBER_DONE" >> $P
while ! grep -q HUBER_DONE results_huber_dev/progress.log 2>/dev/null; do
  if ! pgrep -f '[r]esults_huber_dev/job.sh' > /dev/null; then echo "$(date '+%F %T') huber job gone without DONE; continuing" >> $P; break; fi
  sleep 60
done
echo "$(date '+%F %T') start odebench" >> $P
md5sum benchmarks/odebench_bench.py eqdisc/*.py data/odebench/strogatz_extended.json $O/classes.json $O/library_exponents.json >> $P
for ic in 0 1; do
  out=$O/fit_ic$ic.json
  if [ -f $out ]; then echo "$(date '+%F %T') skip fit_ic$ic" >> $P; continue; fi
  res=""; [ -f $O/fit_ic$ic.partial.json ] && res="--resume"
  echo "$(date '+%F %T') stage fit_ic$ic $res" >> $P
  taskset -c 0-2,10,11 .venv/bin/python benchmarks/odebench_bench.py run --fit-ic $ic --reps 3 --workers 5 \
     --output $out $res >> $O/fit_ic$ic.log 2>&1
  echo "$(date '+%F %T') end fit_ic$ic rc=$?" >> $P
done
echo "$(date '+%F %T') ODEBENCH_DONE" >> $P
setsid nohup bash results_huber_dev/job_ext.sh > /dev/null 2>&1 < /dev/null &
echo "$(date '+%F %T') launched huber extensions" >> $P
