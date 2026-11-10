#!/bin/bash
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
P=.venv/bin/python
O=results_precision
L=$O/a2.log
while pgrep -f '[j]obB1.sh' > /dev/null; do sleep 30; done
date > $L; md5sum eqdisc/*.py benchmarks/precision_*.py benchmarks/final_bench.py >> $L
$P benchmarks/precision_a_dev.py --arms v11 v12 --workers 18 --output $O/a2_dev.json >> $L 2>&1; echo "a2 rc=$?" >> $L
for a in instances instances_colored instances_student; do
  $P benchmarks/final_bench.py --kind poly3 --data data/$a.npz --arms final_v11 final_v12 --workers 18 --output $O/guard_dev_$a.json > $O/guard_dev_$a.log 2>&1; echo "$a rc=$?" >> $L
done
$P benchmarks/final_bench.py --kind nd --data data/nd_dev.npz --arms final_v11 final_v12 --workers 18 --output $O/guard_dev_nd.json > $O/guard_dev_nd.log 2>&1; echo "nd rc=$?" >> $L
echo A2_DONE >> $L
