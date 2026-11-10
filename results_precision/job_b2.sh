#!/bin/bash
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
P=.venv/bin/python
L=results_precision/b2.log
while pgrep -f '[j]obA2.sh' > /dev/null; do sleep 30; done
md5sum eqdisc/*.py benchmarks/precision_*.py > $L
$P benchmarks/precision_b_dev.py runs --arms v11 qr --data data/dysts_precdev.npz --workers 18 --output results_precision/b_dev_qr.json >> $L 2>&1
echo EXIT $? >> $L
