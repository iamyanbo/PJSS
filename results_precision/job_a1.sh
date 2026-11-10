#!/bin/bash
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
P=.venv/bin/python
mkdir -p results_precision
L=results_precision/a1.log
md5sum eqdisc/*.py > $L
$P benchmarks/gen_variants.py --kind white --etas 0 --seed 5150001 --seeds 4 --output data/dev_clean.npz >> $L 2>&1
$P benchmarks/precision_a_dev.py --workers 18 --output results_precision/a1_dev.json >> $L 2>&1
echo EXIT $? >> $L
