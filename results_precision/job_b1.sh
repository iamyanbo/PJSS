#!/bin/bash
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
P=.venv/bin/python
L=results_precision/b1.log
while pgrep -f '[p]recision_a_dev' > /dev/null; do sleep 30; done
md5sum eqdisc/*.py benchmarks/precision_*.py > $L
$P benchmarks/precision_b_dev.py cond --workers 18 --output results_precision/b_cond.json >> $L 2>&1
echo "cond rc=$?" >> $L
SYS="AtmosphericRegime HyperYan IsothermalChemical GuckenheimerHolmes QiChen ChenLee ZhouChen HyperQi Bouali HyperBao Arneodo Lorenz SprottA Halvorsen HyperLu LorenzStenflo"
[ -f data/dysts_precdev.npz ] || $P benchmarks/dysts_generate.py --classes data/dysts_classes.json --output data/dysts_precdev.npz --seed 20261777 --seeds 3 --systems $SYS >> $L 2>&1
$P benchmarks/precision_b_dev.py runs --data data/dysts_precdev.npz --workers 18 --output results_precision/b_dev.json >> $L 2>&1
echo EXIT $? >> $L
