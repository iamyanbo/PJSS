#!/bin/bash
# dev-only hyperparameter sweep (data/instances.npz every 5th, 160 runs); 6 workers
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=.venv_base/bin/python; O=results_baselines/dev; mkdir -p $O
for w in 11 21 41; do for m in fd_stlsq fd_sr3; do $P benchmarks/baselines_bench.py run --method $m --set dev --window $w --workers 6 --output $O/${m}_w$w.json > $O/${m}_w$w.log 2>&1; done; done
for h in 10 20 40; do for m in weak_stlsq weak_sr3; do $P benchmarks/baselines_bench.py run --method $m --set dev --h $h --workers 6 --output $O/${m}_h$h.json > $O/${m}_h$h.log 2>&1; done; done
echo TUNE_DONE > $O/done
