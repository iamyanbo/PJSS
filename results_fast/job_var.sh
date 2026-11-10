#!/bin/bash
# WSINDy sensitivity variants (K = all centres; no pre-smoothing), cores 0-4, after job_ours.sh.
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 PYTHONWARNINGS=ignore
PY=.venv_base/bin/python; O=results_fast; L=$O/progress.log
until grep -q "ours retime done" $L; do sleep 30; done
echo "$(date +%T) wsindy variants start" >> $L
for k in 0 1 2 3 4; do
  taskset -c $k $PY benchmarks/fast_baselines.py run --method wsindy_K1,wsindy_nosmooth --worker $k --nworkers 5 > $O/var$k.log 2>&1 &
done
wait
$PY benchmarks/fast_baselines.py collect >> $L 2>&1
echo "$(date +%T) VARIANTS DONE" >> $L
