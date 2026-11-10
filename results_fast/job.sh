#!/bin/bash
# Fast-baseline pass on the 80 head-to-head runs. DGX Spark, cores 0-4, one process per core.
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 PYTHONWARNINGS=ignore
export OCTAVE_HOME=$HOME/octave_env/oct
PY=.venv_base/bin/python; R=$HOME/fastbl/renv/bin/Rscript; O=results_fast; L=$O/progress.log
$PY benchmarks/fast_baselines.py argos-export --outdir $O/argos_csv --nworkers 5 --subset 10
worker() {
  k=$1
  echo "$(date +%T) w$k python start" >> $L
  taskset -c $k $PY benchmarks/fast_baselines.py run --method all --worker $k --nworkers 5 > $O/w$k.log 2>&1
  echo "$(date +%T) w$k argos B=200 start" >> $L
  taskset -c $k $R benchmarks/argos_run.R $O/argos_csv/manifest_$k.tsv $O/argos_b200 lasso 200 >> $O/w$k.log 2>&1
  echo "$(date +%T) w$k argos B=2000 subset start" >> $L
  taskset -c $k $R benchmarks/argos_run.R $O/argos_csv/manifest_subset.tsv $O/argos_b2000 lasso 2000 >> $O/w$k.log 2>&1
  if [ $k = 0 ]; then
    taskset -c 0 $PY benchmarks/wsindy_octave_check.py --wsindy $HOME/fastbl/WSINDy_ODE --octave $OCTAVE_HOME/bin/octave-cli \
      --positions 0-79 --out $O/wsindy_octave_check.json > $O/octcheck80.log 2>&1
  fi
  echo "$(date +%T) w$k done" >> $L
}
for k in 0 1 2 3 4; do worker $k & done
wait
$PY benchmarks/fast_baselines.py collect >> $L 2>&1
echo "$(date +%T) ALL DONE" >> $L
