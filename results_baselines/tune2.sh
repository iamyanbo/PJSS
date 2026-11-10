#!/bin/bash
# dev sweep, part 2: raw-unit preprocessing (plain pysindy usage) + SFD window 5
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=.venv_base/bin/python; O=results_baselines/dev
for m in fd_stlsq fd_sr3; do
  $P benchmarks/baselines_bench.py run --method $m --set dev --window 5 --workers 6 --output $O/${m}_w5.json > $O/${m}_w5.log 2>&1
  for w in 5 11 21; do $P benchmarks/baselines_bench.py run --method $m --set dev --scale raw --window $w --workers 6 --output $O/${m}_raw_w$w.json > $O/${m}_raw_w$w.log 2>&1; done; done
for m in weak_stlsq weak_sr3; do for h in 10 20 40; do $P benchmarks/baselines_bench.py run --method $m --set dev --scale raw --h $h --workers 6 --output $O/${m}_raw_h$h.json > $O/${m}_raw_h$h.log 2>&1; done; done
echo TUNE2_DONE > $O/done2
