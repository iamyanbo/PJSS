#!/bin/bash
# continuation after the Spark rebooted at ~21:03 during dysts21_odeformer (250 records kept via --resume)
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
B=.venv_base/bin/python; S=benchmarks/baselines_bench.py; O=results_baselines; L=$O/progress.log
run() { name=$1; py=$2; shift 2; echo "$(date +%T) start $name" >> $L
        $py $S run "$@" --resume --workers 6 --output $O/$name.json > $O/$name.log 2>&1; echo "$(date +%T) end $name rc=$?" >> $L; }
echo "$(date +%T) Spark rebooted (~21:03); job3.sh resumes" >> $L
run dysts21_odeformer  .venv_odeformer/bin/python --method odeformer --set dysts7 --every 3
run dysts49_pysr       .venv_pysr/bin/python --method pysr --set dysts7 --every 7 --window 11 --budget 60
run dysts21_fd_esindy_raw_w11   $B --method fd_esindy   --set dysts7 --every 3 --scale raw --window 11
run dysts21_weak_esindy_h10     $B --method weak_esindy --set dysts7 --every 3 --h 10
run dysts21_weak_esindy_raw_h20 $B --method weak_esindy --set dysts7 --every 3 --scale raw --h 20
echo "$(date +%T) ALL_DONE" >> $L
