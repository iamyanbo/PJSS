#!/bin/bash
# Rerun of task nd 165 (Lorenz-96, D = 10): the first attempt hit DEDi's x_10 parsing bug (SyntaxError), and the job
# log said "exit 0" because $(date) reset $? before it was printed. cost_anatomy.py now swaps in a fixed
# prepare_vectorized_lambda for D >= 10 only (results_anatomy/check_patch.py: identical for D = 6, exact for D = 10).
cd ~/eqdisc_v3
O=results_anatomy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "$(date "+%F %T") nd 165 failed on DEDi's x_10 parsing bug; rerun on core 11 with the D >= 10 patch" >> $O/progress.log
timeout -k 60 21600 taskset -c 11 ~/eqdisc_dedi_gpu/.venv/bin/python benchmarks/cost_anatomy.py dedi --ext ~/eqdisc_dedi \
  --device gpu --data data/nd_test.npz --index 165 --lib nd --pairs row --out $O/dedi_nd.jsonl >> $O/dedi_rerun165.log 2>&1
rc=$?
echo "$(date "+%F %T") rerun nd 165 exit code $rc" >> $O/progress.log
