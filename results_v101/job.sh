#!/bin/bash
# v1.0.1 verification on the Spark (19 workers): equivalence test, ODEBench + lynx-hare rerun, then re-runs of the
# stored frozen-v1.0 test archives with the v1.0.1 package (compared by benchmarks/v101_compare.py).
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 TF_CPP_MIN_LOG_LEVEL=3
P=.venv/bin/python
O=results_v101
L=$O/progress.log
W=19
st() { echo "$(date '+%F %T') $*" >> $L; }
st "start"; md5sum eqdisc/*.py benchmarks/final_bench.py benchmarks/odebench_bench.py benchmarks/dysts_bench.py benchmarks/real_lynx.py tests/*.py >> $L
st "equivalence test"
EQDISC_FROZEN_SRC=$HOME/eqdisc_v3/frozen_ac7ab20 EQDISC_EQUIV_WORKERS=18 $P -m pytest tests -s -p no:cacheprovider > $O/pytest.log 2>&1
st "end pytest rc=$?"
mkdir -p results_odebench_v101
cp results_odebench/classes.json results_odebench/library_exponents.json results_odebench_v101/
for ic in 0 1; do
  st "odebench fit_ic$ic"
  $P benchmarks/odebench_bench.py run --fit-ic $ic --reps 3 --workers $W --output results_odebench_v101/fit_ic$ic.json \
     > results_odebench_v101/fit_ic$ic.log 2>&1
  st "end odebench fit_ic$ic rc=$?"
done
$P benchmarks/real_lynx.py --output results_real/lynx_hare_v101.json > results_real/lynx_hare_v101.log 2>&1
st "end lynx rc=$?"
fb() { name=$1; shift; st "rerun $name"; $P benchmarks/final_bench.py "$@" --arms final --workers $W --output $O/$name.json > $O/$name.log 2>&1; st "end $name rc=$?"; }
fb test_sealed    --kind poly3 --data data/joint_sealed.npz   --starts results_squeeze/starts_joint_sealed.pkl
fb test_student   --kind poly3 --data data/sealed_student.npz
fb test_colored   --kind poly3 --data data/sealed_colored.npz
fb test_confirm   --kind poly3 --data data/hybrid_confirm.npz --starts results_squeeze/starts_hybrid_confirm.pkl
fb test_heldout2  --kind poly3 --data data/heldout2.npz       --starts results_squeeze/starts_heldout2.pkl
fb test_coarse    --kind poly3 --data data/sealed_coarse.npz
fb test_trig      --kind trig  --data data/trig_test.npz
fb test_mixedpoly --kind trig  --data data/joint_sealed.npz --every 6
fb test_nd        --kind nd    --data data/nd_test.npz
fb hard120_final  --kind poly3 --data data/hard120.npz
EQ_DEG=4 fb test_deg4 --kind poly4 --data data/joint_sealed.npz --every 3
EQ_DEG=5 fb test_deg5 --kind poly5 --data data/joint_sealed.npz --every 3
st "rerun dysts"
$P benchmarks/dysts_bench.py --method final --data data/dysts_test.npz --workers $W --output $O/dysts_test_final.json > $O/dysts_test_final.log 2>&1
st "end dysts rc=$?"
$P benchmarks/v101_compare.py all > $O/compare.md 2>&1
st "V101_DONE"
