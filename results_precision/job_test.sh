#!/bin/bash
# v1.2 pre-registered test pass (results_precision/PREREGISTRATION_v12.md). Spark, 18 workers.
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
P=.venv/bin/python
O=results_precision
L=$O/test_progress.log
W=18
st() { echo "$(date '+%F %T') $*" >> $L; }
while pgrep -f "[j]obB2.sh" > /dev/null; do sleep 30; done
st "start"; md5sum eqdisc/*.py benchmarks/final_bench.py benchmarks/odebench_bench.py benchmarks/gen_variants.py >> $L
sha256sum data/sealed_clean.npz >> $L
grep -q "$(sha256sum data/sealed_clean.npz | cut -c1-64)" data/MANIFEST.sha256 || { st "ARCHIVE SHA256 NOT IN MANIFEST; abort"; st "TEST_DONE"; exit 1; }
fb() { name=$1; shift; st "run $name"; $P benchmarks/final_bench.py "$@" --workers $W --output $O/$name.json > $O/$name.log 2>&1; st "end $name rc=$?"; }
# 1. primary: fresh noise-free archive
fb test_clean --kind poly3 --data data/sealed_clean.npz --arms final_v11 final_v12
# 2. secondary (non-independent): ODEBench sigma = 0, fit IC 0
for v in 1.1 1.2; do
  st "run odebench v$v"
  $P benchmarks/odebench_bench.py run --sigmas 0 --fit-ic 0 --version $v --workers $W --output $O/odebench_s0_v$v.json > $O/odebench_s0_v$v.log 2>&1
  st "end odebench v$v rc=$?"
done
# 3. guards (sealed test archives)
fb guard_white   --kind poly3 --data data/joint_sealed.npz --starts results_squeeze/starts_joint_sealed.pkl --arms final_v11 final_v12
fb guard_colored --kind poly3 --data data/sealed_colored2.npz --arms final_v11 final_v12
fb guard_student --kind poly3 --data data/sealed_student.npz --arms final_v11 final_v12
fb guard_nd      --kind nd    --data data/nd_test.npz --arms final_v11 final_v12
# 4. old versions unchanged: frozen-v1.0 equivalence test
st "run pytest"
EQDISC_FROZEN_SRC=$HOME/eqdisc_v3/frozen_ac7ab20 EQDISC_EQUIV_WORKERS=6 $P -m pytest tests/test_equivalence.py -s > $O/pytest.log 2>&1
st "end pytest rc=$?"
st "TEST_DONE"
