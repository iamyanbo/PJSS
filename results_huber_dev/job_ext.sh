#!/bin/bash
# Exp 1: Huber vs Gaussian on dev archives. Spare Spark cores only (DEDi holds 3-9,13-19).
# Idempotent: finished stages are skipped, unfinished ones resume from <out>.partial.json.
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
O=results_huber_dev
P=$O/progress.log
echo "$(date '+%F %T') start job" >> $P
md5sum benchmarks/huber_dev.py benchmarks/ablation.py eqdisc/*.py >> $P
stage () {  # name set every offset variants...
  local name=$1 set=$2 every=$3 off=$4; shift 4
  local out=$O/$name.json
  if [ -f $out ]; then echo "$(date '+%F %T') skip $name (done)" >> $P; return; fi
  local res=""; [ -f $O/$name.partial.json ] && res="--resume"
  echo "$(date '+%F %T') stage $name $res" >> $P
  taskset -c 0-2,10,11 .venv/bin/python benchmarks/huber_dev.py --set $set --every $every --offset $off \
     --workers 5 --variants "$@" --output $out $res >> $O/$name.log 2>&1
  echo "$(date '+%F %T') end $name rc=$?" >> $P
}
# 0) ODEBench diagnostic arm (not frozen; see results_odebench/PREREGISTRATION.md amendment)
if [ ! -f results_odebench/fit_ic0_emptyfix.json ]; then
  r=""; [ -f results_odebench/fit_ic0_emptyfix.partial.json ] && r="--resume"
  echo "$(date '+%F %T') stage odebench_emptyfix $r" >> $P
  taskset -c 0-2,10,11 .venv/bin/python benchmarks/odebench_bench.py run --fit-ic 0 --reps 3 --dims 1 2      --patch-empty-support --workers 5 --output results_odebench/fit_ic0_emptyfix.json $r >> results_odebench/fit_ic0_emptyfix.log 2>&1
  echo "$(date '+%F %T') end odebench_emptyfix rc=$?" >> $P
fi
# extensions
stage dev_colored_o1 dev_colored 3 1 full gaussian ar1 ar1_gauss
stage dev_colored_o2 dev_colored 3 2 full gaussian ar1 ar1_gauss
stage dev_cmid       dev_cmid    1 0 full gaussian
stage dev_student_o1 dev_student 3 1 full gaussian
echo "$(date '+%F %T') HUBER_EXT_DONE" >> $P
