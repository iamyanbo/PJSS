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
# primary (pre-registered, see PLAN.md)
stage dev_colored_o0 dev_colored 3 0 full gaussian ar1 ar1_gauss
stage dev_student_o0 dev_student 3 0 full gaussian
stage dev_white_o0   dev_white   3 0 full gaussian
echo "$(date '+%F %T') HUBER_DONE" >> $P
