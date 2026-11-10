#!/bin/bash
# Sensitivity sweep (results_sensitivity/PREREGISTRATION.md) on the Spark, ONE worker on core 13 only.
# Waits until results_h2h/ours_short.json exists (the head-to-head job uses core 13 until then).
# The changed eqdisc package is a full copy in results_sensitivity/pkg/eqdisc (PYTHONPATH first), so the code that
# the running h2h / refresh jobs import from ~/eqdisc_v3/eqdisc is not touched. Resumable: rerun this script.
#   launch: cd ~/eqdisc_v3 && setsid nohup bash results_sensitivity/job.sh > results_sensitivity/job.out 2>&1 < /dev/null &
cd ~/eqdisc_v3
O=results_sensitivity; P=~/eqdisc_v2/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$HOME/eqdisc_v3/$O/pkg:$HOME/eqdisc_v3
log() { echo "$(date "+%F %T") $*" >> $O/progress.log; }
log "waiting for results_h2h/ours_short.json"
while [ ! -e results_h2h/ours_short.json ]; do sleep 60; done
sleep 30
log "core 13 free; eqdisc = $(taskset -c 13 $P -c 'import eqdisc; print(eqdisc.__file__, eqdisc.__version__)')"
md5sum $O/pkg/eqdisc/*.py benchmarks/sensitivity_*.py $O/sample.json data/dysts_test.npz >> $O/progress.log
if [ ! -e $O/pytest.log ]; then
  log "pytest (full suite, equivalence with frozen ac7ab20, 1 worker)"
  { taskset -c 13 $P -c 'import eqdisc; print("eqdisc from", eqdisc.__file__)'
    EQDISC_FROZEN_SRC=$HOME/eqdisc_v3/frozen_ac7ab20 EQDISC_EQUIV_WORKERS=1 timeout 5400 taskset -c 13 \
      $P -m pytest -q -s -o pythonpath=$O/pkg tests; echo "pytest exit $?"; } > $O/pytest.tmp 2>&1
  mv $O/pytest.tmp $O/pytest.log
  log "$(tail -1 $O/pytest.log)"
fi
log "sweep start"
taskset -c 13 $P benchmarks/sensitivity_bench.py run --sample $O/sample.json --outdir $O >> $O/sweep.log 2>&1
log "sweep exit $?"
taskset -c 13 $P benchmarks/sensitivity_report.py --outdir $O --output $O/sensitivity_report_spark.md > /dev/null 2>&1
log "DONE"
