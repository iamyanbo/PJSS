#!/bin/bash
# queued after results_v101/job.sh (pytest was missing from the Spark venv at first launch; installed 11:44)
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
O=results_v101; L=$O/progress.log
while ! grep -q V101_DONE $L; do
  if ! pgrep -f '[r]esults_v101/job.sh' > /dev/null; then echo "$(date '+%F %T') job.sh gone without DONE" >> $L; break; fi
  sleep 60
done
echo "$(date '+%F %T') equivalence test (retry)" >> $L
EQDISC_FROZEN_SRC=$HOME/eqdisc_v3/frozen_ac7ab20 EQDISC_EQUIV_WORKERS=18 .venv/bin/python -m pytest tests -s -p no:cacheprovider > $O/pytest.log 2>&1
echo "$(date '+%F %T') end pytest rc=$?" >> $L
echo "$(date '+%F %T') PYTEST_DONE" >> $L
