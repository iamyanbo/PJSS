#!/bin/bash
# Wrapper (not part of the pre-registered job): wait until the v1.0.1 verification jobs are finished (they import
# the v1.0.1 package from this tree), then install the v1.1 code from /tmp/v11code.tar, check the pre-registered
# md5s and run results_final_v11/job.sh.
cd ~/eqdisc_v3
L=results_final_v11/progress.log
while ! grep -q PYTEST_DONE results_v101/progress.log 2>/dev/null; do
  if ! pgrep -f '[r]esults_v101/job' > /dev/null; then echo "$(date '+%F %T') v101 jobs gone without PYTEST_DONE" >> $L; break; fi
  sleep 60
done
tar -xf /tmp/v11code.tar
md5sum -c results_final_v11/code.md5 >> $L 2>&1 || { echo "$(date '+%F %T') MD5 MISMATCH; abort" >> $L; echo "$(date '+%F %T') V11_DONE" >> $L; exit 1; }
exec bash results_final_v11/job.sh
