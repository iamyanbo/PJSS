#!/bin/bash
# real data 2 (pre-registered in results_real2/PREREGISTRATION.md): 3 single-thread queues
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
E=~/eqdisc_v2/.venv/bin/python; B=.venv_base/bin/python; R=benchmarks/real2_pendulum.py; O=results_real2; L=$O/progress.log
one() { py=$1; part=$2; sel=$3; m=$4; name=$5
  [ -f $O/$name.json ] && return
  echo "$(date +%T) start $name" >> $L
  $py -I $R $part $sel --method $m --output $O/$name.json > $O/$name.log 2>&1
  echo "$(date +%T) end $name rc=$?" >> $L; }
q1() { for a in A0 A1 A2; do one $E single "--arm $a" eqdisc single_${a}_eqdisc; done; }
q2() { for a in A3 A4; do one $E single "--arm $a" eqdisc single_${a}_eqdisc; done; one $E double "--run 1" eqdisc double_r1_eqdisc; }
q3() { for m in fd_stlsq_cv fd_stlsq_aic weak_stlsq_cv weak_stlsq_aic; do
         for a in A0 A1 A2 A3; do one $B single "--arm $a" $m single_${a}_$m; done
         for r in 1 2 3 4; do one $B double "--run $r" $m double_r${r}_$m; done; done
       for r in 2 3 4; do one $E double "--run $r" eqdisc double_r${r}_eqdisc; done; }
q1 & q2 & q3 & wait
echo "$(date +%T) ALL DONE" >> $L
