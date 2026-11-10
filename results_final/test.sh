#!/bin/bash
# ONE frozen test pass of final_model.py (arm "final" = public API with frozen DEFAULTS). Spark, 18 workers.
cd ~/eqdisc_v2
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=.venv/bin/python
W=18
I=results_final
md5sum -c $I/frozen.md5 > $I/frozen_check.txt 2>&1 || { echo "$(date +%T) FROZEN_MISMATCH" >> $I/test_progress.log; exit 1; }
run() { name=$1; shift; echo "$(date +%T) start $name" >> $I/test_progress.log
        "$@" --arms final --workers $W --output $I/$name.json > $I/$name.log 2>&1
        echo "$(date +%T) end $name rc=$?" >> $I/test_progress.log; }
run test_nd        $P final_bench.py --kind nd --data data/nd_test.npz
run test_sealed    $P final_bench.py --kind poly3 --data data/joint_sealed.npz --starts results_squeeze/starts_joint_sealed.pkl
run test_confirm   $P final_bench.py --kind poly3 --data data/hybrid_confirm.npz --starts results_squeeze/starts_hybrid_confirm.pkl
run test_heldout2  $P final_bench.py --kind poly3 --data data/heldout2.npz --starts results_squeeze/starts_heldout2.pkl
run test_colored   $P final_bench.py --kind poly3 --data data/sealed_colored.npz
run test_student   $P final_bench.py --kind poly3 --data data/sealed_student.npz
run test_coarse    $P final_bench.py --kind poly3 --data data/sealed_coarse.npz
EQ_DEG=4 run test_deg4 $P final_bench.py --kind poly4 --data data/joint_sealed.npz --every 3
EQ_DEG=5 run test_deg5 $P final_bench.py --kind poly5 --data data/joint_sealed.npz --every 3
run test_mixedpoly $P final_bench.py --kind trig --data data/joint_sealed.npz --every 6
run test_trig      $P final_bench.py --kind trig --data data/trig_test.npz
echo "$(date +%T) TEST_DONE" >> $I/test_progress.log
