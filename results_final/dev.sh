#!/bin/bash
# Dev crossed experiments for final_model.py (Spark, 18 workers). Logs to results_final/<name>.log
cd ~/eqdisc_v2
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=.venv/bin/python
W=18
run() { name=$1; shift; echo "$(date +%T) start $name" >> results_final/dev_progress.log
        "$@" --workers $W --output results_final/$name.json > results_final/$name.log 2>&1
        echo "$(date +%T) end $name rc=$?" >> results_final/dev_progress.log; }
# (a) prior x AR(1)
run dev_white     $P final_bench.py --kind poly3 --data data/instances.npz --starts results_squeeze/starts_dev.pkl --every 2 --arms C Cp AR ARp
run dev_colored   $P final_bench.py --kind poly3 --data data/instances_colored.npz --every 2 --arms C Cp AR ARp ARp_e14 ARp_e12
run dev_cmid      $P final_bench.py --kind poly3 --data data/dev_colored_mid.npz --arms C Cp AR ARp
run dev_student   $P final_bench.py --kind poly3 --data data/instances_student.npz --every 4 --arms Cp ARp
run dev_coarse    $P final_bench.py --kind poly3 --data data/instances_coarse.npz --every 4 --arms Cp ARp
run dev_heldout3  $P final_bench.py --kind poly3 --data data/heldout.npz --starts results_squeeze/starts_heldout3.pkl --every 3 --arms Cp ARp
# (b) prior x big library
EQ_DEG=4 run dev_deg4 $P final_bench.py --kind poly4 --data data/instances.npz --every 2 --arms off p15 pmid pmean
EQ_DEG=5 run dev_deg5 $P final_bench.py --kind poly5 --data data/instances.npz --every 2 --arms off p15 pmid pmean
run dev_trig      $P final_bench.py --kind trig --data data/trig_dev.npz --arms off p15
run dev_mixedpoly $P final_bench.py --kind trig --data data/instances.npz --every 4 --arms off p15
# (c) prior x D dimensions
run dev_nd        $P final_bench.py --kind nd --data data/nd_dev.npz --every 2 --arms off p15 p10
echo "$(date +%T) DEV_DONE" >> results_final/dev_progress.log
