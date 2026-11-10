#!/bin/bash
# Revision experiments, pre-registered in results_rev/PREREGISTRATION.md. E1 and E2 together, then E3 on an idle machine.
cd ~/eqdisc_rev
O=results_rev
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_rev PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
OURS=~/eqdisc_v2/.venv/bin/python; GPU=~/eqdisc_dedi_gpu/.venv/bin/python; NAT=~/mm/cholmod/bin/python
log(){ echo "$(date "+%F %T") $* load $(cut -d' ' -f1 /proc/loadavg)" >> $O/progress.log; }
log start; git log -1 --format=%h 2>/dev/null >> $O/progress.log
# E1 on the slow cores 0-3
(for s in sealed:5 colored:9 hard40:1 dysts:17; do
  n=${s%:*}; ev=${s#*:}
  taskset -c 0-3 nice -n 5 $OURS benchmarks/ablation.py --set $n --every $ev --workers 4 --variants no_schur refit_only --output $O/abl_$n.json >> $O/abl.log 2>&1
  taskset -c 0-3 nice -n 5 $OURS benchmarks/ablation.py --set $n --every $ev --limit 20 --workers 4 --variants full --output $O/abl_check_$n.json >> $O/abl.log 2>&1
done) &
E1=$!
# E2: ours on core 4, DEDi on 10 GPU workers
taskset -c 4 $OURS benchmarks/rev_ours.py --order $O/order_dysts.json --truth-json data/dysts_test.json --output $O/ours_dysts.json >> $O/ours.log 2>&1 &
mkdir -p $O/e2/claims $O/e2/pairs $O/e2/runs
k=0; for core in 5 6 7 8 9 15 16 17 18 19; do
  timeout -k 60 108000 taskset -c $core $GPU benchmarks/dedi_h2h.py --device gpu --ext ~/eqdisc_dedi --order $O/order_dysts.json \
    --outdir $O/e2 --worker $k --truth-json data/dysts_test.json > $O/gpu$k.err 2>&1 & k=$((k+1))
done
wait
log E1E2_done
# E3 on the idle machine
mkdir -p $O/e3cpu/claims $O/e3cpu/pairs $O/e3cpu/runs $O/e3gpu/claims $O/e3gpu/pairs $O/e3gpu/runs
for c in 15 16 17; do taskset -c $c $OURS benchmarks/rev_ours.py --order $O/order_timing.json --repeats 3 --output $O/ours_timing_c$c.json >> $O/ours.log 2>&1; done
log E3_ours_done
k=0; for core in 15 16 17; do
  taskset -c $core $NAT benchmarks/dedi_h2h.py --device cpu --ext ~/eqdisc_dedi_native --order $O/order_timing.json --outdir $O/e3cpu --worker $k > $O/e3cpu$k.err 2>&1 & k=$((k+1))
done
wait
log E3_cpu_done
taskset -c 15 $GPU benchmarks/dedi_h2h.py --device gpu --ext ~/eqdisc_dedi --order $O/order_timing.json --outdir $O/e3gpu --worker 0 > $O/e3gpu.err 2>&1
log DONE
