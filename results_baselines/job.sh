#!/bin/bash
# main baseline pass (6 workers). Configs = dev-best per rule (results_baselines/dev, see docs/reports/baselines.md)
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
B=.venv_base/bin/python; S=benchmarks/baselines_bench.py; O=results_baselines; L=$O/progress.log
run() { name=$1; py=$2; shift 2; echo "$(date +%T) start $name" >> $L
        $py $S run "$@" --workers 6 --output $O/$name.json > $O/$name.log 2>&1; echo "$(date +%T) end $name rc=$?" >> $L; }
mkdir -p $O/dev
for h in 10 20; do [ -f $O/dev/weak_miosr_h$h.json ] || run dev/weak_miosr_h$h $B --method weak_miosr --set dev --h $h; done
HM=$($B -c "
import json,numpy as np
s={h:np.mean([r[\"cv\"][\"exact\"] for r in json.load(open(f\"$O/dev/weak_miosr_h{h}.json\"))[\"records\"] if r[\"status\"]==\"ok\"]) for h in (10,20)}
print(max(s,key=s.get))")
echo "$(date +%T) miosr h=$HM" >> $L
for set in hard40 dysts7; do
  run ${set}_fd_stlsq_raw_w11  $B --method fd_stlsq  --set $set --scale raw --window 11
  run ${set}_fd_stlsq_raw_w5   $B --method fd_stlsq  --set $set --scale raw --window 5
  run ${set}_fd_sr3_w11        $B --method fd_sr3    --set $set --window 11
  run ${set}_fd_sr3_w5         $B --method fd_sr3    --set $set --window 5
  run ${set}_fd_sr3_w21        $B --method fd_sr3    --set $set --window 21
  run ${set}_weak_stlsq_h10    $B --method weak_stlsq --set $set --h 10
  run ${set}_weak_stlsq_raw_h20 $B --method weak_stlsq --set $set --scale raw --h 20
  run ${set}_weak_sr3_h10      $B --method weak_sr3  --set $set --h 10
  run ${set}_weak_sr3_h20      $B --method weak_sr3  --set $set --h 20
  run ${set}_weak_miosr_h$HM   $B --method weak_miosr --set $set --h $HM
done
run hard40_pysr        .venv_pysr/bin/python --method pysr --set hard40 --window 11 --budget 60
run hard40_odeformer   .venv_odeformer/bin/python --method odeformer --set hard40
for set in hard40 dysts7; do
  run ${set}_weak_esindy_h10     $B --method weak_esindy --set $set --h 10
  run ${set}_weak_esindy_raw_h20 $B --method weak_esindy --set $set --scale raw --h 20
  run ${set}_fd_esindy_raw_w11   $B --method fd_esindy   --set $set --scale raw --window 11
done
run dysts21_odeformer  .venv_odeformer/bin/python --method odeformer --set dysts7 --every 3
run dysts49_pysr       .venv_pysr/bin/python --method pysr --set dysts7 --every 7 --window 11 --budget 60
echo "$(date +%T) ALL_DONE" >> $L
