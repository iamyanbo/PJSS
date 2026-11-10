#!/bin/bash
# v1.1 pre-registered test pass (results_final_v11/PREREGISTRATION.md). Queued behind the v1.0.1 verification
# (needs results_v101/test_*.json for the guards). Spark, 19 workers.
cd ~/eqdisc_v3
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
P=.venv/bin/python
O=results_final_v11
L=$O/progress.log
W=19
st() { echo "$(date '+%F %T') $*" >> $L; }
st "queued; waiting for PYTEST_DONE"
while ! grep -q PYTEST_DONE results_v101/progress.log 2>/dev/null; do
  if ! pgrep -f '[r]esults_v101/job' > /dev/null; then st "v101 jobs gone without PYTEST_DONE; continuing"; break; fi
  sleep 60
done
st "start"; md5sum eqdisc/*.py benchmarks/final_bench.py benchmarks/v11_report.py benchmarks/gen_variants.py >> $L
sha256sum data/sealed_colored2.npz >> $L
grep -q "$(sha256sum data/sealed_colored2.npz | cut -c1-64)" data/MANIFEST.sha256 || { st "ARCHIVE SHA256 NOT IN MANIFEST; abort"; st "V11_DONE"; exit 1; }
fb() { name=$1; shift; st "run $name"; $P benchmarks/final_bench.py "$@" --workers $W --output $O/$name.json > $O/$name.log 2>&1; st "end $name rc=$?"; }
fb repro_dev_colored --kind poly3 --data data/instances_colored.npz --index-file $O/repro_idx.json --arms final final_v101
fb test_colored2     --kind poly3 --data data/sealed_colored2.npz --arms final final_v101
fb guard_sealed      --kind poly3 --data data/joint_sealed.npz --starts results_squeeze/starts_joint_sealed.pkl --arms final
fb guard_student     --kind poly3 --data data/sealed_student.npz --arms final
for a in coarse heldout2; do
  $P -c "import json; r=json.load(open('results_v101/test_$a.json'))['records']; json.dump(sorted(x['index'] for x in r if x['arms']['final'].get('noise_model')=='ar1'), open('$O/idx_${a}_ar1.json','w'))"
  st "idx_${a}_ar1: $(cat $O/idx_${a}_ar1.json)"
done
[ "$(cat $O/idx_coarse_ar1.json)" != "[]" ] && fb guard_coarse_ar1 --kind poly3 --data data/sealed_coarse.npz --index-file $O/idx_coarse_ar1.json --arms final
[ "$(cat $O/idx_heldout2_ar1.json)" != "[]" ] && fb guard_heldout2_ar1 --kind poly3 --data data/heldout2.npz --starts results_squeeze/starts_heldout2.pkl --index-file $O/idx_heldout2_ar1.json --arms final
fb support_colored   --kind poly3 --data data/sealed_colored.npz --arms final
$P benchmarks/v11_report.py > $O/report.md 2>&1
st "V11_DONE"
