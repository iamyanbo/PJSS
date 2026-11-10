cd /c/Users/yanbo/downloads/eqdisc
export PYTHONWARNINGS=ignore
for a in joint_sealed hybrid_confirm heldout2; do
  python squeeze_bench.py starts --data data/$a.npz --workers 3 --output results_squeeze/starts_$a.pkl > results_squeeze/starts_$a.log 2>&1
  python squeeze_bench.py run --data data/$a.npz --starts results_squeeze/starts_$a.pkl --variant deg2 --workers 3 --output results_squeeze/test_${a}_deg2.json > results_squeeze/test_${a}_deg2.log 2>&1
  tail -n 1 results_squeeze/test_${a}_deg2.log
done
