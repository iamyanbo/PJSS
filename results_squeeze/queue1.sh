cd /c/Users/yanbo/downloads/eqdisc
export PYTHONWARNINGS=ignore
for v in deg4 ev10 deg2 anneal swaps; do
  python squeeze_bench.py run --data data/instances.npz --every 3 --starts results_squeeze/starts_dev.pkl --variant $v --workers 3 --output results_squeeze/dev3_$v.json > results_squeeze/dev3_$v.log 2>&1
  tail -n 1 results_squeeze/dev3_$v.log
done
