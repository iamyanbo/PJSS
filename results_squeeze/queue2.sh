cd /c/Users/yanbo/downloads/eqdisc
export PYTHONWARNINGS=ignore
python squeeze_bench.py starts --data data/heldout.npz --every 3 --workers 3 --output results_squeeze/starts_heldout3.pkl > results_squeeze/starts_heldout3.log 2>&1
for v in deg2 deg3; do
  python squeeze_bench.py run --data data/heldout.npz --every 3 --starts results_squeeze/starts_heldout3.pkl --variant $v --workers 3 --output results_squeeze/ho3_$v.json > results_squeeze/ho3_$v.log 2>&1
  tail -n 1 results_squeeze/ho3_$v.log
done
python squeeze_bench.py run --data data/instances.npz --starts results_squeeze/starts_dev.pkl --variant deg2 --workers 3 --output results_squeeze/dev_deg2.json > results_squeeze/dev_deg2.log 2>&1
tail -n 1 results_squeeze/dev_deg2.log
