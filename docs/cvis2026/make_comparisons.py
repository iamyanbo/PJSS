"""Recompute CVIS tables from saved paired records, without fitting models."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
inputs = {}

def read(name):
    data = (ROOT / name).read_bytes()
    inputs[name] = hashlib.sha256(data).hexdigest()
    return json.loads(data)

def read_jsonl(name):
    data = (ROOT / name).read_bytes()
    inputs[name] = hashlib.sha256(data).hexdigest()
    return [json.loads(line) for line in data.splitlines() if line.strip()]

ours = {r['index']: r for r in read('results_dysts/test_final.json')['records']}
spec = [
    ('Stlsq', 'dysts7_fd_stlsq_raw_w11', 'cv', 269, 784),
    ('Srthree', 'dysts7_fd_sr3_w11', 'cv', 153, 784),
    ('Esindy', 'dysts21_fd_esindy_raw_w11', 'cv', 86, 257),
    ('Pysr', 'dysts49_pysr', 'best', 56, 117),
    ('Odeformer', 'dysts21_odeformer', 'best', 2, 257),
]
macros = []
table = []
for key, name, selector, expected, expected_ours in spec:
    d = read('results_baselines/' + name + '.json')
    rows = sorted(d['records'], key=lambda r: r['index'])
    count = sum(int(r.get(selector, {}).get('exact', 0)) if r['status'] == 'ok' else 0 for r in rows)
    oc = sum(int(ours[r['index']]['final']['exact']) for r in rows)
    assert (count, oc) == (expected, expected_ours), (name, count, oc)
    for suffix, val in [('Count', count), ('Ours', oc)]:
        macros.append('\\newcommand{\\' + key + suffix + '}{' + str(val) + '}')
    table.append(dict(method=name, selector=selector, n=len(rows), baseline=count, pjss=oc,
                      configuration=d['args'], indices=[r['index'] for r in rows]))

paired = {}
for label, path, n, expected in [
    ('core', 'results_h2h/rows.json', 80, (66, 55, 61)),
    ('new', 'results_rev/e2_rows.json', 64, (53, 36, 41)),
]:
    rows = read(path)
    counts = tuple(sum(r[k] for r in rows) for k in ['ours', 'full grid|val_error', 'full grid|bic'])
    assert len(rows) == n and counts == expected
    groups = {}
    for r in rows:
        group = r['cond'] if label == 'core' else ('low' if r['noise'] <= .05 else 'high')
        groups.setdefault(group, []).append(r)
    paired[label] = dict(n=n, counts=counts, groups={g: {k: sum(r[k] for r in rr)
        for k in ['ours', 'full grid|val_error', 'full grid|bic']} for g, rr in groups.items()})
    if label == 'core':
        rr = [r for r in rows if r['cond'] != 'coloured']
        assert sum(r['ours'] for r in rr) == sum(r['full grid|bic'] for r in rr) == 56
        assert sum(r['device'] == 'gpu' for r in rows) == 73
    else:
        assert sum(r['ours'] and not r['full grid|bic'] for r in rows) == 15
        assert sum(not r['ours'] and r['full grid|bic'] for r in rows) == 3

cost = {}
for name, selector, count in [('wsindy', 'mstls', 24), ('fd_stlsq', 'cv', 17),
                               ('fd_esindy', 'cv', 17), ('argos_b200', 'argos', 17)]:
    rows = read('results_fast/' + name + '.json')['records']
    assert sum(r[selector]['exact'] for r in rows) == count
    sec = [r['seconds'] for r in rows]
    cost[name] = dict(n=len(rows), count=count, median_recorded_seconds=float(np.median(sec)))

rt = []
for name in ['hard', 'white', 'coloured', 't3', 'short']:
    for r in read('results_fast/ours_retime_' + name + '.json')['records']:
        rt.append(r['arms']['final_v12']['seconds'] + r.get('starts_seconds', 0) + r.get('a_seconds', 0))
cost['PJSS'] = dict(n=len(rt), count=66, median_recorded_seconds=float(np.median(rt)))
rr = read('results_h2h/rows.json')
cost['DEDi'] = dict(n=len(rr), count=61, median_recorded_seconds=float(np.median([r['full grid|seconds'] for r in rr])))
assert round(cost['PJSS']['median_recorded_seconds'], 1) == 14.4
assert round(cost['DEDi']['median_recorded_seconds'], -1) == 5930

# Separate a ratio of pooled medians from the original paired GPU-subset ratio.
gpu = [r for r in rr if r['device'] == 'gpu']
timing = dict(
    pooled_ratio_of_medians=cost['DEDi']['median_recorded_seconds']/cost['PJSS']['median_recorded_seconds'],
    gpu_runs=len(gpu),
    gpu_median_paired_ratio=float(np.median([r['full grid|seconds']/r['ours_s'] for r in gpu])),
    paired_denominator='original concurrent PJSS timings, not the subsequent retiming',
    hardware='loaded NVIDIA GB10; PJSS one CPU core; DEDi ten shared GPU processes or CPU workers',
)
assert round(timing['pooled_ratio_of_medians']) == 411
assert len(gpu) == 73 and round(timing['gpu_median_paired_ratio']) == 343
anatomy_ours = {r['key']:r for r in read_jsonl('results_anatomy/ours_h2h.jsonl')}
anatomy_dedi = read_jsonl('results_anatomy/dedi_anatomy.jsonl')
anatomy_keys = read('results_anatomy/runs_dedi_anatomy.json')
anatomy_rows = []
for key in anatomy_keys:
    pairs = [r for r in anatomy_dedi if r['key'] == key]
    assert len(pairs) == 35 and len({(r['lam'],r['R']) for r in pairs}) == 35
    ours_count = anatomy_ours[key]['lm_fits']
    dedi_count = sum(r['lm_solves'] for r in pairs)
    anatomy_rows.append(dict(key=key, pjss_fits=ours_count, dedi_fits=dedi_count,
                             ratio=dedi_count/ours_count))
timing['operation_count_study'] = dict(
    pjss_runs=len(anatomy_ours), pjss_median_joint_fits=float(np.median([r['lm_fits'] for r in anatomy_ours.values()])),
    paired_runs=len(anatomy_rows), median_paired_fit_ratio=float(np.median([r['ratio'] for r in anatomy_rows])),
    rows=anatomy_rows)
assert len(anatomy_ours) == 80 and len(anatomy_rows) == 10
assert timing['operation_count_study']['pjss_median_joint_fits'] == 21
assert round(timing['operation_count_study']['median_paired_fit_ratio']) == 49

abl = {}
totals = dict(full=0, no_proposals=0, greedy=0, no_schur=0)
for name, n in [('sealed',160), ('colored',89), ('hard40',40), ('dysts',287)]:
    old = read('results_ablation/' + name + '.json')
    new = read('results_rev/abl_' + name + '.json')
    oldrows = {r['index']:r for r in old['records']}
    newrows = {r['index']:r for r in new['records']}
    assert len(oldrows) == n and oldrows.keys() == newrows.keys()
    for index, r in oldrows.items():
        for variant in ['full','no_proposals','greedy']:
            totals[variant] += int(r['variants'][variant]['exact'])
        totals['no_schur'] += int(newrows[index]['variants']['no_schur']['exact'])
    abl[name] = dict(n=n, configuration=old['args'], indices=sorted(oldrows))
assert totals == dict(full=490, no_proposals=458, greedy=462, no_schur=480)

(HERE / 'comparison-numbers.tex').write_text('\n'.join(macros) + '\n')
(HERE / 'evidence' / 'comparison_counts.json').write_text(json.dumps(
    dict(standard=table, dedi=paired, fast=cost, timing=timing, ablation=abl, ablation_counts=totals, input_sha256=inputs), indent=2))
print('Verified paired comparison counts:', [(x['method'], x['baseline'], x['pjss']) for x in table])
print('DEDi:', paired)
print('Timing:', {k:v for k,v in timing.items() if k != 'operation_count_study'})
print('Joint fits:', timing['operation_count_study']['pjss_median_joint_fits'],
      'median paired ratio:',timing['operation_count_study']['median_paired_fit_ratio'])
