"""Targeted post-hoc reproducibility check; historical result files are read-only.

python -m benchmarks.audit_weak_seed --output results_submission_audit/weak_seed.json
This is not a replacement benchmark or an estimate of a full-archive effect.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
from threadpoolctl import threadpool_limits
from benchmarks import baselines_bench as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f'Refusing to overwrite {output}')
    sources = {
        'weak_stlsq': ('results_baselines/dysts7_weak_stlsq_h10.json', 8),
        'weak_sr3': ('results_baselines/dysts7_weak_sr3_h10.json', 4),
        'weak_esindy': ('results_baselines/dysts21_weak_esindy_h10.json', 2),
    }
    # Cache the archive once instead of repeatedly decompressing its object array.
    side = json.loads(Path('data/dysts_test.json').read_text())['systems']
    with np.load('data/dysts_test.npz', allow_pickle=True) as archive:
        data_arrays = {k: archive[k] for k in ('X', 'sys', 'noise', 'n', 'dt')}
    runs = {}
    for i in range(0, len(data_arrays['X']), 7):
        system = str(data_arrays['sys'][i])
        runs[i] = dict(index=i, x=np.asarray(data_arrays['X'][i], float),
                       system=system, dt=float(data_arrays['dt'][i]),
                       noise=float(data_arrays['noise'][i]), n=int(data_arrays['n'][i]),
                       truth=side[system]['xi'], names=side[system]['names'])
    result = dict(purpose='post-hoc seed-order check, not a replacement benchmark',
                  selection='one run per evenly spaced noise/N cell, uniform index choice before fitting',
                  selection_seed=20261103,
                  versions={k: importlib.metadata.version(k) for k in
                            ('numpy', 'scipy', 'pysindy', 'scikit-learn')}, records=[], sources={})
    for filename in ('data/dysts_test.npz', 'data/dysts_test.json',
                     'benchmarks/baselines_bench.py', 'benchmarks/audit_weak_seed.py'):
        result['sources'][filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    warnings.filterwarnings('ignore')
    with threadpool_limits(limits=1):
        for method, (filename, count) in sources.items():
            raw = Path(filename).read_bytes()
            data = json.loads(raw)
            result['sources'][filename] = hashlib.sha256(raw).hexdigest()
            historical = {r['index']: r for r in data['records'] if r['status'] == 'ok'}
            cfg = data.get('cfg', data['args'])
            by_cell = {}
            for i in sorted(historical):
                run = runs[i]
                cell = (run['noise'], run['n'])
                if run['x'].shape[1] == 3:
                    by_cell.setdefault(cell, []).append(i)
            cells = sorted(by_cell)
            rng = np.random.default_rng(20261103)
            selected = [int(rng.choice(by_cell[cells[j]])) for j in
                        np.linspace(0, len(cells)-1, count).round().astype(int)]
            first = {}
            for i in selected:
                np.random.seed(i + 123456)
                first[i] = bench.run_path(runs[i], method, cfg)
            for i in reversed(selected):
                np.random.seed(i + 654321)
                np.random.standard_normal(10007)
                start = time.perf_counter()
                second = bench.run_path(runs[i], method, cfg)
                old = historical[i]
                row = dict(method=method, index=i, system=runs[i]['system'],
                           noise=runs[i]['noise'], n=runs[i]['n'],
                           repeat_identical=first[i] == second,
                           repeat_differing_fields=[k for k in first[i] if first[i][k] != second[k]],
                           historical={k: old[k]['exact'] for k in ('cv', 'aic')},
                           corrected={k: second[k]['exact'] for k in ('cv', 'aic')},
                           changed_support={k: not np.array_equal(np.asarray(old[k+'_xi']) != 0,
                                                np.asarray(second[k+'_xi']) != 0) for k in ('cv', 'aic')},
                           historical_oracle=old['oracle'], corrected_oracle=second['oracle'],
                           corrected_record=second,
                           first_record_if_different=first[i] if first[i] != second else None,
                           second_run_seconds=time.perf_counter()-start)
                result['records'].append(row)
                print(json.dumps(row), flush=True)
    result['all_repeats_identical'] = all(r['repeat_identical'] for r in result['records'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if not result['all_repeats_identical']:
        raise SystemExit('Corrected path is still order-dependent; inspect audit.')


if __name__ == '__main__':
    main()
