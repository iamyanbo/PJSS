"""Audit archived paper evidence without rerunning methods or replacing data.

python benchmarks/audit_archive_evidence.py --output results_submission_audit/archive_evidence.json

Records input SHA-256 hashes, original data-manifest integrity (including the
historical overnight/ relocation), complete DEDi grids and coefficient-derived
metrics, and the profiling ablation's stored endpoints. Missing coefficient
arrays are reported as limitations, not silently treated as independently scored.
The destination must not already exist.
"""
from pathlib import Path
import argparse,hashlib,json,sys,collections

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
args=parser.parse_args()
target=Path(args.output).resolve()
if target.exists():
    raise SystemExit(f'Refusing to overwrite existing report: {target}')
input_hashes={}
def load(path):
    path=Path(path)
    content=path.read_bytes()
    input_hashes[str(path.resolve())]=hashlib.sha256(content).hexdigest()
    return json.loads(content)

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eqdisc.core import SYSTEMS,metrics
sys.path.insert(0,str(ROOT/'benchmarks'))
from h2h_report import LAMBDAS,RS,pick

GRID={(l,r) for l in LAMBDAS for r in RS}
side=load(ROOT/'data/dysts_test.json')['systems']
truth_ext={k:np.asarray(v['xi']) for k,v in side.items()}
out={}
for folder,truth,summary_path in [
    ('results_h2h/pairs',SYSTEMS,'results_h2h/rows.json'),
    ('results_rev/e2/pairs',truth_ext,'results_rev/e2_rows.json'),
    ('results_rev/e3cpu/pairs',SYSTEMS,None),
    ('results_rev/e3gpu/pairs',SYSTEMS,None)]:
    records=[load(p) for p in (ROOT/folder).glob('*.json')]
    groups=collections.defaultdict(dict);errors=[];metrics_bad=[];duplicates=[]
    for rec in records:
        key=rec['key'];lr=(rec['lam'],rec['R'])
        if lr in groups[key]:duplicates.append([key,lr])
        groups[key][lr]=rec
        if 'error' in rec:errors.append(key);continue
        actual=metrics(np.asarray(rec['xi']),truth[rec['system']])
        for k in ('exact','f1','coef_err','fp','fn'):
            if not np.isclose(actual[k],rec['dedi'][k],rtol=1e-9,atol=1e-12,equal_nan=True):
                metrics_bad.append([key,list(lr),k,float(actual[k]),rec['dedi'][k]])
    z={'pairs':len(records),'runs':len(groups),'pairs_per_run':dict(collections.Counter(len(g) for g in groups.values())),
       'errors':errors,'duplicate_pairs':duplicates,'bad_grids':[k for k,g in groups.items() if set(g)!=GRID],
       'metric_mismatches':metrics_bad}
    if summary_path:
        summary={r['key']:r for r in load(ROOT/summary_path)}
        summary_bad=[];totals=collections.Counter()
        for key,g in groups.items():
            for criterion in ('val_error','bic'):
                lr=pick(g,[(l,r) for l in LAMBDAS for r in RS],criterion)
                exact=int(g[lr].get('dedi',{}).get('exact',0))
                totals[criterion]+=exact
                if summary[key]['full grid|'+criterion]!=exact:summary_bad.append([key,criterion])
            oracle=int(any(r.get('dedi',{}).get('exact',0) for r in g.values()))
            totals['oracle']+=oracle
            if summary[key]['oracle']!=oracle:summary_bad.append([key,'oracle'])
        z['selected_exact_totals']=dict(totals);z['summary_mismatches']=summary_bad
    out[folder]=z

ours_bad=[];ours_counts={}
for p in (ROOT/'results_h2h').glob('ours_*.json'):
    data=load(p);recs=data['records'];ours_counts[p.name]=len(recs)
    for r in recs:
        arm=r['arms']['final_v12'];actual=metrics(np.asarray(arm['xi']),SYSTEMS[r['system']])
        for k in ('exact','f1','coef_err','fp','fn'):
            if not np.isclose(actual[k],arm[k],rtol=1e-9,atol=1e-12,equal_nan=True):ours_bad.append([p.name,r['index'],k])
out['pjss_h2h']={'counts':ours_counts,'metric_mismatches':ours_bad}

ablation={};pool=collections.Counter()
for name in ('sealed','colored','hard40','dysts'):
    def records(p):return load(ROOT/p)['records']
    old=records(f'results_ablation/{name}.json');new=records(f'results_rev/abl_{name}.json');checks=records(f'results_rev/abl_check_{name}.json')
    a={r['index']:r for r in old};b={r['index']:r for r in new}
    w=l=full=variant=0;bad=[]
    for i in a:
        f=a[i]['variants']['full']['exact'];v=b[i]['variants']['no_schur']['exact']
        full+=f;variant+=v;w+=f>v;l+=v>f
        if 'error' in b[i]['variants']['no_schur']:bad.append(i)
    ablation[name]={'full_records':len(old),'revision_records':len(new),'unique_indices':[len(a),len(b)],
                    'same_indices':set(a)==set(b),'full_exact':full,'no_schur_exact':variant,'wins':w,'losses':l,
                    'errors':bad,'check_records':len(checks),
                    'check_exact_matches':sum(c['variants']['full']['exact']==a[c['index']]['variants']['full']['exact'] for c in checks),
                    'coefficient_arrays_stored':any('xi' in r['variants']['no_schur'] for r in new)}
    pool.update(n=len(old),full=full,no_schur=variant,wins=w,losses=l)
out['profiling_ablation']={'sets':ablation,'pooled':dict(pool)}
out['e2_pjss_coefficients_stored']=any('xi' in r for r in load(ROOT/'results_rev/ours_dysts.json'))

manifest=ROOT/'data/MANIFEST.sha256'
input_hashes[str(manifest)]=hashlib.sha256(manifest.read_bytes()).hexdigest()
manifest_rows=[]
for line in manifest.read_text().splitlines():
    if not line.strip():continue
    expected,name=line.split(maxsplit=1)
    name=name.lstrip('*').removeprefix('./')
    archive=ROOT/name
    if not archive.exists() and name.startswith('overnight/'):
        archive=ROOT/'archive'/name
    rec={'manifest_path':name,'resolved_path':str(archive),'expected_sha256':expected,'status':'missing'}
    if archive.exists():
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        actual=h.hexdigest()
        rec.update(actual_sha256=actual,bytes=archive.stat().st_size,status='match' if actual==expected else 'mismatch')
        input_hashes[str(archive)]=actual
    manifest_rows.append(rec)
out['data_manifest']={'entries':len(manifest_rows),'counts':dict(collections.Counter(r['status'] for r in manifest_rows)),'rows':manifest_rows}
for script in ['benchmarks/audit_archive_evidence.py','benchmarks/h2h_report.py','eqdisc/core.py']:
    path=ROOT/script
    input_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
out['input_sha256']=input_hashes
out['audit_limits']=[
    'Stored coefficients permit independent recomputation of support/parameter metrics, not of optimization or provenance.',
    'External-comparison PJSS records and profiling variant records omit coefficient arrays; their endpoint aggregates can be checked, but their metrics cannot be rescored from those records alone.',
    'Missing archives are not reconstructed or replaced by this audit.'
]
target.parent.mkdir(parents=True,exist_ok=True)
with target.open('x',encoding='utf-8',newline='\n') as f:
    json.dump(out,f,indent=2)
print('Saved',target)
print('Manifest:',out['data_manifest']['counts'])
for folder in ['results_h2h/pairs','results_rev/e2/pairs','results_rev/e3cpu/pairs','results_rev/e3gpu/pairs']:
    r=out[folder]
    print(folder, 'pairs',r['pairs'],'runs',r['runs'],'errors',len(r['errors']),'metric mismatches',len(r['metric_mismatches']))
print('Profiling:',out['profiling_ablation']['pooled'])
