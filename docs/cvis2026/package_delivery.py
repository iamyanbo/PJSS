"""Package the CVIS source/evidence and original input archives separately."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)

def sha(data):
    return hashlib.sha256(data).hexdigest()

files = set()
for folder, patterns in [('eqdisc', ['*.py','*.json']), ('benchmarks', ['*.py','*.R'])]:
    for pattern in patterns:
        files.update(p for p in (ROOT / folder).rglob(pattern) if '__pycache__' not in p.parts)
for name in ['pjss-extended-abstract.tex','comparison-numbers.tex','illustration-numbers.tex',
             'illustration-protocol.json','make_illustrations.py','make_gallery.py',
             'make_comparisons.py','make_architecture.py','build.py','package_delivery.py','README.txt',
             'REPRODUCE.txt','CVIS-requirements.txt','opening-abstract.txt']:
    files.add(HERE / name)
for name in ['support.pdf','predictions.pdf','attractor_gallery.pdf','recovery_systems.pdf',
             'recovery_noise.pdf','ablation.pdf','architecture.pdf','architecture.svg','architecture.png']:
    files.add(HERE / 'figures' / name)
files.update(p for p in (HERE/'evidence').rglob('*') if p.is_file())
counts=json.loads((HERE/'evidence/comparison_counts.json').read_text())
files.update(ROOT / name for name in counts['input_sha256'])
for name in ['data/dysts_test.json','data/dysts_classes.json',
             'results_h2h/order.json','results_h2h/PREREGISTRATION.md',
             'results_rev/order_dysts.json','results_rev/PREREGISTRATION.md',
             'results_rev/ours_dysts.json','results_rev/job.sh',
             'results_final/hard40_idx.json','results_final/frozen.md5',
             'results_squeeze/starts_joint_sealed.pkl','results_squeeze/starts_hybrid_confirm.pkl',
             'docs/reports/dedi_gpu.md','docs/reports/baselines.md','docs/reports/fast_baselines.md',
             'docs/reports/anatomy.md']:
    files.add(ROOT / name)
for folder in ['results_h2h/pairs','results_rev/e2/pairs']:
    files.update((ROOT/folder).glob('*.json'))

entries = {p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted(files)}
# Package metadata must not reveal submission authors. Solver code is unchanged.
project=(ROOT/'pyproject.toml').read_text()
project=re.sub(r'^authors\s*=.*\n','',project,flags=re.M)
project=project.replace('readme = "README.md"','readme = "docs/cvis2026/README.txt"')
entries['pyproject.toml']=project.encode()
entries['START-HERE.txt']=(
    'Read docs/cvis2026/README.txt, then docs/cvis2026/REPRODUCE.txt.\n'
    'This archive preserves repository-relative paths.\n'
    'Original observations are in cvis2026-pjss-reproduction-data.zip.\n'
    'Historical external DEDi dependencies are not bundled; see the guide.\n').encode()
for name, data in entries.items():
    if name.endswith(('.py','.tex','.txt','.md','.toml','.json','.R')):
        assert not re.search(rb'[A-Za-z0-9._%+-]+@gmail\.com',data,re.I),name
entries['SOURCE-MANIFEST.sha256']=''.join(f'{sha(v)}  {k}\n' for k,v in sorted(entries.items())).encode()
source_zip=OUT/'cvis2026-pjss-extended-abstract-source.zip'
with zipfile.ZipFile(source_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,data in sorted(entries.items()):z.writestr(name,data)
with zipfile.ZipFile(source_zip) as z:assert z.testzip() is None
print('SOURCE',len(entries),'files',source_zip.stat().st_size,'bytes',flush=True)

data_names=['dysts_test','hard120','joint_sealed','sealed_colored2','sealed_student',
            'sealed_colored','hybrid_confirm']
data_zip=OUT/'cvis2026-pjss-reproduction-data.zip'
manifest=[]
with zipfile.ZipFile(data_zip,'w',zipfile.ZIP_STORED) as z:
    for name in data_names:
        p=ROOT/'data'/(name+'.npz')
        entry=p.relative_to(ROOT).as_posix()
        manifest.append(f'{sha(p.read_bytes())}  {entry}\n')
        z.write(p,entry)
    z.writestr('DATA-MANIFEST.sha256',''.join(manifest))
(HERE/'evidence/data-manifest.sha256').write_text(''.join(manifest))
with zipfile.ZipFile(data_zip) as z:assert z.testzip() is None
print('DATA',len(data_names),'archives',data_zip.stat().st_size,'bytes',flush=True)
