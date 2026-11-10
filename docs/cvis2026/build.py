"""Build and render four content pages plus one references-only page."""
from pathlib import Path
import re
import shutil
import subprocess
import pymupdf

root = Path(__file__).resolve().parents[2]
source = Path(__file__).resolve().parent / 'pjss-extended-abstract.tex'
build = root / 'tmp' / 'pdfs' / 'cvis_pjss_extended_abstract_build'
output = root / 'output' / 'pdf'
build.mkdir(parents=True, exist_ok=True)
output.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, build / source.name)
shutil.copy2(source.parent / 'illustration-numbers.tex', build / 'illustration-numbers.tex')
shutil.copy2(source.parent / 'comparison-numbers.tex', build / 'comparison-numbers.tex')
shutil.copytree(source.parent / 'figures', build / 'figures', dirs_exist_ok=True)
command = ['pdflatex', '--disable-installer', '-interaction=nonstopmode',
           '-halt-on-error', source.name]
for i in range(2):
    result = subprocess.run(command, cwd=build, capture_output=True, text=True,
                            errors='replace')
    (build / f'pass-{i+1}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise SystemExit(result.stdout[-5000:] + result.stderr)
log = (build / 'pjss-extended-abstract.log').read_text(encoding='utf-8', errors='replace')
for phrase in ('undefined references', 'undefined citations', 'Overfull'):
    assert phrase not in log, f'Layout/citation check failed: {phrase}'
pdf = build / 'pjss-extended-abstract.pdf'
doc = pymupdf.open(pdf)
print('PAGES:', len(doc))
assert len(doc) == 5, f'Expected four content pages plus references, got {len(doc)}'
assert doc[4].get_text().strip().startswith('References'), 'Page five must contain only references'
for i, page in enumerate(doc):
    boxes = [b[:4] for b in page.get_text('blocks') if b[1] < 750]
    assert all(b[0] >= 50 and b[2] <= 562 and b[3] <= 740 for b in boxes), f'Content exceeds bounds on page {i+1}'
    page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(build / f'page-{i+1}.png')
    print(f'PAGE {i+1}: {len(page.get_text().split())} words')
text = '\n'.join(p.get_text() for p in doc)
assert not doc.metadata.get('author')
assert not re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)
assert not re.search(r'\[\?\?\]', text)
source_text = source.read_text(encoding='utf-8')
cited = {key for group in re.findall(r'\\cite\{([^}]+)\}', source_text) for key in group.split(',')}
references = set(re.findall(r'\\bibitem\{([^}]+)\}', source_text))
assert cited == references, 'References must all be cited and all citations resolved'
shutil.copy2(pdf, output / 'cvis2026-pjss-extended-abstract.pdf')
print('OUTPUT:', output / 'cvis2026-pjss-extended-abstract.pdf')
