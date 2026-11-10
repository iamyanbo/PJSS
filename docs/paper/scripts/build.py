"""Build the paper with LaTeX/BibTeX, retaining intermediate files under tmp/.

Run from any directory: python docs/paper/scripts/build.py
Requires pdflatex and bibtex on PATH; no benchmark is rerun.
"""
import os
from pathlib import Path
import shutil
import subprocess


def main():
    paper = Path(__file__).resolve().parents[1]
    build = paper.parents[1] / 'tmp' / 'pdfs' / 'build'
    build.mkdir(parents=True, exist_ok=True)
    # Build from an isolated source copy so old .aux/.bbl files beside main.tex
    # cannot shadow the newly generated references in the output directory.
    for name in ('main.tex', 'refs.bib'):
        shutil.copy2(paper / name, build / name)
    for name in ('sections', 'figures'):
        shutil.copytree(paper / name, build / name, dirs_exist_ok=True)
    for name in ('pdflatex', 'bibtex'):
        if not shutil.which(name):
            raise SystemExit(f'Required executable missing: {name}')
    version = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True, check=True).stdout
    latex = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             f'-output-directory={build.as_posix()}', 'main.tex']
    if 'MiKTeX' in version:
        latex.insert(1, '--disable-installer')
    env = dict(os.environ)
    env['BIBINPUTS'] = str(paper) + os.pathsep + env.get('BIBINPUTS', '')
    steps = [(latex, build), (['bibtex', 'main'], build), (latex, build), (latex, build)]
    for i, (command, cwd) in enumerate(steps, 1):
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                                errors='replace')
        (build / f'step-{i}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise SystemExit(f'Build failed at step {i}:\n{result.stdout[-5000:]}\n{result.stderr}')
    log = (build / 'main.log').read_text(encoding='utf-8', errors='replace')
    failures = ['There were undefined references', 'There were undefined citations',
                'multiply defined', 'Label(s) may have changed']
    if any(message in log for message in failures):
        raise SystemExit(f'Unresolved references; inspect {build / "main.log"}')
    shutil.copy2(build / 'main.pdf', paper / 'main.pdf')
    print(f'Built {paper / "main.pdf"}')
    for line in log.splitlines():
        if 'Overfull' in line or 'Underfull' in line or 'Output written' in line:
            print(line)


if __name__ == '__main__':
    main()
