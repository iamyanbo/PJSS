"""Fidelity check of benchmarks/wsindy_port.py against the authors' MATLAB WSINDy_ODE run unmodified in GNU Octave.

  OCTAVE_HOME=... python benchmarks/wsindy_octave_check.py --wsindy ~/fastbl/WSINDy_ODE --octave $OCTAVE_HOME/bin/octave-cli \
      --order results_h2h/order.json --positions 0-9 --out results_fast/wsindy_octave_check.json

For each run, calls wsindy_ode_fun.m with the wsindy_ode_script.m defaults (polys = 0:3, everything else as
shipped) and compares the weight matrix with the port's: support equality and max relative difference.
Octave 9 lacks rms/range in core (they live in Octave Forge packages); two one-line shims supply them.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsindy_port import wsindy_ode  # noqa: E402

DRIVER = r"""
addpath('{shim}'); addpath('{src}');
xobs = dlmread('{x}'); tobs = dlmread('{t}'); tobs = tobs(:);
lambda = 10.^(linspace(-4,0,100));
tic;
[w,~,~,~,~,~,~,~,pts,mts,~,~,~,~,~,tags] = wsindy_ode_fun(xobs,tobs,[], 0:3,[],[],{{}}, ...
    1,1,10^-16,-1,1000,1,0, 2,0,lambda,0,0.8, 0.8,[1;0],[1;0],0,2,ceil(length(tobs)/100));
el = toc;
dlmwrite('{w}', [real(tags) w], 'precision', 17);
dlmwrite('{m}', [mts(:) pts(:); el el], 'precision', 17);
"""


def rng(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsindy", required=True)
    ap.add_argument("--octave", required=True)
    ap.add_argument("--order", default="results_h2h/order.json")
    ap.add_argument("--positions", default="0-9")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from eqdisc.core import EXP_ARR, SYSTEMS, metrics
    order = json.loads(Path(a.order).read_text())
    tmp = Path(tempfile.mkdtemp())
    shim = tmp / "shim"
    shim.mkdir()
    (shim / "rms.m").write_text("function y = rms(x)\n  y = sqrt(mean(abs(x).^2));\nend\n")
    (shim / "range.m").write_text("function y = range(x)\n  y = max(x) - min(x);\nend\n")
    recs = []
    for pos in rng(a.positions):
        r = order[pos]
        with np.load(r["data"], allow_pickle=True) as d:
            X = np.asarray(d["X"][r["index"]], float)
            dt = float(d["dt"][r["index"]])
        t = np.arange(len(X)) * dt
        f = {k: tmp / f"{pos}_{k}.csv" for k in ("x", "t", "w", "m")}
        np.savetxt(f["x"], X, delimiter=",", fmt="%.17g")
        np.savetxt(f["t"], t, delimiter=",", fmt="%.17g")
        drv = tmp / f"drv{pos}.m"
        drv.write_text(DRIVER.format(shim=shim, src=a.wsindy, **{k: str(v) for k, v in f.items()}))
        subprocess.run([a.octave, "--no-gui", "-q", str(drv)], check=True, capture_output=True, text=True)
        out = np.loadtxt(f["w"], delimiter=",")
        tags, Wm = out[:, :3].astype(int), out[:, 3:]
        Wo = np.zeros((len(EXP_ARR), 3))
        for e, row in zip(tags, Wm):
            Wo[[tuple(x) for x in EXP_ARR.tolist()].index(tuple(e))] = row
        mp = np.loadtxt(f["m"], delimiter=",")
        t0 = time.perf_counter()
        Wp, info = wsindy_ode(X, t, EXP_ARR, return_info=True)
        tp = time.perf_counter() - t0
        truth = SYSTEMS[r["system"]]
        rel = np.abs(Wp - Wo).max() / max(np.abs(Wo).max(), 1e-300)
        rec = dict(pos=pos, key=f"{r['cond']}_{r['index']}", same_support=bool(((Wp != 0) == (Wo != 0)).all()),
                   max_rel_diff=float(rel), octave_mts=mp[:3, 0].astype(int).tolist(),
                   octave_pts=mp[:3, 1].astype(int).tolist(), port_mts=info["mts"], port_pts=info["pts"],
                   octave_exact=metrics(Wo, truth)["exact"], port_exact=metrics(Wp, truth)["exact"],
                   octave_seconds=float(mp[3, 0]), port_seconds=tp)
        print(json.dumps(rec), flush=True)
        recs.append(rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(recs, indent=1))
    print(f"same support {sum(r['same_support'] for r in recs)}/{len(recs)}; "
          f"max rel diff {max(r['max_rel_diff'] for r in recs):.2e}")


if __name__ == "__main__":
    main()
