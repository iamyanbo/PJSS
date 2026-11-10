"""One-at-a-time sensitivity sweep of the fixed constants of final_model.discover (v1.2 default).

Pre-registration: results_sensitivity/PREREGISTRATION.md (sample, arms, metrics, claim rule; committed before any
sweep result existed). Report: benchmarks/sensitivity_report.py -> docs/reports/sensitivity.md.

    python benchmarks/sensitivity_bench.py sample --output results_sensitivity/sample.json
    python benchmarks/sensitivity_bench.py run --sample results_sensitivity/sample.json --outdir results_sensitivity

`sample`: runs from data/dysts_test.npz on the 49 dysts systems never used in development (sidecar overlap = None),
stratified over (system, noise level): every one of the 49 x 4 cells gets one run, then cells drawn without
replacement get a second, until N runs; inside a cell the run (length, seed) is uniform. Order shuffled.
`run`: for each sampled run in order, the default arm first, then every perturbed arm (so at any moment all arms
cover the same prefix of the random order: a kill leaves a valid paired comparison). One process, one BLAS thread.
Output: <outdir>/arms/<arm>.json ({"records": {index: record}}), rewritten atomically after every run; a restart
skips what is done. Per-call wall limit (SIGALRM) --timeout seconds -> status "timeout" (counts as not exact).

Copies (no recomputation) where the perturbed constant provably cannot act (same deterministic code path):
  ar_threshold arms: the routing decision mean(a_hat) > thr is the same as the default's (a_hat does not depend on
                     the threshold), or D != 3 (joint_nd has no AR(1) path);
  pen_exp arms:      the default routed to the white model (pen_exp only enters joint_colored), or D != 3.
The first --verify-copies (default 3) copy-eligible runs of each such arm are recomputed anyway and must be
bit-identical to the default (else copying is switched off for that arm and it is recorded).
Truth (sidecar xi) is used only for scoring.
"""
import argparse
import json
import os
import signal
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = "data/dysts_test.npz"

# arm -> discover() keyword arguments. Frozen values: degw .2, centre 1.5, ar_threshold .6, pen_exp 1/3,
# huber (white path) 2, rho .03, kmax 5, finalists 10, Savitzky-Golay window 21.
ARMS = {
    "default": {},
    "degw_x0.5": {"degw": 0.1}, "degw_x2": {"degw": 0.4},
    "centre_x0.5": {"centre": 0.75}, "centre_x2": {"centre": 3.0},
    "arthr_0.45": {"ar_threshold": 0.45}, "arthr_0.75": {"ar_threshold": 0.75},
    "penexp_x0.5": {"pen_exp": 1 / 6}, "penexp_x2": {"pen_exp": 2 / 3},
    "huber_x0.5": {"huber_white": 1.0}, "huber_x2": {"huber_white": 4.0},
    "rho_x0.5": {"rho": 0.015}, "rho_x2": {"rho": 0.06},
    "kmax_x0.5": {"kmax": 3}, "kmax_x2": {"kmax": 10},
    "finalists_x0.5": {"finalists": 5}, "finalists_x2": {"finalists": 20},
    "sgwin_x0.5": {"sg_window": 11}, "sgwin_x2": {"sg_window": 41},
}
EXHAUSTIVE_KMAX = 7  # flow_discovery._top_subsets refuses > 250k subsets: sum_{k<=8} C(20, k) > 250k for poly3


def make_sample(n_runs, seed, out):
    side = json.loads((REPO / "data/dysts_test.json").read_text())["systems"]
    rng = np.random.default_rng(seed)
    with np.load(REPO / DATA, allow_pickle=True) as d:
        sysn, noise = d["sys"].astype(str), d["noise"].astype(float)
        new = sorted({s for s in sysn if side[s]["overlap"] in (None, "None")})
        cells = {}
        for i in range(len(sysn)):
            if sysn[i] in new:
                cells.setdefault((sysn[i], float(noise[i])), []).append(i)
    keys = sorted(cells)
    assert len(new) == 49 and len(keys) == 196, (len(new), len(keys))
    picked = [int(rng.choice(cells[k])) for k in keys]
    extra = n_runs - len(picked)
    for j in rng.choice(len(keys), size=extra, replace=False):
        rest = [i for i in cells[keys[j]] if i not in picked]
        picked.append(int(rng.choice(rest)))
    picked = [int(i) for i in rng.permutation(picked)]
    with np.load(REPO / DATA, allow_pickle=True) as d:
        X, nn = d["X"], d["n"]  # load each array once (npz re-reads on every d[...] access)
    meta = [{"index": i, "system": str(sysn[i]), "noise": float(noise[i]), "n": int(nn[i]),
             "D": int(np.asarray(X[i]).shape[1])} for i in picked]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"data": DATA, "seed": seed, "n_runs": n_runs, "systems": new,
                                     "runs": meta}, indent=0))
    print(f"{len(picked)} runs, {len(new)} systems, n=2000: {sum(m['n'] == 2000 for m in meta)}, "
          f"D=3: {sum(m['D'] == 3 for m in meta)}")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def call(x, dt, kw, timeout):
    from eqdisc import final_model as fm
    big = x.shape[1] == 3 and kw.get("kmax", 5) > EXHAUSTIVE_KMAX
    t0 = time.perf_counter()
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout))
    try:
        if big:  # beam-search proposals (joint_biglib, as final_model uses for poly4/poly5): exhaustive is refused
            with fm._big_search():
                r = fm.discover(x, dt, **kw)
        else:
            r = fm.discover(x, dt, **kw)
        out = {"status": "ok", "xi": np.asarray(r.xi), "noise_model": r.info.get("noise_model"),
               "ar_estimate": r.info.get("ar_estimate"), "gap": float(r.gap), "ambiguous": bool(r.ambiguous)}
    except _Timeout:
        out = {"status": "timeout"}
    except Exception as exc:
        out = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    out["seconds"] = time.perf_counter() - t0
    out["beam_search"] = big
    return out


def record(meta, res, truth):
    from eqdisc.core import metrics
    rec = dict(meta)
    rec.update({k: v for k, v in res.items() if k != "xi"})
    if res["status"] == "ok":
        xi = res["xi"]
        if xi.shape != truth.shape:
            rec.update(status="error", error=f"shape {xi.shape} != truth {truth.shape}")
        else:
            m = metrics(xi, truth)
            rec["metrics"] = {k: float(v) for k, v in m.items()}
            rec["xi"] = xi.tolist()
    if rec["status"] != "ok":
        rec["metrics"] = {"exact": 0.0}
    return rec


def copy_eligible(arm, d_rec):
    """True if the arm's constant provably cannot change this run's answer (see module docstring)."""
    if d_rec["status"] != "ok":
        return False
    if arm.startswith("arthr") or arm.startswith("penexp"):
        if d_rec["D"] != 3:
            return True
        if arm.startswith("penexp"):
            return d_rec.get("noise_model") == "white"
        a = d_rec.get("ar_estimate")
        if a is None:
            return False
        thr = ARMS[arm]["ar_threshold"]
        return (float(np.mean(a)) > thr) == (d_rec.get("noise_model") == "ar1")
    return False


def _save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def run(sample, outdir, timeout, verify_copies, arms):
    from threadpoolctl import threadpool_limits
    smp = json.loads(Path(sample).read_text())
    side = json.loads((REPO / "data/dysts_test.json").read_text())["systems"]
    adir = Path(outdir) / "arms"
    adir.mkdir(parents=True, exist_ok=True)
    store = {}
    for a in arms:
        p = adir / f"{a}.json"
        store[a] = json.loads(p.read_text()) if p.exists() else {"arm": a, "kwargs": ARMS[a], "records": {},
                                                                    "copy_check": {"verified": 0, "failed": False}}
    import hashlib
    import eqdisc
    pkg = Path(eqdisc.__file__).parent
    prov = {"started": time.strftime("%F %T"), "eqdisc": str(pkg), "version": eqdisc.__version__,
            "md5": {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(pkg.glob("*.py"))},
            "data_md5": hashlib.md5((REPO / smp["data"]).read_bytes()).hexdigest()}
    with open(Path(outdir) / "provenance.jsonl", "a") as fh:
        fh.write(json.dumps(prov) + "\n")
    d = np.load(REPO / smp["data"], allow_pickle=True)
    X, DT = d["X"], d["dt"]
    t_start = time.perf_counter()
    for j, meta in enumerate(smp["runs"]):
        key = str(meta["index"])
        todo = [a for a in arms if key not in store[a]["records"]]
        if not todo:
            continue
        x, dt = np.asarray(X[meta["index"]], float), float(DT[meta["index"]])
        truth = np.asarray(side[meta["system"]]["xi"], float)
        with threadpool_limits(limits=1):
            if "default" in todo:
                store["default"]["records"][key] = record(meta, call(x, dt, {}, timeout), truth)
            dref = store["default"]["records"][key]
            for a in todo:
                if a == "default":
                    continue
                cc = store[a]["copy_check"]
                elig = copy_eligible(a, dref) and not cc["failed"]
                if elig and cc["verified"] >= verify_copies:
                    rec = dict(dref)
                    rec["copied_from_default"] = True
                    rec["seconds"] = 0.0
                else:
                    rec = record(meta, call(x, dt, ARMS[a], timeout), truth)
                    if elig:
                        same = rec["status"] == "ok" and rec.get("xi") == dref.get("xi") and rec["gap"] == dref["gap"]
                        cc["verified"] += int(same)
                        cc["failed"] |= not same
                        cc.setdefault("checked", []).append({"index": meta["index"], "identical": same})
                store[a]["records"][key] = rec
        for a in todo:
            _save(adir / f"{a}.json", store[a])
        el = time.perf_counter() - t_start
        ex = [f"{a}:{int(store[a]['records'][key]['metrics']['exact'])}" for a in arms]
        print(f"[{j + 1}/{len(smp['runs'])}] {meta['index']} {meta['system']} D={meta['D']} {meta['noise']} "
              f"{meta['n']} default exact {store['default']['records'][key]['metrics']['exact']:.0f} "
              f"run {sum(store[a]['records'][key]['seconds'] for a in todo):.0f}s elapsed {el / 3600:.2f}h | "
              + " ".join(ex), flush=True)
    (Path(outdir) / "DONE").write_text(time.strftime("%F %T"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n-runs", type=int, default=300)
    s.add_argument("--seed", type=int, default=20261027)
    s.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sample", required=True)
    r.add_argument("--outdir", required=True)
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--verify-copies", type=int, default=3)
    r.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    if a.cmd == "sample":
        make_sample(a.n_runs, a.seed, a.output)
    else:
        assert a.arms[0] == "default", "the default arm must come first"
        run(a.sample, a.outdir, a.timeout, a.verify_copies, a.arms)


if __name__ == "__main__":
    main()
