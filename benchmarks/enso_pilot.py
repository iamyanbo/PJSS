"""El Nino pilot: does the frozen model find the recharge oscillator in monthly reanalysis indices?
Protocol in results_enso/PREREGISTRATION.md. eqdisc/ is not modified.

    python benchmarks/enso_pilot.py control --seeds 20 --output results_enso/control.json
    python benchmarks/enso_pilot.py real --output results_enso/real.json

Needs xarray, netCDF4 and XRO (pypi XRO 1.0.4), which are only used to read the data and for the comparators.
Time is in years, T = Nino34 (deg C), h = WWV (m). Arms:
E2  states (T, h), degree-2 library;
E4  states (T, h, c, s), c = cos(2 pi t), s = sin(2 pi t): the seasonal cycle embedded as two more polynomial states,
    with white noise of sd JITTER on c and s.
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

DATA = HERE.parent / "data" / "enso" / "XRO_indices_oras5.nc"
DT = 1 / 12
OM = 2 * np.pi
TRAIN = ("1979-01", "2011-12")
JITTER = 0.1
JITTER_SEEDS = 5


def load():
    import xarray as xr
    return xr.open_dataset(DATA)[["Nino34", "WWV"]].load()


def fit_ro(ds):
    """Comparator XRO-RO: seasonal linear two-variable recharge oscillator."""
    from XRO import XRO
    m = XRO(ncycle=12, ac_order=2)
    return m, m.fit_matrix(ds, maskb=[], maskNT=[])


def embed(y, t0, arm, jitter=0.0, seed=0):
    """jitter: sd of white noise on the clock states; the noise-scale estimate fails on exactly noise-free states."""
    if arm == "E2":
        return y
    t = t0 + np.arange(len(y)) * DT
    rng = np.random.default_rng(seed)
    return np.column_stack([y, np.cos(OM * t) + jitter * rng.standard_normal(len(t)),
                            np.sin(OM * t) + jitter * rng.standard_normal(len(t))])


def run_eqdisc(y, arm, t0=0.0, seed=0):
    from eqdisc import discover
    tic = time.time()
    r = discover(embed(y, t0, arm, JITTER, seed), DT, confidence=False)
    return np.asarray(r.xi, float), [str(n) for n in r.names], time.time() - tic


def run_bic(y, arm, t0=0.0, kmax=5):
    """Comparator: exhaustive best-subset regression of forward differences per equation, BIC pick (exact clock)."""
    from eqdisc.joint_nd import PolyLib
    z = embed(y, t0, arm)
    lib = PolyLib(z.shape[1], 2)
    A, B = lib(z[:-1]), (z[1:] - z[:-1]) / DT
    n = len(A)
    G, Xi = A.T @ A, np.zeros((lib.M, z.shape[1]))
    for j in range(z.shape[1]):
        g, yy, best = A.T @ B[:, j], B[:, j] @ B[:, j], (np.inf, None, None)
        for k in range(1, kmax + 1):
            for sup in itertools.combinations(range(lib.M), k):
                ix = list(sup)
                try:
                    c = np.linalg.solve(G[np.ix_(ix, ix)], g[ix])
                except np.linalg.LinAlgError:  # c^2 + s^2 = 1 with the exact clock
                    continue
                bic = n * np.log(max(yy - c @ g[ix], 1e-300) / n) + k * np.log(n)
                if bic < best[0]:
                    best = (bic, ix, c)
        Xi[best[1], j] = best[2]
    return with_exact_clock(Xi, [str(v) for v in lib.names]), [str(v) for v in lib.names]


def with_exact_clock(Xi, names):
    """Replace the clock equations by dc = -2 pi s, ds = 2 pi c (forward differences distort them)."""
    Xi = Xi.copy()
    if Xi.shape[1] == 4:
        Xi[:, 2:] = 0
        Xi[names.index("x3"), 2], Xi[names.index("x2"), 3] = -OM, OM
    return Xi


def core(Xi, names):
    """Recharge-oscillator core: dT contains +h and dh contains -T (seasonal modulation and extras allowed)."""
    i0, i1 = names.index("x0"), names.index("x1")
    F1, F2 = float(Xi[i1, 0]), -float(Xi[i0, 1])
    return dict(F1=F1, F2=F2, R=float(Xi[i0, 0]), eps=-float(Xi[i1, 1]), core=bool(F1 > 0 and F2 > 0),
                k=[int(np.count_nonzero(Xi[:, j])) for j in range(Xi.shape[1])])


def clock_ok(Xi, names):
    """True iff dc = -2 pi s and ds = 2 pi c within 10%, each with no other term."""
    if Xi.shape[1] != 4:
        return False
    ic, is_ = names.index("x2"), names.index("x3")
    a, b = Xi[:, 2], Xi[:, 3]
    return bool(np.count_nonzero(a) == 1 and np.count_nonzero(b) == 1 and abs(a[is_] + OM) < 0.1 * OM
                and abs(b[ic] - OM) < 0.1 * OM)


def describe(Xi, names):
    return [{names[i]: float(Xi[i, j]) for i in np.flatnonzero(Xi[:, j])} for j in range(Xi.shape[1])]


# ------------------------------------------------------------------ forecasts
def poly_forecast(Xi, arm, y0, t0, leads=18, sub=30):
    """Integrate the discovered polynomial ODE (RK4) from an observed state; T for leads 1..leads.
    A run with |T| > 10 stops; its remaining leads are 0 (climatology), and the second return value is True."""
    from eqdisc.joint_nd import PolyLib
    x = embed(np.asarray(y0, float)[None], t0, arm)[0]
    lib = PolyLib(len(x), 2)
    f = lambda v: lib(v[None])[0] @ Xi
    h, out = DT / sub, np.zeros(leads)
    for m in range(leads):
        for _ in range(sub):
            k1 = f(x)
            k2 = f(x + h / 2 * k1)
            k3 = f(x + h / 2 * k2)
            k4 = f(x + h * k3)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all() or abs(x[0]) > 10:
            return out, True
        out[m] = x[0]
    return out, False


def skill(F, obs_t, inits, leads=18, block=24, nboot=2000, ref=None, seed=0):
    """F: (inits, leads) forecasts. ACC and RMSE per lead; with ref, a moving-block bootstrap of the mean-ACC gap."""
    V = np.column_stack([obs_t[inits + l] for l in range(1, leads + 1)])
    acc = lambda a, b, ix: np.mean([np.corrcoef(a[ix, l], b[ix, l])[0, 1] for l in range(leads)])
    out = dict(acc=[float(np.corrcoef(F[:, l], V[:, l])[0, 1]) for l in range(leads)],
               rmse=[float(np.sqrt(np.mean((F[:, l] - V[:, l]) ** 2))) for l in range(leads)])
    out["mean_acc"] = float(np.mean(out["acc"]))
    if ref is not None:
        rng, n = np.random.default_rng(seed), len(F)
        gaps = []
        for _ in range(nboot):
            ix = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, n // block + 1)])[:n]
            gaps.append(acc(F, V, ix) - acc(ref, V, ix))
        out["gap_ci"] = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
    return out


def xro_forecast(ds_all, train, inits, leads=18, full=False):
    from XRO import XRO
    m = XRO(ncycle=12, ac_order=2)
    fit = m.fit_matrix(train, maskb=["IOD"], maskNT=["T2", "TH"]) if full else m.fit_matrix(train, maskb=[], maskNT=[])
    f = m.reforecast(fit_ds=fit, init_ds=ds_all.isel(time=inits), n_month=leads + 1, ncopy=1, noise_type="zero")
    return np.squeeze(f["Nino34"].sel(lead=list(range(1, leads + 1))).transpose("init", "lead", ...).values)


def evaluate(models, ds_two, ds_ten, t_train_end):
    """models: {name: (Xi, arm)}. Inits are every month after training with all 18 leads inside the record."""
    leads = 18
    T = ds_two.Nino34.values.astype(float)
    Y = np.column_stack([T, ds_two.WWV.values.astype(float)])
    inits = np.arange(t_train_end, len(T) - leads)
    tt = np.arange(len(T)) * DT
    train2, train10 = ds_two.isel(time=slice(0, t_train_end)), ds_ten.isel(time=slice(0, t_train_end))
    ref = xro_forecast(ds_two, train2, inits)
    fc = {"persistence": np.tile(T[inits][:, None], (1, leads)), "xro_ro": ref,
          "xro_full": xro_forecast(ds_ten, train10, inits, full=True)}
    div = {}
    for name, (Xi, arm) in models.items():
        r = [poly_forecast(Xi, arm, Y[i], tt[i], leads) for i in inits]
        fc[name] = np.array([a for a, _ in r])
        div[name] = int(sum(b for _, b in r))
    S = {k: skill(v, T, inits, leads, ref=None if k == "xro_ro" else ref) for k, v in fc.items()}
    return dict(n_inits=len(inits), diverged=div, skill=S)


# ------------------------------------------------------------------ commands
def simulate(m, f, ds, seed):
    s = m.simulate(fit_ds=f, X0_ds=ds.isel(time=0), nyear=38, ncopy=1, is_xi_stdac=False, seed=seed)
    return np.column_stack([s.Nino34.values.ravel(), s.WWV.values.ravel()])[60:]


def lotka(seed, n=396, eta=0.1):
    """Deterministic 3.9-year cycle with white measurement noise: the assumptions of the method hold."""
    from scipy.integrate import solve_ivp
    fn = lambda t, v: [1.6 * v[0] * (1 - v[1]), 1.6 * v[1] * (v[0] - 1)]
    x = solve_ivp(fn, (0, (n - 1) * DT), [1.5, 0.7], t_eval=np.arange(n) * DT, rtol=1e-10, atol=1e-10).y.T
    return x + eta * np.sqrt(np.mean(x ** 2, 0)) * np.random.default_rng(seed).standard_normal(x.shape)


def control_job(a):
    kind, seed = a
    if kind == "lotka":
        Xi, names, sec = run_eqdisc(lotka(seed), "E2")
        lib = {(0, "x0"): 1.6, (0, "x0*x1"): -1.6, (1, "x1"): -1.6, (1, "x0*x1"): 1.6}
        got = {(int(j), names[i]): Xi[i, j] for i, j in zip(*np.nonzero(Xi))}
        return dict(kind=kind, seed=seed, exact=set(got) == set(lib),
                    max_err=float(max(abs(got.get(k, 0) / v - 1) for k, v in lib.items())), seconds=sec)
    ds = load().sel(time=slice(*TRAIN))
    m, f = fit_ro(ds)
    y = simulate(m, f, ds, seed)
    Xi, names, sec = run_eqdisc(y, "E4")
    Xb, nb = run_bic(y, "E4")
    return dict(kind=kind, seed=seed, eqdisc=core(Xi, names), eqdisc_clock=clock_ok(Xi, names),
                eqdisc_terms=describe(Xi, names), bic=core(Xb, nb),
                bic_terms=describe(Xb, nb), seconds=sec)


def cmd_control(a):
    from multiprocessing import Pool
    jobs = [("ro", 100 + s) for s in range(1, a.seeds + 1)] + [("lotka", 100 + s) for s in range(1, 11)]
    with Pool(a.procs) as p:
        rows = p.map(control_job, jobs, chunksize=1)
    ro = [r for r in rows if r["kind"] == "ro"]
    lv = [r for r in rows if r["kind"] == "lotka"]
    summ = dict(n=len(ro), eqdisc_core=sum(r["eqdisc"]["core"] for r in ro), bic_core=sum(r["bic"]["core"] for r in ro),
                eqdisc_clock=sum(r["eqdisc_clock"] for r in ro),
                lotka_n=len(lv), lotka_exact=sum(r["exact"] for r in lv))
    print(summ)
    Path(a.output).write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))


def real_job(a):
    arm, seed = a
    ds = load().sel(time=slice(*TRAIN))
    y = np.column_stack([ds.Nino34.values, ds.WWV.values]).astype(float)
    Xi, names, sec = run_eqdisc(y, arm, seed=seed)
    return dict(arm=arm, seed=seed, core=core(Xi, names), clock=clock_ok(Xi, names), terms=describe(Xi, names),
                seconds=sec, xi=Xi.tolist(), names=names)


def cmd_real(a):
    import xarray as xr
    from multiprocessing import Pool
    ten = xr.open_dataset(DATA).load()
    two = ten[["Nino34", "WWV"]]
    nt = two.sel(time=slice(*TRAIN)).sizes["time"]
    y = np.column_stack([two.Nino34.values[:nt], two.WWV.values[:nt]]).astype(float)
    with Pool(a.procs) as p:
        rows = p.map(real_job, [("E4", s) for s in range(JITTER_SEEDS)] + [("E2", 0)], chunksize=1)
    Xb, nb = run_bic(y, "E4")
    m, f = fit_ro(two.isel(time=slice(0, nt)))
    res = dict(rows=rows, bic=dict(core=core(Xb, nb), terms=describe(Xb, nb)),
               xro_ro_annual_L=f.Lac.mean("cycle").values.tolist())
    models = {f"eqdisc_E4_s{r['seed']}": (np.array(r["xi"]), "E4") for r in rows if r["arm"] == "E4"}
    names = rows[0]["names"]
    models.update({f"eqdisc_E4_s{r['seed']}_clock": (with_exact_clock(np.array(r["xi"]), names), "E4")
                   for r in rows if r["arm"] == "E4"})
    models["eqdisc_E2"] = (np.array(rows[-1]["xi"]), "E2")
    models["bic_E4"] = (Xb, "E4")
    res["forecast"] = evaluate(models, two, ten, nt)
    Path(a.output).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["control", "real"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    (cmd_control if a.cmd == "control" else cmd_real)(a)
