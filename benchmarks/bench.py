"""Benchmark: generate fixed noisy instances, evaluate methods over a lambda grid.

python bench.py gen          -> data/instances.npz
python bench.py run METHOD   -> results/METHOD.npz   (per-instance, per-lambda metrics)
python bench.py report       -> table: oracle-lambda and globally tuned lambda
"""
import sys
import os
import numpy as np
from multiprocessing import Pool
from eqdisc.core import *

NOISES = [0.01, 0.05, 0.1, 0.2]
LENGTHS = [500, 2000]
SEEDS = 10
LAMS = np.geomspace(0.005, 0.5, 20)
VARIANT = os.environ.get("EQ_VARIANT", "base")  # base | colored | student | coarse
SUFFIX = "" if VARIANT == "base" else f"_{VARIANT}"
RESDIR = "results" if VARIANT == "base" else f"results/{VARIANT}"
if DEG != 3:  # same instances, bigger library
    RESDIR += f"_deg{DEG}"


def make_noise(Xc, eta, rng):
    sd = Xc.std(0)
    if VARIANT == "colored":  # AR(1), rho=0.8, unit marginal variance
        e = rng.standard_normal(Xc.shape)
        for i in range(1, len(e)):
            e[i] = 0.8 * e[i - 1] + 0.6 * e[i]
    elif VARIANT == "student":  # heavy tails, df=3, unit variance
        e = rng.standard_t(3, Xc.shape) / np.sqrt(3.0)
    else:
        e = rng.standard_normal(Xc.shape)
    return Xc + eta * sd * e


def gen():
    rng = np.random.default_rng(1234)
    out = dict(X=[], sys=[], noise=[], n=[], dt=[])
    for name, Xi in SYSTEMS.items():
        ref = simulate(Xi, X0[name], DT[name], 20000, burn=2000)
        for n in LENGTHS:
            for seed in range(SEEDS):
                # random start on the attractor
                x0 = ref[rng.integers(0, len(ref))]
                stride = 4 if VARIANT == "coarse" else 1
                Xc = simulate(Xi, x0, DT[name], n * stride)[::stride]
                assert Xc is not None and np.isfinite(Xc).all(), name
                for eta in NOISES:
                    out["X"].append(make_noise(Xc, eta, rng))
                    out["sys"].append(name); out["noise"].append(eta); out["n"].append(n)
                    out["dt"].append(DT[name] * stride)
    os.makedirs("data", exist_ok=True)
    np.savez(f"data/instances{SUFFIX}.npz", X=np.array(out["X"], dtype=object), sys=out["sys"], noise=out["noise"],
             n=out["n"], dt=out["dt"])
    print(len(out["X"]), "instances")


def load():
    d = np.load(f"data/instances{SUFFIX}.npz", allow_pickle=True)
    return [dict(X=d["X"][i], sys=str(d["sys"][i]), noise=float(d["noise"][i]), n=int(d["n"][i]),
                 dt=float(d["dt"][i])) for i in range(len(d["sys"]))]


# ---- methods: f(inst, lam) -> Xi (original units) ----
def problem(inst, kind, h):
    s = scale_factors(inst["X"])
    Xn = inst["X"] / s
    A, B = fd_problem(Xn, inst["dt"], window=h) if kind == "fd" else weak_problem(Xn, inst["dt"], half_width=h)
    return A, B, s


def problem_h(inst, h):
    s = scale_factors(inst["X"])
    Xn = inst["X"] / s
    A, B = weak_problem(Xn, inst["dt"], half_width=h, sig=estimate_sigma(Xn))
    return A, B, s


def make_method(kind, h, ens=False):
    def f(inst, lams):
        A, B, s = problem_h(inst, h) if kind == "hweak" else problem(inst, kind, h)
        if ens:
            return [xi_from_scaled(esindy(A, B, lam, rng=np.random.default_rng(0)), s) for lam in lams]
        return [xi_from_scaled(stlsq(A, B, lam), s) for lam in lams]
    return f


METHODS = {}
for h in [9, 21, 41]:
    METHODS[f"fd{h}"] = make_method("fd", h)
for h in [10, 20, 40]:
    METHODS[f"weak{h}"] = make_method("weak", h)
    METHODS[f"eweak{h}"] = make_method("weak", h, ens=True)
    METHODS[f"hweak{h}"] = make_method("hweak", h)


def make_sel(crit, h=20):
    from eqdisc.select_k import discover
    return lambda inst, lams: [discover(inst["X"], inst["dt"], half_width=h, criterion=crit)]


for crit in ["bic", "cv", "wbic", "chi2", "chi2s"]:
    METHODS[f"bs_{crit}"] = make_sel(crit)


def _ps_fit(inst, opt, h=20):
    import warnings
    import pysindy as ps
    warnings.filterwarnings("ignore")
    s = scale_factors(inst["X"])
    Xn = inst["X"] / s
    t = np.arange(len(Xn)) * inst["dt"]
    lib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=DEG), spatiotemporal_grid=t,
                            K=max(3 * NLIB, len(Xn) // (h // 2)), p=4, H_xt=[h * inst["dt"]])
    m = ps.SINDy(feature_library=lib, optimizer=opt)
    try:
        m.fit(Xn, t=t)
    except Exception:
        return np.zeros((NLIB, DIM))
    names = [n.replace("x0", "x").replace("x1", "y").replace("x2", "z").replace(" ", "*") for n in m.get_feature_names()]
    Xu = np.zeros((NLIB, DIM))
    C = m.coefficients()
    for k, nm in enumerate(names):
        Xu[IDX[nm]] = C[:, k]
    return xi_from_scaled(Xu, s)


def ps_weak(inst, lams):
    import pysindy as ps
    return [_ps_fit(inst, ps.STLSQ(threshold=th, alpha=1e-8)) for th in np.geomspace(0.01, 10, 20)]


def ps_miosr(inst, lams):
    import pysindy as ps
    return [_ps_fit(inst, ps.MIOSR(target_sparsity=k, alpha=1e-6, normalize_columns=True)) for k in range(3, 15)]


def ps_miosr_k(inst, lams):  # oracle number of terms per equation (upper bound)
    import pysindy as ps
    k = tuple(int(v) for v in (SYSTEMS[inst["sys"]] != 0).sum(0))
    return [_ps_fit(inst, ps.MIOSR(group_sparsity=k, alpha=1e-6, normalize_columns=True))]


METHODS.update(ps_weak=ps_weak, ps_miosr=ps_miosr, ps_miosr_k=ps_miosr_k)


def _eval(args):
    name, i = args
    inst = INSTS[i]
    Xis = METHODS[name](inst, LAMS)
    return [list(metrics(Xi, SYSTEMS[inst["sys"]]).values()) for Xi in Xis]


def run(name):
    with Pool(12, initializer=_init) as p:
        res = p.map(_eval, [(name, i) for i in range(len(load()))], chunksize=4)
    os.makedirs(RESDIR, exist_ok=True)
    np.save(f"{RESDIR}/{name}.npy", np.array(res))  # (n_inst, n_lam, 5): exact, f1, coef_err, fp, fn


def _init():
    global INSTS
    INSTS = load()


def report(names=None):
    insts = load()
    noise = np.array([d["noise"] for d in insts]); n = np.array([d["n"] for d in insts])
    names = names or sorted(f[:-4] for f in os.listdir(RESDIR) if f.endswith(".npy"))
    rows = []
    for name in names:
        if not os.path.exists(f"{RESDIR}/{name}.npy"):
            continue
        R = np.load(f"{RESDIR}/{name}.npy")
        if R.ndim == 2:  # lambda-free method
            R = R[:, None, :]
        best = R[:, :, 0].mean(0).argmax() if R.shape[1] > 1 else 0  # global lambda tuned for exact recovery
        orc = (R[:, :, 0] * 10 + R[:, :, 1]).argmax(1)  # oracle per instance (exact first, then F1)
        line = f"{name:14s}"
        for eta in NOISES:
            for nn in LENGTHS:
                m = (noise == eta) & (n == nn)
                g = R[m, best]
                o = R[m, orc[m]]
                line += f" | {g[:, 0].mean():.2f}/{g[:, 1].mean():.2f} ({o[:, 0].mean():.2f})"
        rows.append(line)
    hdr = f"{'method':14s}" + "".join(f" | eta={e:<4} N={nn:<5}  " for e in NOISES for nn in LENGTHS)
    print(f"[{VARIANT}] cells: exact/F1 at globally tuned lambda (exact at per-instance oracle lambda)")
    print(hdr)
    print("\n".join(rows))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen()
    elif cmd == "run":
        for nm in sys.argv[2:]:
            run(nm); print("done", nm, flush=True)
    elif cmd == "report":
        report(sys.argv[2:] or None)
