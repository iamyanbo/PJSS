"""Frozen dysts test archives for the in-library systems found by dysts_classify.py (main env; dysts not needed).

    python dysts_generate.py --classes data/dysts_classes.json --output data/dysts_test.npz  --seed 20261401
    python dysts_generate.py --classes data/dysts_classes.json --output data/dysts_smoke.npz --seed 20261400 --smoke

Protocol (same as heldout_systems.generate / nd_systems.generate, i.e. the sealed-archive protocol):
  * truth = dysts' default parameters, read symbolically and verified to machine precision by dysts_classify;
    trajectories are integrated here from that coefficient matrix (LSODA, rtol = atol = 1e-10) in the raw ODE
    state (dysts' optional `_postprocessing` coordinate maps are NOT applied);
  * dt by our rule: dt = 1/(128 f_peak) on a 1-2-5 grid, f_peak = max over components of the spectral peak of a
    long clean trajectory (this rule reproduces 18/21 of our hand-set DT values; the other three differ by one
    grid step). dysts' own recommendation would be period/100 with its 'period' field; it is stored for reference;
  * starts: reference trajectory of 10000 samples after a burn-in of 100 dominant periods from dysts' initial
    condition; each run starts at a uniformly random reference sample;
  * per seed and per N in {500, 2000}: one clean trajectory, Gaussian noise eta * per-component std for
    eta in {0.01, 0.05, 0.1, 0.2} (same clean trajectory for all four noise levels, as before);
  * systems whose reference run collapses to an equilibrium or whose finite-time largest Lyapunov exponent
    times the dominant period is <= --min-lyap (0.01) are excluded (listed in the sidecar);
  * a run whose clean trajectory fails to integrate, is non-finite or leaves 10x the reference range is
    discarded and a new start is drawn (counted in the sidecar as `discarded`).
--smoke: one run per system, (noise, N) cycling over the 8 cells so runtimes cover all cells.

Sidecar <output>.json: per system truth coefficients (library ordering + names), D, degree, dt, class,
overlap with our SYSTEMS / HELDOUT / nd_systems, coefficient range, smallest relative term contribution, discards.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

NOISE = (0.01, 0.05, 0.1, 0.2)
LENGTHS = (500, 2000)


def library_for(D):
    if D == 3:
        from eqdisc.core import EXP_ARR, NAMES
        return np.array(EXP_ARR), list(NAMES)
    from eqdisc.joint_nd import PolyLib
    L = PolyLib(D, 2)
    return np.array(L.E), list(L.names)


def lib_eval(X, E):
    return np.prod(X[..., None, :] ** E, axis=-1)


def simulate(Xi, E, x0, dt, n, burn=0):
    f = lambda t, x: lib_eval(x[None], E)[0] @ Xi
    tt = np.arange(burn + n) * dt
    with np.errstate(all="ignore"):
        sol = solve_ivp(f, (0, tt[-1]), np.asarray(x0, float), t_eval=tt, method="LSODA", rtol=1e-10, atol=1e-10)
    if not sol.success or sol.y.shape[1] != len(tt) or not np.isfinite(sol.y).all():
        return None
    return sol.y.T[burn:]


def nice(v):
    e = np.floor(np.log10(v))
    m = v / 10 ** e
    g = min((1, 2, 5, 10), key=lambda q: abs(np.log(m / q)))
    return float(f"{g * 10 ** e:.3g}")


def f_peak(x, dt):
    f = []
    for i in range(x.shape[1]):
        s = x[:, i] - x[:, i].mean()
        p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), dt)
        p[:3] = 0
        f.append(fr[np.argmax(p)])
    return max(f)


def prepare(name, r):
    D = r["D"]
    E, names = library_for(D)
    assert names == r["names"], f"{name}: library ordering mismatch"
    Xi = np.array(r["xi"])
    per = r["period"] or 1.0
    fine = per / 512
    long = simulate(Xi, E, r["ic"], fine, 64 * 512, burn=int(50 * 512))
    if long is None:
        raise RuntimeError(f"{name}: long clean run failed")
    dt = nice(1 / (128 * f_peak(long, fine)))
    ref = simulate(Xi, E, long[-1], dt, 10000, burn=int(np.ceil(100 * per / dt)))
    if ref is None:
        raise RuntimeError(f"{name}: reference run failed")
    # smallest relative contribution of a true term: |c| rms(term) / rms(rhs_j) on the reference attractor
    Th = lib_eval(ref, E)
    dX = Th @ Xi
    rc = [float(abs(Xi[m, j]) * np.sqrt(np.mean(Th[:, m] ** 2)) / np.sqrt(np.mean(dX[:, j] ** 2)))
          for m, j in zip(*np.nonzero(Xi))]
    # stiffness proxy at our dt: max |eig J| * dt over the reference attractor
    ev = []
    for y in ref[::50]:
        J = np.zeros((D, D))
        for l in range(D):
            e2 = E.copy()
            a = e2[:, l].astype(float)
            e2[:, l] = np.maximum(e2[:, l] - 1, 0)
            J[:, l] = (a * np.prod(y[None] ** e2, axis=-1)) @ Xi
        ev.append(np.abs(np.linalg.eigvals(J)).max())
    # finite-time largest Lyapunov exponent (two nearby trajectories, renormalised every dominant period)
    a = ref[-1]
    rng = np.random.default_rng(0)
    b = a + 1e-8 * rng.standard_normal(D)
    unit = max(int(round(per / dt)), 2)
    acc, T = 0.0, 0.0
    for _ in range(100):
        ta, tb = simulate(Xi, E, a, dt, unit + 1), simulate(Xi, E, b, dt, unit + 1)
        if ta is None or tb is None:
            break
        a, b = ta[-1], tb[-1]
        dd = max(np.linalg.norm(b - a), 1e-300)
        acc += np.log(dd / 1e-8)
        T += unit * dt
        b = a + (b - a) * (1e-8 / dd)
    lyap = acc / T if T else float("nan")
    collapsed = bool((ref.std(0) < 1e-6 * (np.abs(ref).max() + 1e-12)).any())
    return dict(E=E, Xi=Xi, dt=dt, ref=ref, min_rel_contrib=min(rc), names=names, stiff=float(max(ev) * dt),
                lyap=float(lyap), lyap_per_period=float(lyap * per), collapsed=collapsed)


# dysts name -> our system, with how close the dysts instance is to ours (checked by hand from the coefficients)
OVERLAP = {
    # identical up to rounding of parameters (max |coef diff| <= 3.3e-4, Lorenz beta 2.667 vs 8/3)
    **{k: ("identical", v) for k, v in {
        "Lorenz": "lorenz", "Rossler": "rossler", "Chen": "chen", "Halvorsen": "halvorsen", "SprottA": "sprottA",
        "SprottB": "sprottB", "SprottC": "sprottC", "SprottD": "sprottD", "SprottE": "sprottE",
        "GenesioTesi": "genesio", "Dadras": "dadras", "SprottF": "sprottF", "SprottG": "sprottG",
        "SprottH": "sprottH", "Rucklidge": "rucklidge", "ChenLee": "chenlee", "HyperRossler": "hyper_rossler"}.items()},
    # same system and support, different default parameters
    **{k: ("same_system_other_params", v) for k, v in {
        "BurkeShaw": "burkeshaw", "LuChen": "lu", "RabinovichFabrikant": "rabinovich", "RikitakeDynamo": "rikitake",
        "Hadley": "hadley", "ShimizuMorioka": "shimizu"}.items()},
    # same family, different dimension (ours: D = 5, 6, 10, F = 8; dysts: D = 4, F = 20)
    "Lorenz96": ("same_family_other_D", "l96_5/6/10"),
}


def overlap_kind(name, r):
    if name in OVERLAP:
        return OVERLAP[name]
    if r.get("same_support_as"):
        return ("support_twin", "/".join(r["same_support_as"]))
    return (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="data/dysts_classes.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--min-lyap", type=float, default=0.01, help="exclude if finite-time lyap * period <= this")
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    C = json.load(open(args.classes))
    rng = np.random.default_rng(args.seed)
    arch = {k: [] for k in ("X", "sys", "noise", "n", "dt")}
    side = {"dysts_version": C["dysts_version"], "seed": args.seed, "smoke": args.smoke,
            "noise": list(NOISE), "lengths": list(LENGTHS), "seeds": 1 if args.smoke else args.seeds,
            "protocol": "see dysts_generate.py docstring", "systems": {}}
    names = [n for n, r in C["systems"].items() if r["cls"] in ("in3", "innd") and r.get("verify") == "ok"]
    side["excluded"] = {}
    if args.systems:
        names = [n for n in names if n in args.systems]
    cells = [(eta, L) for L in LENGTHS for eta in NOISE]
    for k, name in enumerate(names):
        r = C["systems"][name]
        P = prepare(name, r)
        if P["collapsed"] or not P["lyap_per_period"] > args.min_lyap:
            why = "collapses to an equilibrium" if P["collapsed"] else f"lyap*period {P['lyap_per_period']:.3g}"
            side.setdefault("excluded", {})[name] = why
            print(f"{name:20} EXCLUDED: {why}", flush=True)
            continue
        lo, hi = P["ref"].min(0), P["ref"].max(0)
        span = hi - lo
        discarded = 0

        def draw(L):
            nonlocal discarded
            for _ in range(20):
                clean = simulate(P["Xi"], P["E"], P["ref"][rng.integers(len(P["ref"]))], P["dt"], L)
                if clean is not None and (clean > hi + 10 * span).sum() == 0 and (clean < lo - 10 * span).sum() == 0:
                    return clean
                discarded += 1
            raise RuntimeError(f"{name}: 20 consecutive discarded runs")

        def push(x, eta, L):
            arch["X"].append(x); arch["sys"].append(name); arch["noise"].append(eta)
            arch["n"].append(L); arch["dt"].append(P["dt"])

        if args.smoke:
            eta, L = cells[k % len(cells)]
            clean = draw(L)
            push(clean + eta * clean.std(0) * rng.standard_normal(clean.shape), eta, L)
        else:
            for _ in range(args.seeds):
                for L in LENGTHS:
                    clean = draw(L)
                    for eta in NOISE:
                        push(clean + eta * clean.std(0) * rng.standard_normal(clean.shape), eta, L)
        side["systems"][name] = {
            "class": r["cls"], "D": r["D"], "degree": r["degree"], "n_terms": r["n_terms"],
            "names": P["names"], "xi": P["Xi"].tolist(), "dt": P["dt"], "dysts_period": r["period"],
            "dysts_recommended_dt": r["period"] / 100 if r["period"] else None,
            "dysts_params": r["params"], "citation": r["citation"], "lyap_max_dysts": r["lyap_max"],
            "source": f"dysts {C['dysts_version']} flows.{name} (default parameters; raw ODE state, no postprocessing)",
            "postprocessing_in_dysts": r["postprocessing"], "coef_range": r["coef_range"],
            "min_rel_contrib": P["min_rel_contrib"], "stiff_eig_dt_at_our_dt": P["stiff"],
            "overlap": overlap_kind(name, r)[0], "overlap_with": overlap_kind(name, r)[1],
            "same_support_as": r.get("same_support_as"), "discarded": discarded,
            "lyap_ours": P["lyap"], "lyap_per_period_ours": P["lyap_per_period"],
            "verify_relerr": r["verify_relerr"]}
        print(f"{name:20} D={r['D']} dt={P['dt']:<7} lyap*T={P['lyap_per_period']:.3f} minrc={P['min_rel_contrib']:.2e} stiff={P['stiff']:.2f} discarded={discarded}",
              flush=True)
    X = np.empty(len(arch["X"]), object)
    X[:] = arch["X"]
    np.savez(out, X=X, **{k: np.array(v) for k, v in arch.items() if k != "X"})
    out.with_suffix(".json").write_text(json.dumps(side, indent=1))
    print(f"saved {out}: {len(X)} runs, {len(side['systems'])} systems ({len(side['excluded'])} excluded)")


if __name__ == "__main__":
    main()
