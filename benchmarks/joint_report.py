"""Offline selection rules over joint_bench candidates.

Rules (score = lower is better, over a run's candidates at one rho):
  bic      data + model + val + k log(3n)      (the rule built into joint_discover)
  aic      data + model + val + 2k
  val      val only
  valbic   val + k log(3n)
'auto' picks rho per run by the held-out misfit of each rho's BIC winner.
"""
import argparse
import json

import numpy as np

RULES = ("bic", "aic", "val", "valbic")


def pick(cands, rule, n, phi=1.0):
    pen = {"bic": np.log(3 * n), "aic": 2.0, "val": 0.0, "valbic": np.log(3 * n)}[rule]
    base = (lambda c: c["val"]) if rule in ("val", "valbic") else (lambda c: c["data"] + c["model"] + c["val"])
    return min(cands, key=lambda c: base(c) + phi * pen * c["k"])


def evaluate(records, rho, rule):
    out = []
    for r in records:
        if rho == "auto":
            best = None
            for j in r["joint"].values():
                if "candidates" in j and j["candidates"]:
                    c = pick(j["candidates"], "bic", r["n"], j.get("phi", 1.0))
                    if best is None or c["val"] < best["val"]:
                        best = c
            out.append(best["exact"] if best else 0.0)
        else:
            j = r["joint"].get(str(rho), {})
            out.append(pick(j["candidates"], rule, r["n"], j.get("phi", 1.0))["exact"] if j.get("candidates") else 0.0)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    records = [r for f in args.files for r in json.load(open(f))["records"]]
    rhos = sorted({k for r in records for k in r["joint"]}, key=float)
    cols = ["weak", "flow", "hyb_bic10"] + [f"j{rho}" for rho in rhos] + ["jauto", "oracle_pool"]

    def row(rs):
        vals = [np.mean([r.get(m, {}).get("exact", 0) for r in rs]) for m in ("weak", "flow", "hyb_bic10")]
        vals += [evaluate(rs, rho, "bic").mean() for rho in rhos] + [evaluate(rs, "auto", "bic").mean()]
        vals.append(np.mean([any(c["exact"] for j in r["joint"].values() for c in j.get("candidates", []))
                             for r in rs]))
        return vals

    print("cell            " + " ".join(f"{c:>9}" for c in cols))
    for key in sorted({(r["noise"], r["n"]) for r in records}):
        rs = [r for r in records if (r["noise"], r["n"]) == key]
        print(f"{key[0]:<5} {key[1]:<5} n={len(rs):<3}" + " ".join(f"{v:9.2f}" for v in row(rs)))
    print("all             " + " ".join(f"{v:9.3f}" for v in row(records)))
    print()
    for s in sorted({r["system"] for r in records}):
        rs = [r for r in records if r["system"] == s]
        print(f"{s:15} " + " ".join(f"{v * len(rs):9.0f}" for v in row(rs)) + f"  /{len(rs)}")
    print("\nrule x rho (all):")
    for rule in RULES:
        print(f"  {rule:7}" + " ".join(f" {rho}={evaluate(records, rho, rule).mean():.3f}" for rho in rhos))
    secs = [j["seconds"] for r in records for j in r["joint"].values() if "seconds" in j]
    print(f"joint seconds per rho: median {np.median(secs):.1f}, max {np.max(secs):.1f}; "
          f"errors {sum('error' in j for r in records for j in r['joint'].values())}")


if __name__ == "__main__":
    main()
