"""Tables for nd_bench results: per system x noise x N, weak-BIC, STLSQ (dev-chosen lambda and oracle), joint."""
import json
import sys

import numpy as np

LAMS = ("0.02", "0.05", "0.1", "0.2", "0.5", "1.0")


def main(path, lam_fixed=None):
    R = json.load(open(path))["records"]
    sl = {l: np.mean([r["stlsq"][l]["exact"] for r in R]) for l in LAMS}
    lam = lam_fixed or max(sl, key=sl.get)
    print(f"{len(R)} runs; STLSQ overall by lambda: " + " ".join(f"{l}:{v:.3f}" for l, v in sl.items())
          + f"  -> fixed lambda {lam}")
    hdr = f"{'system':15} {'D':>2} {'runs':>4} {'weakBIC':>7} {'STLSQ':>6} {'STLSQor':>7} {'joint':>6} {'W/L':>6} " \
          f"{'flag%':>5} {'acc|unfl':>8} {'err%':>5} {'s/run':>6}"
    print(hdr)
    for s in sorted({r["system"] for r in R}, key=lambda s: (R[[r["system"] for r in R].index(s)]["D"], s)):
        rs = [r for r in R if r["system"] == s]
        j = np.array([r["joint"].get("exact", 0) for r in rs])
        st = np.array([r["stlsq"][lam]["exact"] for r in rs])
        orc = np.array([max(r["stlsq"][l]["exact"] for l in LAMS) for r in rs])
        w = np.array([r["weak_bic"]["exact"] for r in rs])
        best = np.maximum(st, w)
        fl = np.array([(r.get("info", {}).get("gap") or 0) < 2 for r in rs])
        err = [r["joint"]["coef_err"] for r in rs if r["joint"].get("exact")]
        print(f"{s:15} {rs[0]['D']:2d} {len(rs):4d} {w.mean():7.3f} {st.mean():6.3f} {orc.mean():7.3f} {j.mean():6.3f} "
              f"{int((j > best).sum()):3d}/{int((j < best).sum()):<3d}{fl.mean() * 100:5.0f} "
              f"{(j[~fl].mean() if (~fl).any() else float('nan')):8.3f} "
              f"{100 * np.median(err) if err else float('nan'):5.2f} {np.median([r['joint_seconds'] for r in rs]):6.1f}")
    print("\nby noise x N (weakBIC / STLSQ / joint):")
    for eta in sorted({r["noise"] for r in R}):
        line = f"  {eta:<5}"
        for n in sorted({r["n"] for r in R}):
            rs = [r for r in R if r["noise"] == eta and r["n"] == n]
            line += f"  N={n}: {np.mean([r['weak_bic']['exact'] for r in rs]):.2f} " \
                    f"{np.mean([r['stlsq'][lam]['exact'] for r in rs]):.2f} {np.mean([r['joint'].get('exact', 0) for r in rs]):.2f}"
        print(line)
    j = np.array([r["joint"].get("exact", 0) for r in R])
    print(f"\nall: weakBIC {np.mean([r['weak_bic']['exact'] for r in R]):.3f}  STLSQ(lam={lam}) "
          f"{np.mean([r['stlsq'][lam]['exact'] for r in R]):.3f}  STLSQ oracle-lam "
          f"{np.mean([max(r['stlsq'][l]['exact'] for l in LAMS) for r in R]):.3f}  joint {j.mean():.3f}; "
          f"errors {sum('error' in r['joint'] for r in R)}")
    print("median joint seconds by D:", {D: round(float(np.median([r['joint_seconds'] for r in R if r['D'] == D])), 1)
                                          for D in sorted({r['D'] for r in R})})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
