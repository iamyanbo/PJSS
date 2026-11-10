# EXPLORATORY: frozen model on noise-free data simulated from the accepted double-pendulum model (run-1 start)
import sys, json, numpy as np
sys.path[:0] = ["benchmarks", "."]
import real2_pendulum as rp
from scipy.integrate import solve_ivp
root = "data/real2/multiarm_pendulum"
P = rp.double_params(root); Y = rp.double_data(root, 1)
t = np.arange(0, 20, 0.01)
s = solve_ivp(rp.double_rhs(P), (0, t[-1]), Y[0], t_eval=t, rtol=1e-10, atol=1e-12, method="LSODA")
from eqdisc import discover
r = discover(s.y.T, 0.01, library="auto", noise="auto")
print(r.equations()); print("gap", r.gap, "noise", r.info.get("noise_model"))
json.dump({"eqs": r.equations(), "gap": float(r.gap)}, open("results_real2/explore/double_synthetic_eqdisc.json", "w"))
