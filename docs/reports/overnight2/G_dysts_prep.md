# Track G: dysts benchmark (prepared, not run)

**Status (2026-10-25):** the archive is frozen and the scripts are smoke-tested on the PC. The full test set has **not** been run on any method.

## Source

- **Package:** `dysts` **0.96** (William Gilpin's chaotic-systems database, from pip), in a separate venv `.venv_dysts` (numpy 2.5.4, scipy 1.18.1, sympy 1.14.0). Numba is not installed, so dysts runs as plain Python. That is what makes the symbolic step possible.
- **Which venv needs dysts:** only `dysts_classify.py`. The generator, bench and report use the project env, and truth comes from the JSON sidecar.
- **Systems:** dysts 0.96 lists 135 continuous-time systems (`get_attractor_list()`; maps are separate and not counted).

## Classification

`dysts_classify.py` → `data/dysts_classes.json`, log in `results_dysts/classify.log`.

- **Symbolic step:** `rhs(X, t)` is called with sympy symbols, using dysts' default parameters, then expanded. Delay systems are classified from metadata.
- **Polynomial test:** every component must be polynomial in the state and free of `t`. Any sympy failure counts as non-polynomial (np.sin/tanh/exp/heaviside on a symbol, or branching on a symbol).
- **Numerical cross-check:** a degree-≤4 least-squares fit to dysts' own rhs on a clean dysts trajectory. It agrees with the symbolic degree for every non-delay polynomial system. The one false "polynomial" is Torus, whose rhs depends on t only; the fit hit degree 1 by coincidence, and it is classified non-autonomous.
- **Verification:** for each in-library system, the library is integrated (LSODA, tol 1e-10) and dysts' rhs is evaluated on those points. Max relative error is ≤ 7e-16 for all 78 systems (`verify_relerr` in the JSON).

| Class | Count |
| --- | --- |
| In library, 3-D method (D = 3, polynomial, degree ≤ 3, 20-term core library) | **62** |
| In library, joint_nd (4 ≤ D ≤ 10, polynomial, degree ≤ 2) | **16** (15 with D = 4, LorenzCoupled with D = 6) |
| Polynomial but outside | **4**: Aizawa (D = 3, deg 4), LorenzBounded (D = 3, deg 4), NuclearQuadrupole (D = 4, deg 3), Qi (D = 4, deg 3) |
| Non-polynomial | **53**: 16 non-autonomous or forced, 12 rational/abs/fractional (e.g. Chua, Blasius, HastingsPowell), 6 delay, 5 trig (Thomas, ThomasLabyrinth, ABC, DoublePendulum, SwingingAtwood), 5 tanh/cosh (Hopfield, WindmiReduced, YuWang2, BickleyJet, SanUmSrisuchinwong), 5 piecewise/sign/branch (Colpitts, AnishchenkoAstakhov, BelousovZhabotinsky, MacArthur, SprottMore), 4 exp (BeerRNN, ExcitableCell, JerkCircuit, YuWang) |

No dysts system has 4 ≤ D ≤ 10 with degree ≤ 2 apart from the 16 listed (no D = 5, 7–10 instances). Lorenz96 defaults to D = 4 and F = 20.

**Excluded:** 1 of the 78. PanXuZhou at dysts defaults collapses to an equilibrium after our 100-period burn-in (transient chaos), so **77 systems** go into the archive.

### Overlap with our systems (not new; reported separately by `dysts_report.py`)

- **Identical to ours (17):** Lorenz (dysts β = 2.667, not 8/3), Rossler, Chen, Halvorsen, SprottA–H, GenesioTesi, Dadras, Rucklidge, ChenLee, HyperRossler.
- **Same system, different default parameters (6):** BurkeShaw, LuChen, RabinovichFabrikant, RikitakeDynamo, Hadley, ShimizuMorioka. Hadley is in `HELDOUT` but was never in our archives, because it has no DT.
- **Same family, other D (1):** Lorenz96 (D = 4, F = 20; ours is D = 5/6/10, F = 8).
- **Support twins (4):** different systems with exactly the support of one of ours. LiuChen ≡ chenlee support, Lorenz84 ≡ hadley, NoseHoover ≡ sprottA (Sprott A is the a = 1 Nosé–Hoover), RayleighBenard ≡ lu.
- **Not overlap:** dysts LorenzCoupled is not our coupled_lorenz. It is one-way coupled with ρ₁ = 35, ρ₂ = 1.15, ε = 2.85, and has a different support, so it counts as new.

**New in-library systems: 49** (35 with D = 3, 14 with D = 4–6). The full in-library list is at the end.

## Archive

`dysts_generate.py` writes `data/dysts_test.npz` (seed 20261401) and `data/dysts_smoke.npz` (seed 20261400), each with a `.json` sidecar.

- **Format:** same as before: keys `X` (object array of (N, D)), `sys`, `noise`, `n`, `dt`.
- **Test archive:** 77 systems × 10 seeds × N ∈ {500, 2000} × η ∈ {0.01, 0.05, 0.1, 0.2} = **6160 runs** (4880 with D = 3, 1280 with D ≥ 4). Each seed and N has one clean trajectory with four noise levels, as in `heldout_systems.generate` and `nd_systems.generate`.
- **Noise:** Gaussian, η × the per-component std of the clean trajectory.
- **Smoke archive:** 77 runs, one per system, with (η, N) cycling over the 8 cells.
- **Checksums (sha256):** test `fd92b588e2ff8d80060e7f52116983c7c5c7231217ce129fb6524e8ddb03f3cf`, smoke `f43742e938923abdcf5de842be5d6b416833cd423a503cfd15a407c6a65e4757`.
- **dt rule (ours, not dysts'):** dt = 1/(128 f_peak) on a 1-2-5 grid, where f_peak is the maximum over components of the spectral peak of a long clean run. On our 21 systems this rule reproduces 18 of the hand-set DT values; Lorenz, Rabinovich and SprottF are off by one grid step.
  - **Why:** it keeps samples per cycle comparable with the sealed and held-out archives.
  - **dysts' alternative:** period/100 using its `period` field, stored in the sidecar as `dysts_recommended_dt`.
  - **Consequence:** dysts Lorenz gets dt = 0.005 here, against 0.01 in our sealed set.
- **Starts:**
  - Burn-in of 100 dominant periods from dysts' initial condition, then a 10,000-sample reference; each run starts at a uniformly random reference point.
  - A run is discarded and redrawn if it fails, is non-finite, or leaves 10× the reference range. No run was discarded.
  - A system is excluded if it collapses, or if its finite-time largest Lyapunov exponent × period ≤ 0.01. Our estimate agrees with dysts' metadata: the lowest are NoseHoover 0.04, SprottA 0.04 and HenonHeiles 0.05 per period, all kept.
- **States:** the raw ODE state. dysts' `_postprocessing` coordinate maps are not applied; no in-library system has one anyway.
- **Sidecar contents per system:** truth `xi` in library ordering plus `names` (core.EXPS for D = 3, joint_nd.PolyLib(D, 2) for D ≥ 4, asserted equal), D, degree, term counts, dt, dysts period and parameters, citation, class, overlap kind, coefficient range, smallest relative term contribution, stiffness proxy, Lyapunov estimate, discards.

## Systems that may need care

- **Small terms.** `min_rel_contrib` = min over true terms of |c|·rms(term)/rms(rhs) on the attractor:
  - AtmosphericRegime 0.006, ZhouChen 0.011, HyperYan 0.017, GuckenheimerHolmes 0.020, Rossler 0.022 (known hard), MooreSpiegel 0.028, WangSun 0.028, QiChen 0.032, Bouali2 0.039, Bouali 0.043, HyperRossler 0.046.
  - At 10–20% noise, exact support may simply be unidentifiable for these.
- **Coefficient ranges (max/min |c|):** HindmarshRose 830, Bouali2 750, AtmosphericRegime 500, HyperBao 360, DequanLi 344, IsothermalChemical 258, VallisElNino 102, WangSun 100. DequanLi also has |x| up to about 235 at dt = 0.0005.
- **Stiffness at our dt (max |eig J|·dt):** AtmosphericRegime 3.5, HyperRossler 1.7, HyperQi 1.2, SprottTorus 1.1; everything else is < 0.85. Joint's automatic sub-steps (cap 12) should cover these, but AtmosphericRegime is the slowest smoke run (88 s at N = 500) and it fails.
- **Conservative or weakly chaotic:** HenonHeiles (Hamiltonian), NoseHoover/SprottA, SprottTorus.

## Scripts

- **`dysts_bench.py --method weak|c|c_prior|final`** (ProcessPoolExecutor, single-threaded workers, longest runs first, partial saves). Truth is read from the sidecar.
  - **weak:** select_k chi2s for D = 3; joint_nd.weak_starts[0] for D ≥ 4.
  - **c:** for D = 3, squeeze_bench starts plus `joint_squeeze.discover_sq()` (config C); for D ≥ 4, `joint_nd.discover` with weak_starts.
  - **c_prior:** `discover_sq(degw=0.2)` for D = 3. For D ≥ 4 it records `status: unsupported`, because the prior exists for joint_nd only inside final_model.
  - **final:** `final_model.discover(x, dt)`, imported lazily in the worker.
  - c and c_prior also record paired weak and hybrid exact flags.
- **`dysts_report.py f1.json [f2.json ...]`:** for each file, overall, new/overlap/twin, D = 3 / D ≥ 4, by cell and per system; then paired W/L against the first file with a sign test.
- **`dysts_classify.py`** (needs `.venv_dysts`) and **`dysts_generate.py`** regenerate everything above.

## Smoke test

PC, `dysts_smoke.npz`, 3 workers. One run per system, so this is a plumbing test, not a result.

| Method | Runs | Exact | New systems | Overlap | Mean s/run (max) |
| --- | --- | --- | --- | --- | --- |
| weak | 77 | 58/77 = 0.753 | 0.694 | 0.833 | 0.1 |
| c | 77 | **70/77 = 0.909** | 42/49 = 0.857 | 24/24 | 15.5 (175, ChenLee N = 2000) |
| c_prior (D = 3 only) | 61 | 55/61 = 0.902 | 29/35 | 22/22 | 18.6 (186) |

- **c fails on:** AtmosphericRegime, Bouali, IsothermalChemical, SprottJerk (N = 2000, η = 0.2, 9 false positives), Tsucs2, HyperCai, HyperYan.
- **c_prior vs c on D = 3:** W/L 0/1. The one loss is Bouali2 at N = 500, η = 0.2.
- **final:** a 3-run check (Lorenz, SprottM, HyperLu) ran through the lazy import: 3/3 exact, 6–12 s.

Outputs are in `results_dysts/smoke_{weak,c,c_prior,final_3sys}.json` with logs alongside.

## Launching the full run on the Spark

On the PC (Git Bash):

```bash
cd /c/Users/yanbo/downloads/eqdisc
K=~/.ssh/YOUR_KEY; H=USER@YOUR_COMPUTE_HOST
scp -i $K dysts_bench.py dysts_report.py overnight2/G_dysts_job.sh $H:eqdisc_v2/
scp -i $K data/dysts_test.npz data/dysts_test.json data/dysts_smoke.npz data/dysts_smoke.json $H:eqdisc_v2/data/
ssh -i $K $H 'cd ~/eqdisc_v2 && sha256sum data/dysts_test.npz && sed -i "s/\r$//" G_dysts_job.sh && W=8 setsid nohup bash G_dysts_job.sh > /dev/null 2>&1 < /dev/null & disown'
```

- **Expected checksum:** `fd92b588e2ff8d80060e7f52116983c7c5c7231217ce129fb6524e8ddb03f3cf`.
- **Options:** `METHODS="c c_prior"` restricts the methods; `W` sets the workers. The job skips methods whose output already exists and writes `results_dysts/DONE` at the end.
- **Progress:** `ssh -i $K $H 'tail -n 2 ~/eqdisc_v2/results_dysts/test_*.log'`.
- **Stopping it:** kill by PID from `ps -u iamyanbo -o pid,args | grep dysts_bench`, never `pkill -f`.
- **final_model on the Spark:** the job runs whatever `final_model.py` is in `~/eqdisc_v2` at launch. Record its state, and do not launch `final` until the other agent has frozen it.

## Runtime estimate

Based on PC smoke timings, roughly per core:

| Method | Runs | Mean s/run | CPU-hours |
| --- | --- | --- | --- |
| c | 6160 | 15.5 | ≈ 26.5 |
| c_prior | 4880 (D = 3 only) | 18.6 | ≈ 25 |
| final | 6160 | unmeasured; assume ≈ c | ≈ 26 |
| weak | 6160 | 0.1 | < 0.2 |

- **Wall time:** about 78 CPU-hours in total. That is about 10 h at W = 8 and about 5 h at W = 16, if a Spark core matches a PC core and the machine is otherwise idle.
- **Current load:** at 14:20 the Spark load average was 17.9 of 20 cores.
- **Tail:** the long tail (175 s runs) is small next to the total.
- **Cheaper option:** `--every 2` (5 seeds) halves everything.

## In-library systems (77 in the archive)

Columns: degree (deg), number of true terms (k), coefficient range (|c|), smallest relative term contribution (min rel), stiffness at our dt (\|eig J\|·dt), our Lyapunov estimate × period (λ·T), overlap.

| System | D | deg | k | dt | \|c\| range | min rel | \|eig J\|·dt | λ·T | Overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{'X': ((6160,), dtype('O')), 'sys': ((6160,), dtype('<U19')), 'noise': ((6160,), dtype('float64')), 'n': ((6160,), dtype('int64')), 'dt': ((6160,), dtype('float64'))}
True [((np.int64(500), np.float64(0.01)), 770), ((np.int64(500), np.float64(0.05)), 770), ((np.int64(500), np.float64(0.1)), 770)] 77
{'PanXuZhou': 'collapses to an equilibrium'}
| Arneodo | 3 | 3 | 6 | 0.02 | 1–5.5 | 0.416 | 0.06 | 0.64 |  |
| AtmosphericRegime | 3 | 2 | 11 | 0.5 | 0.01–5 | 0.006 | 3.50 | 0.97 |  |
| Bouali | 3 | 3 | 8 | 0.02 | 0.05–4 | 0.043 | 0.13 | 0.09 |  |
| Bouali2 | 3 | 3 | 6 | 0.02 | 0.004–3 | 0.039 | 0.12 | 0.11 |  |
| Coullet | 3 | 3 | 6 | 0.05 | 0.45–1.1 | 0.376 | 0.09 | 0.11 |  |
| DequanLi | 3 | 2 | 9 | 0.0005 | 0.16–55 | 0.069 | 0.07 | 0.30 |  |
| Finance | 3 | 2 | 7 | 0.05 | 0.2–5 | 0.743 | 0.19 | 0.86 |  |
| GuckenheimerHolmes | 3 | 3 | 13 | 0.002 | 0.4–20.2 | 0.020 | 0.05 | 0.14 |  |
| HindmarshRose | 3 | 3 | 10 | 0.01 | 0.04–33.3 | 0.112 | 0.79 | 0.35 |  |
| IsothermalChemical | 3 | 3 | 9 | 0.005 | 0.3–76.9 | 0.613 | 0.27 | 0.22 |  |
| KawczynskiStrizhak | 3 | 3 | 9 | 0.2 | 0.2–4.2 | 0.312 | 0.75 | 0.08 |  |
| Laser | 3 | 3 | 7 | 0.005 | 1–10 | 0.530 | 0.18 | 0.64 |  |
| LuChenCheng | 3 | 2 | 7 | 0.01 | 1–18.1 | 0.238 | 0.29 | 0.27 |  |
| MooreSpiegel | 3 | 3 | 6 | 0.002 | 1–10 | 0.028 | 0.05 | 2.11 |  |
| NewtonLiepnik | 3 | 2 | 8 | 0.05 | 0.17–10 | 0.222 | 0.22 | 1.32 |  |
| PehlivanWei | 3 | 2 | 8 | 0.02 | 1–2 | 0.326 | 0.13 | 0.31 |  |
| QiChen | 3 | 2 | 8 | 0.002 | 1–80 | 0.032 | 0.21 | 1.73 |  |
| Sakarya | 3 | 2 | 8 | 0.02 | 0.3–1 | 0.235 | 0.16 | 1.37 |  |
| SprottI | 3 | 2 | 6 | 0.05 | 0.2–1 | 0.685 | 0.09 | 0.12 |  |
| SprottJ | 3 | 2 | 6 | 0.05 | 1–2 | 0.264 | 0.19 | 0.34 |  |
| SprottJerk | 3 | 2 | 5 | 0.1 | 1–2.02 | 1.000 | 0.34 | 0.68 |  |
| SprottK | 3 | 2 | 6 | 0.05 | 0.3–1 | 0.378 | 0.11 | 0.32 |  |
| SprottL | 3 | 2 | 6 | 0.02 | 0.9–3.9 | 0.228 | 0.09 | 0.47 |  |
| SprottM | 3 | 2 | 6 | 0.05 | 1–1.7 | 0.867 | 0.10 | 0.23 |  |
| SprottN | 3 | 2 | 6 | 0.05 | 1–2 | 0.316 | 0.19 | 0.35 |  |
| SprottO | 3 | 2 | 6 | 0.05 | 1–2.7 | 0.430 | 0.08 | 0.47 |  |
| SprottP | 3 | 2 | 6 | 0.05 | 1–2.7 | 0.727 | 0.09 | 0.54 |  |
| SprottQ | 3 | 2 | 6 | 0.02 | 0.5–3.1 | 0.291 | 0.04 | 0.44 |  |
| SprottR | 3 | 2 | 6 | 0.05 | 0.4–1 | 0.159 | 0.13 | 0.47 |  |
| SprottS | 3 | 2 | 6 | 0.02 | 1–4 | 0.611 | 0.05 | 0.64 |  |
| SprottTorus | 3 | 2 | 9 | 0.2 | 1–2 | 0.819 | 1.10 | 0.85 |  |
| Tsucs2 | 3 | 2 | 8 | 0.002 | 0.5–40 | 0.083 | 0.18 | 0.28 |  |
| VallisElNino | 3 | 2 | 7 | 0.01 | 1–102 | 0.188 | 0.13 | 1.20 |  |
| WangSun | 3 | 2 | 7 | 0.2 | 0.01–1 | 0.028 | 0.57 | 1.72 |  |
| ZhouChen | 3 | 2 | 8 | 0.01 | 0.15–8.78 | 0.011 | 0.67 | 0.47 |  |
| BurkeShaw | 3 | 2 | 6 | 0.005 | 1–13 | 0.072 | 0.17 | 1.82 | other params (burkeshaw) |
| Chen | 3 | 2 | 7 | 0.005 | 1–35 | 0.435 | 0.19 | 1.18 | identical (chen) |
| ChenLee | 3 | 2 | 6 | 0.05 | 0.33–10 | 0.377 | 0.70 | 0.34 | identical (chenlee) |
| Dadras | 3 | 2 | 8 | 0.02 | 1–9 | 0.162 | 0.45 | 1.30 | identical (dadras) |
| GenesioTesi | 3 | 2 | 6 | 0.05 | 0.44–1.1 | 0.388 | 0.07 | 0.66 | identical (genesio) |
| Hadley | 3 | 2 | 11 | 0.01 | 0.2–4 | 0.228 | 0.09 | 0.29 | other params (hadley) |
| Halvorsen | 3 | 2 | 12 | 0.01 | 1–4 | 0.300 | 0.16 | 1.07 | identical (halvorsen) |
| LiuChen | 3 | 2 | 6 | 0.02 | 0.4–12 | 0.137 | 0.81 | 3.16 | support twin (chenlee) |
| Lorenz | 3 | 2 | 7 | 0.005 | 1–28 | 0.142 | 0.11 | 1.36 | identical (lorenz) |
| Lorenz84 | 3 | 2 | 11 | 0.01 | 1–7.91 | 0.124 | 0.18 | 2.99 | support twin (hadley) |
| LuChen | 3 | 2 | 6 | 0.005 | 1–36 | 0.867 | 0.16 | 1.27 | other params (lu) |
| NoseHoover | 3 | 2 | 5 | 0.05 | 1–1.5 | 0.554 | 0.28 | 0.04 | support twin (sprottA) |
| RabinovichFabrikant | 3 | 3 | 10 | 0.02 | 0.87–3 | 0.621 | 0.10 | 0.35 | other params (rabinovich) |
| RayleighBenard | 3 | 2 | 6 | 0.002 | 1–30 | 1.070 | 0.07 | 1.67 | support twin (lu) |
| RikitakeDynamo | 3 | 2 | 7 | 0.05 | 1–1 | 0.496 | 0.25 | 0.79 | other params (rikitake) |
| Rossler | 3 | 2 | 7 | 0.05 | 0.2–5.7 | 0.022 | 0.73 | 0.39 | identical (rossler) |
| Rucklidge | 3 | 2 | 6 | 0.02 | 1–6.7 | 0.930 | 0.09 | 1.12 | identical (rucklidge) |
| ShimizuMorioka | 3 | 2 | 6 | 0.1 | 0.5–1 | 0.998 | 0.19 | 0.41 | other params (shimizu) |
| SprottA | 3 | 2 | 5 | 0.05 | 1–1 | 0.652 | 0.21 | 0.04 | identical (sprottA) |
| SprottB | 3 | 2 | 5 | 0.05 | 1–1 | 0.573 | 0.15 | 2.36 | identical (sprottB) |
| SprottC | 3 | 2 | 5 | 0.05 | 1–1 | 0.775 | 0.16 | 1.31 | identical (sprottC) |
| SprottD | 3 | 2 | 5 | 0.05 | 1–3 | 1.000 | 0.27 | 0.49 | identical (sprottD) |
| SprottE | 3 | 2 | 5 | 0.02 | 1–4 | 0.445 | 0.07 | 0.65 | identical (sprottE) |
| SprottF | 3 | 2 | 6 | 0.2 | 0.5–1 | 1.109 | 0.49 | 2.21 | identical (sprottF) |
| SprottG | 3 | 2 | 6 | 0.05 | 0.4–1 | 0.460 | 0.09 | 0.30 | identical (sprottG) |
| SprottH | 3 | 2 | 6 | 0.05 | 0.5–1 | 0.814 | 0.12 | 2.48 | identical (sprottH) |
| HenonHeiles | 4 | 2 | 7 | 0.05 | 1–2 | 0.407 | 0.07 | 0.05 |  |
| HyperBao | 4 | 2 | 9 | 0.005 | 0.1–36 | 0.105 | 0.17 | 0.54 |  |
| HyperCai | 4 | 2 | 9 | 0.005 | 1–27.5 | 0.121 | 0.16 | 1.05 |  |
| HyperJha | 4 | 2 | 10 | 0.005 | 1–28 | 0.110 | 0.13 | 0.36 |  |
| HyperLorenz | 4 | 2 | 10 | 0.005 | 1–28 | 0.114 | 0.13 | 0.55 |  |
| HyperLu | 4 | 2 | 9 | 0.005 | 1–36 | 0.329 | 0.18 | 0.70 |  |
| HyperPang | 4 | 2 | 9 | 0.005 | 1–36 | 0.109 | 0.17 | 1.19 |  |
| HyperQi | 4 | 2 | 12 | 0.002 | 1–50 | 0.093 | 1.20 | 1.02 |  |
| HyperWang | 4 | 2 | 8 | 0.005 | 1–40 | 0.150 | 0.14 | 0.61 |  |
| HyperXu | 4 | 2 | 9 | 0.005 | 1–40 | 0.499 | 0.14 | 1.12 |  |
| HyperYan | 4 | 2 | 13 | 0.002 | 1–38 | 0.017 | 0.08 | 1.41 |  |
| HyperYangChen | 4 | 2 | 8 | 0.005 | 1–35 | 0.197 | 0.22 | 0.50 |  |
| LorenzStenflo | 4 | 2 | 10 | 0.01 | 0.7–26 | 0.269 | 0.10 | 1.47 |  |
| HyperRossler | 4 | 2 | 9 | 0.02 | 0.05–3 | 0.046 | 1.68 | 0.15 | identical (hyper_rossler) |
| Lorenz96 | 4 | 2 | 16 | 0.02 | 1–20 | 0.155 | 0.52 | 2.81 | family, other D (l96_5/6/10) |
| LorenzCoupled | 6 | 2 | 15 | 0.005 | 1–35 | 0.123 | 0.12 | 1.63 |  |
