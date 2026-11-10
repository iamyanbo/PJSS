# DEDi on the Spark's GPU (2026-10-27)

**Status: setup and benchmark only.** The big comparison was not launched.

**Bottom line**
- **Setup works.** A separate env runs DEDi's unmodified code on the GB10 GPU in float64.
- **Speed.** One GPU process (one host core) is about **5× faster per (λ, R) pair** than one CPU core: 5,404 s against 29,175 s over 38 pairs.
- **Throughput.** Up to ~10 GPU processes can share the GPU, one host core each. That gives about **2.2 full-grid runs/h**, against 1.3 runs/h on 18 CPU cores.
- **~70 full-grid runs:** about 52 h on 18 CPU cores, about 32 h on 10 GPU processes, and about 25 h if GPU and CPU workers are combined.
- **Results are not bit-identical, and on GPU not even run-to-run reproducible.**
  - 37 of 38 pairs keep the same support. Their coefficients agree to 1e-11 relative or better, except two pairs at 3e-3 and 1e-5.
  - One pair changes its support between repeated GPU runs.
  - For run 43 the primary selection (validation error) picks the same (λ, R) and the same coefficients on GPU. The secondary BIC selection changed, from an exact model to a wrong one.
  - Treat GPU DEDi as a re-run with different numerics, not a reproduction of the CPU numbers.

## Setup

- **New env** `~/eqdisc_dedi_gpu/.venv`, using the same system Python 3.12.3.
  - It holds the exact `pip freeze` of `~/eqdisc_dedi/.venv` (saved in `~/eqdisc_dedi_gpu/base_freeze.txt`) plus `jax[cuda13]==0.12.0.1`: jax-cuda13-plugin/pjrt 0.12.0.1, CUDA runtime 13.4 wheels and cuDNN 9.27.
  - The CPU env was not touched.
- **GPU check.** The driver is 580.173 (CUDA 13.0). `jax.devices()` returns `[CudaDevice(id=0)]`, compute capability 12.1. float64 matmul and Cholesky work.
- **Precision.** DEDi's `ode_find` defaults to `precision_64_bit=True` and calls `jax.config.update("jax_enable_x64", True)`, so both CPU and GPU run in float64. This was confirmed in every result record (`x64: true`).
- **Choosing the device.**
  - `ode_find` itself also calls `jax.config.update('jax_platform_name', cpu_or_gpu)`, with default `'cpu'`. So `benchmarks/dedi_fullgrid.py` runs on CPU even in a CUDA env.
  - The helper `benchmarks/dedi_gpu_bench.py` passes `cpu_or_gpu` through. That is the only change, and it is an argument DEDi itself exposes. The DEDi code is not modified.
- **Required setting:** `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Without it, each process reserves 75% of the GPU memory, which on the GB10 is the shared 121 GB system RAM. With it, a process uses about 1–1.5 GB RSS.
- Same `XLA_FLAGS`, single-thread BLAS and `taskset` pinning as the full-grid run.

## Where the time goes

- In each LM iteration, JAX computes the gradient and the coloured-HVP sparse Hessian. The step is then solved on the host by our CHOLMOD replacement, `shim/sksparse/cholmod.py` (scipy banded Cholesky plus a Schur complement).
- On GPU, the JAX part shrinks a lot. The host-side solve now takes 50–95% of each pair's wall time (`shim_seconds`). GPU utilisation for one process is only about 10%.
- **So a GPU process is bound by its host core.** That is why throughput scales with the number of host cores given to GPU processes.

## Speed per pair (one process per core; GPU processes on the slow A725 cores 10–11)

| Set | Pairs | CPU (stored full-grid run) | GPU | Ratio |
| --- | --- | --- | --- | --- |
| Run 43 (Chen), all 35 pairs | 35 | 25,307 s | 4,684 s | 5.4× |
| Run 109 (Rössler), 3 pairs | 3 | 3,868 s | 720 s | 5.4× |
| Like-for-like: CPU rerun on core 11 vs GPU on core 10, run concurrently (4 pairs) | 4 | 1,918 s | 505 s | 3.8× |

- Per-pair ratios range from 2.7× to 7.6×. The speedup is largest for λ ≥ 10, which were the slowest pairs on CPU relative to their size.
- **GPU pair times are noisy:** the same pair took between 94 s and 225 s across repeats. The like-for-like row includes one of the slow repeats.
- The stored CPU times come from a mix of slow and fast cores, just like the 13.4 core-h median per run. So about 5× is the fair figure against that median.
- Implied cost of one median full-grid run on GPU: about 13.4 / 5.4 ≈ **2.5 GPU-process hours**, against 13.4 core-hours on CPU.

## Several processes on one GPU (run 43, 2 pairs per process; each process pinned to its own slow core; Spark otherwise idle)

| Processes K | Makespan | Mean pair time vs K=1 | Pairs/h | GPU util (mean) |
| --- | --- | --- | --- | --- |
| 1 | 289 s | 1.0× | 25 | 10% |
| 4 | 373 s | 1.2× | 77 | 41% |
| 7 | 440 s | 1.3× | 115 | 60% |
| 10 | 521 s | 1.6× | 138 | 71% |

- **The rule is one process per host core.** Four GPU processes sharing two cores were slower than two processes on the same two cores: 544 s against 424 s makespan for the same 4 pairs.
- Results were identical at every K.
- vmap over the grid was not tried. It would need DEDi's code changed, because its LM loop is host-side Python with data-dependent iteration counts and scipy solves. It could not be shown to keep results identical.

## Estimated time for ~70 full-grid runs (median 13.4 CPU core-h per run)

| Configuration | Runs/h | ~70 runs |
| --- | --- | --- |
| 18 CPU cores (as now) | 1.34 | ~52 h |
| GPU, 2 host cores (alongside other jobs) | ~0.75 | ~95 h |
| GPU, 10 processes on the 10 slow cores | ~2.2 | ~32 h |
| 10 GPU processes (slow cores) + 8 CPU workers (fast cores) | ~2.8 | ~25 h |

- These are extrapolations. The CPU and GPU workers in the hybrid row were not measured together.
- Run cost varies about 2.5× between systems: Chen runs cost about 7 core-h, Rössler runs about 15.

## Identical results?

Every pair was compared with the stored full-grid CPU result for the same (run, λ, R), using `benchmarks/dedi_gpu_compare.py`.

- **CPU reruns:** 4 of 4 pairs were bit-identical to the stored results, on a different core and a day later. This confirms the determinism already reported in `dedi_fullgrid.md`.
- **GPU, 38 distinct pairs:**
  - 35 have the same support, validation error equal to about 10 digits, and coefficients within 1e-11 relative (most within 1e-15).
  - 2 have the same support but larger drift: run 109 (0.1, 10⁻²), a 30-term non-sparse fit, at 3e-3; and run 43 (0.1, 0.1) at 1e-5.
  - 1 changed support: run 43 (λ, R) = (1, 1) gave 8 terms (not exact) instead of the stored 7 terms (exact).
- **The flipped pair is non-deterministic on GPU.**
  - In 5 fresh GPU processes it gave 8 terms 3 times and the CPU's 7 terms 2 times.
  - Adding `--xla_gpu_deterministic_ops=true` did not fix it (1 of 2 runs gave 8 terms).
  - Other pairs repeated exactly. For example (1, 10⁻²) was the same in all 5 runs.
  - The likely cause is the thread-timing dependence already suspected in `dedi_fullgrid.md`: DEDi runs its two LM solves in two Python threads that share the model object. GPU async dispatch changes their interleaving. This was not proven.
- **Selection for run 43, all 35 pairs on GPU:**
  - **Primary (validation error):** same pick, (10⁻³, 1), 9 terms, not exact, with identical coefficients to 4 decimal places. The runner-up is also the same. The rank order of all 35 pairs differs only among near-ties.
  - **Secondary (validation BIC):** the CPU picks (1, 1), which is exact. The GPU realisation with the 8-term (1, 1) result picks (10⁻³, 10⁻²), which is not exact. So with BIC selection, this run's outcome would flip in that realisation.
  - **Oracle (any pair exact):** 1 of 35 pairs is exact on CPU, against 0 or 1 on GPU depending on the realisation.
- **Implication.** DEDi's discrete term-dropping amplifies differences at roundoff level. On hard-120 the selection margins are tiny: the best and second-best validation errors differ by 10⁻¹⁰ to 10⁻³ relative. So a GPU rerun of the 17-run comparison would agree on most runs, but not necessarily on all.
  - If GPU DEDi is used, rerun everything on GPU, including the 17 already-analysed runs. Do not mix CPU and GPU pairs within one run.
  - Report it as a separate execution. Alternatively, run each pair 2–3 times and report how stable it is.

## Could our own method use the GPU?

Probably not much.
- The final model's numerics are scipy banded Cholesky solves (`solveh_banded`, bandwidth about 2D) with a small dense coefficient border, plus small dense lstsq and solves on 20–60 columns. The time series has N ≈ 2,000.
- Banded factorisation is a sequential recursion with O(N·b²) work per solve, roughly 10⁵ flops. That is far too little to cover GPU launch and transfer costs. cuSOLVER also has no efficient banded Cholesky; batched tridiagonal/pentadiagonal solvers are the nearest.
- A run already takes a median 8.6 s on one core, and the runs are independent. Running 18 processes on the CPU cores is the simple way to scale.
- The only plausible GPU gain would be batching thousands of independent small solves (many runs or candidate supports at once) into one batched kernel. That would mean a rewrite for a workload that is already cheap, and it was not pursued.

## Files

- `benchmarks/dedi_gpu_bench.py`: runs chosen (run, λ, R) pairs on `--device cpu|gpu` and records the result, wall time and host-solver time. `DEDI_XLA_EXTRA` appends XLA flags.
- `benchmarks/dedi_gpu_compare.py`: compares result JSONL against `results_dedi_fullgrid/main/pairs`.
- The raw results and the driver scripts `bench1–5.sh` are on the Spark in `~/eqdisc_v3/results_dedi_gpu/`:
  - `gpu1`/`cpu1`: single-process GPU vs CPU;
  - `gpu2`: 2 or 4 processes on 2 cores;
  - `gpu3`: run 43, full grid;
  - `gpu4`: repeat test;
  - `gpu5`: scaling with K processes.
