#!/bin/bash
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3 OCTAVE_HOME=$HOME/octave_env/oct
taskset -c 0 .venv_base/bin/python benchmarks/wsindy_octave_check.py --wsindy $HOME/fastbl/WSINDy_ODE --octave $OCTAVE_HOME/bin/octave-cli --positions 0-14 --out results_fast/wsindy_octave_check.json
