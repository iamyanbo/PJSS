"""A run's weak domains must not depend on earlier jobs in the worker."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from benchmarks import baselines_bench as bench


def test_weak_features_and_selection_ignore_previous_worker_randomness(monkeypatch):
    pytest.importorskip('pysindy')
    t = np.linspace(0, 8, 160)
    run = dict(index=37, x=np.column_stack((np.sin(t), np.cos(t))), dt=t[1]-t[0],
               names=['1', 'x0', 'x1', 'x0^2', 'x0*x1', 'x1^2'],
               truth=[[0, 0], [0, -1], [1, 0], [0, 0], [0, 0], [0, 0]])
    cfg = dict(scale='rms', window=11, h=10)
    monkeypatch.setattr(bench, 'THS', np.array([0.01, 0.1]))
    original_features = bench.features
    seen = []

    def capture(*args, **kwargs):
        A, b, names = original_features(*args, **kwargs)
        seen.append((A.copy(), b.copy()))
        return A, b, names

    monkeypatch.setattr(bench, 'features', capture)
    with threadpool_limits(limits=1):
        np.random.seed(19)
        first = bench.run_path(run, 'weak_stlsq', cfg)
        np.random.seed(927)
        np.random.standard_normal(10007)
        second = bench.run_path(run, 'weak_stlsq', cfg)
    for a, b in zip(seen[0], seen[1]):
        np.testing.assert_array_equal(a, b)
    assert first == second

