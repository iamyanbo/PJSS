"""Numerical invariants for the scientific method, not implementation snapshots."""
import numpy as np
import pytest
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits
from eqdisc.core import DIM, NLIB, SYSTEMS, simulate, IDX
from eqdisc.flow_discovery import (integrate, project_initial_states, fit_trajectories,
                            flow_dictionary, _weak_seed, discover)


@pytest.fixture(autouse=True)
def one_blas_thread():
    with threadpool_limits(limits=1):
        yield


def test_variational_derivatives_including_zero_states_and_inactive_terms():
    xi = SYSTEMS["lorenz"]
    z = np.array([[0., 2., 0.], [-2., 1., 5.]])
    active = np.arange(NLIB*DIM)
    _, jac = integrate(z, xi, .01, 12, active, substeps=2)
    eps = 1e-5
    for index in [0, 3, 7, 17, 31, 58]:
        plus, minus = xi.copy(), xi.copy()
        plus.ravel()[index] += eps
        minus.ravel()[index] -= eps
        fp, _ = integrate(z, plus, .01, 12, [])
        fm, _ = integrate(z, minus, .01, 12, [])
        np.testing.assert_allclose(jac[..., index], (fp-fm)/(2*eps), rtol=2e-6, atol=2e-7)
    for j in range(DIM):
        zp, zm = z.copy(), z.copy()
        zp[:, j] += eps
        zm[:, j] -= eps
        fp, _ = integrate(zp, xi, .01, 12, [])
        fm, _ = integrate(zm, xi, .01, 12, [])
        np.testing.assert_allclose(jac[..., len(active)+j], (fp-fm)/(2*eps), rtol=2e-6, atol=2e-7)


def test_rk4_matches_independent_adaptive_solver():
    initial = np.array([[1., 2., 3.]])
    predicted, _ = integrate(initial, SYSTEMS["lorenz"], .01, 40, [], substeps=4)
    reference = simulate(SYSTEMS["lorenz"], initial[0], .01, 40)
    np.testing.assert_allclose(predicted[0], reference, atol=1e-5, rtol=1e-5)


def test_projection_matches_full_joint_least_squares():
    rng = np.random.default_rng(7)
    r = rng.normal(size=(4, 30))
    a = rng.normal(size=(4, 30, 7))
    z = rng.normal(size=(4, 30, 3))
    rp, ap = project_initial_states(r, a, z)
    profiled = np.linalg.lstsq(ap, -rp, rcond=None)[0]
    joint = np.column_stack((a.reshape(-1, 7), block_diag(*z)))
    full = np.linalg.lstsq(joint, -r.ravel(), rcond=None)[0]
    np.testing.assert_allclose(profiled, full[:7], atol=1e-12)
    contaminated = r + np.einsum("wmd,wd->wm", z, rng.normal(size=(4, 3)))
    rp2, _ = project_initial_states(contaminated, a, z)
    np.testing.assert_allclose(rp2, rp, atol=1e-12)


def test_joint_fit_recovers_coefficients_and_unknown_starts():
    true = SYSTEMS["lorenz"]
    starts = [[1., 2., 3.], [-2., 3., 8.], [10., -3., 21.]]
    observed = np.stack([simulate(true, z, .01, 30) for z in starts])
    fit = fit_trajectories(observed, .01, .9*true, np.full(3, .1), substeps=4)
    assert fit.converged
    np.testing.assert_allclose(fit.xi, true, atol=3e-5)
    np.testing.assert_allclose(fit.initial, starts, atol=3e-5)
    a, b = flow_dictionary(observed, .01, fit, np.full(3, .1), substeps=4)
    assert np.linalg.norm(a@fit.xi.ravel()-b) < 1e-4


def test_validation_profiles_starts_but_never_changes_coefficients():
    true = SYSTEMS["lorenz"]
    observed = np.stack([simulate(true, [1., 2., 3.], .01, 30)])
    fit = fit_trajectories(observed, .01, true, np.ones(3), fit_coefficients=False,
                           initial=np.array([[1.2, 1.8, 3.1]]), substeps=4)
    np.testing.assert_array_equal(fit.xi, true)
    assert fit.rss < 1e-8


def test_heldout_measurements_do_not_enter_weak_seed_or_proposals():
    x = simulate(SYSTEMS["lorenz"], [1., 2., 3.], .01, 400, burn=100)
    x /= np.sqrt(np.mean(x*x, axis=0))
    mask = np.repeat(np.arange(10)%3 != 2, 40)
    altered = x.copy()
    altered[~mask] += np.random.default_rng(8).normal(size=altered[~mask].shape)*20
    seed1, pools1 = _weak_seed(x, .01, mask, np.full(3, .05), 12, 3, 1)
    seed2, pools2 = _weak_seed(altered, .01, mask, np.full(3, .05), 12, 3, 1)
    np.testing.assert_allclose(seed1, seed2, atol=1e-10)
    assert [[s.tolist() for s in p] for p in pools1] == [[s.tolist() for s in p] for p in pools2]


def test_zero_dynamics_is_a_valid_candidate():
    xi = np.zeros((NLIB, DIM))
    blocks = np.broadcast_to(np.array([1., 2., 3.]), (3, 20, DIM)).copy()
    fit = fit_trajectories(blocks, .01, xi, np.ones(DIM))
    assert fit.rss < 1e-20


@pytest.mark.parametrize("data,dt", [(np.ones((400, 2)), .01), (np.ones((20, 3)), .01),
                                     (np.full((400, 3), np.nan), .01), (np.ones((400, 3)), 0.)])
def test_bad_observation_inputs_fail_explicitly(data, dt):
    with pytest.raises(ValueError):
        discover(data, dt)


def test_divergent_model_returns_failed_fit():
    xi = np.zeros((NLIB, DIM))
    xi[IDX["x^3"], 0] = 1e8
    fit = fit_trajectories(np.ones((2, 20, DIM)), .1, xi, np.ones(DIM))
    assert np.isinf(fit.rss)
    assert not fit.converged


def test_end_to_end_result_is_serializable_and_sparse():
    import json
    from eqdisc.core import metrics
    x = simulate(SYSTEMS["lorenz"], [-8., 6., 25.], .01, 800, burn=100)
    x += .03*x.std(axis=0)*np.random.default_rng(91).normal(size=x.shape)
    result = discover(x, .01, kmax=3, rounds=1, finalists=4)
    assert metrics(result.coefficients, SYSTEMS["lorenz"])["exact"] == 1.
    assert result.diagnostics["train_segments"] == 14
    assert result.diagnostics["validation_segments"] == 6
    json.dumps({"diagnostics": result.diagnostics, "alternatives": result.alternatives}, allow_nan=False)
