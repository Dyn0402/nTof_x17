"""COL_GUARD (opt-in): unobservable depth columns get no charge; a well-observed solve is unchanged."""
import numpy as np

from wft import model as wm


def _solve(M, W, noise, guard):
    old, old_ns = wm.COL_GUARD, wm.NSAMP
    try:
        wm.COL_GUARD = guard
        wm.NSAMP = W.shape[1]
        return wm._solve_nnls(M, W, noise, np.zeros(W.shape, bool), censor=False)
    finally:
        wm.COL_GUARD, wm.NSAMP = old, old_ns


def test_guard_zeroes_unobservable_column():
    rng = np.random.default_rng(3)
    n, ns, k = 4, 8, 3
    M = rng.uniform(0.5, 1.0, (n * ns, k))
    M[:, 2] *= 1e-9                       # a column seen only through the tip of its leading edge
    W = (M[:, :2] @ np.array([100.0, 50.0])).reshape(n, ns) + rng.normal(0, 1, (n, ns))
    noise = np.ones(n)
    chi0, q0 = _solve(M, W, noise, 0.0)
    chi1, q1 = _solve(M, W, noise, 1e-2)
    assert q1[2] == 0.0
    assert np.isfinite(chi1) and chi1 >= chi0 - 1e-9


def test_guard_is_identity_when_every_column_is_live():
    rng = np.random.default_rng(4)
    n, ns, k = 4, 8, 3
    M = rng.uniform(0.5, 1.0, (n * ns, k))
    W = (M @ np.array([10.0, 20.0, 30.0])).reshape(n, ns)
    noise = np.ones(n)
    chi0, q0 = _solve(M, W, noise, 0.0)
    chi1, q1 = _solve(M, W, noise, 1e-2)
    assert np.array_equal(q0, q1) and chi0 == chi1
