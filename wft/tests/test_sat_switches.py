"""Saturation study switches (opt-in, rerun/calib/saturation): SAT_RAIL / SAT_MARGIN / SAT_FLOOR / SAT_TAIL /
SAT_STRIP in prep_plane's censor mask, SIG_VAR_K in the chi2 weights.  All off = bit-identical."""
import contextlib

import numpy as np
from scipy.optimize import nnls

from wft import model as wm

NAMES = ('SAT_RAIL', 'SAT_MARGIN', 'SAT_FLOOR', 'SAT_TAIL', 'SAT_TAIL_SKIP', 'SAT_TAIL_MINLEN', 'SAT_STRIP', 'SAT_CLIP_FREE',
         'SIG_VAR_K', 'ROBUST_C', 'MODEL_FRAC', 'GAIN', 'DEAD', 'HOT', 'NSAMP', 'SAT', 'BAND_NSIG', 'SAT_SOLVE', 'COL_GUARD')


@contextlib.contextmanager
def switches(**kw):
    old = {k: getattr(wm, k) for k in NAMES}
    try:
        wm.GAIN = {'x': np.ones(512), 'y': np.ones(512)}
        wm.DEAD, wm.HOT, wm.SAT, wm.BAND_NSIG = {}, {}, 3550.0, 0.0
        for k, v in kw.items():
            setattr(wm, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(wm, k, v)


def window():
    """Three strips x 10 samples: strip 1 clips for 3 samples (raw 4095) and then falls to raw 0; strip 0 has a
    clipped sample the CM pulled under 3550; strip 2 is unclipped but sits >= 3550 once."""
    ped = np.array([300.0, 280.0, 320.0])
    cm = np.zeros((3, 10)); cm[0, 4] = 300.0
    raw = np.full((3, 10), 300.0); raw[0] = 280.0 + 0 * raw[0]; raw[0] = ped[0]
    raw[1] = [280, 900, 3000, 4095, 4095, 4095, 1500, 0, 0, 0]
    raw[0, 3:6] = [3000, 4095, 3500]
    raw[2, 2:6] = [1000, 3890, 3000, 1000]
    W = raw - ped[:, None] - cm
    return dict(W=W, ch=np.array([10, 11, 12]), pos=np.array([0.0, 0.78, 1.56]), noise=np.full(3, 8.0),
                rail_hi=W + 4095.0 - raw, rail_lo=W - raw), raw


def test_all_off_is_the_3550_rule():
    P, raw = window()
    with switches(NSAMP=10):
        _, _, _, sat = wm.prep_plane(P, 'x')
    assert np.array_equal(sat, P['W'] >= 3550.0)


def test_rail_censors_exactly_the_clipped_samples():
    P, raw = window()
    with switches(NSAMP=10, SAT_RAIL=True):
        _, _, _, sat = wm.prep_plane(P, 'x')
    assert np.array_equal(sat, raw >= 4095)
    assert sat[0, 4] and P['W'][0, 4] < 3550                    # CM-pulled clip: censored now
    assert not sat[2, 3] and P['W'][2, 3] >= 3550               # real sample >= 3550: kept as data now


def test_rails_absent_falls_back():
    P, raw = window(); P.pop('rail_hi'); P.pop('rail_lo')
    with switches(NSAMP=10, SAT_RAIL=True, SAT_FLOOR=True):
        _, _, _, sat = wm.prep_plane(P, 'x')
    assert np.array_equal(sat, P['W'] >= 3550.0)


def test_floor_tail_strip():
    P, raw = window()
    with switches(NSAMP=10, SAT_RAIL=True, SAT_FLOOR=1):
        sat = wm.prep_plane(P, 'x')[3]
    assert np.array_equal(sat, (raw >= 4095) | (raw <= 0))
    assert np.array_equal(sat.ub, raw <= 0) and not sat.free.any()          # floor = upper bounds
    with switches(NSAMP=10, SAT_RAIL=True, SAT_FLOOR=2):
        sat = wm.prep_plane(P, 'x')[3]
    assert np.array_equal(sat.free, raw <= 0) and not sat.ub.any()
    with switches(NSAMP=10, SAT_RAIL=True, SAT_TAIL=True):
        sat = wm.prep_plane(P, 'x')[3]
    assert sat[1, 3:].all() and not sat[1, :3].any()            # plateau + everything after it, rising edge kept
    assert sat[0, 4:].all() and not sat[0, :4].any()
    assert sat.free[1, 6:].all() and not sat.free[1, 3:6].any()  # the tail is free, the plateau a lower bound
    with switches(NSAMP=10, SAT_RAIL=True, SAT_TAIL=True, SAT_FLOOR=1):
        sat = wm.prep_plane(P, 'x')[3]
    assert sat.ub[1, 7:].all() and not sat.free[1, 7:].any() and sat.free[1, 6]   # floor keeps its bound in the tail
    with switches(NSAMP=10, SAT_RAIL=True, SAT_TAIL=True, SAT_TAIL_SKIP=1):
        sat = wm.prep_plane(P, 'x')[3]
    assert not sat[1, 6] and sat[1, 7:].all()                   # first sample after the plateau kept
    with switches(NSAMP=10, SAT_RAIL=True, SAT_TAIL=True, SAT_TAIL_MINLEN=2):
        sat = wm.prep_plane(P, 'x')[3]
    assert sat[1, 6:].all() and not sat[0, 5:].any()            # strip 0's single-sample clip: no tail censoring
    with switches(NSAMP=10, SAT_RAIL=True, SAT_STRIP=True):
        sat = wm.prep_plane(P, 'x')[3]
    assert sat[0].all() and sat[1].all() and not sat[2].any()
    assert np.array_equal(sat.free[1], raw[1] < 4095)


def _bound_problem():
    rng = np.random.default_rng(9)
    n, ns, k = 3, 8, 2
    M = rng.uniform(0.5, 1.0, (n * ns, k))
    W = (M @ np.array([100.0, 60.0])).reshape(n, ns) + rng.normal(0, 2, (n, ns))
    return M, W, np.full(n, 3.0)


def test_plain_mask_equals_censor_with_empty_subsets():
    M, W, noise = _bound_problem()
    sat = np.zeros(W.shape, bool); sat[1, 2:4] = True; W[1, 2:4] += 50.0
    c = sat.copy().view(wm.Censor); c.ub = np.zeros(W.shape, bool); c.free = np.zeros(W.shape, bool)
    with switches(NSAMP=W.shape[1], COL_GUARD=0.0, SAT_SOLVE=True):
        a = wm._solve_nnls(M, W, noise, sat)
        b = wm._solve_nnls(M, W, noise, c)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


def test_upper_bound_penalty_and_active_set():
    M, W, noise = _bound_problem()
    W2 = W.copy(); W2[0, 5] = -200.0                                # a floor sample far below the model
    ub = np.zeros(W.shape, bool); ub[0, 5] = True
    c = ub.copy().view(wm.Censor); c.ub = ub; c.free = np.zeros(W.shape, bool)
    f = ub.copy().view(wm.Censor); f.ub = np.zeros(W.shape, bool); f.free = ub.copy()
    with switches(NSAMP=W.shape[1], COL_GUARD=0.0, SAT_SOLVE=True):
        chi_ub, q_ub = wm._solve_nnls(M, W2, noise, c)
        chi_fr, q_fr = wm._solve_nnls(M, W2, noise, f)
        chi_lb, q_lb = wm._solve_nnls(M, W2, noise, ub)              # plain mask = lower bound: inactive here
    assert np.allclose(q_fr, q_lb) and np.isclose(chi_fr, chi_lb)    # model above -200: a lower bound is free
    model = M @ q_ub
    assert model.reshape(W.shape)[0, 5] <= -200.0 + 1e-6 or chi_ub > chi_fr   # the upper bound pulls the model down
    assert chi_ub >= chi_fr - 1e-9


def test_sig_var_weights():
    rng = np.random.default_rng(5)
    n, ns, k = 3, 8, 2
    M = rng.uniform(0.5, 1.0, (n * ns, k))
    W = (M @ np.array([300.0, 100.0])).reshape(n, ns) + rng.normal(0, 5, (n, ns))
    noise = np.array([6.0, 8.0, 10.0]); sat = np.zeros(W.shape, bool)
    with switches(NSAMP=ns, COL_GUARD=0.0):
        c0, q0 = wm._solve_nnls(M, W, noise, sat)
    with switches(NSAMP=ns, COL_GUARD=0.0, SIG_VAR_K=0.0):
        c1, q1 = wm._solve_nnls(M, W, noise, sat)
    assert c0 == c1 and np.array_equal(q0, q1)
    K = 11.5
    with switches(NSAMP=ns, COL_GUARD=0.0, SIG_VAR_K=K):
        c2, q2 = wm._solve_nnls(M, W, noise, sat)
    wt = 1.0 / np.sqrt(noise[:, None] ** 2 + K * np.maximum(W, 0.0)).reshape(-1)
    q_ref, rn = nnls(M * wt[:, None], W.reshape(-1) * wt)
    assert np.allclose(q2, q_ref) and np.isclose(c2, rn * rn)


def test_clip_free():
    P, raw = window()
    with switches(NSAMP=10, SAT_RAIL=True, SAT_CLIP_FREE=True, SAT_FLOOR=1):
        sat = wm.prep_plane(P, 'x')[3]
    assert np.array_equal(sat.free, raw >= 4095) and np.array_equal(sat.ub, raw <= 0)


def test_robust_huber():
    rng = np.random.default_rng(11)
    n, ns, k = 4, 8, 2
    M = rng.uniform(0.5, 1.0, (n * ns, k))
    qt = np.array([300.0, 100.0])
    W = (M @ qt).reshape(n, ns) + rng.normal(0, 1, (n, ns))
    W[2, 3] += 500.0                                             # one outlier ("delta blob" sample)
    noise = np.ones(n); sat = np.zeros(W.shape, bool)
    with switches(NSAMP=ns, COL_GUARD=0.0):
        c0, q0 = wm._solve_nnls(M, W, noise, sat)
    with switches(NSAMP=ns, COL_GUARD=0.0, ROBUST_C=3.0):
        c1, q1 = wm._solve_nnls(M, W, noise, sat)
    assert np.abs(q1 - qt).sum() < np.abs(q0 - qt).sum()          # the outlier pulls the Huber fit less
    assert c1 < c0
    with switches(NSAMP=ns, COL_GUARD=0.0, ROBUST_C=0.0):
        c2, q2 = wm._solve_nnls(M, W, noise, sat)
    assert c2 == c0 and np.array_equal(q2, q0)
