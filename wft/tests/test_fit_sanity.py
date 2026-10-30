"""Fits that are not tracks are dropped on load, and the z-scan width survives an absurd residual (pass-1 review P1)."""
import numpy as np
import pandas as pd

from wft import compat


def _events():
    tan = [0.1, -0.5, 2.9, 3.5, 6e8, np.nan, 0.2, 0.3]
    p0 = [100.0, 200.0, 50.0, 60.0, 70.0, 80.0, 1.4e6, -19.0]
    return pd.DataFrame({'x_ok': True, 'x_tan_theta': tan, 'x_p0': p0,
                         'y_ok': [True] * 7 + [False], 'y_tan_theta': 0.1, 'y_p0': 150.0})


def test_insane_fits_dropped():
    d = compat.apply_fit_sanity(_events())
    assert d['x_ok'].tolist() == [True, True, True, False, False, False, False, True]
    # an unfitted view stays unfitted; sane views are untouched
    assert d['y_ok'].tolist() == [True] * 7 + [False]


def test_robust_sigma_ignores_one_absurd_residual():
    cm = compat.cm
    rng = np.random.default_rng(3)
    for n in (30, 400):
        v = list(rng.normal(0.0, 0.5, n)) + [6e16]
        assert 0.3 < cm._robust_sigma(v) < 0.8
    assert np.isnan(cm._robust_sigma([]))


def test_bow_uw_ignores_an_off_plate_view():
    from types import SimpleNamespace as S
    from wft import reco as wr
    cal = S(w0={'x': 0.0, 'y': 0.0}, kw={'x': 1.0, 'y': 1.0}, v_drift=35.0, angle_map=None)
    U, W = wr.bow_UW(cal, S(p0=1.3e6, w=0.0), S(p0=250.0, w=0.0))
    assert U == 0.0 and abs(W - 0.5) < 1e-9          # the absurd X point does not set the Y view's bow
    U, W = wr.bow_UW(cal, S(p0=150.0, w=0.0), S(p0=250.0, w=0.0))
    assert abs(U + 0.5) < 1e-9 and abs(W - 0.5) < 1e-9
