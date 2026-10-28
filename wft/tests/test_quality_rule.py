"""quality_ok is relative to the table (wft.reco.CHI2DOF_REL x the median chi2/dof per view; GE 2026-10-28)."""
import numpy as np
import pandas as pd

from wft.reco import CHI2DOF_REL, quality_flags


def _events(scale, n=200, seed=1):
    rng = np.random.default_rng(seed)
    r = scale * rng.lognormal(0.0, 0.5, n)
    r[:10] *= 3 * CHI2DOF_REL                      # ten views far above their table's median
    dof = np.full(n, 100)
    return pd.DataFrame({'x_chi2': r * dof, 'x_dof': dof, 'x_ok': True,
                         'y_chi2': r * dof, 'y_dof': dof, 'y_ok': True})


def test_flag_is_relative():
    # the same relative population at a median of 10 or of 700 flags the same views
    a, ma = quality_flags(_events(10.0))
    b, mb = quality_flags(_events(700.0))
    assert np.array_equal(a['x_quality_ok'], b['x_quality_ok'])
    assert abs(mb['x'] / ma['x'] - 70.0) < 1e-6
    assert not a['x_quality_ok'][:10].any() and a['x_quality_ok'][10:].mean() > 0.95


def test_unfitted_view_is_not_ok_and_does_not_set_the_median():
    d = _events(10.0)
    d.loc[:49, 'y_ok'] = False
    d.loc[:49, 'y_chi2'] = 1e9
    f, m = quality_flags(d)
    assert not f['y_quality_ok'][:50].any()
    assert m['y'] < 100


def test_candidates_take_given_medians():
    c = pd.DataFrame({'plane': ['x', 'x', 'y', 'y'], 'rank': [0, 1, 0, 1],
                      'chi2': [100.0, 1000.0, 100.0, 1000.0], 'dof': [10, 10, 10, 10]})
    f, m = quality_flags(c, {'x': 10.0, 'y': 1.0})
    assert list(f['quality_ok']) == [True, False, False, False]
    f2, m2 = quality_flags(c)                      # own medians: the rank-0 winners
    assert m2 == {'x': 10.0, 'y': 10.0} and list(f2['quality_ok']) == [True, False, True, False]
