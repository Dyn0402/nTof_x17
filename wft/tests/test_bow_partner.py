"""apply_bow(skip=...): a fit already mapped with its pair partner keeps that mapping (tracking audit 2026-11-22,
finding 21: the event winners were re-mapped together after the pairs, so a pair with exactly one rank-0 member took
the other pair's coordinate in its bow term)."""
from types import SimpleNamespace

from wft import reco


def _cal():
    # a bow block whose tilt depends on the OTHER view's coordinate, so a wrong partner shows in the tangent
    bow = {'A_x': {'cw': 30.0}, 'A_y': {'cu': 30.0}}
    return SimpleNamespace(bow=bow, v_map=None, w0={'x': 0.0, 'y': 0.0}, kw={'x': 1.0, 'y': 1.0}, v_drift=40.0)


def _fit(p0, w):
    return SimpleNamespace(p0=p0, w=w, tan_theta=0.0, theta_deg=0.0, slope_reliable=True)


def test_skip_keeps_partner_mapping():
    cal = _cal()
    x1, y1 = _fit(100.0, 0.004), _fit(300.0, 0.002)       # pair 1
    x2, y2 = _fit(250.0, -0.003), _fit(120.0, 0.001)      # pair 2
    reco.apply_bow(cal, x1, y1)
    reco.apply_bow(cal, x2, y2)
    t_x1, t_y2 = x1.tan_theta, y2.tan_theta
    # winners x1 (pair 1) and y2 (pair 2): the old code re-mapped them together and changed both
    reco.apply_bow(cal, x1, y2, skip=[x1, y1, x2, y2])
    assert x1.tan_theta == t_x1 and y2.tan_theta == t_y2
    # without skip the cross-pair coordinates do change them (the bug the skip prevents)
    reco.apply_bow(cal, x1, y2)
    assert x1.tan_theta != t_x1 and y2.tan_theta != t_y2
