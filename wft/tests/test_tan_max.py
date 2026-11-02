"""The candidate |tan| limit is per dataset (bundle ``tan_max``, Dylan 2026-11-02): absent = the reco default, set = the
bundle's value, saved and loaded with the bundle, and a WFT_DIAG reco.TAN_MAX override still wins."""
import numpy as np

from wft import diag
from wft import reco as wr
from wft.calib import CalibrationBundle


def bundle(tan_max=None):
    return CalibrationBundle(hyper={'c1': 0.05, 'c2': 0.0, 'tau_s': 100.0, 'sigma_s': 10.0, 'sigma_p0': 0.3, 'Dp': 0.02},
                             v_drift=33.0, grid=np.arange(10.0), tmpl={'x': np.ones(10), 'y': np.ones(10)},
                             gain={p: np.ones(512) for p in 'xy'}, tan_max=tan_max)


def test_default_is_reco_constant():
    assert wr.tan_max(bundle()) == wr.TAN_MAX == 0.6


def test_bundle_value_and_roundtrip(tmp_path):
    cal = bundle(1.0)
    assert wr.tan_max(cal) == 1.0
    cal.save(str(tmp_path / 'b'))
    assert CalibrationBundle.load(str(tmp_path / 'b')).tan_max == 1.0
    bundle().save(str(tmp_path / 'c'))
    assert CalibrationBundle.load(str(tmp_path / 'c')).tan_max is None


def test_diag_override_wins():
    old = dict(diag.ACTIVE)
    try:
        diag.override('reco', TAN_MAX=0.8)
        assert wr.tan_max(bundle(1.0)) == 0.8
    finally:
        wr.TAN_MAX = 0.6
        diag.ACTIVE.clear(); diag.ACTIVE.update(old)
