#!/usr/bin/env python3
"""The kernel-ordering gate: an inverted bundle (c2 > c1) cannot be loaded,
saved or installed.

Added 2026-08-21, when every product built on an inverted kernel was retired.
The +-2 strip is reached only through the +-1 strip, so c2 < c1 always; the
ref-pinned cosmic chi2 is flat in that direction and an unconstrained fit walks
there, which is how det3 (1.14), det2 (1.53), det7 (1.75) and det4 (2.12) got
one in the first place. What must hold:

  1. load() of an inverted bundle raises,
  2. save() of one raises -- a refit cannot write one either,
  3. use_calibration() raises, so no path reaches the forward model,
  4. the slaved form (c2_over_c1) is judged on the EFFECTIVE c2, not the
     stored 0.0,
  5. WFT_ALLOW_INVERTED_KERNEL=1 downgrades it to a warning, for reports
     *about* the defect,
  6. a physical bundle is untouched.

    ../../.venv/bin/python -m pytest wft/tests/test_c2_gate.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from wft.calib import (CalibrationBundle, check_kernel_ordering,  # noqa: E402
                       effective_c2, C2_GATE_ENV)
from wft import model as wm                                       # noqa: E402


def _hyper(c1=0.05, c2=0.03, ratio=None):
    h = dict(c1=c1, c2=c2, kY=1.0, tau_s=140.0, sigma_s=20.0,
             sigma_p0=0.3, Dp=0.013)
    if ratio is not None:
        h['c2_over_c1'] = ratio
        h['c2'] = 0.0
    return h


def _bundle(h):
    g = np.linspace(-200, 1400, 200)
    t = np.exp(-0.5 * ((g - 300) / 120) ** 2)
    return CalibrationBundle(hyper=h, v_drift=36.6, grid=g,
                             tmpl={'x': t, 'y': t},
                             gain={'x': np.ones(512), 'y': np.ones(512)},
                             detector='mx17_test', run_key='unit')


def test_physical_passes():
    check_kernel_ordering(_hyper(0.05, 0.03))
    check_kernel_ordering(_hyper(0.05, ratio=0.6))


def test_inverted_raises():
    with pytest.raises(ValueError, match='inverted sharing kernel'):
        check_kernel_ordering(_hyper(0.0509, 0.0580))


def test_effective_c2_uses_the_ratio():
    # the stored c2 is 0.0 on a slaved bundle; the gate must not be fooled
    assert effective_c2(_hyper(0.05, ratio=0.6)) == pytest.approx(0.03)
    assert effective_c2(_hyper(0.05, ratio=1.4)) == pytest.approx(0.07)
    with pytest.raises(ValueError):
        check_kernel_ordering(_hyper(0.05, ratio=1.4))


def test_save_and_load_are_gated(tmp_path):
    bad = _bundle(_hyper(0.0509, 0.0580))
    with pytest.raises(ValueError):
        bad.save(str(tmp_path / 'bad'))
    good = _bundle(_hyper(0.05, 0.03))
    p = good.save(str(tmp_path / 'good'))
    assert CalibrationBundle.load(p).hyper['c2'] == pytest.approx(0.03)


def test_use_calibration_is_gated():
    with pytest.raises(ValueError):
        wm.use_calibration(_bundle(_hyper(0.0509, 0.0580)))


def test_escape_hatch(monkeypatch, capsys):
    monkeypatch.setenv(C2_GATE_ENV, '1')
    check_kernel_ordering(_hyper(0.0509, 0.0580), where='a parked bundle')
    assert 'WARNING' in capsys.readouterr().out
