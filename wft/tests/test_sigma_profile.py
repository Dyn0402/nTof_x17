#!/usr/bin/env python3
"""The per-event prompt-footprint profile (bundle hyper ``sigma_p0_prof_<plane>``, rerun/det4 2026-10-25).

What must hold:
  1. absent key: fit_plane is the single fit, bit-identical to calling it with the key empty,
  2. a cluster made with a wide footprint is refitted with the wide value when it is offered, at lower chi2,
  3. list-valued hypers survive CalibrationBundle save / load.

    ../../.venv/bin/python -m pytest wft/tests/test_sigma_profile.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import model as wm                          # noqa: E402
from wft import reco as wr                           # noqa: E402
from wft.calib import CalibrationBundle              # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402

HYPER = dict(c1=0.1, c2=0.05, kY=2.0, tau_s=150.0, sigma_s=50.0, sigma_p0=0.25, Dp=0.015)


@pytest.fixture()
def cal():
    c = synth_bundle()
    c.hyper = dict(HYPER)
    wm.use_calibration(c)
    wm.set_share_mode('delay')
    wm.set_nsamp(32)
    wm.set_depth_bins(18)
    yield c


def _plane(sig, w=0.0, seed=0):
    ch = np.arange(100, 140)
    pos = ch * wm.PITCH
    q = np.full(wm.K, 2000.0)
    q[13:] = 0.0
    h = dict(HYPER, sigma_p0_x=sig)
    W = wm.model_waveforms('x', pos, 120 * wm.PITCH, w, 300.0, q, h)
    W = W + np.random.default_rng(seed).normal(0, 3.0, W.shape)
    return dict(W=W, pos=pos, noise=np.full(len(ch), 3.0), ch=ch)


def test_absent_key_is_single_fit(cal):
    P = _plane(0.25)
    a = wr.fit_plane(P, 'x', cal, hyper=dict(HYPER))
    b = wr.fit_plane(P, 'x', cal, hyper=dict(HYPER, sigma_p0_prof_x=[]))
    assert (a.p0, a.w, a.t0, a.chi2) == (b.p0, b.w, b.t0, b.chi2)
    assert a._sigma_p0 == 0.25


def test_wide_cluster_takes_wide_footprint(cal):
    P = _plane(0.55, seed=2)
    one = wr.fit_plane(P, 'x', cal, hyper=dict(HYPER))
    prof = wr.fit_plane(P, 'x', cal, hyper=dict(HYPER, sigma_p0_prof_x=[0.40, 0.55]))
    assert prof._sigma_p0 == 0.55
    assert prof.chi2 < one.chi2
    assert abs(prof.w) < 1e-3                     # head-on recovered (|tan| < ~0.03)


def test_list_hyper_round_trip(cal, tmp_path):
    cal.hyper = dict(HYPER, sigma_p0_prof_x=[0.40, 0.55])
    cal.save(str(tmp_path / 'b'))
    back = CalibrationBundle.load(str(tmp_path / 'b'))
    assert back.hyper['sigma_p0_prof_x'] == [0.40, 0.55]
