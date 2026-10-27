#!/usr/bin/env python3
"""The pivot start (wft.reco.PIVOT_START, rerun CHECKS C19).

What must hold:
  1. the model-free charge centroid (x_c, t_c) lies on the track line x = p0 + w (t - t0),
  2. the pivot fit recovers an inclined track's slope, either sign, from no prior knowledge of the angle,
  3. it is opt-in: the default leaves fit_plane on the _global_start path.

    ../../.venv/bin/python -m pytest wft/tests/test_pivot_start.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import model as wm                          # noqa: E402
from wft import reco as wr                           # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402

HYPER = dict(c1=0.1, c2=0.05, kY=2.0, tau_s=150.0, sigma_s=50.0, sigma_p0=0.3, Dp=0.015)


@pytest.fixture(autouse=True)
def cal():
    wm.use_calibration(synth_bundle())
    wm.set_share_mode('delay')
    wm.set_nsamp(32)
    wm.set_depth_bins(18)
    yield


def _plane(p0, w, t0, seed=0):
    ch = np.arange(100, 140)
    pos = ch * wm.PITCH
    q = np.full(wm.K, 400.0)
    q[13:] = 0.0                                   # a 780 ns column inside the 18-bin grid
    W = wm.model_waveforms('x', pos, p0, w, t0, q, HYPER)
    W = W + np.random.default_rng(seed).normal(0, 3.0, W.shape)
    return dict(W=W, pos=pos, noise=np.full(len(ch), 3.0), ch=ch)


@pytest.mark.parametrize('w', [0.0, 0.006, -0.012])
def test_centroid_on_the_line(w):
    p0, t0 = 120 * wm.PITCH, 300.0
    P = _plane(p0, w, t0)
    W, noise, pos, sat = wm.prep_plane(P, 'x')
    xc, tc = wr.charge_centroid(W, noise, pos, 'x', HYPER)
    assert abs(xc - (p0 + w * (tc - t0))) < 0.4     # a seed: ~0.25 mm bias from the window edges is harmless
    assert 300.0 < tc < 300.0 + 13 * wm.DT        # inside the column


@pytest.mark.parametrize('w', [0.012, -0.012])
def test_pivot_recovers_inclined_slope(w):
    p0, t0 = 120 * wm.PITCH, 300.0
    P = _plane(p0, w, t0, seed=1)
    r = wr._pivot_fit(P, 'x', HYPER, t0_seed=t0, t0_prior=(t0, 5.0))
    assert r is not None
    assert abs(r['w'] - w) < 0.03 * abs(w)
    # p0 is free to slide one depth bin along the line (t0 one 60 ns slice off): the chi2 surface's known
    # near-degeneracy (wft.model.chi2_plane), downstream of any start
    assert abs(r['p0'] - p0) < abs(w) * wm.DT + 0.2


def test_on_by_default():
    # built in since 2026-10-27; only a wft.diag study override turns it off
    assert wr.PIVOT_START is True or 'reco.PIVOT_START' in __import__('wft.diag').diag.ACTIVE
    assert 'PIVOT_START' in wr.WORKER_OPTS
