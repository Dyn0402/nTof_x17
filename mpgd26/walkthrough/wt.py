#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wt.py -- shared machinery for the det3 forward-fit walkthrough.

One real cosmic muon (event 1663 of the ref-pinned calibration cache, the same
event the deck's "One muon through the forward fit" slide uses), taken through
every stage of wft's forward model with nothing re-implemented: the design
matrix, the kernel and the fit all come from ``wft.model`` itself, so if the
model moves, these figures move with it.

Calibration: calib_bundle_r06 (det3, Saturday long run, resistive 490 V /
drift 1000 V) -- the refit with c2 slaved to 0.6 x c1, which is the ratio the
H4 head-on beam measures (0.45 +- 0.02) and near-vertical bench cosmics confirm
(0.63 +- 0.09).  The FROZEN production bundle calib_bundle_lp2_t0p carries
c2/c1 = 1.14 -- a +-2 copy LARGER than the +-1 copy, which cannot happen on a
resistive film because the +-2 strip is reached only through the +-1 strip.  It
appears in exactly one place here, section 10, as the thing that was replaced.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
           os.path.join(REPO, 'cosmic_bench_analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402

ANALYSIS = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
            'mx17_det3_saturday_scan_6-27-26/'
            'long_run_resist_490V_drift_1000V/mx17_3')
# THE calibration this walkthrough runs on: c2 slaved to 0.6 x c1, the
# physically ordered kernel (19_ratio_recal.py / 20_make_ratio_bundle.py).
BUNDLE = os.path.join(ANALYSIS, 'wft', 'calib_bundle_r06')
# The FROZEN production bundle, kept only so section 10 can show what it was:
# it carries c2/c1 = 1.14, i.e. a +-2 copy larger than the +-1 copy.
BUNDLE_PROD = os.path.join(ANALYSIS, 'wft', 'calib_bundle_lp2_t0p')
CACHE = os.path.join(ANALYSIS, 'wft', 'calib_work', 'calib_cache.pkl')
ANGLES = os.path.join(ANALYSIS, 'wft', 'angles', 'angular_resolution.json')
FIG = os.path.join(HERE, 'figures')

EID = 1663
PLANE = 'y'

# the colour rule, identical to mpgd26/make_share.py and deck slides 9-9b:
# blue = the strip's own charge, vermillion = +-1 copies, purple = +-2.
OWN = '#0072B2'
N1 = '#D55E00'
N2 = '#8a3f8f'
DATA = '#111827'
MODEL = '#0072B2'
REF = '#16a34a'
ACC = '#b45309'
GREY = '#6b7280'


def style():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.bbox': 'tight',
        'font.size': 10.5, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5,
        'axes.edgecolor': '#9ca3af', 'axes.labelcolor': '#374151',
        'axes.titlecolor': '#111827', 'text.color': '#374151',
        'xtick.color': '#6b7280', 'ytick.color': '#6b7280',
        'axes.grid': True, 'grid.color': '#9ca3af', 'grid.alpha': 0.18,
        'grid.linewidth': 0.7, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False,
        'figure.dpi': 130,
    })


def save(fig, name):
    p = os.path.join(FIG, name + '.png')
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('  wrote', os.path.relpath(p, HERE))
    return p


def load(bundle=BUNDLE):
    from wft.calib import CalibrationBundle
    from wft import model as wm
    cal = CalibrationBundle.load(bundle)
    wm.use_calibration(cal)
    return cal, wm


def events():
    with open(CACHE, 'rb') as f:
        return pickle.load(f)


def trim(P, keep_frac=0.02, pad=3):
    """The strips production's window extraction hands the fit."""
    W = np.asarray(P['W'], float)
    noise = np.maximum(np.asarray(P['noise'], float), 3.0)
    amp = W.max(axis=1)
    live = np.where((amp > 5 * noise) & (amp > keep_frac * amp.max()))[0]
    if len(live) == 0:
        live = np.array([int(np.argmax(amp))])
    sl = slice(max(0, live.min() - pad), min(W.shape[0] - 1, live.max() + pad) + 1)
    return dict(W=W[sl], pos=np.asarray(P['pos'], float)[sl],
                noise=np.asarray(P['noise'], float)[sl],
                ch=np.asarray(P['ch'])[sl])


def _zero_c2(h):
    h = dict(h)
    h['c2'] = 0.0
    h.pop('c2_over_c1', None)
    return h


def _zero_all(h):
    h = _zero_c2(h)
    h['c1'] = 0.0
    return h


def fit_event(cal, wm, ev, plane=PLANE, hyper=None):
    """Fit one plane the way production fits it: (p0, w, t0) free, the bundle's
    absolute-t0 prior on t0, charges by NNLS. Returns everything the figures
    need, including the model split into own / +-1 / +-2."""
    P = trim(ev[plane])
    wm.set_nsamp(np.asarray(P['W']).shape[1])
    W, noise, pos, sat = wm.prep_plane(P, plane)
    h = dict(hyper if hyper is not None else cal.hyper)
    t0_pred = cal.t0_abs[plane][ev[f'ftst_{plane}']]
    r = wm.fit_plane_raw(P, plane, ev[f'ref_mesh_{plane}'],
                         ev[f'tan_{plane}'] * cal.v_drift * 1e-3, t0_pred,
                         hyper=h, t0_prior=(t0_pred, cal.t0_prior_sigma))
    q, t0, p0, w = r['q'], r['t0'], r['p0'], r['w']

    def build(hh):
        M = wm.build_matrix(plane, pos, p0, w, t0, hh)
        return (M @ q).reshape(len(pos), wm.NSAMP)

    own = build(_zero_all(h))
    with1 = build(_zero_c2(h))
    full = build(h)
    return dict(P=P, W=W, raw=np.asarray(P['W'], float), pos=pos, noise=noise,
                ch=np.asarray(P['ch']), sat=sat,
                t=np.arange(wm.NSAMP) * wm.SNS, nsamp=int(wm.NSAMP),
                own=own, sh1=with1 - own, sh2=full - with1, full=full,
                q=np.asarray(q, float), t0=float(t0), p0=float(p0), w=float(w),
                chi2=float(r['chi2']), dof=int(r['dof']),
                tan=float(w * 1e3 / cal.v_drift),
                tan_ref=float(ev[f'tan_{plane}']),
                p0_ref=float(ev[f'ref_mesh_{plane}']),
                t0_pred=float(t0_pred), hyper=h, plane=plane,
                eid=int(ev['eid']))


def chi2_at(wm, st, p0, w, t0, hyper=None):
    h = hyper if hyper is not None else st['hyper']
    c, _ = wm.chi2_plane(st['plane'], st['W'], st['noise'], st['pos'],
                         st['sat'], p0, w, t0, h, snap_t0=False)
    return float(c)


def core_index(st):
    """Index of the strip with the largest measured amplitude."""
    return int(np.argmax(st['raw'].max(axis=1)))
