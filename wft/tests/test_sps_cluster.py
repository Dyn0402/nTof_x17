"""SPS head-on cluster (rerun/model_audit/sps_cluster.py): centroid, footprint, censored strips, pile-up out of time."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                'rerun', 'model_audit'))
import sps_cluster as sc                                         # noqa: E402


def _win(x0=10.3, n=21, ns=32, amps=(0.3, 1.0, 0.3), t=12, extra=None):
    pos = np.arange(n) * 0.78; ch = np.arange(100, 100 + n)
    W = np.zeros((n, ns)); tt = np.arange(ns)
    i0 = int(np.argmin(np.abs(pos - x0)))
    for d, a in zip((-1, 0, 1), amps):
        W[i0 + d] = 800 * a * np.exp(-0.5 * ((tt - t) / 2.5) ** 2)
    if extra:                                                      # (strip, sample, amp): pile-up
        i, s, a = extra
        W[i] += a * np.exp(-0.5 * ((tt - s) / 2.5) ** 2)
    return dict(W=W, pos=pos, ch=ch), pos[i0], ch[i0]


def test_symmetric_cluster():
    P, x, _ = _win()
    c = sc.cluster(P)
    assert c['n_strips'] == 3 and abs(c['pos'] - x) < 1e-6
    assert abs(c['lead_frac'] - 1 / 1.6) < 1e-3 and abs(c['nb_lo'] - 0.3) < 1e-3 and abs(c['nb_hi'] - 0.3) < 1e-3


def test_sharing_moves_centroid():
    P, x, _ = _win(amps=(0.1, 1.0, 0.5))
    assert sc.cluster(P)['pos'] > x


def test_out_of_time_pileup_ignored():
    P, x, _ = _win(extra=(18, 28, 2000.0))                          # brighter, but 16 samples later and far away
    c = sc.cluster(P)
    assert c['n_strips'] == 1 and abs(c['pos'] - 18 * 0.78) < 1e-6   # the lead is the brightest good strip ...
    P, x, _ = _win(extra=(14, 28, 300.0))                           # ... a later pulse next to the cluster does not join it
    c = sc.cluster(P)
    assert c['n_strips'] == 3 and abs(c['pos'] - x) < 1e-6


def test_censored_strip_bridged_not_counted():
    P, x, ch0 = _win(amps=(0.6, 1.0, 0.6))
    c = sc.cluster(P, bad=[ch0 + 1])
    assert c['n_strips'] == 2 and np.isnan(c['nb_hi'])
    c = sc.cluster(P, bad=[ch0])                                    # lead censored: the next good strip leads, bridges
    assert c['n_strips'] == 2 and abs(c['pos'] - x) < 1e-6


def test_below_threshold():
    P, _, _ = _win(amps=(0.0, 0.05, 0.0))
    assert sc.cluster(P) is None
