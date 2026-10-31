#!/usr/bin/env python3
"""estimators.py -- column-start timing estimators on the strip sum S(t) that colsum.py stores per trigger and view.

S(t) is the summed waveform of the track's cluster (+- 2 strips), 60 ns samples.  Summed over strips every lateral term of
the forward model drops out (colsum.py docstring), so S is the column's charge-arrival profile convolved with the
electronics template, for any track angle.  The column START is the arrival of the charge made next to the mesh, i.e. the
particle's crossing time plus a fixed delay: it does not depend on the track angle or depth.  Estimators:

  box      colsum's own fit: box start t0 AND duration T free (with a linear charge gradient).  t0 and T trade off.
  fixT     the same box model with T fixed to the row's median T (= gap / v, one number per row and view): only the
           start (and the two linear amplitudes) are fitted.  Removes the t0-T valley.  1 ns grid + parabola.
  edge     the rising edge alone: a step a G(t - t0) fitted to the samples up to the first sample above EDGE_FRAC of
           the maximum of S (the box's far end and its gradient cannot pull it).  1 ns grid + parabola.
  cf       constant fraction on S: where S first crosses CF_FRAC of its maximum, linearly interpolated (model-free).

Absolute times add 10 ns x ftst (the DREAM fine timestamp; t0_abs in the bundles falls 10 ns per ftst step).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fa_gas_v_tilt'))
from colsum import Kern, DTS  # noqa: E402

EDGE_FRAC = 0.8
CF_FRAC = 0.3
FTST_NS = 10.0
T0_FINE = np.arange(-200.0, 1200.0 + 1e-9, 1.0)


def _parab(chi, i, step):
    if 0 < i < len(chi) - 1 and np.all(np.isfinite(chi[i - 1:i + 2])):
        c0, c1, c2 = chi[i - 1], chi[i], chi[i + 1]
        den = c0 - 2 * c1 + c2
        if den > 0:
            return 0.5 * (c0 - c2) / den * step
    return 0.0


class FixT:
    """box start with T fixed: chi2(t0) over a 1 ns grid, (a, b) linear, a > 0, a + b > 0."""

    def __init__(self, kern, ns, T):
        ts = np.arange(ns) * DTS
        B = np.stack([kern.basis(ts, t0, T) for t0 in T0_FINE])           # [n_t0, ns, 2]
        M = np.einsum('isk,isl->ikl', B, B)
        det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] ** 2
        self.good = det > 1e-9 * np.maximum(M[:, 0, 0] * M[:, 1, 1], 1e-30)
        inv = np.zeros_like(M)
        inv[:, 0, 0] = M[:, 1, 1]; inv[:, 1, 1] = M[:, 0, 0]; inv[:, 0, 1] = inv[:, 1, 0] = -M[:, 0, 1]
        self.inv = inv / np.where(self.good, det, 1.0)[:, None, None]
        self.B = B

    def fit(self, y):
        Bty = np.einsum('isk,s->ik', self.B, y)
        c = np.einsum('ikl,il->ik', self.inv, Bty)
        chi = y @ y - np.einsum('ik,ik->i', c, Bty)
        chi = np.where(self.good & (c[:, 0] > 0) & (c[:, 0] + c[:, 1] > 0), chi, np.inf)
        i = int(np.argmin(chi))
        if not np.isfinite(chi[i]):
            return np.nan
        return float(T0_FINE[i] + _parab(chi, i, 1.0))


class Edge:
    """rising edge: y[:k] ~ a G(t - t0), k = first sample above EDGE_FRAC max(y) (+1); a by least squares."""

    def __init__(self, kern, ns):
        ts = np.arange(ns) * DTS
        self.G = np.stack([np.interp(ts - t0, kern.g, kern.G, left=0.0) for t0 in T0_FINE])   # [n_t0, ns]

    def fit(self, y):
        m = y.max()
        if not m > 0:
            return np.nan
        k = int(np.argmax(y > EDGE_FRAC * m)) + 1
        k = max(k, 3)
        G = self.G[:, :k]; yy = y[:k]
        gg = (G * G).sum(1)
        a = np.where(gg > 0, (G @ yy) / np.maximum(gg, 1e-30), 0.0)
        chi = ((yy[None, :] - a[:, None] * G) ** 2).sum(1)
        chi = np.where(a > 0, chi, np.inf)
        i = int(np.argmin(chi))
        if not np.isfinite(chi[i]):
            return np.nan
        return float(T0_FINE[i] + _parab(chi, i, 1.0))


def cf(y, frac=CF_FRAC):
    m = y.max()
    if not m > 0:
        return np.nan
    thr = frac * m
    k = int(np.argmax(y >= thr))
    if k == 0:
        return 0.0
    y0, y1 = y[k - 1], y[k]
    return float((k - 1 + (thr - y0) / max(y1 - y0, 1e-9)) * DTS)


def kernel(grid, tmpl):
    return Kern(grid, tmpl)
