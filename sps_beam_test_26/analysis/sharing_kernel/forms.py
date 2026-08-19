#!/usr/bin/env python3
"""forms.py -- the candidate sharing-kernel forms, and a fit that needs no
deconvolution.

THE IDENTITY.  At normal incidence every strip is driven by the same signal
C(t) (ionisation column x amplifier) passed through a lateral network:

        W_d = n_d (*) C            n_d = the strip-d transfer function

so for any two offsets the measured traces obey the CROSS-RELATION

        n_0 (*) W_d  ==  n_d (*) W_0                                    (*)

C cancels identically and nothing has to be inverted -- no Wiener filter, no
regularisation, no choice of lambda.  Both sides are the measured data
convolved with a MODEL filter, so the fit is well posed even where |W_0(f)| is
small.  Causality also makes the window truncation harmless: (n (*) W)[i] only
ever reaches back to W[<= i], and the pre-pulse region really is zero (the
clean stacks sit at 1.7 % of peak there), so the missing tail beyond the window
is never needed.

THE NETWORK.  The primary cloud puts a geometric fraction q_j on strip j (the
strip integral of the avalanche footprint; a nuisance parameter here, and the
one thing that MUST move with drift field).  Each of those then disperses
laterally through the resistive layer:

        n_d = sum_j q_j k_(|d-j|)

and the three candidate k are:

  cascade   k_m = c^m * (m-fold cascade of a one-pole of time constant tau)
            the RC-ladder form.  wft implements this as share_mode='lp'.
            Enforces c2 = c1^2 -- the ladder cannot invert.
  delay     k_m = c_m * delta(t - m*tau) smeared by sigma_s
            wft's shipped share_mode='delay'.  c1 and c2 independent.
  ladder    the cascade with the +-2 step's amplitude set FREE instead of
            c^2 -- the test of whether the ladder constraint itself holds.
  geom      k_m = 0 for m > 0 -- no lateral transfer at all, every neighbour
            signal is direct geometric charge.  The null.
"""
from __future__ import annotations

import numpy as np

SNS = 60.0


def one_pole(n, tau, dt=SNS):
    """Discrete one-pole impulse response on the sample grid, unit area."""
    a = np.exp(-dt / max(tau, 1e-3))
    h = np.zeros(n)
    acc = 0.0
    for i in range(n):
        acc = acc * a + (1.0 - a if i == 0 else 0.0)
        h[i] = acc
    return h


def cascade_k(m, tau, c, n, dt=SNS):
    """c^m x the m-fold cascade of a one-pole. m=0 -> a unit impulse."""
    k = np.zeros(n)
    k[0] = 1.0
    for _ in range(m):
        k = np.convolve(k, one_pole(n, tau, dt))[:n]
    return (c ** m) * k


def delay_k(m, tau, c, sigma_s, n, dt=SNS):
    """c x a unit impulse translated to m*tau and Gaussian-smeared."""
    if m == 0:
        k = np.zeros(n)
        k[0] = 1.0
        return k
    t = np.arange(n) * dt
    g = np.exp(-0.5 * ((t - m * tau) / max(sigma_s, dt / 3)) ** 2)
    s = g.sum()
    return (c * g / s) if s > 0 else g


def build_n(form, dmax, q, par, n, jmax=None):
    """{d: n_d} for d in -dmax..dmax.  q is {j: q_j} with q[0] == 1."""
    jmax = jmax if jmax is not None else max(abs(j) for j in q)
    out = {}
    for d in range(-dmax, dmax + 1):
        acc = np.zeros(n)
        for j, qj in q.items():
            m = abs(d - j)
            if form == 'geom':
                if m == 0:
                    acc[0] += qj
                continue
            if m == 0:
                acc[0] += qj
                continue
            if form == 'cascade':
                acc += qj * cascade_k(m, par['tau'], par['c'], n)
            elif form == 'ladder':
                k = cascade_k(m, par['tau'], 1.0, n)
                amp = par['c'] if m == 1 else (par['c2'] if m == 2
                                               else par['c'] ** m)
                acc += qj * amp * k
            elif form == 'delay':
                cm = par['c1'] if m == 1 else (par['c2'] if m == 2 else 0.0)
                if cm:
                    acc += qj * delay_k(m, par['tau'], cm, par['sigma_s'], n)
            else:
                raise ValueError(form)
        out[d] = acc
    return out


def cross_resid(nn, W, dlist, lo, hi):
    """The cross-relation residual (*) over the fit window [lo, hi] samples."""
    n0 = nn[0]
    L = len(W[0])
    r = []
    for d in dlist:
        a = np.convolve(nn[0], W[d])[:L]
        b = np.convolve(nn[d], W[0])[:L]
        r.append((a - b)[lo:hi])
    return np.concatenate(r)
