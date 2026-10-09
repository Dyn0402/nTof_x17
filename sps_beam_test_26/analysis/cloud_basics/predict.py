#!/usr/bin/env python3
"""predict.py -- from a gas composition to the signals the detector records.

No free loss rate.  For a composition (water %, air %) and the drift field, Magboltz
(gasmodel.GasGrid) gives v, eta and D_L.  A track crossing the gap leaves uniform
ionisation; the charge from depth z arrives at t0 + z/v, spread by
sigma_t = D_L sqrt(z) / v, and survives with probability exp(-eta z).  That current
convolved with the measured electronics template and averaged over the 60 ns sampling
phase is the head-on stack.  Free per field: the amplitude and the trigger latency t0.
The gap is fixed at 30 mm.

    headon_curve(t, comp, E, amp, t0, grid)       -> predicted samples
    fit_headon(t, y, err, comp, E, grid)          -> amp, t0, chi2, curve
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
GAP_MM = 30.0
DT = 10.0
_T = json.load(open(os.path.join(RES, 'template_det3_x.json')))
TG, TM = np.array(_T['grid']), np.array(_T['tmpl_x'])
FINE = np.arange(-400.0, 64 * 60 + 2500, DT)
ZS = np.linspace(0.0, GAP_MM, 601)                 # 50 um depth slices


def current(gas_pt, gap=GAP_MM):
    """arrival current on FINE (t0 = 0), per unit charge."""
    v, eta, DL = gas_pt['v'], gas_pt['eta'], gas_pt['DL']
    z = ZS[ZS <= gap]
    tz = z * 1e3 / v
    sig = np.maximum(DL * np.sqrt(z / 10.0) / v, 1.0)          # ns
    w = np.exp(-eta * z / 10.0) * (z[1] - z[0])
    # each slice: a Gaussian in time on the FINE grid (vectorised in chunks)
    I = np.zeros(len(FINE))
    for c in range(0, len(z), 100):
        tt, ss, ww = tz[c:c + 100, None], sig[c:c + 100, None], w[c:c + 100, None]
        I += (ww * np.exp(-0.5 * ((FINE[None, :] - tt) / ss) ** 2) / (ss * np.sqrt(2 * np.pi))).sum(0)
    return I * DT


TT = np.arange(0.0, 3000.0, DT)


def gamma_template(par):
    """parametric shaper: (t/tau)^n e^(-t/tau) (CR-RC^n), minus a slow undershoot u (1 - e^(-t/tu)) e^(-t/tu2)
    normalised to unit area of the positive part.  par = (n, tau, u, tu)."""
    n, tau, u, tu = par
    h = (TT / tau) ** n * np.exp(-TT / tau)
    h = h / (h.sum() * DT)
    tail = np.exp(-TT / tu) - np.exp(-TT / max(tau, 1.0))
    tail = np.clip(tail, 0, None)
    if tail.sum() > 0:
        h = h - u * tail / (tail.sum() * DT)
    return h


def current_field(lookup, E0, gap=GAP_MM, k=0.0):
    """arrival current for a linear drift-field profile E(z) = E0 (1 + k (1/2 - z/gap)), z = depth
    from the mesh (k > 0: stronger field near the mesh).  lookup(E) -> gasmodel point.  Electrons
    from depth z cross every field between z and the mesh: t(z) = int dz/v, survival
    exp(-int eta dz), sigma_t^2 = int 2 D dz / v^3 (D_L as um/sqrt(cm))."""
    z = ZS[ZS <= gap]
    dz = z[1] - z[0]
    Es = np.linspace(E0 * (1 - 0.5 * abs(k)) * 0.98, E0 * (1 + 0.5 * abs(k)) * 1.02, 9) if k else [E0]
    tab = [lookup(e) for e in Es]
    Ez = E0 * (1 + k * (0.5 - z / gap))
    if k:
        v = np.interp(Ez, Es, [p['v'] for p in tab]); eta = np.interp(Ez, Es, [p['eta'] for p in tab])
        DL = np.interp(Ez, Es, [p['DL'] for p in tab])
    else:
        v = np.full(len(z), tab[0]['v']); eta = np.full(len(z), tab[0]['eta']); DL = np.full(len(z), tab[0]['DL'])
    tz = np.concatenate([[0.0], np.cumsum((1e3 * dz / v)[:-1])])               # ns
    surv = np.exp(-np.concatenate([[0.0], np.cumsum((eta * dz / 10.0)[:-1])]))
    # longitudinal variance: sigma_z^2 = DL^2 * z_cm  per slice traversed -> time via local v
    var_t = np.concatenate([[0.0], np.cumsum(((DL ** 2) * (dz / 10.0) / v ** 2)[:-1])])
    sig = np.maximum(np.sqrt(var_t), 1.0)
    w = surv * dz
    I = np.zeros(len(FINE))
    for c in range(0, len(z), 100):
        tt, ss, ww = tz[c:c + 100, None], sig[c:c + 100, None], w[c:c + 100, None]
        I += (ww * np.exp(-0.5 * ((FINE[None, :] - tt) / ss) ** 2) / (ss * np.sqrt(2 * np.pi))).sum(0)
    return I * DT


def shaped(I, par=None):
    """current (x) electronics (x) 60 ns sampling phase.  par=None: the measured bench template;
    else the parametric shaper (gamma_template)."""
    if par is None:
        s = np.convolve(I, TM)[:len(FINE)]
        off = TG[0]
    else:
        s = np.convolve(I, gamma_template(par))[:len(FINE)] * DT
        off = 0.0
    return np.convolve(s, np.ones(6) / 6, mode='same'), off


def headon_curve(t, gas_pt, amp, t0, gap=GAP_MM, par=None):
    s, off = shaped(current(gas_pt, gap), par)
    return amp * np.interp(t, FINE + off + t0, s)


def fit_headon(t, y, err, gas_pt, gap=GAP_MM, window=(500.0, 3800.0), par=None):
    m = (t >= window[0]) & (t <= window[1])
    base, off = shaped(current(gas_pt, gap), par)

    def f(p):
        return (p[0] * np.interp(t[m], FINE + off + p[1], base) - y[m]) / err[m]
    sol = least_squares(f, [1.0 / max(base.max(), 1e-12), 650.0], x_scale=[0.1 / max(base.max(), 1e-12), 30.0])
    amp, t0 = sol.x
    return dict(amp=float(amp), t0=float(t0), chi2=float(np.sum(sol.fun ** 2)), n=int(m.sum()),
                curve=(amp * np.interp(t, FINE + off + t0, base)).tolist())


def fit_electronics(datasets, comp_fn, par0=(2.0, 80.0, 0.03, 600.0)):
    """shared shaper parameters over several (t, y, err, gas_pt) stacks; per-stack amp, t0 profiled."""
    def f(par):
        if par[0] <= 0.3 or par[1] <= 5 or par[3] <= 50:
            return np.full(sum(int(((d[0] >= 500) & (d[0] <= 3800)).sum()) for d in datasets), 1e3)
        res = []
        for t, y, e, g in datasets:
            r = fit_headon(t, y, e, g, par=par)
            m = (t >= 500) & (t <= 3800)
            res.append((np.array(r['curve'])[m] - y[m]) / e[m])
        return np.concatenate(res)
    sol = least_squares(f, par0, x_scale=[0.3, 10.0, 0.01, 100.0], diff_step=[1e-2, 1e-2, 1e-2, 1e-2])
    return sol.x, float(np.sum(sol.fun ** 2))
