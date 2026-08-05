#!/usr/bin/env python3
"""
Part II figures — the measurement that forced the rebuild.

A per-strip hit time is an aggregate of that strip's own charge and delayed,
dispersed copies of its neighbours'. The consequences are re-measured here from
scratch on `sat_det3`:

  * the S-shaped time residual vs drift depth (compression),
  * its estimator-independence (four different time estimators, same S),
  * the implied drift velocity falling with track angle — the signature that
    tells you a chain has the bias,
  * and the same event fitted both ways.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import wftdoc as K
from wftdoc import C, save

from wft import model as wm

CAL = None
EVS = None
SNS = 60.0


def setup():
    global CAL, EVS
    CAL = K.install()
    EVS = K.calib_events()
    return CAL, EVS


# --------------------------------------------------------------- estimators
def t_rising(w, noise, frac=0.20):
    """Leading-edge crossing at `frac` of the strip's own peak."""
    ipk = int(np.argmax(w))
    a = w[ipk]
    if a <= 5 * noise:
        return np.nan
    thr = frac * a
    for k in range(1, ipk + 1):
        if w[k] >= thr > w[k - 1]:
            return SNS * (k - 1 + (thr - w[k - 1]) / (w[k] - w[k - 1]))
    return np.nan


def t_cfd(w, noise):
    return t_rising(w, noise, 0.50)


def t_peak(w, noise):
    ipk = int(np.argmax(w))
    if w[ipk] <= 5 * noise or ipk < 1 or ipk >= len(w) - 1:
        return np.nan
    a, b, c = w[ipk - 1], w[ipk], w[ipk + 1]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den != 0 else 0.0
    return SNS * (ipk + d)


def t_matched(w, noise, tmpl, grid):
    """Matched filter against the calibrated impulse response: the shift that
    maximises the template's correlation with the strip's waveform."""
    ipk = int(np.argmax(w))
    if w[ipk] <= 5 * noise:
        return np.nan
    ts = np.arange(len(w)) * SNS
    shifts = np.arange(-200, 1400, 10.0)
    best, bt = -np.inf, np.nan
    for s in shifts:
        h = np.interp(ts - s, grid, tmpl, left=0, right=0)
        n = np.dot(h, h)
        if n <= 0:
            continue
        c = np.dot(h, w) ** 2 / n
        if c > best:
            best, bt = c, s
    # the template is aligned on its own 50 % crossing, so the returned shift
    # is directly comparable with the CFD estimator
    return bt


# ------------------------------------------------- 1. the S-curve, re-measured
def collect_residuals(n=320, plane='x', tan_min=0.10):
    """For every strip of every clean reference-matched event: the strip's hit
    time from four estimators, minus the drift time the reference line implies
    for that strip's position. t0 is floated per event (median offset)."""
    v = CAL.v_drift
    grid = np.asarray(CAL.grid, float)
    out = {k: ([], []) for k in ('rising 20 %', 'CFD 50 %', 'peak',
                                 'matched filter')}
    for eid in sorted(EVS)[:n]:
        e = EVS[eid]
        if plane not in e:
            continue
        tan = e[f'tan_{plane}']
        if abs(tan) < tan_min:
            continue
        P = K.trim_window(e[plane])
        W = np.asarray(P['W'], float)
        pos = np.asarray(P['pos'], float)
        noise = np.maximum(np.asarray(P['noise'], float), 3.0)
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        tmpl = np.asarray(CAL.tmpl[plane], float)
        p0 = e[f'ref_mesh_{plane}']
        # drift time the reference line implies for a strip at position p:
        #   p = p0 + w u  ->  u = (p - p0) / w
        w_ref = tan * v * 1e-3
        u_ref = (pos - p0) / w_ref
        for name, fn in (('rising 20 %', lambda x, s: t_rising(x, s, 0.20)),
                         ('CFD 50 %', t_cfd),
                         ('peak', t_peak),
                         ('matched filter',
                          lambda x, s: t_matched(x, s, tmpl, grid))):
            ts = np.array([fn(W[i], noise[i]) for i in range(len(pos))])
            m = np.isfinite(ts) & (u_ref > -60) & (u_ref < 1000) & \
                (W.max(axis=1) > 6 * noise)
            if m.sum() < 4:
                continue
            resid = ts[m] - u_ref[m]
            resid = resid - np.median(resid)      # float t0 per event
            out[name][0].extend(u_ref[m])
            out[name][1].extend(resid)
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


def fig_compression(res):
    bins = np.array([0, 120, 250, 380, 510, 640, 770, 900])
    ctr = 0.5 * (bins[:-1] + bins[1:])
    v = CAL.v_drift
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.7),
                            gridspec_kw=dict(width_ratios=[1.15, 1]))

    ax = axs[0]
    cols = [C['blue'], C['orange'], C['green'], C['purple']]
    tbl = {}
    for (name, (u, r)), col in zip(res.items(), cols):
        med = [np.median(r[(u >= lo) & (u < hi)])
               if ((u >= lo) & (u < hi)).sum() > 20 else np.nan
               for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot(ctr, med, 'o-', color=col, label=name)
        tbl[name] = np.round(med, 0)
        # slope of the residual vs depth = the compression
        m = np.isfinite(med)
        s = np.polyfit(ctr[m], np.array(med)[m], 1)[0]
        print(f'[hits] {name:15s} residual slope {s:+.3f} ns/ns  '
              f'-> ladder compressed to {1+s:.2f} of true, '
              f'implied v = {v/(1+s):.1f} um/ns')
    ax.axhline(0, color=K.CHROME, lw=0.8)
    ax.set_xlabel('true drift time of that strip [ns]   (reference line)')
    ax.set_ylabel('hit time − true drift time [ns]')
    ax.set_title('every estimator shows the same S:\nlate at the mesh, early '
                 'at the cathode', loc='left')
    ax.legend(fontsize=7.5)
    sec = ax.secondary_xaxis('top', functions=(lambda x: x * v * 1e-3,
                                               lambda z: z / (v * 1e-3)))
    sec.set_xlabel('drift depth [mm]', color=K.CHROME)
    sec.tick_params(colors=K.CHROME)

    ax = axs[1]
    u, r = res['matched filter']
    ax.hexbin(u, r, gridsize=42, cmap='magma', mincnt=1,
              extent=(0, 950, -400, 400))
    ax.axhline(0, color='w', lw=1.0)
    ax.set_xlabel('true drift time [ns]')
    ax.set_ylabel('hit time − true drift time [ns]')
    ax.set_title(f'the matched filter, per strip (n = {len(u):,})', loc='left')
    ax.grid(False)
    save(fig, 'compression')
    return tbl


# ------------------------------------------------------ 2. the same event
def ladder_tan(P, plane, v):
    """The production recipe, reproduced: amplitude-weighted line through
    (position, hit time), anchored at the earliest hit."""
    W = np.asarray(P['W'], float)
    pos = np.asarray(P['pos'], float)
    noise = np.maximum(np.asarray(P['noise'], float), 3.0)
    t_hit = np.array([t_cfd(W[i], noise[i]) for i in range(len(pos))])
    amp = W.max(axis=1)
    m = np.isfinite(t_hit) & (amp > 6 * noise) & (amp > 0.10 * amp.max())
    if m.sum() < 3:
        return np.nan, t_hit, m, np.nan
    pw, tw, aw = pos[m], t_hit[m], amp[m]
    i0 = int(np.argmin(tw))
    A = np.vstack([pw - pw[i0], np.ones_like(pw)]).T
    Wt = np.sqrt(aw)
    sol, *_ = np.linalg.lstsq(A * Wt[:, None], (tw - tw[i0]) * Wt, rcond=None)
    tan = 1.0 / (sol[0] * v * 1e-3) if sol[0] != 0 else np.nan
    return tan, t_hit, m, tw[i0]


def fig_ladder_population(n=380):
    """Where the claim actually lives: the ratio of reconstructed to true slope
    over the population, both chains, on identical events."""
    v = CAL.v_drift
    h = dict(CAL.hyper)
    rh, rf = [], []
    for eid in sorted(EVS)[:n]:
        e = EVS[eid]
        if 'x' not in e or abs(e['tan_x']) < 0.10:
            continue
        P = K.trim_window(e['x'])
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        tan_h, _t, _m, _a = ladder_tan(P, 'x', v)
        if not np.isfinite(tan_h):
            continue
        try:
            r = wm.fit_plane_raw(P, 'x', e['ref_mesh_x'],
                                 e['tan_x'] * v * 1e-3, 400.0, hyper=h)
        except Exception:
            continue
        rh.append(tan_h / e['tan_x'])
        rf.append((r['w'] * 1e3 / v) / e['tan_x'])
    rh, rf = np.array(rh), np.array(rf)
    print(f'[hits] slope ratio to truth — hits ladder median {np.median(rh):.3f}, '
          f'forward fit {np.median(rf):.3f}  (n={len(rh)})')

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    b = np.linspace(0.3, 2.2, 70)
    ax.hist(rh, bins=b, histtype='step', lw=2, color=C['prod'],
            label=f'hits ladder (median {np.median(rh):.2f})')
    ax.hist(rf, bins=b, histtype='step', lw=2, color=C['orange'],
            label=f'forward fit (median {np.median(rf):.2f})')
    ax.axvline(1.0, color=C['ref'], ls='--', lw=1.5, label='truth')
    ax.set_xlabel(r'reconstructed tan$\theta$ / reference tan$\theta$')
    ax.set_ylabel('planes')
    ax.set_title('the same events through both chains', loc='left')
    ax.legend(fontsize=8)
    save(fig, 'ladder_population')


def fig_two_ladders(eid=1663, plane='x'):
    e = EVS[eid]
    P = K.trim_window(e[plane])
    W = np.asarray(P['W'], float)
    pos = np.asarray(P['pos'], float)
    noise = np.maximum(np.asarray(P['noise'], float), 3.0)
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    v = CAL.v_drift
    h = dict(CAL.hyper)
    tan_ref, p0_ref = e[f'tan_{plane}'], e[f'ref_mesh_{plane}']

    r = wm.fit_plane_raw(P, plane, p0_ref, tan_ref * v * 1e-3, 400.0, hyper=h)
    t_hit = np.array([t_cfd(W[i], noise[i]) for i in range(len(pos))])
    amp = W.max(axis=1)
    m = np.isfinite(t_hit) & (amp > 6 * noise) & (amp > 0.10 * amp.max())

    # the production ladder: amplitude-weighted line through (pos, time),
    # anchored at the earliest hit  (cosmic_micro_tpc_analysis._fit_single_axis)
    pw, tw, aw = pos[m], t_hit[m], amp[m]
    i0 = int(np.argmin(tw))
    A = np.vstack([pw - pw[i0], np.ones_like(pw)]).T
    Wt = np.sqrt(aw)
    sol, *_ = np.linalg.lstsq(A * Wt[:, None], (tw - tw[i0]) * Wt, rcond=None)
    slope_ns_mm = sol[0]                       # ns per mm
    tan_hits = 1.0 / (slope_ns_mm * v * 1e-3) if slope_ns_mm != 0 else np.nan
    print(f'[hits] event {eid} {plane}: reference tan {tan_ref:+.3f}, '
          f'hits ladder tan {tan_hits:+.3f}, forward fit '
          f'{r["w"]*1e3/v:+.3f}')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axs[0]
    t = np.arange(wm.NSAMP) * SNS
    off = 0.42 * W.max()
    for i in range(len(pos)):
        col = C['blue'] if m[i] else C['grey']
        ax.plot(t, W[i] + i * off, color=col, lw=1.1, alpha=0.9)
        if m[i]:
            ax.plot(t_hit[i], 0.5 * amp[i] + i * off, 'v', ms=6,
                    color=C['red'])
    ax.plot([], [], 'v', color=C['red'], ms=6, label='CFD 50 % hit time')
    ax.set_xlabel('time [ns]')
    ax.set_yticks([])
    ax.set_ylabel('strip (offset)')
    ax.set_title(f'event {eid} {plane}: one time per strip', loc='left')
    ax.legend(fontsize=8)

    ax = axs[1]
    z_hit = (t_hit[m] - tw[i0]) * v * 1e-3
    ax.plot(pw, z_hit, 'o', color=C['prod'], ms=7,
            label=f'hits ladder, tan = {tan_hits:+.3f}')
    zz = np.array([0, 29.0])
    ax.plot(p0_ref + tan_ref * zz, zz, color=C['ref'], lw=2,
            label=f'M3 reference, tan = {tan_ref:+.3f}')
    ax.plot(r['p0'] + (r['w'] * 1e3 / v) * zz, zz, color=C['orange'], lw=2,
            ls='--', label=f'forward fit, tan = {r["w"]*1e3/v:+.3f}')
    q = np.asarray(r['q'], float)
    zc = wm.UK * v * 1e-3
    ax.scatter(r['p0'] + r['w'] * wm.UK, zc, s=90 * q / max(q.max(), 1e-9) + 3,
               color=C['orange'], alpha=0.35, zorder=1,
               label='fitted charge profile')
    ax.set_xlabel('transverse position [mm]')
    ax.set_ylabel('drift depth [mm]')
    ax.set_ylim(-1, 31)
    ax.invert_yaxis()
    ax.set_title('the hits ladder reads steeper than the truth', loc='left')
    ax.legend(fontsize=7.5)
    save(fig, 'two_ladders')


# ------------------------------------------------- 3. implied-v vs angle
def fig_implied_v(n=380):
    """The signature test. A geometrically honest reconstruction gives the same
    drift velocity in every angle bin. The hits ladder does not."""
    v = CAL.v_drift
    h = dict(CAL.hyper)
    rows = []
    for eid in sorted(EVS)[:n]:
        e = EVS[eid]
        if 'x' not in e:
            continue
        tan = e['tan_x']
        if abs(tan) < 0.08:
            continue
        P = K.trim_window(e['x'])
        W = np.asarray(P['W'], float)
        pos = np.asarray(P['pos'], float)
        noise = np.maximum(np.asarray(P['noise'], float), 3.0)
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        t_hit = np.array([t_cfd(W[i], noise[i]) for i in range(len(pos))])
        amp = W.max(axis=1)
        m = np.isfinite(t_hit) & (amp > 6 * noise) & (amp > 0.10 * amp.max())
        if m.sum() < 3:
            continue
        pw, tw, aw = pos[m], t_hit[m], amp[m]
        i0 = int(np.argmin(tw))
        A = np.vstack([pw - pw[i0], np.ones_like(pw)]).T
        Wt = np.sqrt(aw)
        sol, *_ = np.linalg.lstsq(A * Wt[:, None], (tw - tw[i0]) * Wt,
                                  rcond=None)
        if sol[0] == 0:
            continue
        v_hits = 1.0 / (sol[0] * tan) * 1e3        # um/ns implied by the ladder
        try:
            r = wm.fit_plane_raw(P, 'x', e['ref_mesh_x'], tan * v * 1e-3,
                                 400.0, hyper=h)
        except Exception:
            continue
        rows.append((abs(tan), v_hits, r['w'] * 1e3 / tan))
    a = np.array(rows)
    bins = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]
    ctr, mh, mf = [], [], []
    for lo, hi in bins:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        ctr.append(0.5 * (lo + hi))
        mh.append(np.median(a[m, 1]) if m.sum() > 3 else np.nan)
        mf.append(np.median(a[m, 2]) if m.sum() > 3 else np.nan)
    print(f'[hits] implied v, hits ladder: {np.round(mh,1)} '
          f'(spread {np.nanmax(mh)-np.nanmin(mh):.1f})')
    print(f'[hits] implied v, forward fit: {np.round(mf,1)} '
          f'(spread {np.nanmax(mf)-np.nanmin(mf):.1f})')

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(ctr, mh, 'o-', color=C['prod'], ms=7,
            label=f'hits ladder  (spread {np.nanmax(mh)-np.nanmin(mh):.0f} µm/ns)')
    ax.plot(ctr, mf, 'o-', color=C['orange'], ms=7,
            label=f'forward fit  (spread {np.nanmax(mf)-np.nanmin(mf):.1f} µm/ns)')
    ax.axhline(v, color=K.CHROME, ls='--', lw=1,
               label=f'calibration v = {v:.1f} µm/ns')
    ax.set_xlabel(r'|tan$\theta$|  (reference)')
    ax.set_ylabel(r'implied drift velocity, median $w/\tan\theta$ [µm/ns]')
    ax.set_title('the compression signature: a real velocity does not\n'
                 'depend on how steep the track was', loc='left')
    ax.legend(fontsize=8)
    save(fig, 'implied_v')


def main():
    setup()
    print('[hits] measuring per-strip time residuals ...')
    res = collect_residuals()
    fig_compression(res)
    fig_two_ladders()
    fig_ladder_population()
    fig_implied_v()


if __name__ == '__main__':
    main()
