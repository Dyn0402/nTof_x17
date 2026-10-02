#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_qsum_runaway_figures.py -- figures for `qsum_runaway`.

    python -m sept26_prelim_analysis.make_qsum_runaway_figures
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import qsum_runaway as qr  # noqa: E402

CLS_LABEL = {'time': 'runaway, time-censored', 'space': 'runaway, space-censored',
             'normal': 'normal', 'other': 'runaway, other'}


def fig_t0(od, fd):
    h = pd.read_csv(od / 'census_t0_hist.csv')
    fig, (a0, a1) = plt.subplots(2, 1, figsize=fs.FIG, sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.6, 1]))
    rows = []
    for p, c in (('x', fs.ACCENT), ('y', fs.COPPER)):
        s = h[h.plane == p].pivot(index='t0_lo', columns='big', values='n').fillna(0)
        n = s.sum(1)
        f = s[True] / n.where(n > 200)
        x = s.index + 10
        a0.plot(x, f, color=c, lw=1.4, label=f'{p} plane')
        a1.plot(x, n / n.sum(), color=c, lw=1.1)
        rows.append(pd.DataFrame(dict(plane=p, t0=x, n=n.values, frac_big=f.values)))
    for a in (a0, a1):
        a.axvline(qr.LATE_T0, color=fs.MUTED, lw=0.8, ls='--')
    a0.set_ylabel('fraction with q_sum > 1e6')
    a0.set_ylim(0, 1)
    a0.legend(frameon=False, loc='upper left')
    a1.set_ylabel('share of tracks')
    a1.set_xlabel('fitted t0 [ns]')
    fs.title(a0, 'The runaway rate switches on with t0',
             'gated tracks, whole campaign; dashed: the 300 ns class boundary')
    fs.preliminary(a0)
    fs.save(fig, fd / 'runaway_vs_t0', data=pd.concat(rows))


def _bins_table(ex):
    m = ex['meta']
    return pd.DataFrame(dict(arm=m['arm'], event_id=m['event_id'], plane=m['plane'],
                             k=np.arange(len(ex['q_prod'])), arrival_ns=ex['arr'],
                             centre_mm=ex['pk'], column_peak=ex['peak'],
                             q_prod=ex['q_prod'], q_grd=ex['q_grd']))


def fig_example(ex, name, fd, headline):
    m = ex['meta']
    W = ex['W']
    fig = plt.figure(figsize=fs.FULL)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.45, wspace=0.35)
    ts = np.arange(W.shape[1]) * 60.0
    ext = [ts[0] - 30, ts[-1] + 30, ex['pos'][0], ex['pos'][-1]]
    vmax = np.percentile(W, 99.5)
    for j, (img, lab) in enumerate(((W, 'data'), (ex['model_prod'], 'production model'),
                                    (ex['model_grd'], 'guarded model'))):
        ax = fig.add_subplot(gs[j // 2, j % 2]) if j < 2 else fig.add_subplot(gs[1, 0])
        ax.imshow(img, aspect='auto', origin='lower', extent=ext, cmap='magma',
                  vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_title(lab, fontsize=fs.BASE_PT * 0.9)
        ax.set_xlabel('sample time [ns]')
        ax.set_ylabel('strip position [mm]')
    ax = fig.add_subplot(gs[1, 1])
    i = int(np.argmax(W.max(1)))
    ax.plot(ts, W[i], color=fs.INK, lw=1.2, label='data')
    ax.plot(ts, ex['model_prod'][i], color=fs.ACCENT, lw=1.1, label='production')
    ax.plot(ts, ex['model_grd'][i], color=fs.COPPER, lw=1.1, ls='--', label='guarded')
    ax.set_title('brightest strip', fontsize=fs.BASE_PT * 0.9)
    ax.set_xlabel('sample time [ns]')
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.75)
    ax = fig.add_subplot(gs[:, 2])
    k = np.arange(len(ex['q_prod']))
    ax.semilogy(ex['arr'], np.maximum(ex['q_prod'], 1e-1), 'o-', color=fs.ACCENT,
                ms=4, lw=1, label='q, production')
    ax.semilogy(ex['arr'], np.maximum(ex['q_grd'], 1e-1), 's--', color=fs.COPPER,
                ms=3.5, lw=1, label='q, guarded')
    ax.semilogy(ex['arr'], np.maximum(ex['peak'], 1e-20) * 1e3, ':', color=fs.MUTED,
                lw=1.2, label='column peak x 1e3')
    ax.axvline(ex['t_last'] - ex['t_peak'], color=fs.TRACK, lw=0.8)
    ax.text(ex['t_last'] - ex['t_peak'], ax.get_ylim()[1], ' pulse peak\n after last sample',
            fontsize=fs.BASE_PT * 0.7, color=fs.TRACK, va='top')
    ax.set_xlabel('depth-bin arrival time t0 + u [ns]')
    ax.set_ylabel('fitted charge per bin [ADC]')
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.75, loc='lower left')
    _ = k
    fs.fig_title(fig, headline,
                 f"arm {m['arm']} {m['plane']}, event {m['event_id']}: q_sum {m['q_sum_prod']:.2g}"
                 f" -> {m['q_sum_grd']:.3g}; tan {m['tan_theta_prod']:+.3f} -> "
                 f"{m['tan_theta_grd']:+.3f}; t0 {m['t0_prod']:.0f} -> {m['t0_grd']:.0f} ns")
    fs.save(fig, fd / name, data=_bins_table(ex))


def fig_geometry(R, fd):
    R = R.copy()
    R['dtan'] = R.tan_theta_grd - R.tan_theta_prod
    R['dp0'] = R.p0_grd - R.p0_prod
    R['grp'] = np.where(R.cls == 'normal',
                        np.where(R.t0_prod > qr.LATE_T0, 'normal, t0 > 300', 'normal, t0 <= 300'),
                        R.cls.map(CLS_LABEL))
    order = ['normal, t0 <= 300', 'normal, t0 > 300', CLS_LABEL['space'], CLS_LABEL['time']]
    cols = [fs.MUTED, fs.LINE, fs.COPPER, fs.ACCENT]
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE)
    for g, c in zip(order, cols):
        s = R[R.grp == g]
        if not len(s):
            continue
        for ax, v, lim in ((axs[0], s.dtan.abs(), 1e-4), (axs[1], s.dp0.abs(), 1e-3)):
            x = np.sort(np.maximum(v.values, lim))
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=c, lw=1.4,
                    label=f'{g} ({len(s)})')
    axs[0].set_xscale('log'); axs[1].set_xscale('log')
    axs[0].set_xlabel('|tan guarded - tan production|')
    axs[1].set_xlabel('|p0 guarded - p0 production| [mm]')
    axs[0].set_ylabel('cumulative fraction of plane fits')
    axs[0].legend(frameon=False, fontsize=fs.BASE_PT * 0.75, loc='upper left')
    fs.fig_title(fig, 'Everything late moves when the unobservable bins go; early normal fits do not',
                 'run_145 sample, four arms; values below the left edge are piled there')
    fs.preliminary(axs[1])
    fs.save(fig, fd / 'geometry_shift',
            data=R[['arm', 'event_id', 'plane', 'grp', 't0_prod', 't0_grd',
                    'tan_theta_prod', 'tan_theta_grd', 'p0_prod', 'p0_grd']])


def fig_flat(R, fd):
    L = R[R.t0_prod > qr.LATE_T0]
    fig, ax = plt.subplots(figsize=fs.FIG)
    bins = np.linspace(0, 1.5, 61)
    rows = []
    for sel, lab, c in ((L.cls == 'time', 'time-censored runaways', fs.ACCENT),
                        (L.cls == 'normal', 'normal tracks, same t0 band', fs.MUTED)):
        for fit, ls in (('prod', '-'), ('grd', '--')):
            v = L[sel][f'tan_theta_{fit}'].abs().clip(upper=1.5)
            h, _ = np.histogram(v, bins)
            h = h / max(h.sum(), 1)
            ax.step(bins[:-1], h, where='post', color=c, ls=ls, lw=1.3,
                    label=f'{lab}, {"production" if fit == "prod" else "guarded"}')
            rows.append(pd.DataFrame(dict(sample=lab, fit=fit, tan_lo=bins[:-1], frac=h)))
    ax.set_xlabel('|tan theta| (raw, last bin = overflow)')
    ax.set_ylabel('fraction')
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.8)
    fs.title(ax, 'The flat tracks are made by the runaway',
             'fitted t0 > 300 ns; guarded = same fit without the unobservable bins')
    fs.preliminary(ax, 'upper left')
    fs.save(fig, fd / 'flat_tracks', data=pd.concat(rows))


def fig_qobs(R, fd):
    s = R[R.cls != 'normal']
    fig, ax = plt.subplots(figsize=fs.FIG)
    rows = []
    for cls, c, mk in (('space', fs.COPPER, 'o'), ('time', fs.ACCENT, 's')):
        t = s[s.cls == cls]
        ax.loglog(t.q_sum_grd.clip(lower=10), t.q_obs.clip(lower=10), mk, ms=3,
                  color=c, alpha=0.6, label=f'{CLS_LABEL[cls]} ({len(t)})')
        rows.append(t[['arm', 'event_id', 'plane', 'cls', 'q_sum_prod', 'q_obs', 'q_sum_grd']])
    lim = [10, 1e6]
    ax.plot(lim, lim, color=fs.LINE, lw=1)
    ax.set_xlabel('guarded-fit q_sum [ADC]')
    ax.set_ylabel('production q_obs (observable bins only) [ADC]')
    ax.legend(frameon=False)
    fs.title(ax, 'Dropping the unobservable bins after the fact recovers the charge',
             'space-censored: on the diagonal; time-censored: the geometry moved too')
    fs.preliminary(ax, 'lower right')
    fs.save(fig, fd / 'q_obs', data=pd.concat(rows))


def main() -> int:
    fs.use()
    od = qr.out_dir()
    fd = od / 'figures'
    fd.mkdir(exist_ok=True)
    fig_t0(od, fd)
    R = pd.read_parquet(od / 'refit.parquet')
    fig_geometry(R, fd)
    fig_flat(R, fd)
    fig_qobs(R, fd)
    with open(od / 'examples.pkl', 'rb') as f:
        E = pickle.load(f)
    picks = {}
    for ex in E:
        m = ex['meta']
        c = m['cls']
        if c == 'time' and m['t0_prod'] > qr.LATE_T0 and abs(m['tan_theta_prod']) < qr.FLAT_TAN:
            key = 'time'
        elif c == 'space':
            key = 'space'
        else:
            continue
        # the clearest case: largest q runaway with a visible signal
        score = np.log10(max(m['q_sum_prod'], 1)) + np.log10(max(m['wmax'], 1))
        if key not in picks or score > picks[key][0]:
            picks[key] = (score, ex)
    if 'time' in picks:
        fig_example(picks['time'][1], 'example_time', fd,
                    'Time-censored: deep bins arrive after the readout stops')
    if 'space' in picks:
        fig_example(picks['space'][1], 'example_space', fd,
                    'Space-censored: deep bins step off the strip window')
    (od / 'figures.meta.json').write_text(json.dumps(
        {k: dict(arm=v[1]['meta']['arm'], event_id=v[1]['meta']['event_id'],
                 plane=v[1]['meta']['plane'], win=v[1]['meta']['win']) for k, v in picks.items()},
        indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
