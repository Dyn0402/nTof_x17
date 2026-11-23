#!/usr/bin/env python3
"""make_position_res.py -- Paper I v2: bench and beam position resolution in one figure (agent, 2026-11-10).

    .venv/bin/python papers/paper1_v2/figs/make_position_res.py

Top row (a), (b): the bench panels of rerun/paper1_floor/position_ms_fig.py fig() (reads results/position_ms_bins.json);
bottom row (c), (d), (e): the beam panels of rerun/paper1_spsfigs/make_figs.py sps_resolution() (reads
rerun/sps_region/results/region.json).  Plotting code copied from those two, drawn into sub-figures; changes:
bench (b) x label in plain words and the legend without the star (the star is plotted in (a) only, now its own legend
there); no other restyling.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / 'papers'), str(REPO / 'rerun' / 'paper1_spsfigs')]
import paper_style as ps                                           # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402

BINS = REPO / 'rerun' / 'paper1_floor' / 'results' / 'position_ms_bins.json'
REGION = REPO / 'rerun' / 'sps_region' / 'results' / 'region.json'
Z_DET4, Z_BACK = 1120.0, 1370.0


def pointing_mm(f, pitch=1.0):
    return float(np.sqrt((Z_DET4 / Z_BACK) ** 2 * (f * pitch) ** 2 + (1 - Z_DET4 / Z_BACK) ** 2 * (1.0 / np.sqrt(12)) ** 2))


ps.use()
D = json.load(open(BINS))
G = json.load(open(REGION))
r63 = G['resolution']['run63']
sps = dict(x=r63['R|p_mid']['X'], y=r63['R|p0']['Y'])

fig = plt.figure(figsize=(6.1, 5.7), layout='constrained')
top, bot = fig.subfigures(2, 1, height_ratios=[2.6, 2.45])
rows = []

# ------------------------------------------------------------------------------------------- (a), (b) bench
a1, a2 = top.subplots(1, 2)
c3 = ps.color('det3')
for p in ('x', 'y'):
    d = D[f'det3|{p}']
    kw = ps.det('det3', view=p, line=False)
    x = [b['chi2'] for b in d['bins']]; y = [b['sigma_um'] for b in d['bins']]; e = [b['err_um'] for b in d['bins']]
    a1.errorbar(x, y, yerr=e, fmt=kw['marker'], ms=4, color=c3, mfc=kw.get('mfc', c3), mec=c3, zorder=5)
    cx = np.linspace(0, max(x) * 1.05, 100)
    a1.plot(cx, 1e3 * np.sqrt(np.maximum(d['ic_mm2'] + d['slope_mm2'] * cx, 0)), ls=ps.VIEW_LS[p], color=c3, lw=1.0, zorder=4)
    a1.errorbar([0], [d['sigma0_um']], yerr=[d['sigma0_err_um']], fmt='*', ms=8, color=c3,
                mfc=c3 if p == 'x' else 'white', mec=c3, zorder=6, clip_on=False)
    a1.axhline(d['P_um'], color=ps.MUTED, lw=0.9, ls=ps.VIEW_LS[p], zorder=2)
    a1.text(0.74, d['P_um'] + (6 if p == 'x' else -6), f'M3 pointing $P_{p.upper()}$',
            fontsize=ps.BASE_PT * 0.8, color=ps.MUTED, va='bottom' if p == 'x' else 'top', ha='right')
    for b in d['bins']:
        rows.append(dict(panel='a', det='det3', view=p, chi2_ref=b['chi2'], sigma_core_um=b['sigma_um'], err_um=b['err_um'], n=b['n']))
    rows.append(dict(panel='a', det='det3', view=p, chi2_ref=0.0, sigma_core_um=d['sigma0_um'], err_um=d['sigma0_err_um'], n=d['n']))
a1.set_xlim(-0.015, 0.75); a1.set_ylim(0, 900)
a1.set_xlabel(r'M3 $\chi^2$ of the matching coordinate')
a1.set_ylabel(r'residual core $\sigma$ [µm]')
a1.legend([Line2D([], [], ls='none', marker='*', ms=8, color=ps.INK)], [r'$\sigma_0$ ($\chi^2 \to 0$)'],
          loc='lower left', fontsize=ps.BASE_PT * 0.85, handletextpad=0.2, borderaxespad=0.3)

s = np.linspace(0.8, 1.2, 161)
a2.axvspan(0.9, 1.1, color=ps.GRID, lw=0, zorder=0)
for det in ('det2', 'det3'):
    for p in ('x', 'y'):
        d = D[f'{det}|{p}']
        s0, e0, P = d['sigma0_um'], d['sigma0_err_um'], d['P_um']
        ch = np.sqrt(np.maximum(s0 ** 2 - (s * P) ** 2, 0))
        kw = ps.det(det, view=p)
        if det == 'det3':
            lo = np.sqrt(np.maximum((s0 - e0) ** 2 - (s * P) ** 2, 0))
            hi = np.sqrt(np.maximum((s0 + e0) ** 2 - (s * P) ** 2, 0))
            a2.fill_between(s, lo, hi, color=c3, alpha=0.10 if p == 'x' else 0.07, lw=0, zorder=1)
        a2.plot(s, ch, ls=kw['ls'], color=kw['color'], lw=kw['lw'], zorder=kw['zorder'])
        a2.plot([1.0], [d['chamber_um']], marker=kw['marker'], ms=4.5, color=kw['color'],
                mfc=kw.get('mfc', kw['color']), mec=kw['color'], ls='', zorder=kw['zorder'] + 1)
        for si, ci in zip(s[::20], ch[::20]):
            rows.append(dict(panel='b', det=det, view=p, P_scale=si, chamber_um=ci, sigma0_um=s0, P_um=P))
sx, sy = sps['x'], sps['y']
a2.axhspan(min(sy['sigma_det4'], sx['sigma_det4']) * 1e3, max(sy['sigma_det4'], sx['sigma_det4']) * 1e3,
           color=ps.color('det4'), alpha=0.25, lw=0, zorder=0)
a2.text(0.81, sy['sigma_det4'] * 1e3 - 3, 'det4 at SPS', fontsize=ps.BASE_PT * 0.8, color=ps.MUTED, va='top')
a2.set_xlim(0.8, 1.2); a2.set_ylim(0, 330)
a2.set_xlabel('assumed reference pointing error\n(1.0 = nominal estimate)')
a2.set_ylabel(r'chamber $\sqrt{\sigma_0^2 - P^2}$ [µm]')
a2.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
ps.panel_labels([a1, a2])
top.legend(handles=ps.det_handles(('det3', 'det2')) + ps.view_handles(), loc='outside upper center', ncol=4, frameon=False)

# ------------------------------------------------------------------------------------------- (c), (d), (e) beam
fc, frng = G['f_common']['f'], G['f_common']['range']
V = {'X': (r63['R|p_mid']['X'], 'res_b2f2'), 'Y': (r63['R|p0']['Y'], 'res_b2')}
for v, (o, _) in V.items():
    assert abs(pointing_mm(fc) - o['pointing']) < 1e-4, 'pointing model does not reproduce region.json'
axs = bot.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
edges = np.arange(-1.5, 1.5001, 0.05); xc = 0.5 * (edges[1:] + edges[:-1])
c4 = ps.color('det4'); ink = ps.INK
g = lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2)
for ax, v, pl in zip(axs[:2], 'XY', 'cd'):
    o, ub = V[v]
    for key, s_key, lab in (('hist', 'sigma_res', 'all tracks'), ('hist_b2', 'res_b2', '2-strip back cluster')):
        h = np.array(o[key], float); hn = h / h.sum() / 0.05
        sg = o[s_key] if key == 'hist' else o['res_b2'][0]
        core = np.abs(xc) < 1.2
        (a, m), _ = curve_fit(lambda x, a, m: g(x, a, m, sg), xc[core], hn[core], p0=(hn.max(), 0.0))
        full = key == 'hist'
        col = ink if full else ps.ACCENT
        ax.errorbar(xc, hn, yerr=np.sqrt(np.maximum(h, 1)) / h.sum() / 0.05, fmt='o', ms=2.4, lw=0.5, color=col,
                    mfc=col if full else 'white', mec=col, zorder=4 if full else 3)
        xx = np.linspace(-1.2, 1.2, 200)
        ax.plot(xx, g(xx, a, m, sg), '-' if full else '--', color=ps.MODEL, lw=0.9, zorder=5)
        for xi, yi, hi in zip(xc, hn, h):
            rows.append(dict(panel=pl, view=v, sample=key, residual_mm=xi, density=yi, count=hi))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 3.25)
    ax.set_xlabel(f'{v} residual $\\Delta {v.lower()}$ (mm)')
axs[0].set_ylabel('density (mm$^{-1}$)')
ax = axs[2]
ff = np.linspace(0.20, 0.34, 141)
ax.axvspan(frng[0], frng[1], color=ps.GRID, lw=0, zorder=0)
ax.axvline(fc, color=ps.MUTED, lw=0.7, zorder=1)
for v in 'XY':
    o, ub = V[v]
    s_ = o['sigma_res']
    d4 = np.array([np.sqrt(max(s_ ** 2 - pointing_mm(f) ** 2, 0)) for f in ff]) * 1e3
    kw = ps.det('det4', view=v.lower())
    ax.plot(ff, d4, ls=kw['ls'], color=ink, lw=1.2, zorder=3)
    ax.errorbar([fc], [o['sigma_det4'] * 1e3], yerr=[o['stat'] * 1e3], fmt='o', ms=4.5, color=ink,
                mfc=ink if v == 'X' else 'white', mec=ink, zorder=5)
    b, be = o[ub][0] * 1e3, o[ub][1] * 1e3
    ax.axhspan(b - be, b + be, color=ps.ACCENT, alpha=0.18 if v == 'X' else 0.10, lw=0, zorder=1)
    ax.axhline(b, color=ps.ACCENT, lw=0.9, ls=kw['ls'], zorder=2)
    for fi, di in zip(ff[::10], d4[::10]):
        rows.append(dict(panel='e', view=v, f=fi, sigma_det4_um=di, upper_bound_um=b, upper_bound_err_um=be))
ax.set_xlim(0.20, 0.34); ax.set_ylim(0, 300)
ax.set_xlabel(r"back-plane $f$ ($\sigma$ / pitch)"); ax.set_ylabel('det4 resolution $\\sigma$ (µm)')
ax.legend(ps.view_handles(color=ink), ['X', 'Y'], loc='upper right')
ps.panel_labels(axs, start='c')
bot.legend(handles=[Line2D([], [], ls='', marker='o', ms=3, color=ink),
                    Line2D([], [], ls='', marker='o', ms=3, color=ps.ACCENT, mfc='white'),
                    Line2D([], [], color=ps.MODEL, lw=0.9), Patch(fc=ps.ACCENT, alpha=0.18, lw=0)],
           labels=['all tracks', '2-strip back-plane cluster', 'Gaussian core', 'upper bound (2-strip)'],
           loc='outside upper center', ncol=4, frameon=False, handlelength=1.6)

out = HERE / 'out' / 'position_res'
ps.save(fig, out, data=pd.DataFrame(rows))
summ = {'bench': {k: {kk: D[k][kk] for kk in ('sigma0_um', 'sigma0_err_um', 'P_um', 'chamber_um', 'chamber_err_um')}
                  for k in D if k.endswith(('|x', '|y'))},
        'beam': {v: {kk: V[v][0][kk] for kk in ('sigma_res', 'pointing', 'sigma_det4', 'stat')} for v in 'XY'}, 'f': fc}
json.dump(summ, open(out.with_suffix('.json'), 'w'), indent=1)
print('ok', out)
