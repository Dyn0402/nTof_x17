#!/usr/bin/env python3
"""make_gas_vE_all.py -- Paper I v2: drift velocity against field, chamber A's scan plus one long-run point for each
of chambers B-E (agent, 2026-11-10).

    .venv/bin/python papers/paper1_v2/figs/make_gas_vE_all.py

Copy of rerun/paper1_gas/pg_figs.py vE() (same panels, same chamber-A markers, panel (b) unchanged: chamber A against
its own fitted curve), with in (a) the golden-row timing velocity of B, C, D, E (results/summary.json golden_water: v,
E, water) and a thin Magboltz curve at each chamber's fitted water fraction, labelled by water fraction.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

REPO = Path(__file__).resolve().parents[3]
GAS = REPO / 'rerun' / 'paper1_gas'
sys.path[:0] = [str(GAS), str(REPO / 'papers')]
import gasinv as GI                                                # noqa: E402
import paper_style as ps                                           # noqa: E402

ps.use()
D = json.load(open(GAS / 'results' / 'gas.json'))
S = json.load(open(GAS / 'results' / 'summary.json'))['golden_water']
TW = json.load(open(GAS / 'results' / 'twov.json'))
sc = D['scan']
gap = sc['fixed']['gap']
w0, (wlo, whi) = sc['fixed']['water'], sc['fixed']['water_68']
c3 = ps.DET_COLOR['det3']
r2 = lambda x: f'{x:.2f}'

fig, (ax, ar) = ps.fig_for(4.3, height=3.5, nrows=2, sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1]))
E = np.linspace(25, 410, 160)
vm = np.array([GI.v_of(w0, e) for e in E])
lo = np.array([GI.v_of(wlo, e) for e in E]); hi = np.array([GI.v_of(whi, e) for e in E])
rows = []

# thin curves at the other chambers' water, labelled by water fraction (drawn first, under chamber A's)
OTH = ('det2', 'det6', 'det7', 'det4')                              # B, C, D, E
for det in OTH:
    w = S[det]['water']
    col = ps.DET_COLOR[det]
    ax.plot(E, [GI.v_of(w, e) for e in E], '-', color=col, lw=0.7, alpha=0.9, zorder=2)
ax.fill_between(E, np.minimum(lo, hi), np.maximum(lo, hi), color=ps.MODEL, alpha=0.12, lw=0, zorder=1)
ax.plot(E, vm, '-', color=ps.MODEL, lw=1.1, zorder=2, label=f'Magboltz, {r2(w0)} % H$_2$O (A, fit)')
ax.plot(E, [GI.v_of(0.0, e) for e in E], ':', color=ps.MUTED, lw=0.9, zorder=2, label='Magboltz, dry')

La = sorted(TW['ladder'].values(), key=lambda L: L['drift_V'])
xa = np.array([L['drift_V'] / (gap / 10) for L in La]); ya = np.array([L['v_ang'] for L in La])
ea = np.array([L['v_ang_err'] for L in La])
ax.errorbar(xa, ya, yerr=ea, fmt='s', ms=4, color=c3, mfc='white', mec=c3, zorder=4, label=r'A: angle-scale $v_\mathrm{d}$ (reference ladder)')
P = sorted(sc['points'], key=lambda p: p['E'])
xt = np.array([p['E'] for p in P]); yt = np.array([p['v'] for p in P]); et = np.array([p['v_err'] for p in P])
ax.errorbar(xt, yt, yerr=et, fmt='o', ms=4.5, color=c3, zorder=5, label=r'A: timing $v_\mathrm{d} = g/T$')
for x, y, e in zip(xa, ya, ea):
    rows.append(dict(kind='A_angle_scale_v', E=x, v=y, v_err=e, magboltz=GI.v_of(w0, x)))
for x, y, e in zip(xt, yt, et):
    rows.append(dict(kind='A_timing_v', E=x, v=y, v_err=e, magboltz=GI.v_of(w0, x)))

# chambers B-E: one long-run timing point each, in their own colour and marker
for det in OTH:
    s = S[det]
    col = ps.DET_COLOR[det]
    ax.errorbar([s['E']], [s['v']], yerr=[s['v_err']], fmt=ps.DET_MARKER[det], ms=4.5, color=col, mec=col, zorder=6,
                ecolor=col, mew=0.6)
    rows.append(dict(kind=f'{ps.NAME[det]}_timing_v', E=s['E'], v=s['v'], v_err=s['v_err'], water_pct=s['water'],
                     magboltz_at_own_water=GI.v_of(s['water'], s['E'])))
# curve labels (water fraction), placed on each curve at the field given
LAB = {'det4': (70, 'left'), 'det7': (95, 'left'), 'det2': (130, 'left'), 'det6': (170, 'left')}
LAB = {'det4': 120, 'det7': 100, 'det2': 140, 'det6': 180}
ax.set_ylabel(r'drift velocity $v_\mathrm{d}$ (µm/ns)'); ax.set_ylim(0, 44)
h, l = ax.get_legend_handles_labels()
order = [3, 2, 0, 1]
ax.legend([h[i] for i in order], [l[i] for i in order], loc='lower right')
mz = lambda x: np.array([GI.v_of(w0, e) for e in x])
ar.axhline(1.0, color=ps.MODEL, lw=1.0)
ar.fill_between(E, np.minimum(lo, hi) / vm, np.maximum(lo, hi) / vm, color=ps.MODEL, alpha=0.12, lw=0)
ar.errorbar(xa, ya / mz(xa), yerr=ea / mz(xa), fmt='s', ms=4, color=c3, mfc='white', mec=c3, zorder=4)
ar.errorbar(xt, yt / mz(xt), yerr=et / mz(xt), fmt='o', ms=4.5, color=c3, zorder=5)
ar.set_ylim(0.9, 1.26); ar.set_xlim(20, 415)
ar.set_xlabel(r'drift field $E = V/g$ (V/cm)'); ar.set_ylabel(r'$v_\mathrm{d}\,/\,v_{\mathrm{d}}^{\mathrm{Magboltz}}$')
ps.panel_labels([ax, ar])
if '--nolabels' not in sys.argv:
    exec(open(Path(__file__).with_name('_gas_labels.py')).read())
out = Path(__file__).resolve().parent / 'out' / 'gas_vE_all'
ps.save(fig, out, data=pd.DataFrame(rows))
json.dump({ps.NAME[d]: dict(E=S[d]['E'], v=S[d]['v'], v_err=S[d]['v_err'], water_pct=S[d]['water']) for d in OTH},
          open(out.with_suffix('.json'), 'w'), indent=1)
print('ok', out)
