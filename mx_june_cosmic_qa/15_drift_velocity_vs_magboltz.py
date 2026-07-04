#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_drift_velocity_vs_magboltz.py

Overlay the measured det3 drift-velocity curve (14_drift_velocity_scan.py)
with Magboltz predictions for the nominal gas and candidate explanations of
the observed suppression (water contamination, quencher-rich mixtures).

Usage: ../.venv/bin/python 15_drift_velocity_vs_magboltz.py [sat_det3] [--gap=3.0]
Output: <Analysis>/<run>/drift_velocity/<det>/drift_velocity_vs_magboltz[_gap<G>].png

--gap sets the drift gap in cm used to convert HV -> field. Default 3.0:
17_gap_attachment_test.py shows the mechanical gap is ~30 mm (arrival-time
tail extends to ~29 mm; amplitude decays with depth = attachment), and
v*T_sat = 19.4 mm is only the VISIBLE column (threshold crossing).
"""
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()

GAP_CM = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--gap=')), 3.0)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(REPO, 'garfield_sim', 'results')

out = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                   CFG.RUN, 'drift_velocity', CFG.DET_NAME)
df = pd.read_csv(os.path.join(out, 'drift_velocity_scan.csv'))
df = df[df['drift_hv'] >= 300]          # 100 V invalid (window truncation)
e_meas = df['drift_hv'] / GAP_CM
v_err = np.hypot(df['v_ridge_err'], df['v_ridge_sys'])

fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(e_meas, df['v_ridge'], yerr=v_err, fmt='o', color='black', ms=8,
            capsize=4, zorder=5, label=f'measured (ridge fit, gap = {GAP_CM*10:g} mm)')
star = df[df['is_long_run']]
if len(star):
    ax.plot(star['drift_hv'] / GAP_CM, star['v_ridge'], '*', color='crimson',
            ms=18, zorder=6, label='7 h long run')

# nominal gas
nom = json.load(open(os.path.join(GS, 'drift_velocity_Ar_iC4H10_95_5_Saclay.json')))
E = np.array([p['E_Vcm'] for p in nom['points']])
V = np.array([p['v_um_per_ns'] for p in nom['points']])
ax.plot(E, V, '-', color='steelblue', lw=2.5, label='Magboltz Ar/iC4H10 95/5 (nominal)')

# candidates
styles = {
    'Ar94_iso5_H2O1':      ('tab:cyan', ':', 'Ar/iso 95/5 + 1% H2O'),
    'Ar93_iso5_H2O2':      ('tab:green', '--', 'Ar/iso 95/5 + 2% H2O'),
    'Ar92.5_iso5_H2O2.5':  ('tab:olive', '--', 'Ar/iso 95/5 + 2.5% H2O'),
    'Ar92_iso5_H2O3':      ('tab:orange', '--', 'Ar/iso 95/5 + 3% H2O'),
    'Ar90_iso10':          ('tab:purple', '-.', 'Ar/iso 90/10'),
    'Ar85_iso15':          ('tab:pink', '-.', 'Ar/iso 85/15'),
    'Ar80_iso20':          ('tab:red', '-.', 'Ar/iso 80/20'),
    'Ar90_CO2_10':         ('tab:gray', ':', 'Ar/CO2 90/10'),
    'Ar_iso5_air2':        ('tab:brown', ':', 'Ar/iso 95/5 + 2% dry air'),
}
CAND_FILES = ('drift_velocity_candidates.json', 'drift_velocity_candidates2.json',
              'attachment_air_candidates.json')
plotted = set()
for fn in CAND_FILES:
    p = os.path.join(GS, fn)
    if not os.path.exists(p):
        continue
    cands = json.load(open(p))['mixtures']
    for name, pts in cands.items():
        if name not in styles or name in plotted:
            continue
        plotted.add(name)
        c, ls, lab = styles[name]
        Ec = np.array([q['E_Vcm'] for q in pts])
        Vc = np.array([q['v_um_per_ns'] for q in pts])
        ax.plot(Ec, Vc, ls, color=c, lw=1.6, label=lab)

ax.set_xlabel(f'drift field [V/cm]  (E = HV / {GAP_CM:g} cm)')
ax.set_ylabel('drift velocity [µm/ns]')
ax.set_title(f'{CFG.DET_NAME} measured drift velocity vs Magboltz — {CFG.RUN}\n'
             'Ar/iso 95/5 nominal, Saclay 745.8 Torr')
ax.set_xlim(0, 650)
ax.set_ylim(0, 45)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
suffix = '' if abs(GAP_CM - 3.0) < 1e-6 else f'_gap{GAP_CM:g}'
fig.savefig(os.path.join(out, f'drift_velocity_vs_magboltz{suffix}.png'), dpi=170,
            bbox_inches='tight')
print(f'Written {out}/drift_velocity_vs_magboltz{suffix}.png')

# chi2 per candidate for the record
print(f'\nAgreement with measured points (gap = {GAP_CM:g} cm):')
seen = set()
for fn in CAND_FILES + ('attachment_Ar_iso_H2O.json',):
    p = os.path.join(GS, fn)
    if not os.path.exists(p):
        continue
    for name, pts in json.load(open(p))['mixtures'].items():
        if name in seen:
            continue
        seen.add(name)
        Ec = np.array([q['E_Vcm'] for q in pts])
        Vc = np.array([q['v_um_per_ns'] for q in pts])
        pred = np.interp(e_meas, Ec, Vc)
        rms = float(np.sqrt(np.mean((pred - df['v_ridge'])**2)))
        print(f'  {name:22s} RMS dev = {rms:6.2f} µm/ns')
