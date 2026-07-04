#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_attachment_vs_magboltz.py

Close the attachment loop: compare the MEASURED amplitude decay length
lambda(E) from the drift scan (17_gap_attachment_test.py) with the Magboltz
attachment length 1/eta(E) for air-contaminated Ar/iso 95/5
(mm_attachment_air_candidates.py) under the 30 mm gap field scale.

Also shown: clean Ar/CO2 90/10 attaches nothing (lambda = inf), so the
wrong-bottle hypothesis cannot produce the observed decay.

Usage: ../.venv/bin/python 18_attachment_vs_magboltz.py [sat_det3]
Output: <Analysis>/<run>/drift_velocity/<det>/attachment_vs_magboltz.png
"""
import os
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()

GAP_CM = 3.0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(REPO, 'garfield_sim', 'results')

out = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                   CFG.RUN, 'drift_velocity', CFG.DET_NAME)
df = pd.read_csv(os.path.join(out, 'gap_attachment_test.csv'))
e_meas = df['drift_hv'] / GAP_CM

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(e_meas, df['lam_amp_mid_mm'], 'o', color='black', ms=9, zorder=5,
        label='measured: amplitude decay length (mid-column)')
ax.plot(e_meas, df['lam_amp_tail_mm'], 's', color='dimgray', ms=7, zorder=5,
        label='measured: decay length beyond $T_{sat}$ (threshold-steepened)')

styles = {
    'Ar_iso5_air1':   ('tab:cyan', '--', 'Ar/iso 95/5 + 1% dry air (0.21% O$_2$)'),
    'Ar_iso5_air2':   ('tab:blue', '--', 'Ar/iso 95/5 + 2% dry air (0.42% O$_2$)'),
    'Ar_iso5_O2half': ('tab:purple', ':', 'Ar/iso 95/5 + 0.5% O$_2$'),
}
mix = json.load(open(os.path.join(GS, 'attachment_air_candidates.json')))['mixtures']
for name, (c, ls, lab) in styles.items():
    pts = mix[name]
    E = np.array([p['E_Vcm'] for p in pts])
    eta = np.array([p['eta_per_cm'] for p in pts])
    m = eta > 1e-6
    ax.plot(E[m], 10.0 / eta[m], ls, color=c, lw=2,
            label=f'Magboltz $1/\\eta$: {lab}')
ax.annotate('Ar/CO$_2$ 90/10 and Ar/iso 95/5 + H$_2$O only:\n'
            '$\\eta = 0$ (no attachment) — cannot produce the decay',
            (0.03, 0.03), xycoords='axes fraction', fontsize=9, color='tab:red')

ax.set_xlabel(f'drift field [V/cm]  (E = HV / {GAP_CM:g} cm)')
ax.set_ylabel('attachment / decay length [mm]')
ax.set_xlim(100, 420)
ax.set_ylim(0, 40)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='upper left')
ax.set_title(f'{CFG.DET_NAME} measured signal-decay length vs Magboltz attachment\n'
             '(30 mm gap field scale, Saclay 745.8 Torr)')
fig.tight_layout()
fig.savefig(os.path.join(out, 'attachment_vs_magboltz.png'), dpi=170,
            bbox_inches='tight')
print(f'Written {out}/attachment_vs_magboltz.png')
