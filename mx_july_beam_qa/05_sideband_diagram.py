"""
05_sideband_diagram.py — Explanatory diagram of the combinatorial-background
estimation, drawn on real data (WALC-PSSC, tof > 1 ms, from the 02 cache).

Usage: python 05_sideband_diagram.py [run_stem]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '05_sideband_diagram'
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(BASE / 'cache' / f'02_coinc_{RUN_STEM}.npz')
cen = 0.5 * (d['dt_edges'][:-1] + d['dt_edges'][1:])
hh = d['WALC_PSSC'].sum(axis=0)[5]           # >1ms region, ded+para summed

SB = 100                                      # sideband starts at |dt| > SB
side = np.abs(cen) > SB
base = hh[side].mean()
dpk = cen[np.argmax(hh - base)]
W = 10
win = np.abs(cen - dpk) <= W

C_SIDE = '#93c5fd'      # sideband fill
C_COMB = '#9ca3af'      # combinatorial under peak
C_TRUE = '#fb923c'      # true-coincidence excess
C_LINE = '#1e3a5f'

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.step(cen, hh, where='mid', color=C_LINE, lw=1.3, zorder=5)

# sidebands
for lo, hi in [(cen[0] - 0.5, -SB), (SB, cen[-1] + 0.5)]:
    m = (cen >= lo) & (cen <= hi)
    ax.fill_between(cen[m], 0, hh[m], step='mid', color=C_SIDE, alpha=0.8, lw=0,
                    zorder=2)

# combinatorial + excess inside the window
ax.fill_between(cen[win], 0, np.minimum(hh[win], base), step='mid', color=C_COMB,
                alpha=0.9, lw=0, zorder=3)
ax.fill_between(cen[win], base, np.maximum(hh[win], base), step='mid', color=C_TRUE,
                alpha=0.95, lw=0, zorder=4)

ax.axhline(base, color='#374151', ls='--', lw=1.2, zorder=6)
for x in (dpk - W, dpk + W):
    ax.axvline(x, color='#374151', ls=':', lw=1)

ymax = hh.max()
ax.annotate('sideband  |Δt| > 100 ns\naccidentals only → flat\n'
            f'mean = {base:,.0f} pairs/ns',
            xy=(-125, base * 1.05), xytext=(-140, ymax * 0.42),
            arrowprops=dict(arrowstyle='->', color='#374151'), fontsize=10.5)
ax.annotate('', xy=(125, base * 1.05), xytext=(-68, ymax * 0.43),
            arrowprops=dict(arrowstyle='->', color='#374151'))
ax.annotate('baseline extended\nunder the peak',
            xy=(dpk + 25, base), xytext=(45, ymax * 0.25),
            arrowprops=dict(arrowstyle='->', color='#374151'), fontsize=10.5)

n_tot = hh[win].sum()
n_comb = base * win.sum()
ax.annotate(f'combinatorial in window\n= baseline × {2 * W} ns\n= {n_comb:,.0f} pairs',
            xy=(dpk + 4, base * 0.45), xytext=(35, ymax * 0.08),
            arrowprops=dict(arrowstyle='->', color='#374151'), fontsize=10.5,
            color='#374151')
ax.annotate(f'true coincidences (excess)\n= total − combinatorial\n'
            f'= {n_tot:,.0f} − {n_comb:,.0f} = {n_tot - n_comb:,.0f}',
            xy=(dpk - 2, ymax * 0.55), xytext=(-140, ymax * 0.75),
            arrowprops=dict(arrowstyle='->', color='#c2410c'), fontsize=10.5,
            color='#c2410c')
ax.annotate(f'window: peak {dpk:+.0f} ns ± {W} ns',
            xy=(dpk, ymax * 1.015), ha='center', fontsize=10.5)

ax.set_xlim(-150, 150)
ax.set_ylim(0, ymax * 1.12)
ax.set_xlabel('Δt = t(wall) − t(plastic)  [ns]', fontsize=11)
ax.set_ylabel('pairs / ns', fontsize=11)
ax.set_title(f'{RUN_STEM}  WALC–PSSC, hits >1 ms after γ-flash: '
             'combinatorial background from Δt sidebands', fontsize=12)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / 'sideband_method.png', dpi=150)
print(OUT / 'sideband_method.png')
