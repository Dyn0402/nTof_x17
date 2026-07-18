"""19c_compare_figs.py — run-224489 vs run-224466 comparison figures for the
report/slides (all from existing caches, no run-file access):

figures/19_triples/
  liq_wide_dt.png    the LIQ timing-in discovery plot: WALxLIQ / PSSxLIQ dt
                     over the full +-400 ns first-look window, per arm
  fifo_ratio.png     FIFO/BNC-T same-HV coincident-median ratio per PMT vs V
  offset_shift.png   per-channel wall-plastic offset shift 224489 - 224466
  satellite.png      late-tof dt spectra B/C, both runs: satellite stays at
                     fixed ABSOLUTE dt while the main peak moves -22 ns

Usage: python 19c_compare_figs.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
OUT = BASE / 'figures' / '19_triples'
OUT.mkdir(parents=True, exist_ok=True)
ARM_COLORS = dict(zip('ABCD', plt.cm.tab10.colors))

# ------------------------------------------------------------- liq wide dt
d9 = np.load(BASE / 'cache' / '02_coinc_run224489.npz')
ce = d9['dt_edges_liq']
cen = 0.5 * (ce[:-1] + ce[1:])
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
for ai, st in enumerate('ABCD'):
    for row, pre in enumerate(('WAL', 'PSS')):
        ax = axes[row][ai]
        h = d9[f'{pre}{st}_LIQ{st}'].sum(axis=(0, 1))
        base = h[np.abs(cen) > 350].mean()
        ax.plot(cen, h / 1e6, color=ARM_COLORS[st], lw=1)
        ax.axhline(base / 1e6, color='gray', lw=0.8, ls='--')
        ipk = np.argmax(h - base)
        ax.annotate(f'{cen[ipk]:+.0f} ns', (cen[ipk], h[ipk] / 1e6),
                    textcoords='offset points', xytext=(8, -2), fontsize=9)
        ax.set_title(f'{pre}{st} $\\times$ LIQ{st}')
        ax.grid(alpha=0.3)
        if row == 1:
            ax.set_xlabel(r'dt = $t_{\mathrm{det}} - t_{\mathrm{liq}}$ [ns]')
        if ai == 0:
            ax.set_ylabel('pairs / ns [$10^6$]')
fig.suptitle('run224489: first LIQ coincidence scan (±400 ns window) — '
             'every arm shows a sharp peak on the combinatorial floor', y=0.995)
fig.tight_layout()
fig.savefig(OUT / 'liq_wide_dt.png', dpi=140)
plt.close(fig)

# ------------------------------------------------------------- fifo ratio
def coinc_medians(run):
    d = np.load(BASE / 'cache' / f'12_hvscan_{run}.npz')
    edges = d['amp_edges']
    c = np.sqrt(edges[:-1] * edges[1:])
    sb = float(d['sb_scale'])
    volts = d['step_volts']
    med = np.full((4, 2, len(volts)), np.nan)
    for i in range(len(volts)):
        for ai in range(4):
            for b in range(2):
                sub = np.clip(d['pss_mip'][i, ai, 0, b]
                              - sb * d['pss_mip'][i, ai, 1, b], 0, None)
                cum = np.cumsum(sub)
                if cum[-1] > 200:
                    med[ai, b, i] = c[np.searchsorted(cum, cum[-1] / 2)]
    return volts, med


v6, m6 = coinc_medians('run224466')
v9, m9 = coinc_medians('run224489')
common = sorted(set(v6[v6 > 0]) & set(v9[v9 > 0]))
fig, ax = plt.subplots(figsize=(9, 5.5))
gms = []
for ai, st in enumerate('ABCD'):
    for b, side in enumerate('LR'):
        r = [m9[ai, b, int(np.where(v9 == v)[0][0])]
             / m6[ai, b, int(np.where(v6 == v)[0][0])] for v in common]
        gm = float(np.exp(np.nanmean(np.log(r))))
        gms.append(gm)
        ax.plot(common, r, 'o-', lw=1, ms=4, color=ARM_COLORS[st],
                ls='-' if b == 0 else '--', label=f'{st}{side}  ({gm:.2f})')
fleet = float(np.exp(np.mean(np.log(gms))))
ax.axhline(2.0, color='crimson', lw=1, ls=':', label='naive 2$\\times$ expectation')
ax.axhline(fleet, color='k', lw=1.2, label=f'fleet geo-mean {fleet:.2f}')
ax.set_xlabel('PMT bias [V]')
ax.set_ylabel('coincident median ratio  run224489 (FIFO) / run224466 (BNC-T)')
ax.set_ylim(0.9, 2.2)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=3)
ax.set_title('FIFO vs BNC-T split: same-HV gain ratio per PMT')
fig.tight_layout()
fig.savefig(OUT / 'fifo_ratio.png', dpi=140)
plt.close(fig)

# ------------------------------------------------------------ offset shift
o6 = json.loads((BASE / 'calib' / 'time_offsets_run224466.json').read_text())['stations']
o9 = json.loads((BASE / 'calib' / 'time_offsets_run224489.json').read_text())['stations']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                               gridspec_kw={'width_ratios': [2.2, 1]})
shifts = []
x = 0
for st in 'ABCD':
    xs, ys = [], []
    for wc in range(1, 9):
        for pc in range(1, 3):
            k = f'WAL{st}{wc}_PSS{st}{pc}'
            a, b = o6[st][k]['offset_ns'], o9[st][k]['offset_ns']
            if a is not None and b is not None:
                xs.append(x)
                ys.append(b - a)
                shifts.append(b - a)
            x += 1
    ax1.plot(xs, ys, 'o', ms=4, color=ARM_COLORS[st], label=f'arm {st}')
mu, sd = np.mean(shifts), np.std(shifts)
ax1.axhline(mu, color='k', lw=1)
ax1.set_xlabel('channel pair (arm-ordered)')
ax1.set_ylabel(r'offset shift $\Delta$ = 224489 $-$ 224466 [ns]')
ax1.set_title(f'wall–plastic offset shift: common {mu:+.2f} ns, rms {sd:.2f} ns')
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8, ncol=4)
ax2.hist(shifts, bins=np.arange(-23.2, -21.2, 0.1), color='steelblue')
ax2.set_xlabel(r'$\Delta$ [ns]')
ax2.set_ylabel('channels')
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'offset_shift.png', dpi=140)
plt.close(fig)

# --------------------------------------------------------------- satellite
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, arm in zip(axes, 'BC'):
    for run, lbl, col in (('run224466', '224466 (BNC-T)', 'gray'),
                          ('run224489', '224489 (FIFO)', ARM_COLORS[arm])):
        d = np.load(BASE / 'cache' / f'02_coinc_{run}.npz')
        e = d['dt_edges']
        c = 0.5 * (e[:-1] + e[1:])
        h = d[f'WAL{arm}_PSS{arm}'][:, 4:6].sum(axis=(0, 1))
        base = h[np.abs(c) > 120].mean()
        ax.plot(c, (h - base) / max((h - base).max(), 1), color=col, lw=1.2,
                label=lbl)
    ax.axvspan(-68, -58, color='crimson', alpha=0.12, lw=0)
    ax.text(-63, 0.75, 'satellite\nband', color='crimson', ha='center', fontsize=9)
    ax.set_title(f'WAL{arm}–PSS{arm}, late tof ($>$0.1 ms)')
    ax.set_xlabel(r'dt = $t_{\mathrm{wall}} - t_{\mathrm{pss}}$ [ns]')
    ax.set_xlim(-120, 40)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_ylabel('excess (normalized to main peak)')
fig.suptitle('the −60 ns satellite keeps its ABSOLUTE position while the main '
             'peak moves −22 ns ⇒ not a plastic-cable reflection')
fig.tight_layout()
fig.savefig(OUT / 'satellite.png', dpi=140)
plt.close(fig)

print(f'Figures -> {OUT}: liq_wide_dt, fifo_ratio, offset_shift, satellite')
