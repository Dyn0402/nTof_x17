#!/usr/bin/env python3
"""vtc_figure.py -- figures/vt_crosscheck.png: per flagged row, v_rel(t) from the M3 ladder, the column duration, the
composition-standardised column and the head-on stack, in the same 4 time blocks (results/measure/*.json), plus the
column at the survey's finer blocks; and the gain per block (CROSSCHECK_2026-10-31 §1)."""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'measure')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
ORDER = ['c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 'c26__mx17_det3_det4_overnight_6-23-26__long_run__mx17_4',
         'g_det4', 'c26__mx17_det6_det7_overnight_6-26-26__longer_run__mx17_6', 'g_det6_long',
         'c26__mx17_det6_det7_overnight_6-26-26__longer_run__mx17_7', 'g_det7_long']
LAB = {'lad': ('M3 ladder', '#1f4e9a', 'o'), 'col': ('column duration', '#d0700a', 's'),
       'col_std': ('column, standardised', '#e8b04a', 'D'), 'stack': ('head-on stack', '#4a9a4a', '^')}


def short(k):
    return k.replace('c26__mx17_', '').replace('_overnight_', ' ').replace('__', ' ').replace('_mx17_', ' det')


D = {json.load(open(f))['key']: json.load(open(f)) for f in glob.glob(os.path.join(RES, '*.json'))}
keys = [k for k in ORDER if k in D]
fig, axs = plt.subplots(2, len(keys), figsize=(3.1 * len(keys), 6.2), sharex='col',
                        gridspec_kw=dict(height_ratios=[3, 1]))
for j, k in enumerate(keys):
    d = D[k]
    ax, ag = axs[0, j], axs[1, j]
    B = d['blk']['blocks']
    t = np.array([b['t'] for b in B])
    for n, (lab, c, m) in LAB.items():
        r = np.array([b[n]['r'] for b in B], float); e = np.array([b[n]['r_err'] for b in B], float)
        ok = np.isfinite(r) & np.isfinite(e)
        ax.errorbar(t[ok] + 0.03 * list(LAB).index(n), r[ok], e[ok], fmt=m, color=c, ms=4, lw=1, capsize=0, label=lab)
    S = d['survey']['blocks']
    ax.plot([b['t'] for b in S], [b['col']['r'] for b in S], '-', color='#d0700a', lw=0.8, alpha=0.6)
    L = d['blk']['lines']
    ax.set_title(f"{short(k)}\nladder {100 * L['lad']['dv']:+.1f}±{100 * L['lad']['dv_err']:.1f} %, "
                 f"column {100 * L['col']['dv']:+.1f}±{100 * L['col']['dv_err']:.1f} %", fontsize=8)
    ax.axhline(1, color='0.7', lw=0.6)
    ax.set_ylim(0.85, 1.15)
    ag.plot(t, [b['gain'] for b in B], 'k.-', lw=0.8)
    ag.set_ylim(0.6, 1.6)
    ag.set_xlabel('h since 2026-06-01', fontsize=8)
    if j == 0:
        ax.set_ylabel('v(t) / v(row)'); ag.set_ylabel('gain / row')
        ax.legend(fontsize=7, loc='lower right')
    for a_ in (ax, ag):
        a_.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'vt_crosscheck.png'), dpi=130)
print(os.path.join(FIG, 'vt_crosscheck.png'))
