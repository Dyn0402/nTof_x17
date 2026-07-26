#!/usr/bin/env python3
"""Flagship comparison: per-event angle residual and full-depth line threading
for production raw ladder, SOTA hybrid (unshared+calibrated), forward fit."""
import os, pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
AL = os.path.dirname(BASE) + '/alignment_tpc_veto50'
res = pickle.load(open(os.path.join(BASE, 'freefit.pkl'), 'rb'))
hj = json.load(open(os.path.join(BASE, 'hyper_ref.json')))
V = hj['v']
hyb = pd.read_csv(os.path.join(AL, 'hybrid/hybrid_events.csv'))
hyb = hyb.set_index(['eid', 'plane'])

def robust_sigma(a):
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a)))

GAP = 29.0
recs = []
for r in res:
    for p in ('x', 'y'):
        if p not in r or 'error' in r[p]:
            continue
        d = r[p]
        key = (r['eid'], p)
        th = np.degrees(np.arctan(d['tan_ref']))
        row = dict(eid=r['eid'], plane=p, tan_ref=d['tan_ref'], th_ref=th)
        row['tan_ff'] = d['w'] * 1e3 / V
        row['p0_ff'] = d['p0']; row['p0_ref'] = d['p0_ref']
        row['p0_prod'] = d['p0_prod']
        row['tan_prod'] = (d['w_prod'] * 1e3 / V if np.isfinite(d['w_prod'])
                           else np.nan)
        if key in hyb.index:
            h = hyb.loc[key]
            row['tan_hyb'] = h['tan_hyb']
            row['tan_prod_v34'] = h['tan_prod']
        else:
            row['tan_hyb'] = np.nan; row['tan_prod_v34'] = np.nan
        recs.append(row)
df = pd.DataFrame(recs)
print(f'{len(df)} plane-fits, hybrid match {df["tan_hyb"].notna().mean()*100:.0f}%')

# sign conventions: hybrid tan columns are in ... check correlation sign
for c in ('tan_hyb', 'tan_prod_v34'):
    ok = df[c].notna() & (df['tan_ref'].abs() > 0.05)
    corr = np.corrcoef(df.loc[ok, c], df.loc[ok, 'tan_ref'])[0, 1]
    print(c, 'corr with tan_ref:', round(corr, 3), 'n', ok.sum())

fig, axs = plt.subplots(2, 2, figsize=(13, 9))
summary = {}
for i, p in enumerate(('x', 'y')):
    sub = df[df['plane'] == p]
    ax = axs[0, i]
    b = np.linspace(-8, 8, 100)
    for col, lab, colr in (('tan_prod', 'production raw ladder', 'orange'),
                           ('tan_hyb', 'SOTA hybrid (unshared+cal)', 'green'),
                           ('tan_ff', 'waveform forward fit', 'C0')):
        dth = np.degrees(np.arctan(sub[col])) - np.degrees(np.arctan(sub['tan_ref']))
        s = robust_sigma(dth.to_numpy())
        m = np.nanmedian(dth)
        ax.hist(dth, bins=b, histtype='step', lw=2, color=colr,
                label=f'{lab}: med {m:+.2f}, sig {s:.2f} deg')
        summary[(p, col)] = (m, s)
    ax.set_xlabel('reconstructed - reference angle [deg]')
    ax.set_title(f'{p}: per-event angle residual (same events)')
    ax.legend(fontsize=8)

    # full-depth line deviation census (line vs ref line over 29 mm)
    ax = axs[1, i]
    for col, pcol, lab, colr in (
            ('tan_prod', 'p0_prod', 'production raw ladder', 'orange'),
            ('tan_ff', 'p0_ff', 'waveform forward fit', 'C0')):
        dev0 = np.abs(sub[pcol] - sub['p0_ref'])
        dev1 = np.abs(sub[pcol] + sub[col] * GAP - (sub['p0_ref'] + sub['tan_ref'] * GAP))
        worst = np.maximum(dev0, dev1).to_numpy()
        worst = worst[np.isfinite(worst)]
        ax.hist(worst, bins=np.linspace(0, 5, 120), cumulative=True,
                density=True, histtype='step', lw=2, color=colr,
                label=f'{lab}: <0.5mm {np.mean(worst<0.5)*100:.0f}%, '
                      f'<1mm {np.mean(worst<1)*100:.0f}%')
        summary[(p, col, 'census')] = (np.mean(worst < 0.5), np.mean(worst < 1.0),
                                       np.mean(worst < 1.5))
    ax.set_xlabel(f'{p}: max |line - ref line| over {GAP:.0f} mm depth [mm]')
    ax.set_ylabel('cumulative fraction'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlim(0, 4); ax.axvline(1.0, color='gray', ls=':', lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'three_way_comparison.png'), dpi=110)
for k, v in summary.items():
    print(k, np.round(v, 3))
