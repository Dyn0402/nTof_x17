#!/usr/bin/env python3
"""Paper I main-text figure `angle_resolution`: chamber A (det3), Run 1 (sat_det3) + Run 2 (p2_det3) summed.

Per-track fitted angle (waveform forward-model fit, Analysis_pass1) against the reference (M3) angle for the reference-matched
single tracks of the paper's resolution numbers: effbudget's 03_angles interior_good sample, exactly the one res_rows.py
uses (pass arm; chamber A has no v(t) map).  Left: 2D histogram of theta_fit vs theta_ref (X and Y views stacked, they
are the same physics), diagonal dashed.  Right: median of theta_fit - theta_ref per theta_ref bin, with the s68 band.

Input: per_event.npz of the two rows (x_/y_ eid, tan_ref, tan_fit), EOS x17rerun/paper1_eff/angles/<run>/<subrun>/mx17_3/
(fetch with XRootD, see --data).  Output: figures/angle_resolution.{pdf,png,csv}, results/angle_resolution.json.

    ../../.venv/bin/python angle_resolution.py [--data $WORK_DATA/claude-jobs/77eb854f/angres]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402

RUNS = {'run1': 'sat_det3 (Run 1)', 'run2': 'p2_det3 (Run 2)'}
LIM = 20.0          # the reference telescope's acceptance ends near 20 deg
BW = 2.0            # deviation panel bin width [deg]
MIN_N = 50


def s68(d, q=68.0):
    d = np.asarray(d, float)
    return float(np.percentile(np.abs(d - np.median(d)), q))


def load(data):
    out = {'x': [], 'y': []}
    for run in RUNS:
        z = np.load(os.path.join(data, run + '.npz'))
        for p in 'xy':
            ref = np.degrees(np.arctan(z[f'{p}_tan_ref']))
            fit = np.degrees(np.arctan(z[f'{p}_tan_fit']))
            out[p].append(pd.DataFrame(dict(run=run, ref=ref, fit=fit)))
    return {p: pd.concat(v, ignore_index=True) for p, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'),
                                                   'claude-jobs', '77eb854f', 'angres'))
    a = ap.parse_args()
    D = load(a.data)
    res = {'sample': 'effbudget 03_angles interior_good, pass arm, Run 1 + Run 2 summed; s68 = 68th percentile of '
                     '|d - median(d)|, d = theta_fit - theta_ref'}
    for p in 'xy':
        d = D[p]
        dd = d.fit - d.ref
        r = dict(n=int(len(d)), median=float(np.median(dd)), s68=s68(dd), mean=float(dd.mean()), by_run={}, by_angle=[])
        for run in RUNS:
            k = d.run == run
            r['by_run'][run] = dict(n=int(k.sum()), s68=s68(dd[k]), median=float(np.median(dd[k])))
        for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20)):
            k = (d.ref.abs() >= lo) & (d.ref.abs() < hi)
            r['by_angle'].append(dict(lo=lo, hi=hi, n=int(k.sum()), s68=s68(dd[k]), median=float(np.median(dd[k]))))
        res[p] = r

    ps.use()
    fig, (a1, a2) = ps.fig_for('full', height=2.7, ncols=2, gridspec_kw=dict(width_ratios=[1, 1.15]))
    # --- (a) 2D histogram, X + Y stacked
    ref = np.concatenate([D[p].ref for p in 'xy'])
    fit = np.concatenate([D[p].fit for p in 'xy'])
    edges = np.arange(-LIM, LIM + 0.01, 0.5)
    H, xe, ye = np.histogram2d(ref, fit, bins=[edges, edges])
    cm = plt.get_cmap('viridis').copy()
    cm.set_bad('white'); cm.set_under('white')
    Hm = np.ma.masked_where(H < 1, H)
    im = a1.pcolormesh(xe, ye, Hm.T, cmap=cm, norm=LogNorm(vmin=1, vmax=H.max()), rasterized=True)
    a1.plot([-LIM, LIM], [-LIM, LIM], color='white', lw=0.9, ls=(0, (4, 3)))
    a1.set_xlim(-LIM, LIM); a1.set_ylim(-LIM, LIM); a1.set_aspect('equal')
    a1.set_xlabel(r'$\theta_{\rm ref}$ [deg]'); a1.set_ylabel(r'$\theta_{\rm fit}$ [deg]')
    a1.set_xticks(range(-20, 21, 10)); a1.set_yticks(range(-20, 21, 10))
    cb = fig.colorbar(im, ax=a1, fraction=0.045, pad=0.02)
    cb.set_label('tracks per 0.5$^\\circ\\times$0.5$^\\circ$')
    # --- (b) median deviation, s68 band
    be = np.arange(-LIM, LIM + 0.01, BW)
    c = 0.5 * (be[1:] + be[:-1])
    rows = []
    for p, off in (('x', -0.15), ('y', 0.15)):
        d = D[p]
        dd = (d.fit - d.ref).to_numpy()
        med, sg, n = [], [], []
        for lo, hi in zip(be[:-1], be[1:]):
            k = (d.ref.to_numpy() >= lo) & (d.ref.to_numpy() < hi)
            n.append(int(k.sum()))
            if k.sum() >= MIN_N:
                med.append(float(np.median(dd[k]))); sg.append(s68(dd[k]))
            else:
                med.append(np.nan); sg.append(np.nan)
        med, sg = np.array(med), np.array(sg)
        kw = ps.det('det3', view=p)
        kw.pop('zorder', None); kw.pop('label', None)
        col = ps.DET_COLOR['det3']
        a2.fill_between(c, med - sg, med + sg, color=col, alpha=0.18 if p == 'x' else 0.0, lw=0)
        a2.plot(c, med - sg, color=col, lw=0.7, ls=ps.VIEW_LS[p], alpha=0.8)
        a2.plot(c, med + sg, color=col, lw=0.7, ls=ps.VIEW_LS[p], alpha=0.8)
        kw.update(ms=3.6, lw=1.2)
        a2.plot(c, med, label=f'{p.upper()}', **kw)
        rows += [dict(view=p.upper(), theta_ref_center=ci, n=ni, median_dev_deg=m, s68_deg=s)
                 for ci, ni, m, s in zip(c, n, med, sg)]
        res[p]['by_ref_bin'] = [dict(center=float(ci), n=ni, median=m, s68=s) for ci, ni, m, s in zip(c, n, med, sg)]
    a2.axhline(0, color='0.3', lw=0.6, ls=':')
    a2.set_xlim(-LIM, LIM); a2.set_ylim(-2.0, 2.0); a2.set_yticks([-2, -1, 0, 1, 2])
    a2.set_xticks(range(-20, 21, 10))
    a2.set_xlabel(r'$\theta_{\rm ref}$ [deg]'); a2.set_ylabel(r'$\theta_{\rm fit}-\theta_{\rm ref}$ [deg]')
    a2.legend(loc='upper center', ncol=2, handlelength=2.2, title=None)
    a2.text(0.5, 0.04, r'$s_{68}$: X %.2f$^\circ$, Y %.2f$^\circ$ (band: $\pm s_{68}$ per bin)' % (res['x']['s68'], res['y']['s68']),
            transform=a2.transAxes, ha='center', va='bottom', fontsize=ps.BASE_PT * 0.85, color=ps.INK)
    ps.panel_labels([a1, a2])
    df = pd.DataFrame(rows)
    ps.save(fig, os.path.join(HERE, 'figures', 'angle_resolution'), data=df)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'angle_resolution.json'), 'w'), indent=1)
    for p in 'xy':
        print(p, res[p]['n'], 'median %.3f s68 %.3f' % (res[p]['median'], res[p]['s68']),
              {k: round(v['s68'], 3) for k, v in res[p]['by_run'].items()},
              [(b['lo'], b['hi'], b['n'], round(b['s68'], 3)) for b in res[p]['by_angle']])


if __name__ == '__main__':
    main()
