#!/usr/bin/env python3
"""Paper IIa, figure toy_floor: the angle-resolution floor of the method against track angle.

(a) s68 against angle for chamber A (det3) from the electron-by-electron toy of rerun/model_audit/ws_floor
    (F3, 2026-10-27; results/*.json, nothing re-run): perfect model with 60 ns depth bins (production) and with 20 ns
    bins, the 20 ns fit without diffusion, electronics noise alone (Cramer-Rao), and the measured resolution of
    chamber A against the reference (papers/paper1_v2/figs/out/angle_resolution_v2.json, Runs 1 + 2, folded in |angle|).
(b) the bias of the perfect-model fit, 60 ns and 20 ns bins.

    .venv/bin/python papers/paper2/figs/make_toy_floor.py
Writes papers/paper2/figs/out/toy_floor.{pdf,png,json}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402

RES = os.path.join(REPO, 'rerun', 'model_audit', 'ws_floor', 'results')
MEAS = os.path.join(REPO, 'papers', 'paper1_v2', 'figs', 'out', 'angle_resolution_v2.json')
OUT = os.path.join(HERE, 'out')
C60, C20, CND, CNO = '#b2182b', '#1b9e77', '#6a7583', '#000000'


def curve(name, key='s68'):
    d = json.load(open(os.path.join(RES, name + '.json')))['by_angle']
    a = sorted(int(k) for k in d)
    n = np.array([d[str(k)].get('n', 0) for k in a])
    return np.array(a), np.array([d[str(k)][key] for k in a]), n


def measured(view):
    """fold the signed 2-deg bins of the Paper I json into |angle|, weighting s68 by the bin counts"""
    b = json.load(open(MEAS))['A'][view]['by_ref_bin']
    out = {}
    for r in b:
        c = abs(r['center'])
        o = out.setdefault(c, [0.0, 0, 0.0])
        o[0] += r['s68'] * r['n']; o[1] += r['n']; o[2] += (r['s68_err'] * r['n']) ** 2
    c = np.array(sorted(out))
    return c, np.array([out[k][0] / out[k][1] for k in c]), np.array([np.sqrt(out[k][2]) / out[k][1] for k in c]), \
        np.array([out[k][1] for k in c])


def main():
    ps.use()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.1, 3.0), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.30))
    num = {}
    a, y60, n60 = curve('det3_full')
    _, b60, _ = curve('det3_full', 'bias')
    a2, y20, n20 = curve('det3_full_bin20')
    _, b20, _ = curve('det3_full_bin20', 'bias')
    a3, ynd, _ = curve('det3_no_diff_bin20')
    crb = json.load(open(os.path.join(RES, 'det3_crb.json')))['by_angle']
    ac = np.array(sorted(int(k) for k in crb)); yc = np.array([crb[str(k)]['sig_deg'] for k in ac])
    # s68 error of a toy point: ~ s68 / sqrt(2 n) x 1.3 (percentile of a near-Gaussian core)
    e60, e20 = 1.3 * y60 / np.sqrt(2 * n60), 1.3 * y20 / np.sqrt(2 * n20)
    ax.plot(a, y60, '-', c=C60, lw=1.6, label='toy, perfect model, 60 ns depth bins')
    ax.fill_between(a, y60 - e60, y60 + e60, color=C60, alpha=0.15, lw=0)
    ax.plot(a2, y20, '-', c=C20, lw=1.6, label='toy, perfect model, 20 ns depth bins')
    ax.fill_between(a2, y20 - e20, y20 + e20, color=C20, alpha=0.15, lw=0)
    ax.plot(a3, ynd, '--', c=CND, lw=1.2, label='toy, 20 ns, no diffusion')
    ax.plot(ac, yc, ':', c=CNO, lw=1.2, label='electronics noise alone')
    for v in 'xy':
        c, s, e, n = measured(v)
        k = n >= 200
        st = ps.det('det3', view=v, line=False)
        st.pop('label', None)
        ax.errorbar(c[k], s[k], e[k], ls='none', label=f'chamber A, measured, {v.upper()}', **st)
        num[f'measured_{v}'] = dict(angle=c[k].tolist(), s68=s[k].tolist(), err=e[k].tolist(), n=n[k].tolist())
    ax.set_xlabel('track angle in the view [deg]'); ax.set_ylabel(r'angle resolution $s_{68}$ [deg]')
    ax.set_xlim(-1, 46); ax.set_ylim(0, 2.15)
    ax.legend(loc='upper left', bbox_to_anchor=(0.12, 1.0), fontsize=7.2, handlelength=1.6, labelspacing=0.3)
    ax.text(0.02, 0.97, '(a)', transform=ax.transAxes, fontweight='bold', va='top')
    bx.axhline(0, c='#555555', lw=0.6)
    bx.plot(a, b60, 'o-', c=C60, ms=3, label='60 ns bins')
    bx.plot(a2, b20, 's-', c=C20, ms=3, label='20 ns bins')
    bx.set_xlabel('track angle in the view [deg]'); bx.set_ylabel('toy bias, fit $-$ true [deg]')
    bx.set_xlim(-1, 46); bx.set_ylim(-0.62, 0.25)
    bx.legend(loc='lower left', fontsize=7.5)
    bx.text(0.04, 0.97, '(b)', transform=bx.transAxes, fontweight='bold', va='top')
    ps.save(fig, os.path.join(OUT, 'toy_floor'))
    num.update(angle=a.tolist(), s68_60ns=y60.tolist(), bias_60ns=b60.tolist(), n_60ns=n60.tolist(),
               s68_20ns=y20.tolist(), bias_20ns=b20.tolist(), n_20ns=n20.tolist(), s68_no_diffusion_20ns=ynd.tolist(),
               noise_only_crb=yc.tolist(),
               sources=dict(toy='rerun/model_audit/ws_floor/results/det3_{full,full_bin20,no_diff_bin20,crb}.json',
                            measured='papers/paper1_v2/figs/out/angle_resolution_v2.json (A, by_ref_bin, folded)'),
               note='toy is the X view only (no along-strip RC spreading); measured widths include the reference error')
    json.dump(num, open(os.path.join(OUT, 'toy_floor.json'), 'w'), indent=1)
    print('floor 60ns 0-20 deg', y60[a <= 20].min(), y60[a <= 20].max(), '| 20ns', y20.min(), y20.max())


if __name__ == '__main__':
    main()
