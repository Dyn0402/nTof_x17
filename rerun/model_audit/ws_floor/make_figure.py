#!/usr/bin/env python3
"""make_figure.py -- figures of the angle-resolution floor (floor_toy.py results) against the measurement (measured.py).

    make_figure.py   -> figures/floor_vs_angle_det3.png, figures/floor_breakdown_det3.png, figures/floor_fleet.png,
                        figures/floor_bias_grid_det3.png,
                        results/summary.json
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
C = dict(meas='#1b1b1b', fit='#c0392b', lik='#e67e22', orc='#2471a3', crb='#7f8c8d')
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False})


def load(name):
    p = os.path.join(RES, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def curve(r, key):
    a = sorted(int(k) for k in r['by_angle'])
    return np.array(a), np.array([r['by_angle'][str(k)][key] for k in a])


def fig_det3(meas):
    full, lik, crb = load('det3_full'), load('det3_full_poisson'), load('det3_crb')
    fig, ax = plt.subplots(figsize=(10, 5.6))
    a, y = curve(full, 'oracle_s68')
    ax.plot(a, y, 'o-', c=C['orc'], lw=2, label='information limit: every electron seen (ionisation + diffusion)')
    if crb:
        a2 = np.array(sorted(int(k) for k in crb['by_angle']))
        ax.plot(a2, [crb['by_angle'][str(k)]['sig_deg'] for k in a2], 's--', c=C['crb'], lw=1.5,
                label='electronic noise alone (Cramér–Rao)')
    if lik:
        a, y = curve(lik, 's68')
        ax.plot(a, y, 'D:', c=C['lik'], lw=1.5, label='perfect model, 60 ns bins, signal-variance weighting')
    a, y = curve(full, 's68')
    ax.plot(a, y, 'o-', c=C['fit'], lw=2.5, label='perfect model, production fit (60 ns depth bins)')
    b20 = load('det3_full_bin20')
    if b20:
        a, y = curve(b20, 's68')
        ax.plot(a, y, 'o-', c='#27ae60', lw=2.5, label='perfect model, same fit with 20 ns depth bins')
    for v, mk in (('x', 'o'), ('y', '^')):
        b = meas[f'det3_{v}']['bins']
        ax.plot([r['mid'] for r in b], [r['s68'] for r in b], mk, ms=9, mfc='white', mec=C['meas'], mew=2,
                label=f'measured det3 {v.upper()} (production fit vs M3)')
    ax.set_xlabel('track angle in the view [deg]')
    ax.set_ylabel('angle resolution s68 [deg]')
    ax.set_xlim(-1, 46); ax.set_ylim(0, 2.1)
    ax.set_title('det3 (bench, 333 V/cm): how far the measurement is from the physics floor', loc='left', fontsize=13)
    ax.legend(fontsize=10, frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'floor_vs_angle_det3.png'), dpi=150); plt.close(fig)


LABELS = [('full', 'everything'), ('no_noise', 'no electronic noise'), ('no_gainfl', 'no gain fluctuation'),
          ('no_delta', 'δ-rays point-like'), ('no_landau', 'no cluster-size (Landau) fluct.'),
          ('no_diff', 'no diffusion'), ('only_diff', 'ONLY diffusion + e⁻ count'), ('only_ion', 'ONLY clusters + δ-rays')]


def fig_breakdown():
    # the 20 ns charge grid: the decomposition of the physics without the 60 ns grid artefact at steep angles
    rows = [(lab, load(f'det3_{c}_bin20')) for c, lab in LABELS]
    rows = [(l, r) for l, r in rows if r]
    angs = ('5', '20', '35')
    fig, ax = plt.subplots(figsize=(10, 5.6))
    yv = np.arange(len(rows))[::-1]
    cols = ('#5d6d7e', '#c0392b', '#7d3c98')
    for j, an in enumerate(angs):
        ax.barh(yv + (1 - j) * 0.26, [r['by_angle'][an]['s68'] for _, r in rows], height=0.25, color=cols[j],
                label=f'{an}°')
    ax.set_yticks(yv); ax.set_yticklabels([l for l, _ in rows])
    ax.set_xlabel('angle resolution s68 of the perfect-model fit [deg]')
    ax.set_title('det3, perfect model (20 ns grid): which physics sets the floor', loc='left', fontsize=13)
    ax.legend(title='track angle', frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'floor_breakdown_det3.png'), dpi=150); plt.close(fig)


def fig_fleet(meas):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(DETS))
    fl, lk, orc, mx, my = [], [], [], [], []
    for d in DETS:
        r, l = load(f'{d}_full'), load(f'{d}_full_poisson')
        sel = ('5', '10', '15')       # the bench angle mix: most tracks within 20 deg
        fl.append(np.mean([r['by_angle'][k]['s68'] for k in sel]) if r else np.nan)
        lk.append(np.mean([l['by_angle'][k]['s68'] for k in sel]) if l else np.nan)
        orc.append(np.mean([r['by_angle'][k]['oracle_s68'] for k in sel]) if r else np.nan)
        mx.append(meas[f'{d}_x']['all_s68']); my.append(meas[f'{d}_y']['all_s68'])
    w = 0.17
    ax.bar(x - 2 * w, orc, w, color=C['orc'], label='information limit')
    ax.bar(x - w, lk, w, color=C['lik'], label='perfect model, signal-variance weighting')
    ax.bar(x, fl, w, color=C['fit'], label='perfect model, production fit')
    ax.bar(x + w, mx, w, color='#aab7b8', label='measured X')
    ax.bar(x + 2 * w, my, w, color='#566573', label='measured Y')
    ax.set_xticks(x); ax.set_xticklabels(DETS)
    ax.set_ylabel('angle resolution s68 [deg]')
    ax.set_title('Every chamber: floor (toy, 5–15°) against the measurement (all angles)', loc='left', fontsize=13)
    ax.legend(fontsize=10, frameon=False, ncol=2, loc='upper left'); ax.set_ylim(0, 2.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'floor_fleet.png'), dpi=150); plt.close(fig)
    return dict(zip(DETS, [dict(oracle=o, likelihood=l, production_fit=f, meas_x=a, meas_y=b)
                           for o, l, f, a, b in zip(orc, lk, fl, mx, my)]))


def fig_bias():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name, lab, col in (('det3_full', '60 ns depth bins (production grid)', C['fit']),
                           ('det3_full_bin30', '30 ns', '#8e44ad'), ('det3_full_bin20', '20 ns', '#27ae60')):
        r = load(name)
        if r:
            a, y = curve(r, 'bias')
            ax.plot(a, y, 'o-', c=col, lw=2, label=lab)
    ax.axhline(0, c='k', lw=0.8)
    ax.set_xlabel('track angle in the view [deg]'); ax.set_ylabel('median fitted − true angle [deg]')
    ax.set_title('Perfect model, det3: the 60 ns charge grid reads steep tracks too shallow', loc='left', fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'floor_bias_grid_det3.png'), dpi=150); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    meas = json.load(open(os.path.join(RES, 'measured.json')))
    fig_det3(meas)
    fig_breakdown()
    fig_bias()
    fleet = fig_fleet(meas)
    summ = dict(fleet_5_15deg=fleet, det3={})
    for name in [f'{c}{tag}' for c, _ in LABELS for tag in ('', '_bin20')]:
        r = load(f'det3_{name}')
        if r:
            summ['det3'][name] = {k: dict(s68=v['s68'], far5=v['far5'], bias=v['bias'], oracle=v['oracle_s68'])
                                  for k, v in r['by_angle'].items()}
    for extra in ('det3_full_poisson', 'det3_crb', 'det3_full_bin20', 'det3_full_bin30'):
        r = load(extra)
        if r:
            summ['det3'][extra] = r['by_angle']
    json.dump(summ, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)
    print(json.dumps(fleet, indent=1))


if __name__ == '__main__':
    main()
