#!/usr/bin/env python3
"""pg_figs.py -- the figures of rerun/paper1_gas (results/gas.json from pg_gas.py) -> figures/*.png and the paper's
gas_vE_timing.pdf.

  v626.png       6-26 drift velocity against the clock: the anchored M3 ladder (k v_lad, attachment-blind) and the
                 timing stacks (gap / T), det6 and det7
  water626.png   the water each chamber carried through the night (anchored ladder through Magboltz)
  attach626.png  the visible fraction of the drift column T / T_true and its O2-equivalent attachment
  nights.png     water against time from the timing stacks on det4 6-23 -> 6-24 and det2 6-22
  gas_vE_timing  det3's 6-27 drift scan: timing v (gap / T) and angle-scale v (v_drift x mean kw) against field,
                 Magboltz bench gas at the fitted water, the other chambers' golden timing v at their own field
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import gasinv as GI  # noqa: E402

D = json.load(open(os.path.join(HERE, 'results', 'gas.json')))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)
T626 = 600.0
COL = {'det2': '#4c72b0', 'det3': '#55a868', 'det4': '#c44e52', 'det6': '#8172b2', 'det7': '#dd8452'}


def v626():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for det in ('det6', 'det7'):
        A = D['att_626'][det]
        t = [w['t'] - T626 for w in A['water']]
        ax.plot(t, [w['v_true'] for w in A['water']], '-', color=COL[det], lw=1.5,
                label=f'{det} M3 ladder x k ({A["k"]:.3f})')
        P = A['points']
        ax.errorbar([p['t'] - T626 for p in P], [p['v_stack'] for p in P], yerr=[p['v_stack_err'] for p in P], fmt='o',
                    ms=4, color=COL[det], mfc='white', label=f'{det} timing stack gap/T')
    ax.set_xlabel('hours on 6-26'); ax.set_ylabel('drift velocity (µm/ns)')
    ax.set_title('6-26: the timing stack reads fast while the column is truncated (det6 before ~4 h)', fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'v626.png'), dpi=150); plt.close(fig)


def water626():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for det in ('det6', 'det7'):
        W = D['att_626'][det]['water']
        t = np.array([w['t'] - T626 for w in W])
        ax.plot(t, [w['water'] for w in W], '-o', ms=3, color=COL[det], label=det)
        lo = [w['water_lo'] for w in W]; hi = [w['water_hi'] for w in W]
        ax.fill_between(t, lo, hi, color=COL[det], alpha=0.2)
    S = D['att_626']['shared']
    ax.set_xlabel('hours on 6-26'); ax.set_ylabel('water (%) from Magboltz')
    ax.set_title(f'6-26 water: same time shape (r = {S["corr"]:.2f}), det6 excess = {S["excess_ratio_det6_over_det7"]:.2f} x det7; '
                 f'plateaus {S["plateau_water"]["det6"]:.2f} / {S["plateau_water"]["det7"]:.2f} %', fontsize=8.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'water626.png'), dpi=150); plt.close(fig)


def attach626():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    for det in ('det6', 'det7'):
        P = D['att_626'][det]['points']
        t = [p['t'] - T626 for p in P]
        a1.errorbar(t, [p['frac'] for p in P], yerr=[p['frac'] * p['v_stack_err'] / p['v_stack'] for p in P], fmt='o',
                    ms=4, color=COL[det], label=det)
        q = [(p['t'] - T626, p['o2_equiv_ppm']) for p in P if 'o2_equiv_ppm' in p and p['t'] - T626 < 5]
        if q:
            a2.plot(*zip(*q), 'o-', color=COL[det], label=det)
    a1.axhline(1, color='k', lw=0.8); a1.set_ylabel('T / T_true (visible column)'); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a2.set_ylabel('O2-equivalent (ppm)\n(tau = T / ln 2, x2 model unc.)'); a2.set_xlabel('hours on 6-26'); a2.grid(alpha=0.3)
    a2.set_yscale('log')
    fig.suptitle('6-26 attachment from the truncated drift column (T_true from the anchored ladder)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'attach626.png'), dpi=150); plt.close(fig)


def nights():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, det, lab in ((axs[0], 'det4', 'det4: 6-23 night -> 6-24 (g_det4), 197 V/cm'),
                         (axs[1], 'det2', 'det2: 6-22 night, 321 V/cm (near the v maximum)')):
        P = sorted((p for p in D['points'] if p['det'] == det and p['arm'].startswith('in30|t') and 'water' in p),
                   key=lambda p: p['t'])
        t0 = min(p['t'] for p in P)
        ax.errorbar([p['t'] - t0 for p in P], [p['water'] for p in P],
                    yerr=[[p['water'] - (p['water_lo'] if np.isfinite(p['water_lo']) else p['water']) for p in P],
                          [(p['water_hi'] if np.isfinite(p['water_hi']) else p['water']) - p['water'] for p in P]],
                    fmt='o', color=COL[det])
        for p in P:
            if p.get('ambiguous'):
                ax.annotate('2 branches', (p['t'] - t0, p['water']), fontsize=6)
        ax.set_title(lab, fontsize=9); ax.set_xlabel(f'hours after {t0:.1f}'); ax.set_ylabel('water (%)'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'nights.png'), dpi=150); plt.close(fig)


def vE():
    sc = D['scan']
    gap = sc['fixed']['gap'] if sc else 28.19
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    E = np.linspace(30, 400, 120)
    w0 = sc['fixed']['water'] if sc else 0.95
    for w, ls, lab in ((w0, '-', f'Magboltz {w0:.2f} % water (fit)'), (0.0, ':', 'Magboltz dry'),
                       (w0 - 0.3, '--', f'{w0 - 0.3:.2f} %'), (w0 + 0.3, '--', f'{w0 + 0.3:.2f} %')):
        ax.plot(E, [GI.v_of(max(w, 0), e) for e in E], ls, color='k' if ls == '-' else 'grey', lw=1.2 if ls == '-' else 0.8,
                label=lab)
    # angle-scale v: v_drift x mean kw per drift-scan row (paper1_res rows.csv), E = V / gap
    R = [r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'paper1_res', 'results', 'rows.csv')))
         if 'saturday_scan' in r['key'] and 'drift_scan' in r['key']]
    ang = {}
    for r in R:
        ang.setdefault(r['key'], []).append((float(r['drift_V']), float(r['v_drift']), float(r['kw_pass'])))
    xa = [v[0][0] / (gap / 10) for v in ang.values()]; ya = [v[0][1] * np.mean([x[2] for x in v]) for v in ang.values()]
    ax.plot(xa, ya, 's', color=COL['det3'], mfc='white', label='det3 angle-scale v (v x kw, M3)')
    P = [p for p in D['points'] if p['det'] == 'det3' and p['arm'] == 'in30']
    ax.errorbar([p['E'] for p in P], [p['v'] for p in P], yerr=[p['v_err'] for p in P], fmt='o', color=COL['det3'],
                label='det3 timing v (gap / T)')
    for det, key in (('det2', 'o22_long_det2'), ('det4', 'g_det4'), ('det6', 'g_det6_long'), ('det7', 'g_det7_long')):
        q = [p for p in D['points'] if p['key'] == key and p['arm'] == 'in30']
        if q:
            ax.errorbar([q[0]['E']], [q[0]['v']], yerr=[q[0]['v_err']], fmt='D', ms=5, color=COL[det],
                        label=f'{det} golden timing v')
    ax.set_xlabel('drift field E = V / gap (V/cm)'); ax.set_ylabel('drift velocity (µm/ns)')
    ax.set_title('Bench Ar/iso 95/5: timing v vs angle-scale v, Magboltz (fine grid)', fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gas_vE_timing.png'), dpi=150); fig.savefig(os.path.join(OUT, 'gas_vE_timing.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    for f in (v626, water626, attach626, nights, vE):
        f(); print(f.__name__)
