#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
46c_gas_scale_and_anglebins.py — two more discriminators for the raw-hit
v-scan / gap-filling question (companion to 46/46b).

(1) CROSS-FIELD GAS TEST.  If the visible column always fills the 29 mm gap,
    then at EVERY drift field v(E) = 29 mm / T_sat(E).  Build that series from
    the measured saturated time spans (geometry_vdrift_scan.csv) and compare
    it — together with the unshared slope-vs-reference series — against the
    full Magboltz mixture library (garfield_sim/results/*.json, 25 mixtures:
    watered/dry Ar/iso 95/5, air, wrong-quencher 98/2..75/25, Ar/CO2...).
    RMS ranking decides which series is physical.

(2) SCAN MINIMUM PER ANGLE BIN.  Re-run the offset-floated scan objective of
    46 in bins of |tan theta_ref|.  A real drift velocity gives the same
    minimum in every bin; the spatial-floor artifact predicts
    v_min(tan) ~ v_true + w/(|tan|*T_span).

Usage: ../.venv/bin/python 46c_gas_scale_and_anglebins.py
"""
import os
import sys
import json
import glob
import importlib

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'engineer_package'))
m46 = importlib.import_module('46_vdrift_ref_metric_scan')
from make_event_displays import load_hits                     # noqa: E402
from make_event_displays_3d import load_full_reference        # noqa: E402

OUT = m46.OUT
GS = os.path.join(os.path.dirname(HERE), 'garfield_sim', 'results')
DV = os.path.expanduser(
    '~/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
    'drift_velocity/mx17_3')
GAP_MM = m46.GAP_MM
GAP_CM = 3.0
HV_VALID = [700, 900, 1000, 1100]      # gap-limited fields (500 V window-cut)

# curves to draw (label, file glob key, mixture key, color, ls)
DRAW = [
    ('Ar/iso 95/5 + 1.0% H2O (RMS-best, unshared)', 'drift_velocity_candidates.json',
     'Ar94_iso5_H2O1', 'tab:green', '-'),
    ('Ar/iso 95/5 dry', 'attachment_Ar_iso_H2O.json', 'Ar95_iso5', 'tab:blue', '-'),
    ('Ar/iso 95/5 + 0.3% H2O', 'drift_velocity_candidates.json',
     'Ar95_iso5_H2O0.3', 'tab:cyan', '--'),
    ('Ar/iso 90/10', 'drift_velocity_candidates2.json', 'Ar90_iso10',
     'tab:orange', '-.'),
    ('Ar/iso 80/20 (RMS-best, gap-filling; wrong bottle)',
     'drift_velocity_candidates2.json', 'Ar80_iso20', 'tab:red', ':'),
]


def load_all_mixtures():
    mixes = {}
    for fn in glob.glob(os.path.join(GS, '*.json')):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        for k, pts in d.get('mixtures', {}).items():
            if not pts or 'v_um_per_ns' not in pts[0]:
                continue
            E = np.array([q['E_Vcm'] for q in pts])
            V = np.array([q['v_um_per_ns'] for q in pts])
            o = np.argsort(E)
            mixes[(os.path.basename(fn), k)] = (E[o], V[o])
    return mixes


def fig_gas_scale():
    geo = pd.read_csv(os.path.join(DV, 'geometry_vdrift_scan.csv'))
    slr = pd.read_csv(os.path.join(DV, 'slope_reference_vdrift_scan.csv'))
    geo = geo.set_index('drift_hv')
    slr = slr.set_index('drift_hv')
    E = np.array([hv / GAP_CM for hv in HV_VALID])
    tsat = np.array([geo.loc[hv, 't_sat_x_ns'] for hv in HV_VALID])
    v_gap = GAP_MM * 1000.0 / tsat
    v_uns = np.array([slr.loc[hv, 'v_unshared'] for hv in HV_VALID])
    v_uns_err = np.array([slr.loc[hv, 'v_unshared_err'] for hv in HV_VALID])

    mixes = load_all_mixtures()

    def ranking(series):
        out = []
        for (fn, k), (Ec, Vc) in mixes.items():
            vi = np.interp(E, Ec, Vc)
            out.append((float(np.sqrt(np.mean((vi - series) ** 2))), k, fn))
        out.sort()
        return out

    r_uns, r_gap = ranking(v_uns), ranking(v_gap)
    print('best fits, UNSHARED series:', [f'{r[0]:.2f} {r[1]}' for r in r_uns[:3]])
    print('best fits, GAP-FILLING series:', [f'{r[0]:.2f} {r[1]}' for r in r_gap[:3]])

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    for lab, fn, key, c, ls in DRAW:
        g = mixes.get((fn, key))
        if g is None:
            continue
        Ec, Vc = g
        m = (Ec > 130) & (Ec < 430)
        ax.plot(Ec[m], Vc[m], ls, color=c, lw=2.0, label=lab, alpha=0.9)
    ax.errorbar(E, v_uns, yerr=v_uns_err, fmt='o', color='k', ms=8, capsize=4,
                zorder=6, label=f'measured, UNSHARED slope-vs-ref '
                                f'(best gas RMS {r_uns[0][0]:.2f})')
    ax.plot(E, v_gap, 's', color='#c0392b', ms=9, mfc='none', mew=2.2, zorder=6,
            label=f'GAP-FILLING hypothesis: 29 mm / T_sat(E) '
                  f'(best gas RMS {r_gap[0][0]:.2f} = Ar/iso 80/20)')
    for e, v in zip(E, v_gap):
        ax.annotate('', xy=(e, v), xytext=(e, np.interp(e, E, v_uns)),
                    arrowprops=dict(arrowstyle='->', color='#c0392b', alpha=0.5))
    ax.set_xlabel('drift field E [V/cm]')
    ax.set_ylabel('drift velocity [µm/ns]')
    ax.set_title('Cross-field test: gap-filling requires a gas that does not exist\n'
                 '(29 mm/T_sat rises 1.52× over 233→367 V/cm AND reaches 43-45; '
                 'no surveyed mixture does both)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, 'gas_scale_test.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'-> {p}')
    # dump ranking tables for the report
    with open(os.path.join(OUT, 'gas_ranking.txt'), 'w') as f:
        f.write('series v(E) at 233/300/333/367 V/cm\n')
        f.write(f'unshared    : {np.round(v_uns, 1).tolist()}\n')
        f.write(f'gap-filling : {np.round(v_gap, 1).tolist()}\n\n')
        for name, r in (('UNSHARED', r_uns), ('GAP-FILLING', r_gap)):
            f.write(f'best fits, {name} series:\n')
            for rms, k, fn in r[:8]:
                f.write(f'  RMS {rms:5.2f}  {k:28s} ({fn})\n')
            f.write('\n')
    return v_gap, v_uns


def fig_anglebin_scan(df):
    bins = [(0.08, 0.14), (0.14, 0.22), (0.22, 0.32), (0.32, 0.50)]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    mins = []
    at = np.abs(df['tan'].to_numpy())
    for (a, b), c in zip(bins, ('#9b59b6', '#2e86c1', '#c0392b', '#1a1a2e')):
        sub = df[(at >= a) & (at < b)]
        if sub.eid.nunique() < 40:
            continue
        sc = m46.scan(sub)
        j = sc['j_float'] / sc['j_float'].min()
        vm = sc.v[np.argmin(j)]
        mins.append((0.5 * (a + b), vm, sub.eid.nunique()))
        ax.plot(sc.v, j, lw=2, color=c,
                label=f'|tanθ| {a:.2f}-{b:.2f}  (min {vm:.1f}, '
                      f'{sub.eid.nunique()} ev)')
    ax.axvline(m46.V_NOMINAL, color='#1a9850', ls='--', lw=2, label='v_geom 34')
    ax.axvline(42.1, color='#888', ls='-.', lw=1.6, label='gap-filling 42.1')
    ax.set_xlim(20, 60)
    ax.set_ylim(0.98, 1.6)
    ax.set_xlabel('drift velocity v [µm/ns]')
    ax.set_ylabel('offset-floated median |d|, normalised to min')
    ax.set_title('The scan minimum depends on track angle — a real drift '
                 'velocity cannot\n(floor artifact: v_min → v_true as |tanθ| grows)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, 'anglebin_scan.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'-> {p}')
    for t, vm, n in mins:
        print(f'   |tan| ~{t:.2f}: scan min {vm:.1f} um/ns ({n} events)')


def main():
    fig_gas_scale()
    results, best, ref, by_eid = load_full_reference()
    hits, det = load_hits()
    df = m46.build_hit_table(hits, ref, by_eid, best, res_cut=6.0)
    fig_anglebin_scan(df)


if __name__ == '__main__':
    main()
