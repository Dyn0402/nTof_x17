#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
44_final_vdrift_plot.py

FINAL drift-velocity vs drift-field plot for the det3 Saturday drift scan.

Method decision (see 20/21/23 + 43 window-truncation diagnostic):
  * PRIMARY  : v_geom = extent-slope / T_sat  (21_geometry_vdrift_scan.csv).
               Bias-free (does not use the corrupted strip-time ORDER) and
               truncation-immune by construction (numerator and denominator
               clip at the same recorded depth).
  * X-CHECK 1: v_sig = hybrid signature-regressed angle -> geometry v
               (35_hybrid_drift_scan.csv), a hits-only INDEPENDENT recompute
               that spans every field point.
  * X-CHECK 2: unshared time-fit at the 1000 V operating point (waveform
               chain 26/27): 34.2 / 32.9 um/ns (x/y) -> converges with v_geom,
               proving the geometry value is not a geometry artefact.

Validity range (43): 500-1100 V is defensible; 300 V and 100 V are
WINDOW-LIMITED and shown excluded -- at <=300 V the recorded column
d_win = v * t_window collapses onto the ~3.9 mm charge-spreading floor, so the
extent-vs-angle lever arm vanishes and v is unconstrained.

Overlay: Magboltz curves for the Ar/iC4H10 95/5 + H2O(+air) family; the RMS
best-fit member over the valid range is highlighted.

Usage: ../.venv/bin/python 44_final_vdrift_plot.py sat_det3
Output: <Analysis>/<run>/drift_velocity/<det>/final_vdrift_vs_field.png (+ .csv)
"""
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv

CFG = config_from_argv()
GAP_CM = 3.0
VALID_HV_MIN = 500          # <500 V is window-limited (see 43)
# shade edge at the midpoint between the last excluded (300 V) and first valid
# (500 V) point, so the 500 V point reads clearly as valid
SHADE_EDGE = 400 / GAP_CM                 # 133 V/cm
E_WINDOW_LIMIT = VALID_HV_MIN / GAP_CM   # 167 V/cm (first valid point)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(REPO, 'garfield_sim', 'results')
OUT = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                   CFG.RUN, 'drift_velocity', CFG.DET_NAME)

# 1000 V operating-point unshared time-fit cross-check (waveform chain 26/27)
UNSHARED_1000V = dict(hv=1000, vx=34.2, vy=32.9)

# gas family to overlay: (json file, mixture key, colour, style, label)
GAS_CURVES = [
    ('attachment_Ar_iso_H2O.json', 'Ar95_iso5',      'tab:blue',  '-',  'Ar/iC$_4$H$_{10}$ 95/5 (dry)'),
    ('water_grid.json',            'Ar_iso5_H2O0.8',  'tab:cyan',  '--', '+0.8% H$_2$O'),
    ('attachment_Ar_iso_H2O.json', 'Ar94_iso5_H2O1',  'tab:green', '-',  '+1.0% H$_2$O'),
    ('water_grid.json',            'Ar_iso5_H2O1_air2','tab:olive', '-.', '+1.0% H$_2$O +2% air'),
    ('water_grid.json',            'Ar_iso5_H2O1.2',  'tab:orange','--', '+1.2% H$_2$O'),
    ('attachment_Ar_iso_H2O.json', 'Ar93_iso5_H2O2',  'tab:red',   ':',  '+2.0% H$_2$O'),
]


def load_gas(fn, key):
    p = os.path.join(GS, fn)
    if not os.path.exists(p):
        return None
    mix = json.load(open(p)).get('mixtures', {})
    if key not in mix:
        return None
    pts = mix[key]
    return (np.array([q['E_Vcm'] for q in pts]),
            np.array([q['v_um_per_ns'] for q in pts]))


def main():
    geo = pd.read_csv(os.path.join(OUT, 'geometry_vdrift_scan.csv')).sort_values('drift_hv')
    geo['E'] = geo['drift_hv'] / GAP_CM
    geo['v_err_tot'] = np.hypot(geo['v_geom_err'].abs(), geo.get('v_geom_sys', 0.0))
    valid = geo[geo['drift_hv'] >= VALID_HV_MIN].copy()
    excl = geo[geo['drift_hv'] < VALID_HV_MIN].copy()

    # slope-vs-reference (waveform, unshared) — truncation-robust cross-check +
    # low-field recovery (uses M3 tanθ, so not independent)
    slr = None
    sp = os.path.join(OUT, 'slope_reference_vdrift_scan.csv')
    if os.path.exists(sp):
        slr = pd.read_csv(sp).sort_values('drift_hv')
        slr['E'] = slr['drift_hv'] / GAP_CM

    hyb = None
    hp = os.path.join(OUT, 'hybrid_vdrift_scan.csv')
    if os.path.exists(hp):
        hyb = pd.read_csv(hp).sort_values('drift_hv')
        hyb['E'] = hyb['drift_hv'] / GAP_CM
        hyb = hyb[(hyb['drift_hv'] >= VALID_HV_MIN) & hyb['v_sig'].notna()].copy()

    # PRIMARY = slope-vs-reference UNSHARED (recipe-robust; balanced x/y).
    # Geometry Y-plane extent-slope is fragile under the tight M3 recipe
    # (chi2<1/NClus4) -> show geometry per-plane so X (robust) and Y (fragile)
    # are both visible instead of a misleading combined average.
    slrv = None
    if slr is not None and 'v_unshared' in slr:
        slrv = slr[(slr['drift_hv'] >= VALID_HV_MIN) & slr['v_unshared'].notna()
                   & (slr['v_unshared'] > 0)].copy()

    # ---- best-fit gas over the valid range (RMS to the PRIMARY series) ----
    if slrv is not None and len(slrv):
        e_meas, v_meas = slrv['E'].to_numpy(), slrv['v_unshared'].to_numpy()
        prim_lbl = 'slope method'
    else:
        e_meas, v_meas = valid['E'].to_numpy(), valid['v_geom'].to_numpy()
        prim_lbl = 'v_geom'
    ranking = []
    for fn, key, *_ in GAS_CURVES:
        g = load_gas(fn, key)
        if g is None:
            continue
        Ec, Vc = g
        rms = float(np.sqrt(np.mean((np.interp(e_meas, Ec, Vc) - v_meas) ** 2)))
        ranking.append((rms, key))
    ranking.sort()
    best_key = ranking[0][1] if ranking else None

    # ---------------------------- plot ----------------------------
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.axvspan(0, SHADE_EDGE, color='0.92', zorder=0)
    ax.text(SHADE_EDGE * 0.5, 42.5, 'window-limited\n(drift time > readout\nwindow — excluded)',
            ha='center', va='top', fontsize=8.5, color='0.35')

    # Magboltz family
    for fn, key, c, ls, lab in GAS_CURVES:
        g = load_gas(fn, key)
        if g is None:
            continue
        Ec, Vc = g
        is_best = (key == best_key)
        ax.plot(Ec, Vc, ls, color=c, lw=3.0 if is_best else 1.6,
                alpha=1.0 if is_best else 0.75, zorder=3 if is_best else 2,
                label=lab + (f'  ★ best fit to {prim_lbl} (RMS {ranking[0][0]:.2f})' if is_best else ''))

    # geometry cross-check: X-plane only (Y-plane extent-slope is stats-hungry &
    # recipe-fragile under chi2<1/NClus4 -> noted in the caption, not plotted).
    # Geometry uses the looser match (res<10) because its extent-slope binning
    # needs the statistics; the long-run point is the high-stats anchor.
    gv = geo[geo['drift_hv'] >= VALID_HV_MIN]
    ax.plot(gv['E'], gv['v_geom_x'], '^', mfc='none', mec='tab:blue', ms=10, mew=1.6,
            zorder=5, label='geometry X-plane (cross-check; stats-limited <900 V)')

    # slope-vs-reference RAW (sharing-biased low) — the bias the unsharing fixes
    if slr is not None and 'v_raw' in slr:
        mr = slr['v_raw'].notna() & (slr['v_raw'] > 0)
        ax.plot(slr.loc[mr, 'E'], slr.loc[mr, 'v_raw'], 'x', color='0.6',
                ms=8, mew=1.6, zorder=4, label='slope-vs-ref, RAW times (sharing-biased low)')

    # PRIMARY: slope-vs-reference UNSHARED
    if slrv is not None and len(slrv):
        ax.errorbar(slrv['E'], slrv['v_unshared'], yerr=slrv['v_unshared_err'].abs(),
                    fmt='o', color='black', ms=11, capsize=5, mew=1.5, zorder=8,
                    label='v (slope-vs-ref, UNSHARED) — this work')
        # 300 V slope point (partial recovery, in the excluded band)
        s300 = slr[(slr['drift_hv'] < VALID_HV_MIN) & slr['v_unshared'].notna()
                   & (slr['v_unshared'] > 0)]
        if len(s300):
            ax.errorbar(s300['E'], s300['v_unshared'], yerr=s300['v_unshared_err'].abs(),
                        fmt='o', mfc='none', mec='0.6', ecolor='0.6', ms=10, capsize=4,
                        zorder=4, label='excluded / partial (window-limited)')

    ax.set_xlabel('drift field  E = HV / 3 cm   [V/cm]', fontsize=12)
    ax.set_ylabel('electron drift velocity  [µm/ns]', fontsize=12)
    ax.set_xlim(0, 400); ax.set_ylim(0, 45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.95)
    ax.text(0.985, 0.985, 'primary (slope method) cuts: |Δ$_{ref}$|<2 mm, edge>25 mm, M3 χ²<1 & NClus=4\n'
            'geometry cross-check: looser match (extent-slope needs stats); Y-plane omitted\n'
            '(recipe-fragile). Slope method (per-strip times) is robust to cuts & recipe.',
            transform=ax.transAxes, va='top', ha='right', fontsize=7.2, color='0.4')
    ax.set_title(f'{CFG.DET_NAME} drift velocity vs field — Saturday scan {CFG.RUN}\n'
                 f'slope-vs-reference (primary) + geometry cross-check vs Magboltz',
                 fontsize=12)
    fig.tight_layout()
    outpng = os.path.join(OUT, 'final_vdrift_vs_field.png')
    fig.savefig(outpng, dpi=180, bbox_inches='tight')

    # summary CSV
    if slrv is not None and len(slrv):
        summ = slrv[['drift_hv', 'E', 'v_raw', 'v_unshared', 'v_unshared_err']].copy()
        summ = summ.merge(geo[['drift_hv', 'v_geom_x', 'v_geom_y']], on='drift_hv', how='left')
        summ.to_csv(os.path.join(OUT, 'final_vdrift_vs_field.csv'), index=False)
        print(f'Gas RMS ranking over valid range (>=500 V, vs {prim_lbl}):')
        for rms, key in ranking:
            print(f'   {key:22s} RMS = {rms:5.2f} µm/ns{"  <-- best" if key==best_key else ""}')
        print('\nFinal v (primary = slope method) + geometry per-plane cross-check:')
        print(summ.round(2).to_string(index=False))
    print(f'\nWritten: {outpng}  +  final_vdrift_vs_field.csv')


if __name__ == '__main__':
    main()
