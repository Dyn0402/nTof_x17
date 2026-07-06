#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_amplitude_attachment_plot.py

The attachment money-plot: median strip amplitude vs drift TIME (left) and
vs drift DISTANCE z = v_ridge * t (right), one curve per drift HV.

If signal loss were an electronics/timing effect the curves would line up in
TIME; if it is electron attachment (capture per unit drift LENGTH) they must
line up in DISTANCE with a common exponential decay exp(-z/lambda). The right
panel overlays the fitted lambda and the Magboltz attachment band for
0.21-0.42 % O2 (1-2 % air) in Ar/iso 95/5.

Usage: ../.venv/bin/python 19_amplitude_attachment_plot.py [sat_det3] [--veto=50]
Output: <Analysis>/<run>/drift_velocity/<det>/amplitude_attachment.png
"""
import os
import sys
import json
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths, _Config
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
SAMPLE_NS = 60.0
INCL_DEG = 12.0
RES_CUT_MM = 10.0
CHI2_CUT = 5.0   # M3 v2 recipe (chi2<5; NClus>=3 automatic in M3RefTracking); was 20 pre-v2
MIN_HITS_PER_BIN = 30
FIT_Z_MM = (8.0, 24.0)          # fit window: past the spreading pile-up rise
GAP_CM = 3.0

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATT_JSON = os.path.join(REPO, 'garfield_sim', 'results', 'attachment_air_candidates.json')
ALIGN_SEED = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
VD_CSV = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                      CFG.RUN, 'drift_velocity', CFG.DET_NAME, 'drift_velocity_scan.csv')
OUT = os.path.dirname(VD_CSV)


def cfg_for(subrun):
    return _Config(f'{CFG.KEY}_{subrun}', CFG.RUN, subrun, feus=CFG.MX17_FEUS,
                   det_z=CFG.DET_PLANE_Z, det_name=CFG.DET_NAME,
                   base_path=CFG.BASE_PATH, zero_suppressed=CFG.ZERO_SUPPRESSED)


def inclined_events(cfg, seed):
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not os.path.exists(cache):
        return None
    results = pickle.load(open(cache, 'rb'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = seed.ref_x_sign * np.array(xang)
    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xang, anum)
    ids = set()
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        ax = abs(np.degrees(np.arctan(r.ref_tan_theta_x)))
        ay = abs(np.degrees(np.arctan(r.ref_tan_theta_y)))
        if max(ax, ay) > INCL_DEG:
            ids.add(r.event_id)
    return ids


def median_amp_curve(cfg, ids):
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                            expressions=['eventId', 'feu', 'amplitude', 'sample'],
                            library='pd')
    df = df[df['feu'].isin(cfg.MX17_FEUS)]
    hpe = df.groupby('eventId')['eventId'].transform('size')
    df = df[(hpe <= VETO) & df['eventId'].isin(ids)].copy()
    df['t'] = df['sample'] * SAMPLE_NS
    df['t_rel'] = df['t'] - df.groupby(['eventId', 'feu'])['t'].transform('min')
    bins = np.arange(0.0, 1920.0 + SAMPLE_NS, SAMPLE_NS)
    tc = 0.5 * (bins[:-1] + bins[1:])
    t_rel, amp = df['t_rel'].to_numpy(), df['amplitude'].to_numpy()
    med = np.full(len(tc), np.nan)
    for i in range(len(tc)):
        m = (t_rel >= bins[i]) & (t_rel < bins[i + 1])
        if m.sum() >= MIN_HITS_PER_BIN:
            med[i] = np.median(amp[m])
    return tc, med


def main():
    seed = cm.load_alignment(ALIGN_SEED)
    vd = pd.read_csv(VD_CSV)
    geom_csv = os.path.join(OUT, 'geometry_vdrift_scan.csv')
    if os.path.exists(geom_csv):
        gv = pd.read_csv(geom_csv)[['drift_hv', 'v_geom']]
        vd = vd.merge(gv, on='drift_hv', how='left')
        vd['v_ridge'] = vd['v_geom'].fillna(vd['v_ridge'])
        print('using v_geom for the depth scale')
    vd = vd[vd['drift_hv'] >= 500].sort_values('drift_hv')

    curves = []
    for _, row in vd.iterrows():
        cfg = cfg_for(row['subrun'])
        ids = inclined_events(cfg, seed)
        if ids is None or len(ids) < 50:
            continue
        tc, med = median_amp_curve(cfg, ids)
        curves.append(dict(hv=int(row['drift_hv']), v=row['v_ridge'], tc=tc, med=med))
        print(f"  {row['subrun']}: done")

    # common exponential fit in DEPTH over all HV >= 700 V
    zz, aa = [], []
    for c in curves:
        if c['hv'] < 700:
            continue
        z = c['tc'] * c['v'] / 1000.0
        m = (z > FIT_Z_MM[0]) & (z < FIT_Z_MM[1]) & np.isfinite(c['med'])
        zz.append(z[m]); aa.append(c['med'][m])
    zz, aa = np.concatenate(zz), np.concatenate(aa)
    p = np.polyfit(zz, np.log(aa), 1)
    lam_fit = -1.0 / p[0]
    print(f'common exponential fit ({FIT_Z_MM[0]:g}-{FIT_Z_MM[1]:g} mm, '
          f'700-1100 V combined): lambda = {lam_fit:.1f} mm')

    cols = plt.cm.viridis(np.linspace(0.0, 0.9, len(curves)))
    fig, (axt, axz) = plt.subplots(1, 2, figsize=(14.5, 6))

    for c, col in zip(curves, cols):
        axt.plot(c['tc'], c['med'], 'o-', ms=3.5, lw=1.3, color=col,
                 label=f"{c['hv']} V  (v={c['v']:.1f} µm/ns)")
    axt.set_xlabel('drift time  (strip time − earliest strip) [ns]')
    axt.set_ylabel('median strip amplitude [ADC]')
    axt.set_yscale('log')
    axt.set_xlim(0, 1500)
    axt.set_title('vs drift TIME — curves disagree\n(decay is NOT an electronics timescale)')
    axt.legend(fontsize=9)
    axt.grid(alpha=0.3, which='both')

    for c, col in zip(curves, cols):
        axz.plot(c['tc'] * c['v'] / 1000.0, c['med'], 'o-', ms=3.5, lw=1.3, color=col,
                 label=f"{c['hv']} V")
    zf = np.linspace(FIT_Z_MM[0], 30.0, 50)
    axz.plot(zf, np.exp(np.polyval(p, zf)), 'k-', lw=2.5, zorder=6,
             label=f'common fit  $e^{{-z/\\lambda}}$,  $\\lambda={lam_fit:.1f}$ mm')
    # Magboltz band for 1-2 % air at the operating field
    if os.path.exists(ATT_JSON):
        mix = json.load(open(ATT_JSON))['mixtures']
        lams = []
        for name in ('Ar_iso5_air1', 'Ar_iso5_air2'):
            pts = mix[name]
            E = np.array([q['E_Vcm'] for q in pts])
            eta = np.array([q['eta_per_cm'] for q in pts])
            lams.append(10.0 / np.interp(333.0, E, eta))
        lo, hi = min(lams), max(lams)
        a0 = np.exp(np.polyval(p, 12.0))
        for lam, ls in ((lo, ':'), (hi, ':')):
            axz.plot(zf, a0 * np.exp(-(zf - 12.0) / lam), ls, color='crimson', lw=1.8)
        axz.plot([], [], ':', color='crimson', lw=1.8,
                 label=f'Magboltz 1–2% air (O$_2$ attachment):\n'
                       f'$\\lambda$ = {lo:.0f}–{hi:.0f} mm @ 333 V/cm')
    axz.axvline(19.4, color='gray', ls='--', lw=1.2)
    axz.axvline(30.0, color='k', ls='-', lw=1.2)
    axz.annotate('$z_{vis}$ = 19.4', xy=(19.6, 0.955), xycoords=('data', 'axes fraction'),
                 fontsize=10, color='gray')
    axz.annotate('gap = 30', xy=(30.3, 0.955), xycoords=('data', 'axes fraction'),
                 fontsize=10)
    axz.set_xlabel('drift distance  z = v$_{ridge}$ · t  [mm]')
    axz.set_ylabel('median strip amplitude [ADC]')
    axz.set_yscale('log')
    axz.set_xlim(0, 38)
    axz.set_title('vs drift DISTANCE — curves collapse\n(loss per mm drifted = attachment)')
    axz.legend(fontsize=9)
    axz.grid(alpha=0.3, which='both')

    fig.suptitle(f'{CFG.DET_NAME} — attachment signature: amplitude decay is a function '
                 f'of drift distance, not time ({CFG.RUN}, inclined tracks)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'amplitude_attachment.png'), dpi=170,
                bbox_inches='tight')
    print(f'Written {OUT}/amplitude_attachment.png')


if __name__ == '__main__':
    main()
