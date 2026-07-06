#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_geometry_vdrift_scan.py

Bias-free(er) drift velocity from GEOMETRY per drift-scan point:

    v_geom = (cluster-extent slope vs |tan th_ref|) / T_sat

The extent slope (mm per unit tan, from strip counts x pitch vs the M3 angle)
measures the recorded column depth z_rec + w' with NO time information and no
drift-velocity prior; T_sat (median cluster time span of inclined tracks) is
the drift time across the same recorded column. Their ratio is v, immune to
the amplitude-weighted time-fit bias that pulls the ridge estimator low
(20_ridge_systematics.py: ridge is convex, inner 26 vs outer 32 um/ns, and
extent slope = 23.2 mm vs v_ridge*T_sat = 19.4 mm on the long run).

Works even where the readout window truncates the column (both numerator and
denominator see the same truncated column).

Usage: ../.venv/bin/python 21_geometry_vdrift_scan.py sat_det3 [--veto=50]
Output: <Analysis>/<run>/drift_velocity/<det>/geometry_vdrift_scan.csv/.png
"""
import os
import re
import sys
import pickle
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths, _Config
setup_paths()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
MIN_STRIPS = 4
RES_CUT_MM = 10.0
PITCH_MM = 0.78
CHI2_CUT = 5.0   # M3 v2 recipe (chi2<5; NClus>=3 automatic in M3RefTracking); was 20 pre-v2
SAT_DEG = 10.0
TAN_LO, TAN_HI, TAN_STEP = 0.06, 0.44, 0.04
GAP_CM = 3.0

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(REPO, 'garfield_sim', 'results')
ALIGN_SEED = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
OUT = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                   CFG.RUN, 'drift_velocity', CFG.DET_NAME)


def cfg_for(subrun):
    return _Config(f'{CFG.KEY}_{subrun}', CFG.RUN, subrun, feus=CFG.MX17_FEUS,
                   det_z=CFG.DET_PLANE_Z, det_name=CFG.DET_NAME,
                   base_path=CFG.BASE_PATH, zero_suppressed=CFG.ZERO_SUPPRESSED)


def measure_point(subrun, seed):
    cfg = cfg_for(subrun)
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not os.path.exists(cache):
        print(f'  [SKIP] {subrun}: no cache')
        return None
    results = pickle.load(open(cache, 'rb'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = seed.ref_x_sign * np.array(xang)
    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xang, anum)

    sel = [r for r in results if r.has_x and r.has_y
           and np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_tan_theta_y)
           and r.x_fit.n_strips >= MIN_STRIPS and r.y_fit.n_strips >= MIN_STRIPS
           and np.isfinite(r.radial_residual_mm) and r.radial_residual_mm < RES_CUT_MM]
    if len(sel) < 200:
        print(f'  [SKIP] {subrun}: only {len(sel)} events')
        return None

    tan_rx = np.array([r.ref_tan_theta_x for r in sel])
    tan_ry = np.array([r.ref_tan_theta_y for r in sel])
    th = np.deg2rad(seed.theta_deg)
    t_det = {
        'x': np.cos(th) * tan_rx + np.sin(th) * tan_ry,
        'y': -np.sin(th) * tan_rx + np.cos(th) * tan_ry,
    }
    ns = {'x': np.array([r.x_fit.n_strips for r in sel]),
          'y': np.array([r.y_fit.n_strips for r in sel])}
    dur = {'x': np.array([r.x_fit.latest_time_ns - r.x_fit.earliest_time_ns for r in sel]),
           'y': np.array([r.y_fit.latest_time_ns - r.y_fit.earliest_time_ns for r in sel])}

    vs, zs, row = [], [], dict(subrun=subrun, n_quality=len(sel))
    for p in ('x', 'y'):
        at = np.abs(t_det[p])
        ext = (ns[p] - 1) * PITCH_MM
        bins = np.arange(TAN_LO, TAN_HI + TAN_STEP, TAN_STEP)
        ctr, med, mer = [], [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (at >= b0) & (at < b1)
            if m.sum() >= 40:
                ctr.append(0.5 * (b0 + b1))
                med.append(np.median(ext[m]))
                mer.append(1.253 * np.std(ext[m], ddof=1) / np.sqrt(m.sum()))
        if len(ctr) < 4:
            continue
        ctr, med, mer = map(np.array, (ctr, med, mer))
        w = 1.0 / mer**2
        # weighted straight line through the medians
        W = np.sum(w); Wx = np.sum(w * ctr); Wy = np.sum(w * med)
        Wxx = np.sum(w * ctr**2); Wxy = np.sum(w * ctr * med)
        den = W * Wxx - Wx**2
        slope = (W * Wxy - Wx * Wy) / den
        slope_err = np.sqrt(W / den)
        m_sat = np.abs(np.degrees(np.arctan(t_det[p]))) > SAT_DEG
        if m_sat.sum() < 30:
            continue
        tsat = float(np.median(dur[p][m_sat]))
        tsat_err = float(1.253 * np.std(dur[p][m_sat], ddof=1) / np.sqrt(m_sat.sum()))
        v = slope * 1000.0 / tsat
        v_err = v * np.hypot(slope_err / slope, tsat_err / tsat)
        vs.append((v, v_err)); zs.append(slope)
        row[f'z_slope_{p}_mm'] = float(slope)
        row[f't_sat_{p}_ns'] = tsat
        row[f'v_geom_{p}'] = float(v)
        row[f'v_geom_{p}_err'] = float(v_err)

    if not vs:
        return None
    ws = 1.0 / np.array([e for _, e in vs])**2
    row['v_geom'] = float(np.sum([v * w for (v, _), w in zip(vs, ws)]) / np.sum(ws))
    row['v_geom_err'] = float(np.sqrt(1.0 / np.sum(ws)))
    if len(vs) == 2:
        row['v_geom_sys'] = float(abs(vs[0][0] - vs[1][0]) / 2.0)
    print(f'  {subrun}: v_geom = {row["v_geom"]:.2f} ± {row["v_geom_err"]:.2f} µm/ns   '
          f'z_slopes = {[f"{z:.1f}" for z in zs]} mm')
    return row


def main():
    seed = cm.load_alignment(ALIGN_SEED)
    run_dir = os.path.join(CFG.BASE_PATH, CFG.RUN)
    points = []
    for name in sorted(os.listdir(run_dir)):
        m = re.match(r'drift_scan_resist_(\d+)V_drift_(\d+)V$', name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            points.append((int(m.group(2)), name))
    m = re.search(r'drift_(\d+)V', CFG.SUB_RUN)
    if m:
        points.append((int(m.group(1)), CFG.SUB_RUN))
    points.sort()

    rows = []
    for hv, subrun in points:
        print(f'=== drift {hv} V — {subrun} ===')
        r = measure_point(subrun, seed)
        if r is not None:
            r['drift_hv'] = hv
            r['is_long_run'] = (subrun == CFG.SUB_RUN)
            rows.append(r)

    df = pd.DataFrame(rows).sort_values('drift_hv')
    df.to_csv(os.path.join(OUT, 'geometry_vdrift_scan.csv'), index=False)

    # comparison with the ridge estimator
    ridge = pd.read_csv(os.path.join(OUT, 'drift_velocity_scan.csv'))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    e = df['drift_hv'] / GAP_CM
    ax.errorbar(e, df['v_geom'], yerr=np.hypot(df['v_geom_err'],
                df.get('v_geom_sys', 0.0)), fmt='o', color='black', ms=9,
                capsize=4, zorder=6, label='v_geom = extent-slope / T_sat (this work)')
    er = ridge['drift_hv'] / GAP_CM
    ax.errorbar(er, ridge['v_ridge'], yerr=np.hypot(ridge['v_ridge_err'],
                ridge['v_ridge_sys']), fmt='s', color='gray', ms=6, capsize=3,
                alpha=0.8, zorder=5, label='v_ridge (time-fit; biased low)')
    star = df[df['is_long_run']]
    if len(star):
        ax.plot(star['drift_hv'] / GAP_CM, star['v_geom'], '*', color='crimson',
                ms=18, zorder=7, label='2.4 h long run')

    styles = {
        'Ar95_iso5':        ('steelblue', '-', 'clean Ar/iso 95/5'),
        'Ar94_iso5_H2O1':   ('tab:green', '--', '+ 1% H2O'),
        'Ar93_iso5_H2O2':   ('tab:olive', '--', '+ 2% H2O'),
        'Ar90_CO2_10':      ('tab:gray', ':', 'Ar/CO2 90/10'),
    }
    seen = set()
    for fn in ('attachment_Ar_iso_H2O.json', 'attachment_air_candidates.json',
               'drift_velocity_candidates.json', 'water_grid.json'):
        p = os.path.join(GS, fn)
        if not os.path.exists(p):
            continue
        for name, pts in json.load(open(p))['mixtures'].items():
            if name in seen:
                continue
            seen.add(name)
            st = styles.get(name)
            if st is None and name.startswith('Ar_iso5_H2O'):
                st = ('tab:green', ':', name.replace('Ar_iso5_H2O', '+ ') + '% H2O')
            if st is None:
                continue
            c, ls, lab = st
            Ec = np.array([q['E_Vcm'] for q in pts])
            Vc = np.array([q['v_um_per_ns'] for q in pts])
            ax.plot(Ec, Vc, ls, color=c, lw=1.8, label=lab)

    ax.set_xlabel(f'drift field [V/cm]  (E = HV / {GAP_CM:g} cm)')
    ax.set_ylabel('drift velocity [µm/ns]')
    ax.set_xlim(0, 450); ax.set_ylim(0, 45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title(f'{CFG.DET_NAME} geometry-based drift velocity vs Magboltz — {CFG.RUN}')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'geometry_vdrift_scan.png'), dpi=170,
                bbox_inches='tight')
    print(f'\nWritten {OUT}/geometry_vdrift_scan.png + .csv')

    # RMS ranking against v_geom (drop <=300 V: window truncation kills the
    # extent lever arm there, z_slopes collapse to the min-strips floor)
    dfr = df[df['drift_hv'] >= 500]
    print('\nRMS vs v_geom (E = HV/3, HV >= 500 V):')
    e_meas = dfr['drift_hv'].to_numpy() / GAP_CM
    vm = dfr['v_geom'].to_numpy()
    ranked = []
    seen = set()
    for fn in ('drift_velocity_candidates.json', 'drift_velocity_candidates2.json',
               'attachment_air_candidates.json', 'attachment_Ar_iso_H2O.json',
               'water_grid.json'):
        p = os.path.join(GS, fn)
        if not os.path.exists(p):
            continue
        for name, pts in json.load(open(p))['mixtures'].items():
            if name in seen:
                continue
            seen.add(name)
            Ec = np.array([q['E_Vcm'] for q in pts])
            Vc = np.array([q['v_um_per_ns'] for q in pts])
            ranked.append((float(np.sqrt(np.mean((np.interp(e_meas, Ec, Vc) - vm)**2))), name))
    for rms, name in sorted(ranked):
        print(f'  {name:22s} RMS = {rms:5.2f} µm/ns')


if __name__ == '__main__':
    main()
