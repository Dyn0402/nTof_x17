#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_drift_velocity_scan.py

Measure the electron drift velocity vs drift HV from the micro-TPC angle
correlation, one measurement per drift-scan subrun (plus the long run as the
operating-point measurement).

Estimator (bias-robust; see 13_tpc_angle_bias.py)
-------------------------------------------------
The strip-fit slope in the M3 frame S [µm/ns] relates to the reference track
angle as
    S = v_drift * tanθ_ref  ±  v_drift * (w_extra/gap)
where the intercept term is the (angle-independent) charge-spreading excess.
A straight-line fit of S vs tanθ_ref per track-sign therefore measures
v_drift as the SLOPE, unbiased by the off-diagonal offset that afflicts the
diagonal-projection σ-scan.  Four independent fits (X/Y plane × sign) are
combined by weighted mean.

Cross-check: the cluster TIME SPAN saturates at the full-gap drift time
T_gap = gap / v_drift for inclined tracks; median span at |θ_ref| > 10°
gives an independent 1/v proxy (same gap at every HV).

Usage:
    ../.venv/bin/python 14_drift_velocity_scan.py sat_det3 [--veto=50] [--refit]

Output: <Analysis>/<run>/drift_velocity/<det>/
    drift_velocity_scan.csv, drift_velocity_scan.png
"""
import os
import re
import sys
import pickle
import concurrent.futures

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
REFIT = '--refit' in sys.argv
MIN_STRIPS = 4
RES_CUT_MM = 10.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
TAN_FIT_MIN, TAN_FIT_MAX = 0.06, 0.55   # |tanθ_ref| window for the ridge fit
SAT_DEG = 10.0                           # |θ_ref| above which time span saturates
MAGBOLTZ_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'garfield_sim', 'results',
                             'drift_velocity_Ar_iC4H10_95_5_Saclay.json')

ALIGN_SEED = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')


def robust_line(x, y, n_iter=4, clip=3.0):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    keep = np.ones(len(x), bool)
    p = (np.nan, np.nan)
    for _ in range(n_iter):
        if keep.sum() < 20:
            return np.nan, np.nan, np.nan, 0
        p = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < clip * s
    n = int(keep.sum())
    xk = x[keep]
    resid = y[keep] - np.polyval(p, xk)
    se = np.sqrt(np.sum(resid**2) / max(n - 2, 1) / np.sum((xk - xk.mean())**2))
    return float(p[0]), float(p[1]), float(se), n


def cfg_for(subrun):
    return _Config(f'{CFG.KEY}_{subrun}', CFG.RUN, subrun, feus=CFG.MX17_FEUS,
                   det_z=CFG.DET_PLANE_Z, det_name=CFG.DET_NAME,
                   base_path=CFG.BASE_PATH, zero_suppressed=CFG.ZERO_SUPPRESSED)


def analyse(cfg, det):
    """Per-event micro-TPC analysis with spark veto, cached per subrun."""
    tag = f'_veto{VETO}' if VETO is not None else ''
    cache = os.path.join(cfg.out_dir('cache'), f'event_results{tag}.pkl')
    if os.path.exists(cache) and not REFIT:
        return pickle.load(open(cache, 'rb'))
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs], library='pd')
    df = df[df['feu'].isin(cfg.MX17_FEUS)].copy()
    if VETO is not None:
        hpe = df.groupby('eventId')['channel'].transform('size')
        df = df[hpe <= VETO].copy()
    df = cm._map_strip_positions(df, det)
    eids = df['eventId'].unique()
    g = df.groupby('eventId')
    args = [(g.get_group(e).copy(), int(e)) for e in eids]
    nw = max(1, (os.cpu_count() or 1) - cm.N_FREE_THREADS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=nw) as pool:
        results = list(cm._progress(pool.map(cm._analyse_event_worker, args),
                                    total=len(args), desc=f'  {cfg.SUB_RUN}'))
    pickle.dump(results, open(cache, 'wb'))
    return results


def measure_point(subrun, det, seed):
    """Full v_drift measurement for one subrun. Returns row dict or None."""
    cfg = cfg_for(subrun)
    if not (os.path.isdir(cfg.combined_hits_dir) and os.path.isdir(cfg.m3_tracking_dir)):
        print(f'  [SKIP] {subrun}: missing data dirs')
        return None
    results = analyse(cfg, det)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = seed.ref_x_sign * np.array(xang)
    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xang, anum)

    sel = [r for r in results if r.has_x and r.has_y
           and np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_tan_theta_y)
           and np.isfinite(r.x_fit.slope_mm_per_ns) and np.isfinite(r.y_fit.slope_mm_per_ns)
           and r.x_fit.n_strips >= MIN_STRIPS and r.y_fit.n_strips >= MIN_STRIPS
           and np.isfinite(r.radial_residual_mm) and r.radial_residual_mm < RES_CUT_MM]
    n_q = len(sel)
    if n_q < 100:
        print(f'  [SKIP] {subrun}: only {n_q} quality events')
        return None

    s_x = np.array([r.x_fit.slope_mm_per_ns for r in sel]) * 1000.0
    s_y = np.array([r.y_fit.slope_mm_per_ns for r in sel]) * 1000.0
    tan_rx = np.array([r.ref_tan_theta_x for r in sel])
    tan_ry = np.array([r.ref_tan_theta_y for r in sel])
    dur_x = np.array([r.x_fit.latest_time_ns - r.x_fit.earliest_time_ns for r in sel])
    dur_y = np.array([r.y_fit.latest_time_ns - r.y_fit.earliest_time_ns for r in sel])

    th = np.deg2rad(seed.theta_deg)
    S_x = np.cos(th) * s_x - np.sin(th) * s_y
    S_y = np.sin(th) * s_x + np.cos(th) * s_y

    fits = []
    for S, tr in [(S_x, tan_rx), (S_y, tan_ry)]:
        for lo, hi in [(TAN_FIT_MIN, TAN_FIT_MAX), (-TAN_FIT_MAX, -TAN_FIT_MIN)]:
            m = (tr > lo) & (tr < hi)
            v, b, ve, n = robust_line(tr[m], S[m])
            if np.isfinite(v) and n > 50:
                fits.append((v, b, ve, n))
    if len(fits) < 2:
        print(f'  [SKIP] {subrun}: ridge fits failed')
        return None
    vs = np.array([f[0] for f in fits])
    ws = 1.0 / np.array([f[2] for f in fits])**2
    v_ridge = float(np.sum(vs * ws) / np.sum(ws))
    v_ridge_err = float(np.sqrt(1.0 / np.sum(ws)))
    # spread between the 4 fits as a systematic
    v_ridge_sys = float(np.std(vs, ddof=1) / np.sqrt(len(vs)))
    w_gap = float(np.mean([abs(f[1]) for f in fits]) / v_ridge)

    # time-span saturation (inclined tracks only; per-plane pairing at θ≈90°:
    # X plane (s_x) pairs tan_ref_y, Y plane (s_y) pairs tan_ref_x)
    sat_x = dur_x[np.abs(np.degrees(np.arctan(tan_ry))) > SAT_DEG]
    sat_y = dur_y[np.abs(np.degrees(np.arctan(tan_rx))) > SAT_DEG]
    t_sat = float(np.median(np.concatenate([sat_x, sat_y])))
    t_sat_err = float(1.253 * np.std(np.concatenate([sat_x, sat_y]), ddof=1)
                      / np.sqrt(len(sat_x) + len(sat_y)))

    print(f'  {subrun}: v_ridge = {v_ridge:.2f} ± {v_ridge_err:.2f} (stat) '
          f'± {v_ridge_sys:.2f} (fit spread) µm/ns   w/gap = {w_gap:.3f}   '
          f'T_sat = {t_sat:.0f} ns   (n={n_q:,})')
    return dict(subrun=subrun, n_quality=n_q, v_ridge=v_ridge,
                v_ridge_err=v_ridge_err, v_ridge_sys=v_ridge_sys,
                w_over_gap=w_gap, t_sat_ns=t_sat, t_sat_err_ns=t_sat_err)


def main():
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    seed = cm.load_alignment(ALIGN_SEED)
    print(f'Alignment seed: {seed}')

    run_dir = os.path.join(CFG.BASE_PATH, CFG.RUN)
    points = []
    for name in sorted(os.listdir(run_dir)):
        m = re.match(r'drift_scan_resist_(\d+)V_drift_(\d+)V$', name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            points.append((int(m.group(2)), name))
    points.sort()
    # the long run is the high-stats operating point (append with its drift V)
    m = re.search(r'drift_(\d+)V', CFG.SUB_RUN)
    if m:
        points.append((int(m.group(1)), CFG.SUB_RUN))
    print(f'Drift points: {[(v, s) for v, s in points]}')

    rows = []
    for hv, subrun in points:
        print(f'\n=== drift {hv} V — {subrun} ===')
        r = measure_point(subrun, det, seed)
        if r is not None:
            r['drift_hv'] = hv
            r['is_long_run'] = (subrun == CFG.SUB_RUN)
            rows.append(r)

    df = pd.DataFrame(rows).sort_values('drift_hv')
    out = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                       CFG.RUN, 'drift_velocity', CFG.DET_NAME)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'drift_velocity_scan.csv'), index=False)

    # gap estimate from the long-run point: gap = v_ridge * T_sat
    lr = df[df['is_long_run']]
    ref = lr.iloc[0] if len(lr) else df.iloc[-1]
    gap_mm = ref['v_ridge'] * ref['t_sat_ns'] / 1000.0
    print(f'\nGap estimate from {ref["subrun"]}: v*T_sat = {gap_mm:.1f} mm')

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    d = df[~df['is_long_run']]
    l = df[df['is_long_run']]
    err = np.hypot(df['v_ridge_err'], df['v_ridge_sys'])
    ax = axes[0]
    ax.errorbar(d['drift_hv'], d['v_ridge'],
                yerr=np.hypot(d['v_ridge_err'], d['v_ridge_sys']),
                fmt='o-', color='steelblue', capsize=4, ms=7, label='drift scan (15 min pts)')
    if len(l):
        ax.errorbar(l['drift_hv'], l['v_ridge'],
                    yerr=np.hypot(l['v_ridge_err'], l['v_ridge_sys']),
                    fmt='*', color='crimson', ms=16, capsize=4, label='7 h long run')
    # Magboltz overlay (needs the gap to convert HV → field)
    if os.path.exists(MAGBOLTZ_JSON):
        import json
        mb = json.load(open(MAGBOLTZ_JSON))
        e = np.array([p['E_Vcm'] for p in mb['points']])
        v = np.array([p['v_um_per_ns'] for p in mb['points']])
        hv_mb = e * gap_mm / 10.0   # V = E[V/cm] * gap[cm]
        ax.plot(hv_mb, v, 'k--', lw=1.5,
                label=f'Magboltz {mb["gas"]} (gap={gap_mm:.1f} mm)')
    ax.set_xlabel('drift HV [V]')
    ax.set_ylabel('drift velocity [µm/ns]')
    ax.set_title('v_drift from angle-correlation ridge fit')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_xlim(0, None); ax.set_ylim(0, None)

    ax = axes[1]
    ax.errorbar(df['drift_hv'], df['t_sat_ns'], yerr=df['t_sat_err_ns'], fmt='s-',
                color='darkorange', capsize=4, ms=6, label='median span, |θ|>10°')
    ax2 = ax.twinx()
    ax2.errorbar(df['drift_hv'], gap_mm * 1000.0 / df['t_sat_ns'],
                 yerr=gap_mm * 1000.0 * df['t_sat_err_ns'] / df['t_sat_ns']**2,
                 fmt='^:', color='seagreen', capsize=3, ms=6)
    ax2.set_ylabel(f'gap/T_sat [µm/ns] (gap={gap_mm:.1f} mm)', color='seagreen')
    ax2.tick_params(axis='y', labelcolor='seagreen')
    ax.set_xlabel('drift HV [V]')
    ax.set_ylabel('cluster time span T_sat [ns]', color='darkorange')
    ax.set_title('full-gap drift time (independent 1/v proxy)')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(df['drift_hv'], df['w_over_gap'], 'D-', color='purple', ms=6)
    ax.set_xlabel('drift HV [V]')
    ax.set_ylabel('ridge-fit |intercept|/v  =  w_extra/gap')
    ax.set_title('charge-spreading excess vs drift HV')
    ax.grid(alpha=0.3)

    fig.suptitle(f'{CFG.DET_NAME} drift-velocity scan — {CFG.RUN} '
                 f'(resist 490 V, alignment from long run)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, 'drift_velocity_scan.png'), dpi=170, bbox_inches='tight')
    print(f'\nWritten: {out}/drift_velocity_scan.png  +  .csv')


if __name__ == '__main__':
    main()
