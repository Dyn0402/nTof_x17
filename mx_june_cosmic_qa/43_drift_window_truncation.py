#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
43_drift_window_truncation.py

DIAGNOSTIC (no new physics estimate yet): map the DREAM readout-window
truncation across the det3 drift scan so we can decide, per field point,
whether the geometry v estimator is measuring the *gap* or just the
*window*.

At low drift field the electrons from the TOP of the 30 mm drift gap take
longer to arrive than the 32-sample x 60 ns = 1920 ns readout window, so the
deep part of the ionization column is never recorded.  Symptoms we quantify
here, per drift HV, using ONLY the veto50 hits cache (each StripFitResult
stores absolute earliest/latest_time_ns within the window, n_strips, mesh pos):

  (a) latest_time_ns distribution     -> is the deep edge piling up at a
                                         window ceiling? where is the ceiling?
  (b) earliest_time_ns distribution   -> the t0 / mesh-arrival offset
  (c) extent (n_strips-1)*pitch vs |tan_det|  -> does the column SATURATE
                                         (flatten) above some tan, and at what
                                         extent? (= truncation onset)
  (d) cluster time-span vs |tan_det|  -> same saturation seen in time
  (e) latest_time vs |tan_det|        -> the deep edge should ride up to the
                                         ceiling for inclined tracks once
                                         truncation sets in

Output: <Analysis>/<run>/drift_velocity/<det>/window_truncation/
    window_truncation_overview.png   (ceiling + onset summary, all points)
    window_truncation_profiles.png   (extent/span/latest vs tan, per point)
    window_truncation.csv            (per-point ceiling, onset tan, sat extent)

Usage: ../.venv/bin/python 43_drift_window_truncation.py sat_det3 [--veto=50]
"""
import os
import re
import sys
import pickle

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
GAP_CM = 3.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS

N_SAMPLES = 32
DT_NS = 60.0
WINDOW_NS = N_SAMPLES * DT_NS      # nominal 1920 ns
SAT_DEG = 10.0

ALIGN_SEED = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
OUT = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                   CFG.RUN, 'drift_velocity', CFG.DET_NAME, 'window_truncation')
os.makedirs(OUT, exist_ok=True)


def cfg_for(subrun):
    return _Config(f'{CFG.KEY}_{subrun}', CFG.RUN, subrun, feus=CFG.MX17_FEUS,
                   det_z=CFG.DET_PLANE_Z, det_name=CFG.DET_NAME,
                   base_path=CFG.BASE_PATH, zero_suppressed=CFG.ZERO_SUPPRESSED)


def load_point(subrun, seed):
    """Return per-track arrays for both planes, or None."""
    cfg = cfg_for(subrun)
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not os.path.exists(cache):
        print(f'  [SKIP] {subrun}: no cache'); return None
    results = pickle.load(open(cache, 'rb'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = seed.ref_x_sign * np.array(xang)
    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xang, anum)

    sel = [r for r in results if r.has_x and r.has_y
           and np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_tan_theta_y)
           and r.x_fit.n_strips >= MIN_STRIPS and r.y_fit.n_strips >= MIN_STRIPS
           and np.isfinite(r.radial_residual_mm) and r.radial_residual_mm < RES_CUT_MM]
    if len(sel) < 100:
        print(f'  [SKIP] {subrun}: only {len(sel)} events'); return None

    tan_rx = np.array([r.ref_tan_theta_x for r in sel])
    tan_ry = np.array([r.ref_tan_theta_y for r in sel])
    th = np.deg2rad(seed.theta_deg)
    # physical-plane detector-frame angle that each plane's cluster responds to
    t_det = {'x': np.cos(th) * tan_rx + np.sin(th) * tan_ry,
             'y': -np.sin(th) * tan_rx + np.cos(th) * tan_ry}
    out = {'n': len(sel)}
    for p, fitattr in (('x', 'x_fit'), ('y', 'y_fit')):
        fits = [getattr(r, fitattr) for r in sel]
        out[p] = dict(
            tan=np.abs(t_det[p]),
            ext=(np.array([f.n_strips for f in fits]) - 1) * PITCH_MM,
            span=np.array([f.latest_time_ns - f.earliest_time_ns for f in fits]),
            t_early=np.array([f.earliest_time_ns for f in fits]),
            t_late=np.array([f.latest_time_ns for f in fits]),
        )
    return out


def profile(x, y, bins):
    ctr, med, err, nn = [], [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (x >= b0) & (x < b1)
        if m.sum() >= 25:
            ctr.append(0.5 * (b0 + b1)); med.append(np.median(y[m]))
            err.append(1.253 * np.std(y[m], ddof=1) / np.sqrt(m.sum()))
            nn.append(int(m.sum()))
    return map(np.array, (ctr, med, err, nn))


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

    data = {}
    for hv, sub in points:
        print(f'=== drift {hv} V — {sub} ===')
        d = load_point(sub, seed)
        if d is not None:
            data[hv] = d
    hvs = sorted(data)
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(hvs)))

    # -------- overview: absolute timing distributions + ceiling --------
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    tbins = np.arange(0, WINDOW_NS + DT_NS, DT_NS / 2)
    rows = []
    for c, hv in zip(cmap, hvs):
        both_late = np.concatenate([data[hv]['x']['t_late'], data[hv]['y']['t_late']])
        both_early = np.concatenate([data[hv]['x']['t_early'], data[hv]['y']['t_early']])
        # window ceiling proxy: 99th percentile of latest_time
        ceil99 = np.percentile(both_late, 99)
        ceil_med_incl = None
        # inclined-only latest time (these are the ones that reach deepest)
        inc = []
        for p in ('x', 'y'):
            m = data[hv][p]['tan'] > np.tan(np.deg2rad(SAT_DEG))
            inc.append(data[hv][p]['t_late'][m])
        inc = np.concatenate(inc)
        ceil_med_incl = np.median(inc) if len(inc) else np.nan
        lab = f'{hv} V ({hv/GAP_CM:.0f} V/cm)'
        ax[0, 0].hist(both_late, bins=tbins, histtype='step', color=c, lw=1.6, label=lab)
        ax[0, 1].hist(both_early, bins=tbins, histtype='step', color=c, lw=1.6, label=lab)
        rows.append(dict(drift_hv=hv, E_Vcm=hv / GAP_CM, n=data[hv]['n'],
                         t_late_p99_ns=float(ceil99),
                         t_late_incl_med_ns=float(ceil_med_incl),
                         t_early_med_ns=float(np.median(both_early))))
    for a in (ax[0, 0], ax[0, 1]):
        a.axvline(WINDOW_NS, color='r', ls='--', lw=1, label='1920 ns window')
        a.set_xlabel('time within window [ns]'); a.grid(alpha=0.3); a.legend(fontsize=7)
    ax[0, 0].set_title('latest_time (deep edge of column) — piles up at ceiling if truncated')
    ax[0, 1].set_title('earliest_time (mesh / prompt arrival = t0 offset)')

    # extent-saturation & span-saturation onset vs tan, all points overlaid
    tanb = np.arange(0.0, 0.60, 0.04)
    for c, hv in zip(cmap, hvs):
        for p, a, col in (('x', ax[1, 0], None),):
            xall = np.concatenate([data[hv]['x']['tan'], data[hv]['y']['tan']])
            eall = np.concatenate([data[hv]['x']['ext'], data[hv]['y']['ext']])
            sall = np.concatenate([data[hv]['x']['span'], data[hv]['y']['span']])
            ctr, med, err, nn = profile(xall, eall, tanb)
            ax[1, 0].plot(ctr, med, 'o-', color=c, ms=4, lw=1.4,
                          label=f'{hv} V')
            ctr2, med2, err2, nn2 = profile(xall, sall, tanb)
            ax[1, 1].plot(ctr2, med2, 's-', color=c, ms=4, lw=1.4, label=f'{hv} V')
    ax[1, 0].set_xlabel('|tan θ_det|'); ax[1, 0].set_ylabel('cluster extent [mm]')
    ax[1, 0].set_title('extent vs angle — flattening = column truncated by window')
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=7)
    ax[1, 1].set_xlabel('|tan θ_det|'); ax[1, 1].set_ylabel('cluster time span [ns]')
    ax[1, 1].set_title('time span vs angle — saturates at window, not gap, when truncated')
    ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=7)
    fig.suptitle(f'{CFG.DET_NAME} drift scan — DREAM window truncation diagnostic '
                 f'(veto{VETO}, window={WINDOW_NS:.0f} ns)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'window_truncation_overview.png'), dpi=160,
                bbox_inches='tight')
    plt.close(fig)

    # -------- per-point profile panels (extent + latest_time vs tan) --------
    n = len(hvs)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 8), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, hv in enumerate(hvs):
        xall = np.concatenate([data[hv]['x']['tan'], data[hv]['y']['tan']])
        eall = np.concatenate([data[hv]['x']['ext'], data[hv]['y']['ext']])
        lall = np.concatenate([data[hv]['x']['t_late'], data[hv]['y']['t_late']])
        ctr, med, err, nn = profile(xall, eall, tanb)
        axes[0, j].errorbar(ctr, med, yerr=err, fmt='o-', color='navy', ms=4)
        axes[0, j].set_title(f'{hv} V ({hv/GAP_CM:.0f} V/cm)\nn={data[hv]["n"]}', fontsize=9)
        axes[0, j].grid(alpha=0.3)
        ctr2, med2, err2, nn2 = profile(xall, lall, tanb)
        axes[1, j].errorbar(ctr2, med2, yerr=err2, fmt='s-', color='darkred', ms=4)
        axes[1, j].axhline(WINDOW_NS, color='r', ls='--', lw=1)
        axes[1, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel('extent [mm]'); axes[1, 0].set_ylabel('latest_time [ns]')
    for j in range(n):
        axes[1, j].set_xlabel('|tan θ_det|')
    fig.suptitle(f'{CFG.DET_NAME} per-point: extent (top) & deep-edge arrival (bottom) vs angle',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'window_truncation_profiles.png'), dpi=160,
                bbox_inches='tight')
    plt.close(fig)

    df = pd.DataFrame(rows).sort_values('drift_hv')
    df.to_csv(os.path.join(OUT, 'window_truncation.csv'), index=False)
    print('\n' + df.to_string(index=False))
    print(f'\nWritten: {OUT}/  (overview.png, profiles.png, .csv)')


if __name__ == '__main__':
    main()
