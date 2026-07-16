#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_gap_attachment_test.py

Discriminate two interpretations of the 19.4 mm "gap" measured as v*T_sat:

  H1 (geometric): the drift column really ends 19.4 mm above the mesh
     -> the arrival-time distribution should terminate at 19.4 mm / v
        with nothing but noise floor beyond.
  H2 (attachment in a 30 mm gap): the mechanical gap is ~30 mm but the gas
     (humid -> electronegative) captures electrons during drift, so signal
     from beyond ~19 mm falls below threshold. The arrival-time distribution
     should show an exponential tail BEYOND T_sat terminating at 30 mm / v,
     and the per-strip amplitude should decay with drift time.

For every drift-scan point (plus the long run) this script builds, for
inclined tracks (|theta_ref| > 12 deg), the distribution of
(strip time - earliest strip time in event/plane) and the median strip
amplitude vs that time, expressed in DEPTH units z = v_ridge * t. If H2
holds, the tails of all HV points collapse to a common endpoint at the
mechanical gap depth, and the amplitude decay length gives the attachment
length lambda_att(E).

Usage: ../.venv/bin/python 17_gap_attachment_test.py [sat_det3] [--veto=50]
Output: <Analysis>/<run>/drift_velocity/<det>/gap_attachment_test.png/.csv
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
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
SAMPLE_NS = 60.0
INCL_DEG = 12.0
RES_CUT_MM = 10.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
GAP_CANDIDATES_MM = (19.4, 30.0)
MIN_HITS_PER_BIN = 30

ALIGN_SEED = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
VD_CSV = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                      CFG.RUN, 'drift_velocity', CFG.DET_NAME, 'drift_velocity_scan.csv')
OUT = os.path.dirname(VD_CSV)


def cfg_for(subrun):
    return _Config(f'{CFG.KEY}_{subrun}', CFG.RUN, subrun, feus=CFG.MX17_FEUS,
                   det_z=CFG.DET_PLANE_Z, det_name=CFG.DET_NAME,
                   base_path=CFG.BASE_PATH, zero_suppressed=CFG.ZERO_SUPPRESSED)


def inclined_events(cfg, seed):
    """Event ids of matched inclined tracks, from the cached micro-TPC results."""
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not os.path.exists(cache):
        return None
    results = pickle.load(open(cache, 'rb'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
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


def arrival_data(cfg, ids):
    """(t_rel [ns], amplitude) for every strip hit of the selected events."""
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
    return df['t_rel'].to_numpy(), df['amplitude'].to_numpy()


def analyse_point(row, seed):
    subrun, hv, v, t_sat = row['subrun'], row['drift_hv'], row['v_ridge'], row['t_sat_ns']
    cfg = cfg_for(subrun)
    ids = inclined_events(cfg, seed)
    if ids is None or len(ids) < 50:
        print(f'  [SKIP] {subrun}: no cache / too few inclined events')
        return None
    t_rel, amp = arrival_data(cfg, ids)
    print(f'  {subrun}: {len(ids):,} inclined events, {len(t_rel):,} hits')

    bins = np.arange(0.0, 1920.0 + SAMPLE_NS, SAMPLE_NS)
    cnt, _ = np.histogram(t_rel, bins=bins)
    tc = 0.5 * (bins[:-1] + bins[1:])
    cnt = cnt.astype(float)
    cnt[0] = np.nan          # the t_rel = 0 reference bin (the earliest strip itself)

    # noise floor: beyond the LATEST candidate endpoint, inside the window
    t_floor_lo = min(30.0 / v * 1000.0 + 150.0, 1620.0)
    floor_m = tc > t_floor_lo
    floor = np.nanmedian(cnt[floor_m]) if floor_m.sum() >= 3 else 0.0

    # plateau level: 120 ns .. 0.8*T_sat
    plat_m = (tc > 120) & (tc < 0.8 * t_sat)
    plateau = np.nanmedian(cnt[plat_m])

    # endpoint: last bin above floor + 5 sigma AND 2 % of plateau, after T_sat/2
    sig = cnt - floor
    thresh = max(5.0 * np.sqrt(max(floor, 1.0)), 0.02 * plateau)
    above = np.where((tc > 0.5 * t_sat) & (sig > thresh))[0]
    t_end = float(tc[above[-1]] + 0.5 * SAMPLE_NS) if len(above) else np.nan

    # tail decay constant between T_sat and t_end (exponential region)
    tail_m = (tc > t_sat) & (tc < t_end) & (sig > thresh)
    tau_tail = np.nan
    if tail_m.sum() >= 3:
        p = np.polyfit(tc[tail_m], np.log(sig[tail_m]), 1)
        tau_tail = -1.0 / p[0] if p[0] < 0 else np.nan

    # median amplitude per bin + decay fits in two windows
    med = np.full(len(tc), np.nan)
    for i in range(len(tc)):
        m = (t_rel >= bins[i]) & (t_rel < bins[i + 1])
        if m.sum() >= MIN_HITS_PER_BIN:
            med[i] = np.median(amp[m])

    def amp_lambda(f_lo, f_hi):
        m = (tc > f_lo * t_sat) & (tc < f_hi * t_sat) & np.isfinite(med) & (med > 0)
        if m.sum() < 3:
            return np.nan
        p = np.polyfit(tc[m], np.log(med[m]), 1)
        return -v / (1000.0 * p[0]) if p[0] < 0 else np.nan   # decay length [mm]

    lam_mid = amp_lambda(0.45, 1.0)     # inside the visible column
    lam_tail = amp_lambda(1.0, 1.45)    # beyond T_sat

    return dict(subrun=subrun, drift_hv=hv, v_ridge=v, t_sat_ns=t_sat,
                t_end_ns=t_end, z_end_mm=t_end * v / 1000.0,
                z_sat_mm=t_sat * v / 1000.0,
                tau_tail_ns=tau_tail, lam_tail_from_counts_mm=tau_tail * v / 1000.0
                if np.isfinite(tau_tail) else np.nan,
                lam_amp_mid_mm=lam_mid, lam_amp_tail_mm=lam_tail,
                plateau=plateau, floor=floor,
                tc=tc, cnt=cnt, med=med)


def main():
    seed = cm.load_alignment(ALIGN_SEED)
    vd = pd.read_csv(VD_CSV)
    # prefer the bias-free geometry velocity (21_geometry_vdrift_scan.py) when
    # available: the ridge v is ~20% low (charge-sharing time distortion)
    geom_csv = os.path.join(OUT, 'geometry_vdrift_scan.csv')
    if os.path.exists(geom_csv):
        gv = pd.read_csv(geom_csv)[['drift_hv', 'v_geom']]
        vd = vd.merge(gv, on='drift_hv', how='left')
        vd['v_ridge'] = vd['v_geom'].fillna(vd['v_ridge'])
        print('using v_geom for the depth scale')
    vd = vd[vd['drift_hv'] >= 500].sort_values('drift_hv')   # endpoint needs headroom
    print(f'Points: {list(vd["subrun"])}')

    rows = []
    for _, row in vd.iterrows():
        r = analyse_point(row, seed)
        if r is not None:
            rows.append(r)

    cols = plt.cm.viridis(np.linspace(0.0, 0.9, len(rows)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (A) arrival-time distribution in depth units
    ax = axes[0, 0]
    for r, c in zip(rows, cols):
        z = r['tc'] * r['v_ridge'] / 1000.0
        norm = r['cnt'] / r['plateau']
        lab = f"{r['drift_hv']} V (v={r['v_ridge']:.1f})"
        ax.step(z, norm, where='mid', color=c, lw=1.6, label=lab)
        ax.axhline(r['floor'] / r['plateau'], color=c, lw=0.6, ls=':', alpha=0.5)
    for g, ls in zip(GAP_CANDIDATES_MM, ('--', '-')):
        ax.axvline(g, color='k', ls=ls, lw=1.4, label=f'{g:g} mm')
    ax.set_yscale('log')
    ax.set_xlim(0, 40)
    ax.set_xlabel('drift depth  z = v_ridge · (strip time − earliest) [mm]')
    ax.set_ylabel('hits / bin (plateau-normalised)')
    ax.set_title('arrival-depth distribution — do the tails end at 19.4 or 30 mm?')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (B) median amplitude vs depth
    ax = axes[0, 1]
    for r, c in zip(rows, cols):
        z = r['tc'] * r['v_ridge'] / 1000.0
        ax.plot(z, r['med'], 'o-', ms=3, color=c, lw=1.2,
                label=f"{r['drift_hv']} V  λ_mid={r['lam_amp_mid_mm']:.1f} mm  "
                      f"λ_tail={r['lam_amp_tail_mm']:.1f} mm")
    for g, ls in zip(GAP_CANDIDATES_MM, ('--', '-')):
        ax.axvline(g, color='k', ls=ls, lw=1.4)
    ax.set_yscale('log')
    ax.set_xlim(0, 40)
    ax.set_xlabel('drift depth z [mm]')
    ax.set_ylabel('median strip amplitude [ADC]')
    ax.set_title('amplitude vs drift depth — attachment decay')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (C) endpoint time vs HV against the two gap hypotheses
    ax = axes[1, 0]
    hv = np.array([r['drift_hv'] for r in rows], float)
    ax.plot(hv, [r['t_sat_ns'] for r in rows], 's-', color='darkorange',
            label='T_sat (median span)')
    ax.plot(hv, [r['t_end_ns'] for r in rows], 'o-', color='crimson',
            label='tail endpoint t_end')
    vv = np.array([r['v_ridge'] for r in rows], float)
    order = np.argsort(hv)
    for g, ls in zip(GAP_CANDIDATES_MM, ('--', '-')):
        ax.plot(hv[order], g * 1000.0 / vv[order], ls, color='k', lw=1.2,
                label=f'{g:g} mm / v_ridge')
    ax.set_xlabel('drift HV [V]')
    ax.set_ylabel('time [ns]')
    ax.set_title('endpoint scaling test')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (D) implied depth of T_sat and t_end
    ax = axes[1, 1]
    ax.plot(hv, [r['z_sat_mm'] for r in rows], 's-', color='darkorange',
            label='v·T_sat (visible column)')
    ax.plot(hv, [r['z_end_mm'] for r in rows], 'o-', color='crimson',
            label='v·t_end (tail endpoint)')
    for g, ls in zip(GAP_CANDIDATES_MM, ('--', '-')):
        ax.axhline(g, color='k', ls=ls, lw=1.2)
    ax.set_xlabel('drift HV [V]')
    ax.set_ylabel('depth [mm]')
    ax.set_ylim(0, 40)
    ax.set_title('implied depths (flat ⇒ geometric scale)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f'{CFG.DET_NAME} — drift-gap discrimination: geometric 19.4 mm edge '
                 f'vs attachment-truncated 30 mm gap ({CFG.RUN})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'gap_attachment_test.png'), dpi=170,
                bbox_inches='tight')

    tab = pd.DataFrame([{k: v for k, v in r.items() if k not in ('tc', 'cnt', 'med')}
                        for r in rows])
    tab.to_csv(os.path.join(OUT, 'gap_attachment_test.csv'), index=False)
    print(f'\nWritten {OUT}/gap_attachment_test.png + .csv\n')
    with pd.option_context('display.width', 200):
        print(tab[['drift_hv', 'v_ridge', 't_sat_ns', 't_end_ns', 'z_sat_mm',
                   'z_end_mm', 'lam_amp_mid_mm', 'lam_amp_tail_mm',
                   'lam_tail_from_counts_mm']].to_string(index=False))


if __name__ == '__main__':
    main()
