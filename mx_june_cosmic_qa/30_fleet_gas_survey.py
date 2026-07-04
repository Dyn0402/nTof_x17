#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30_fleet_gas_survey.py

Fleet-and-time survey of the two gas observables across every June run with
a cached analysis:

    v_geom  = extent-slope / T_sat   -> tracks the WATER content (drift speed)
    lambda  = amplitude-vs-depth decay length -> tracks the O2 (attachment)

Motivation: det2 (6-22) reproduced det3's v but had ~2.5x weaker attachment
=> O2 is detector- or time-specific while H2O looks shared.  This script
maps both quantities per detector per date, in particular the SAME physical
det3 across 6-22 / 6-23 / 6-25 / 6-27 / 6-27-night, plus det6/7 (flagged:
lower trust) and det4/det1/det2.

Drift HV per run is read from hv_monitor.csv where present (drift channel =
largest mean vmon above 540 V); otherwise assumed 1000 V and flagged.

Usage: ../.venv/bin/python 30_fleet_gas_survey.py
Output: Analysis/fleet_gas_survey.csv/.png + stdout table
"""
import os
import glob
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import _Config, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

VETO = 50
SAMPLE_NS = 60.0
MIN_STRIPS = 4
RES_CUT_MM = 10.0
CHI2_CUT = 20.0
PITCH_MM = 0.78
SAT_DEG = 10.0
TAN_LO, TAN_HI, TAN_STEP = 0.06, 0.44, 0.04
INCL_DEG = 12.0
LAM_FIT_MM = (8.0, 22.0)
CB = '/home/dylan/x17/cosmic_bench'

ENTRIES = [
    # (base_path, run, subrun, det_name, date_label, day_ord, slot, trust)
    (f'{CB}/det1_det2/', 'mx17_det1_det2_overnight_6-17-26', 'longer_run',
     'mx17_1', '6-17', 17.0, '?', 'low(noisy)'),
    (f'{CB}/det1_det2/', 'mx17_det1_det2_overnight_6-17-26', 'longer_run',
     'mx17_2', '6-17', 17.2, '?', 'low(halfdeadX)'),
    (f'{CB}/det_3/', 'mx17_det3_test_6-22-26', 'short_run',
     'mx17_3', '6-22t', 21.8, '?', 'ok'),
    (f'{CB}/det2_det3/', 'mx17_det2_det3_overnight_6-22-26', 'longer_run',
     'mx17_3', '6-22', 22.0, 'bottom', 'ok'),
    (f'{CB}/det2_det3/', 'mx17_det2_det3_overnight_6-22-26', 'longer_run',
     'mx17_2', '6-22', 22.2, 'top', 'ok'),
    (f'{CB}/det3_det4/', 'mx17_det3_det4_overnight_6-23-26', 'long_run',
     'mx17_3', '6-23', 23.0, 'bottom', 'ok'),
    (f'{CB}/det3_det4/', 'mx17_det3_det4_overnight_6-23-26', 'long_run',
     'mx17_4', '6-23', 23.2, 'top', 'gain-limited'),
    (f'{CB}/det_4day/', 'mx17_det4_day_6-24-26', 'long_run',
     'mx17_4', '6-24', 24.0, '?', 'gain-limited'),
    (f'{CB}/det3/', 'mx17_det3_day_6-25-26', 'long_run',
     'mx17_3', '6-25', 25.0, '?', 'ok'),
    (f'{CB}/det6_det7/', 'mx17_det6_det7_overnight_6-26-26', 'long_run',
     'mx17_6', '6-26', 26.0, 'bottom', 'LOW TRUST'),
    (f'{CB}/det6_det7/', 'mx17_det6_det7_overnight_6-26-26', 'long_run',
     'mx17_7', '6-26', 26.2, 'top', 'LOW TRUST'),
    (f'{CB}/det3/', 'mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V',
     'mx17_3', '6-27', 27.0, 'top', 'REF'),
    (f'{CB}/det3/', 'mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check',
     'mx17_3', '6-27n', 27.5, 'top', 'ok'),
]


def autodetect_feus(cfg, det):
    """Return [feu_x, feu_y] by probing the strip map (a strip maps to x OR y;
    the other coordinate is None/NaN)."""
    fx = fy = None
    for feu in range(16):
        nx = ny = 0
        for ch in (40, 120, 200, 300, 420):
            p = det.map_hit(feu, ch)
            if p is None:
                continue
            x, y = p
            if x is not None and np.isfinite(x):
                nx += 1
            if y is not None and np.isfinite(y):
                ny += 1
        if nx >= 3 and nx > ny:
            fx = feu
        elif ny >= 3 and ny > nx:
            fy = feu
    return fx, fy


# Drift HV decoded by hand from each run's hv_monitor.csv (channel mapping
# calibrated on the 6-27 sat run: 0:7 = top drift = 997 V, 3:4 = top resist
# = 490 V; 0:6 = bottom drift; 0:8-0:11 = M3 at 500 V; 3:8-3:11 at 455 V).
DRIFT_HV = {
    ('mx17_det3_test_6-22-26', 'mx17_3'): (986, 'monitor'),
    ('mx17_det2_det3_overnight_6-22-26', 'mx17_3'): (1000, 'monitor'),
    ('mx17_det2_det3_overnight_6-22-26', 'mx17_2'): (1000, 'monitor'),
    ('mx17_det3_det4_overnight_6-23-26', 'mx17_3'): (600, 'monitor'),
    ('mx17_det3_det4_overnight_6-23-26', 'mx17_4'): (600, 'monitor'),
    ('mx17_det4_day_6-24-26', 'mx17_4'): (600, 'monitor'),
    ('mx17_det3_day_6-25-26', 'mx17_3'): (500, 'monitor?500-ambig'),
    ('mx17_det6_det7_overnight_6-26-26', 'mx17_6'): (700, 'monitor'),
    ('mx17_det6_det7_overnight_6-26-26', 'mx17_7'): (700, 'monitor'),
    ('mx17_det3_saturday_scan_6-27-26', 'mx17_3'): (1000, 'name'),
    ('mx17_det3_p2_det1_overnight_6-27-26', 'mx17_3'): (999, 'monitor'),
}


def drift_hv(base, run, subrun, det_name='?'):
    if (run, det_name) in DRIFT_HV:
        return DRIFT_HV[(run, det_name)]
    return 1000.0, 'ASSUMED'


def water_interp():
    """v(E, w%) tables for pure-H2O contamination of Ar/iso 95/5."""
    import json
    GS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'garfield_sim', 'results')
    tabs = {}
    for fn, names in [
            ('drift_velocity_candidates.json',
             {'Ar95_iso5_H2O0.3': 0.3, 'Ar94_iso5_H2O1': 1.0, 'Ar93_iso5_H2O2': 2.0}),
            ('drift_velocity_candidates2.json',
             {'Ar92.5_iso5_H2O2.5': 2.5, 'Ar92_iso5_H2O3': 3.0}),
            ('water_grid.json',
             {'Ar_iso5_H2O0.6': 0.6, 'Ar_iso5_H2O0.8': 0.8,
              'Ar_iso5_H2O1.2': 1.2, 'Ar_iso5_H2O1.5': 1.5}),
            ('attachment_Ar_iso_H2O.json', {'Ar95_iso5': 0.0})]:
        d = json.load(open(os.path.join(GS, fn)))['mixtures']
        for n, w in names.items():
            if n in d:
                pts = d[n]
                tabs[w] = (np.array([p['E_Vcm'] for p in pts]),
                           np.array([p['v_um_per_ns'] for p in pts]))
    ws = sorted(tabs)

    def implied(v, E):
        vw = np.array([np.interp(E, *tabs[w]) for w in ws])
        if v > vw[0]:
            return 0.0
        if v < vw[-1]:
            return np.nan     # wetter than the 3 % grid edge
        return float(np.interp(-v, -vw, ws))
    return implied


def survey_one(base, run, subrun, det_name, label):
    cfg = _Config(f'survey_{det_name}_{label}', run, subrun, feus=[0, 1],
                  det_z=0.0, det_name=det_name, base_path=base,
                  zero_suppressed=False)
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not os.path.exists(cache):
        print(f'  [SKIP] no cache: {cache}')
        return None
    align = os.path.join(cfg.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
    if not os.path.exists(align):
        align = os.path.join(cfg.OUT_BASE, 'alignment_tpc', 'alignment.json')
        if not os.path.exists(align):
            print('  [SKIP] no alignment')
            return None

    from common.Mx17StripMap import RunConfig
    rc = RunConfig(cfg.run_config_path, cfg.MAP_CSV_PATH)
    det = rc.get_detector(det_name)
    fx, fy = autodetect_feus(cfg, det)
    if fx is None or fy is None:
        print('  [SKIP] FEU autodetect failed')
        return None
    cfg = _Config(f'survey_{det_name}_{label}', run, subrun, feus=[fx, fy],
                  det_z=0.0, det_name=det_name, base_path=base,
                  zero_suppressed=False)

    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)
    ref, dur, ns_ = {}, {}, {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        if r.x_fit.n_strips < MIN_STRIPS or r.y_fit.n_strips < MIN_STRIPS:
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
        dur[r.event_id] = (r.x_fit.latest_time_ns - r.x_fit.earliest_time_ns,
                           r.y_fit.latest_time_ns - r.y_fit.earliest_time_ns)
        ns_[r.event_id] = (r.x_fit.n_strips, r.y_fit.n_strips)
    n_match = len(ref)
    if n_match < 500:
        print(f'  [SKIP] only {n_match} matched events')
        return None

    out = dict(n_matched=n_match)
    vgs = []
    for pi, p in enumerate(('x', 'y')):
        at = np.abs(np.array([v[pi] for v in ref.values()]))
        ext = np.array([(ns_[e][pi] - 1) * PITCH_MM for e in ref])
        dd = np.array([dur[e][pi] for e in ref])
        bins = np.arange(TAN_LO, TAN_HI + TAN_STEP, TAN_STEP)
        ctr, med = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (at >= b0) & (at < b1)
            if m.sum() >= 30:
                ctr.append(0.5 * (b0 + b1))
                med.append(np.median(ext[m]))
        if len(ctr) < 4:
            continue
        slope = np.polyfit(ctr, med, 1)[0]
        msat = np.degrees(np.arctan(at)) > SAT_DEG
        tsat = float(np.median(dd[msat]))
        v = slope * 1000.0 / tsat
        out[f'z_slope_{p}'] = slope
        out[f't_sat_{p}'] = tsat
        out[f'v_geom_{p}'] = v
        vgs.append(v)
    if not vgs:
        return None
    out['v_geom'] = float(np.mean(vgs))

    # lambda from hits
    try:
        fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                    if f.endswith('.root') and '_datrun_' in f)
        hf = uproot.concatenate(
            [f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
            expressions=['eventId', 'feu', 'amplitude', 'sample'], library='pd')
        hf = hf[hf['feu'].isin([fx, fy])]
        hpe = hf.groupby('eventId')['eventId'].transform('size')
        incl = {e for e, v in ref.items()
                if max(abs(np.degrees(np.arctan(v[0]))),
                       abs(np.degrees(np.arctan(v[1])))) > INCL_DEG}
        hf = hf[(hpe <= VETO) & hf['eventId'].isin(incl)].copy()
        hf['t'] = hf['sample'] * SAMPLE_NS
        hf['t_rel'] = hf['t'] - hf.groupby(['eventId', 'feu'])['t'].transform('min')
        t_rel = hf['t_rel'].to_numpy()
        amp = hf['amplitude'].to_numpy()
        bins = np.arange(0.0, 1920.0 + SAMPLE_NS, SAMPLE_NS)
        tc = 0.5 * (bins[:-1] + bins[1:])
        med = np.full(len(tc), np.nan)
        for i in range(len(tc)):
            m = (t_rel >= bins[i]) & (t_rel < bins[i + 1])
            if m.sum() >= 30:
                med[i] = np.median(amp[m])
        z = tc * out['v_geom'] / 1000.0
        mfit = (z > LAM_FIT_MM[0]) & (z < LAM_FIT_MM[1]) & np.isfinite(med) & (med > 0)
        if mfit.sum() >= 4:
            pfit = np.polyfit(z[mfit], np.log(med[mfit]), 1)
            out['lambda_mm'] = float(-1.0 / pfit[0]) if pfit[0] < 0 else np.inf
    except Exception as e:
        print(f'  [WARN] lambda failed: {e}')
    return out


def main():
    implied = water_interp()
    rows = []
    for base, run, subrun, det_name, date, day, slot, trust in ENTRIES:
        print(f'=== {date} {det_name} ({run}/{subrun}) ===')
        r = survey_one(base, run, subrun, det_name, date)
        if r is None:
            continue
        hv, hvsrc = drift_hv(base, run, subrun, det_name)
        r.update(run=run, subrun=subrun, det=det_name, date=date, day=day,
                 slot=slot, trust=trust, drift_hv=hv, hv_source=hvsrc)
        r['h2o_pct'] = implied(r['v_geom'], hv / 3.0)
        rows.append(r)
        print(f"  v_geom = {r['v_geom']:.2f} µm/ns (x {r.get('v_geom_x', np.nan):.1f} / "
              f"y {r.get('v_geom_y', np.nan):.1f})   λ = {r.get('lambda_mm', np.nan):.1f} mm   "
              f"drift = {hv:.0f} V ({hvsrc})   H2O ≈ {r['h2o_pct']:.2f} %   "
              f"n = {r['n_matched']:,}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(CB, 'Analysis', 'fleet_gas_survey.csv')
    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    dets = sorted(df['det'].unique())
    colors = dict(zip(dets, plt.cm.tab10(np.linspace(0, 0.9, len(dets)))))
    H2O_WET = 3.4   # plotting stand-in for the off-grid (>3 %) 6-22 points
    for _, r in df.iterrows():
        c = colors[r['det']]
        mk = 'o' if r['trust'] not in ('LOW TRUST',) else 'x'
        h2o = r['h2o_pct'] if np.isfinite(r['h2o_pct']) else H2O_WET
        axes[0].plot(r['day'], h2o, mk, ms=11, color=c)
        if not np.isfinite(r['h2o_pct']):
            axes[0].annotate('>3%', (r['day'], h2o), textcoords='offset points',
                             xytext=(-4, 10), fontsize=9, color=c)
        if np.isfinite(r.get('lambda_mm', np.nan)):
            axes[1].plot(r['day'], min(r['lambda_mm'], 60), mk, ms=11, color=c)
    d3 = df[df['det'] == 'mx17_3'].sort_values('day')
    axes[0].plot(d3['day'], d3['h2o_pct'].fillna(H2O_WET), '-',
                 color=colors['mx17_3'], alpha=0.5)
    axes[1].plot(d3['day'], d3['lambda_mm'].clip(upper=60), '-',
                 color=colors['mx17_3'], alpha=0.5)
    for ax in axes:
        ax.grid(alpha=0.3)
        for _, r in df.iterrows():
            y = (r['h2o_pct'] if np.isfinite(r['h2o_pct']) else H2O_WET) \
                if ax is axes[0] else min(r.get('lambda_mm', np.nan), 60)
            ax.annotate(f"{r['det'].replace('mx17_', 'd')}@{r['drift_hv']:.0f}V",
                        (r['day'], y), textcoords='offset points',
                        xytext=(6, 4), fontsize=8)
    axes[0].axhspan(0.7, 1.2, color='green', alpha=0.08,
                    label='fleet equilibrium band 0.7–1.2 %')
    axes[0].set_ylabel('implied H$_2$O [%]  (from v_geom at the run\'s field)')
    axes[0].set_title('water content vs date — det3 dries from >3 % to ~1 %')
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel('λ_att [mm] (capped at 60)')
    axes[1].set_title('amplitude-decay length (O$_2$ tracker; '
                      'shorter λ at fixed E = more O$_2$)')
    axes[1].set_xlabel('June date')
    fig.suptitle('Fleet & time survey of gas observables (× = low-trust detector)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(CB, 'Analysis', 'fleet_gas_survey.png'), dpi=160)
    print(f'\nWritten {out_csv} + .png')

    with pd.option_context('display.width', 250):
        cols = ['date', 'det', 'slot', 'trust', 'drift_hv', 'hv_source',
                'v_geom_x', 'v_geom_y', 'v_geom', 'h2o_pct', 'lambda_mm',
                't_sat_x', 't_sat_y', 'n_matched']
        print(df[[c for c in cols if c in df.columns]].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
