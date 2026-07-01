#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_hv_scan_efficiency.py

Detector efficiency vs. mesh HV for the det1 resistor-HV scan that lives inside
the overnight run (subruns ``resist_<NNN>V_drift_1000V``).  FEU 3/4 mesh HV was
stepped 450 -> 520 V in 10 V steps (drift fixed 1000 V), 30 min each.

This is the *June-convention* HV scan: it reuses the same building blocks as the
rest of the suite (``cosmic_micro_tpc_analysis``) with ``ref_x_sign = +1`` and the
established det1 rotation (theta ~= 89.5 deg, z ~= 243 mm), NOT the legacy
``cosmic_bench_analysis/hv_scan_efficiency.py`` (which hardcodes the det_4 / n_TOF
theta~=0, ref_x_sign=-1 recipe and would mirror the per-event match on the
90-deg-rotated det1 -> bogus-low efficiencies).

Method, per subrun (mirrors 08_efficiency_maps.py):
  1. Load detector hits (FEU 3/4), map strips, run analyse_event per event.
  2. Load M3 reference tracks (chi2 < 20).
  3. Seed z/theta/centre/ref_x_sign from the long_run det1 alignment; re-run
     translation-only alignment so each subrun is centred on the M3 frame.
  4. For every clean M3 single track, project to the aligned plane and ask:
     is there a reconstructed det X+Y hit within R mm? (a track with no DREAM
     event is a genuine MISS, kept in the denominator).
  5. Integrated efficiency inside a FIXED active box (same box for every HV so
     the denominator region is identical across the scan).
  6. Plot efficiency vs HV.

Usage:
  ../venv/bin/python 10_hv_scan_efficiency.py [ovn_det1] [--r=5] [--minvalid=20]

The run/detector (FEUs, det_name, base_path, run_config) come from the qa_config
key (default ovn_det1 = det1, FEU 3/4); the subruns are discovered from the run
directory by the ``resist_<NNN>V`` pattern.
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions

CFG = config_from_argv()

R = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--r=')), 5.0)
MIN_VALID = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--minvalid=')), 20)
SPARK_THRESH = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--spark=')), 50)
M3_CHI2_CUT = 20.0

# Alignment seed (z/theta/centre/handedness; translation re-run per subrun). Defaults to
# this run's own long_run alignment, but --seed=<alignment.json> overrides it -- needed
# for dedicated HV-scan runs that have NO long_run subrun (seed from another run of the
# same detector, e.g. the 6-26 overnight long_run for the 6-26 hv_scan run).
_seed_override = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--seed=')), None)
ALIGN_SEED = _seed_override or os.path.join(
    os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis', CFG.RUN, 'long_run',
    CFG.DET_NAME, 'alignment_tpc_veto50', 'alignment.json')
# Fallback if the long_run alignment isn't on disk (from memory: z=243, th=89.5, ref_x_sign=+1).
SEED_DEFAULT = cm.AlignmentParams(z_x=243.0, z_y=243.0, theta_deg=89.5,
                                  centre_x=200.0, centre_y=200.0,
                                  x_offset=-209.0, y_offset=-195.0, ref_x_sign=1.0)


def out_dir(*parts):
    base = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')),
                        'Analysis', CFG.RUN, 'hv_scan', CFG.DET_NAME)
    d = os.path.join(base, *parts)
    os.makedirs(d, exist_ok=True)
    return d


# Detector number from the qa_config det_name (mx17_6 -> '6'). Used for the
# per-detector HV naming resist_det6_505V_det7_480V_... (the 6-26 scan steps BOTH
# detectors at once, at different voltages). The single-value 6-22/6-23 scans use
# resist_<NNN>V where the one value applies to every stepped detector (in 6-22,
# channels 3:3=det3 and 3:4=det2 step together).
_dnm = re.search(r'(\d+)$', CFG.DET_NAME)
DET_NUM = _dnm.group(1) if _dnm else ''


def extract_hv(name):
    m = re.search(rf'det{DET_NUM}_(\d+)V', name)   # per-detector naming first
    if m:
        return int(m.group(1))
    m = re.search(r'resist_(\d+)V', name)          # single-value naming
    return int(m.group(1)) if m else None


def find_subruns():
    run_dir = os.path.join(CFG.BASE_PATH, CFG.RUN)
    pairs = []
    for name in sorted(os.listdir(run_dir)):
        hv = extract_hv(name)
        if hv is not None and name.startswith('resist_') \
                and os.path.isdir(os.path.join(run_dir, name)):
            pairs.append((name, hv))
    return sorted(pairs, key=lambda x: x[1])      # ascending HV


def analyse_subrun(subrun, det, seed):
    """Run the per-event pipeline + translation alignment for one subrun.

    Returns dict with reco positions, ray projections, has_any set, params; or
    None if too few valid events to align.
    """
    hits_dir = f'{CFG.BASE_PATH}{CFG.RUN}/{subrun}/combined_hits_root/'
    rays_dir = f'{CFG.BASE_PATH}{CFG.RUN}/{subrun}/m3_tracking_root/'
    if not (os.path.isdir(hits_dir) and os.path.isdir(rays_dir)):
        print(f'  [SKIP] {subrun}: missing hits/tracking dir')
        return None

    hit_files = sorted(f for f in os.listdir(hits_dir)
                       if f.endswith('.root') and '_datrun_' in f)
    if not hit_files:
        print(f'  [SKIP] {subrun}: no hit files')
        return None

    df = uproot.concatenate([f'{hits_dir}{f}:hits' for f in hit_files], library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)].copy()
    df = cm._map_strip_positions(df, det)

    rays = M3RefTracking(rays_dir, chi2_cut=M3_CHI2_CUT)
    xa, _ya, an = get_xy_angles(rays.ray_data)
    xa = seed.ref_x_sign * np.array(xa)

    results = []
    for eid in cm._progress(df['eventId'].unique(), desc=f'  {subrun}'):
        results.append(cm.analyse_event(df[df['eventId'] == eid], event_id=eid, plot=False))

    n_valid = sum(r.has_both for r in results)
    print(f'  {n_valid}/{len(results)} events with valid X+Y hits')
    if n_valid < MIN_VALID:
        print(f'  [SKIP] {subrun}: too few valid events ({n_valid} < {MIN_VALID})')
        return None

    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xa, an)

    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm)
            for r in results if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}

    # raw "any hit on det FEUs" event set (sparks included)
    raw = uproot.concatenate([f'{hits_dir}{f}:hits' for f in hit_files],
                             expressions=['eventId', 'feu'], library='pd')
    det_raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    has_any = set(int(e) for e in det_raw['eventId'].unique())
    # spark metric: # strips fired per detector-firing event (one row per hit); an event
    # is a spark/discharge if multiplicity > SPARK_THRESH (same threshold as the veto).
    mult = det_raw.groupby('eventId').size()
    n_firing = int(len(mult))
    n_spark = int((mult > SPARK_THRESH).sum())

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = seed.ref_x_sign * np.array(xr)
    py = np.array(yr)

    rows = []
    for e, x, y in zip((int(v) for v in evn), px, py):
        within = e in reco and np.hypot(x - reco[e][0], y - reco[e][1]) <= R
        rows.append((e, x, y, bool(within), e in has_any))
    d = pd.DataFrame(rows, columns=['event_id', 'x', 'y', 'within', 'has_any'])
    d = d[np.isfinite(d['x']) & np.isfinite(d['y'])]

    # per-event residuals (aligned det - M3 ref) for the resolution-vs-HV curve
    resid = pd.DataFrame(
        [(r.event_id, r.ref_x_mm, r.ref_y_mm, r.residual_x_mm, r.residual_y_mm)
         for r in results if r.has_both
         and np.isfinite(r.residual_x_mm) and np.isfinite(r.residual_y_mm)],
        columns=['event_id', 'x', 'y', 'res_x', 'res_y'])

    return {'subrun': subrun, 'rays': d, 'reco': np.array(list(reco.values())),
            'resid': resid, 'params': params, 'n_valid': n_valid,
            'n_spark': n_spark, 'n_firing': n_firing}


def main():
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)

    if os.path.exists(ALIGN_SEED):
        seed = cm.load_alignment(ALIGN_SEED)
        print(f'Alignment seed from {ALIGN_SEED}\n  {seed}  ref_x_sign={seed.ref_x_sign:+.0f}')
    else:
        seed = SEED_DEFAULT
        print(f'Alignment seed (default, long_run align not on disk): {seed}')

    subruns = find_subruns()
    if not subruns:
        print(f'No resist_<NNN>V subruns under {CFG.BASE_PATH}{CFG.RUN}/'); return
    print(f'HV-scan subruns ({len(subruns)}): ' +
          ', '.join(f'{n} ({hv}V)' for n, hv in subruns))

    analysed = []
    for subrun, hv in subruns:
        print(f'\n{"="*64}\n{subrun}  (HV = {hv} V)\n{"="*64}')
        r = analyse_subrun(subrun, det, seed)
        if r is not None:
            r['hv'] = hv
            analysed.append(r)

    if not analysed:
        print('\nNo subruns had enough valid events to analyse.'); return

    # ---- Fixed active box: from the subrun with the most reco points (highest HV) ----
    ref = max(analysed, key=lambda a: len(a['reco']))
    rp = ref['reco']
    ax0, ax1 = np.percentile(rp[:, 0], [1, 99])
    ay0, ay1 = np.percentile(rp[:, 1], [1, 99])
    print(f'\nFixed active box from {ref["subrun"]} ({len(rp)} reco pts): '
          f'x[{ax0:.0f},{ax1:.0f}] y[{ay0:.0f},{ay1:.0f}]')

    # ---- Integrated efficiency per HV inside the fixed box ----
    rows = []
    for a in analysed:
        d = a['rays']
        inbox = ((d['x'] >= ax0) & (d['x'] <= ax1) &
                 (d['y'] >= ay0) & (d['y'] <= ay1))
        da = d[inbox]
        n_tracks = len(da)
        n_hit = int(da['within'].sum())
        n_any = int(da['has_any'].sum())
        eff = n_hit / n_tracks if n_tracks else np.nan
        err = np.sqrt(eff * (1 - eff) / n_tracks) if n_tracks else np.nan
        eff_any = n_any / n_tracks if n_tracks else np.nan
        # core spatial resolution inside the same fixed box
        rr = a['resid']
        rbox = ((rr['x'] >= ax0) & (rr['x'] <= ax1) &
                (rr['y'] >= ay0) & (rr['y'] <= ay1))
        fx = cm.fit_residual_peak(rr.loc[rbox, 'res_x'].to_numpy())
        fy = cm.fit_residual_peak(rr.loc[rbox, 'res_y'].to_numpy())
        sx = fx.resolution if fx is not None else np.nan
        sy = fy.resolution if fy is not None else np.nan
        spark_frac = a['n_spark'] / a['n_firing'] if a['n_firing'] else np.nan
        rows.append(dict(hv=a['hv'], subrun=a['subrun'], n_valid=a['n_valid'],
                         n_tracks=n_tracks, n_hit=n_hit, n_any=n_any,
                         eff_reco=eff, eff_reco_err=err, eff_anyhit=eff_any,
                         sigma_x_mm=sx, sigma_y_mm=sy,
                         n_spark=a['n_spark'], n_firing=a['n_firing'], spark_frac=spark_frac))
        print(f'  HV={a["hv"]}V  eff(reco<{R:g}mm)={eff:.3f}+-{err:.3f}  '
              f'eff(any hit)={eff_any:.3f}  sigma=({sx:.2f},{sy:.2f})mm  '
              f'spark={spark_frac*100:.1f}%  ({n_hit}/{n_tracks} in box)')

    df = pd.DataFrame(rows).sort_values('hv').reset_index(drop=True)
    od = out_dir()
    df.to_csv(os.path.join(od, 'efficiency_vs_hv.csv'), index=False)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df['hv'], df['eff_reco'], yerr=df['eff_reco_err'], fmt='o-',
                color='steelblue', capsize=4, lw=2, ms=7,
                label=f'reco within {R:g} mm')
    ax.plot(df['hv'], df['eff_anyhit'], 's--', color='darkorange', ms=6, alpha=0.8,
            label='any hit on detector')
    ax.set_xlabel('Resist HV [V]')
    ax.set_ylabel('Efficiency (fixed active box)')
    ax.set_title(f'{CFG.DET_NAME} efficiency + spark rate vs resist HV — {CFG.RUN}\n'
                 f'(r<{R:g} mm; integrated over fixed active area)')
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    # spark fraction on a twin axis (the high-HV roll-off mechanism)
    axs = ax.twinx()
    axs.plot(df['hv'], df['spark_frac'], 'x:', color='crimson', ms=8, lw=1.5,
             label=f'spark fraction (mult>{SPARK_THRESH})')
    axs.set_ylabel('Spark fraction of firing events', color='crimson')
    axs.tick_params(axis='y', labelcolor='crimson')
    _smax = float(np.nanmax(df['spark_frac'])) if len(df) else 0.0
    axs.set_ylim(0, max(0.05, _smax * 1.25))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axs.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, 'efficiency_vs_hv.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- Resolution vs HV ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(df['hv'], df['sigma_x_mm'], 'o-', color='steelblue', lw=2, ms=7, label='σ_x')
    ax2.plot(df['hv'], df['sigma_y_mm'], 's--', color='darkorange', lw=2, ms=6, label='σ_y')
    ax2.set_xlabel('Resist HV [V]')
    ax2.set_ylabel('Core spatial resolution σ [mm]')
    ax2.set_title(f'{CFG.DET_NAME} resolution vs resist HV — {CFG.RUN}\n'
                  f'(Gaussian core of in-box residuals)')
    ax2.set_ylim(0, None)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(od, 'resolution_vs_hv.png'), dpi=200, bbox_inches='tight')
    plt.close(fig2)

    print(f'\n{"HV[V]":>6}  {"eff_reco":>9}  {"+-err":>6}  {"eff_any":>8}  '
          f'{"tracks":>7}  {"valid":>6}')
    for _, r in df.iterrows():
        print(f'{r.hv:>6.0f}  {r.eff_reco:>9.3f}  {r.eff_reco_err:>6.3f}  '
              f'{r.eff_anyhit:>8.3f}  {r.n_tracks:>7.0f}  {r.n_valid:>6.0f}')
    print(f'\nWritten to: {od}')


if __name__ == '__main__':
    main()
