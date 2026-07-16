#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hv_scan.py

HV-scan analysis: take multiple subruns of one run at varying HV, analyse each,
and plot spark rate and efficiency vs HV.

Because the detector does not move during a scan, the geometry (z, theta, offset)
is HV-independent: we align ONCE on a high-stats reference subrun and apply that
alignment to every HV point. Per HV point we then measure
  - spark rate  : events with > SPARK_MAX raw hits, per second (and as a fraction)
  - efficiency  : per-ray, in the active area (08/09 method):
        within_R : reconstructed X+Y hit within R mm of the M3 projection
        has_any  : detector fired any strip (responded at all)

Scans are registered in SCANS below. Usage:
    python hv_scan.py ovn6-17_det1
    python hv_scan.py ovn6-17_det2

Output -> <cosmic_bench>/Analysis/<run>/HV_scan/<det>/ : hv_scan.png, hv_scan.csv
"""
import os, sys, pickle, concurrent.futures
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from qa_config import setup_paths, _Config
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions

# pre-v2/veto50-era script, superseded by 10_hv_scan_efficiency.py (the golden HV-scan
# chain); CHI2_CUT intentionally NOT bumped to the qa_config.M3_CHI2_CUT recipe here.
SPARK_MAX = 50; MAXDROP = 2; R = 5.0; CHI2_CUT = 20.0; CENTRE_XY = 200.0
N_ITER = 3; REF_X_SIGN = +1.0

_DET12_BASE = '/home/dylan/x17/cosmic_bench/det1_det2/'
_RESIST = [450, 460, 470, 480, 490, 500, 510, 520]   # resist HV [V], drift fixed 1000 V

SCANS = {
    'ovn6-17_det1': dict(run='mx17_det1_det2_overnight_6-17-26', base=_DET12_BASE,
                         det_name='mx17_1', feus=[3, 4], det_z=232.0,
                         ref_subrun='longer_run', hv_label='resist HV [V]',
                         points=[(v, f'resist_{v}V_drift_1000V') for v in _RESIST]),
    'ovn6-17_det2': dict(run='mx17_det1_det2_overnight_6-17-26', base=_DET12_BASE,
                         det_name='mx17_2', feus=[7, 8], det_z=702.0,
                         ref_subrun='longer_run', hv_label='resist HV [V]',
                         points=[(v, f'resist_{v}V_drift_1000V') for v in _RESIST]),
}


def cfg_for(scan, subrun):
    return _Config(f"{scan['det_name']}_{subrun}", scan['run'], subrun,
                   feus=scan['feus'], det_z=scan['det_z'], det_name=scan['det_name'],
                   base_path=scan['base'])


def analyse(cfg, veto=None):
    """Per-event micro-TPC analysis (cached). veto drops events > veto raw hits."""
    tag = f'_veto{veto}' if veto is not None else ''
    cache = os.path.join(cfg.out_dir('cache'), f'event_results{tag}.pkl')
    rc = RunConfig(cfg.run_config_path, cfg.MAP_CSV_PATH); det = rc.get_detector(cfg.DET_NAME)
    if os.path.exists(cache):
        return pickle.load(open(cache, 'rb')), det
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs], library='pd')
    df = df[df['feu'].isin(cfg.MX17_FEUS)].copy()
    if veto is not None:
        hpe = df.groupby('eventId')['channel'].transform('size'); df = df[hpe <= veto].copy()
    df = cm._map_strip_positions(df, det)
    eids = df['eventId'].unique(); g = df.groupby('eventId')
    args = [(g.get_group(e).copy(), int(e)) for e in eids]
    nw = max(1, (os.cpu_count() or 1) - cm.N_FREE_THREADS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=nw) as pool:
        results = list(pool.map(cm._analyse_event_worker, args))
    pickle.dump(results, open(cache, 'wb'))
    return results, det


def compute_alignment(scan):
    cfg = cfg_for(scan, scan['ref_subrun'])
    print(f"Aligning on reference subrun '{scan['ref_subrun']}' ...")
    results, det = analyse(cfg, veto=SPARK_MAX)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, yang, anum = get_xy_angles(rays.ray_data); xang = REF_X_SIGN * np.array(xang)
    rot0 = float(det.orientation.get('z', 0.0) or 0.0) or 90.0
    align = [r for r in results if r.has_both and (r.x_fit.n_dropped + r.y_fit.n_dropped) <= MAXDROP]
    init = cm.AlignmentParams(z_x=scan['det_z'], z_y=scan['det_z'], theta_deg=rot0,
                              centre_x=CENTRE_XY, centre_y=CENTRE_XY, ref_x_sign=REF_X_SIGN)
    zs = np.linspace(scan['det_z'] - 60, scan['det_z'] + 60, 121)
    ts = np.linspace(rot0 - 2, rot0 + 2, 81)
    best = cm.run_alignment(align, rays, initial_params=init, n_iterations=N_ITER,
                            z_values=zs, theta_values=ts, plot_each=False, plot_final=False)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    rp = np.array([(r.det_x_aligned_mm, r.det_y_aligned_mm) for r in results if r.has_both
                   and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)])
    box = (np.percentile(rp[:, 0], 0.5), np.percentile(rp[:, 0], 99.5),
           np.percentile(rp[:, 1], 0.5), np.percentile(rp[:, 1], 99.5))
    print(f'  aligned: {best}\n  active box: x[{box[0]:.0f},{box[1]:.0f}] y[{box[2]:.0f},{box[3]:.0f}]')
    return best, box


def spark_rate(cfg):
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel', 'trigger_timestamp_ns'], library='pd')
    raw = raw[raw['feu'].isin(cfg.MX17_FEUS)]
    g = raw.groupby('eventId'); hpe = g['channel'].size(); ts = g['trigger_timestamp_ns'].first()
    n_ev = len(hpe); n_spark = int((hpe > SPARK_MAX).sum())
    dur = float((ts.max() - ts.min()) / 1e9) if len(ts) > 1 else float('nan')
    rate = n_spark / dur if dur and np.isfinite(dur) and dur > 0 else float('nan')
    return dict(n_events=n_ev, n_spark=n_spark, spark_frac=n_spark / max(n_ev, 1),
                duration_s=dur, spark_rate_hz=rate,
                spark_rate_err=(np.sqrt(n_spark) / dur if dur and dur > 0 else float('nan')))


def efficiency(cfg, params, box):
    results, det = analyse(cfg, veto=None)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data); xang = REF_X_SIGN * np.array(xang)
    cm.attach_reference_positions(results, rays, params, xang, anum)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm) for r in results if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu'], library='pd')
    det_hit = set(int(e) for e in raw.loc[raw['feu'].isin(cfg.MX17_FEUS), 'eventId'].unique())
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = REF_X_SIGN * np.array(xr); py = np.array(yr)
    ax0, ax1, ay0, ay1 = box
    n = within = hasany = 0
    for e, x, y in zip((int(v) for v in evn), px, py):
        if not (np.isfinite(x) and np.isfinite(y) and ax0 <= x <= ax1 and ay0 <= y <= ay1):
            continue
        n += 1
        if e in reco and np.hypot(x - reco[e][0], y - reco[e][1]) <= R:
            within += 1
        if e in det_hit:
            hasany += 1
    nan = float('nan')
    n_reco = len(reco)
    err = lambda k: np.sqrt(max(k, 1) / n * (1 - k / n) / n) if n else nan
    if n == 0:                       # no M3 reference (e.g. empty tracking file)
        return dict(n_rays=0, n_reco=n_reco, eff_within=nan, eff_within_err=nan,
                    eff_has_any=nan, eff_has_any_err=nan, flag='no_m3')
    flag = '' if n_reco > 0 else 'no_reco(plane dropout?)'
    return dict(n_rays=n, n_reco=n_reco,
                eff_within=(within / n if n_reco > 0 else nan),
                eff_within_err=(err(within) if n_reco > 0 else nan),
                eff_has_any=hasany / n, eff_has_any_err=err(hasany), flag=flag)


def main():
    key = next((a for a in sys.argv[1:] if not a.startswith('-')), None)
    if key not in SCANS:
        sys.exit(f'usage: hv_scan.py <{"|".join(SCANS)}>')
    scan = SCANS[key]
    best, box = compute_alignment(scan)

    rows = []
    for hv, sub in scan['points']:
        cfg = cfg_for(scan, sub)
        if not os.path.isdir(cfg.combined_hits_dir):
            print(f'  skip {sub}: no data'); continue
        sp = spark_rate(cfg); ef = efficiency(cfg, best, box)
        rows.append(dict(hv=hv, subrun=sub, **sp, **ef))
        print(f"  HV={hv:>4} V: spark {sp['spark_rate_hz']:6.3f} Hz "
              f"({100*sp['spark_frac']:4.1f}%)  eff_within {100*ef['eff_within']:5.1f}%  "
              f"eff_has_any {100*ef['eff_has_any']:5.1f}%  (n_rays={ef['n_rays']}, "
              f"n_reco={ef['n_reco']}) {ef['flag']}")
    df = pd.DataFrame(rows).sort_values('hv')
    flagged = [f"{int(r.hv)}V: {r.flag}" for r in df.itertuples() if getattr(r, 'flag', '')]

    cb_root = os.path.dirname(scan['base'].rstrip('/'))
    out_dir = os.path.join(cb_root, 'Analysis', scan['run'], 'HV_scan', scan['det_name'])
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'hv_scan.csv'), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    hv = df['hv'].values
    axes[0].errorbar(hv, df['spark_rate_hz'], yerr=df['spark_rate_err'], fmt='o-', color='crimson', capsize=3)
    axes[0].set_xlabel(scan['hv_label']); axes[0].set_ylabel('spark rate [Hz]')
    axes[0].set_title('Spark rate vs HV'); axes[0].grid(alpha=0.3)
    axes[1].plot(hv, 100 * df['spark_frac'], 's-', color='darkorange')
    axes[1].set_xlabel(scan['hv_label']); axes[1].set_ylabel('spark fraction [%]')
    axes[1].set_title('Spark fraction vs HV'); axes[1].grid(alpha=0.3)
    axes[2].errorbar(hv, 100 * df['eff_within'], yerr=100 * df['eff_within_err'],
                     fmt='o-', color='green', capsize=3, label=f'within {R:g} mm (reco)')
    axes[2].errorbar(hv, 100 * df['eff_has_any'], yerr=100 * df['eff_has_any_err'],
                     fmt='s-', color='steelblue', capsize=3, label='has_any (detection)')
    axes[2].set_xlabel(scan['hv_label']); axes[2].set_ylabel('efficiency [%]')
    axes[2].set_title('Efficiency vs HV (active area)'); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle(f"HV scan — {scan['det_name']}  {scan['run']}  (drift 1000 V)")
    if flagged:
        fig.text(0.5, 0.005, 'excluded/flagged: ' + '; '.join(flagged),
                 ha='center', fontsize=8, color='gray')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(out_dir, 'hv_scan.png'), dpi=150, bbox_inches='tight')
    print(f'\nWritten: {out_dir}/hv_scan.png  +  hv_scan.csv')


if __name__ == '__main__':
    main()
