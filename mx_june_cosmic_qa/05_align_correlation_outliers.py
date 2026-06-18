#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_align_correlation_outliers.py

(1) Refine the translation using the position-correlation line (robust, outlier-
    rejecting) so the correlation sits on the diagonal — the upstream median
    translation is biased by the off-diagonal outliers.
(2) Characterise the significant outliers: are they the spark events, or do they
    have some other signature (low n_strips, bad chi2, long cluster, etc.)?

Reads the cached per-event results + alignment.json from a prior
  03_alignment_and_tpc.py <key> --flipy   (no veto) run.

Usage:
    python 05_align_correlation_outliers.py ovn_det1 --flipy
Products (output/<run>/<det>/correlation_outliers/):
  correlation_fit.png, residual_vs_multiplicity.png,
  outlier_feature_comparison.png, summary.txt
"""

import os, sys, pickle
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()

import uproot
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking

FLIPY = '--flipy' in sys.argv
OUTLIER_MM = 15.0          # radial residual above which an event is an "outlier"
tag = '_flipy' if FLIPY else ''


def sigma_clipped_median(v, nsig=3.0, iters=5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    for _ in range(iters):
        m, s = np.median(v), np.std(v)
        keep = np.abs(v - m) <= nsig * s
        if keep.all() or keep.sum() < 10:
            break
        v = v[keep]
    return float(np.median(v))


def robust_line(x, y, nsig=3.0, iters=5):
    """Robust slope/intercept of y vs x with iterative sigma clipping."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    for _ in range(iters):
        a, b = np.polyfit(x, y, 1)
        resid = y - (a * x + b)
        s = np.std(resid)
        keep = np.abs(resid) <= nsig * s
        if keep.all() or keep.sum() < 10:
            break
        x, y = x[keep], y[keep]
    return a, b, len(x)


def main():
    out_dir = CFG.out_dir('correlation_outliers')
    align_dir = CFG.out_dir(f'alignment_tpc{tag}')
    cache_path = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')

    results = pickle.load(open(cache_path, 'rb'))
    params = cm.load_alignment(os.path.join(align_dir, 'alignment.json'))

    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=20.0)
    from M3RefTracking import get_xy_angles
    xang, yang, anum = get_xy_angles(rays.ray_data); xang = -np.array(xang)
    cm.attach_reference_positions(results, rays, params, xang, anum)

    # ---- per-event table ----
    rows = []
    for r in results:
        if not r.has_both: continue
        if np.isnan(r.ref_x_mm) or np.isnan(r.ref_y_mm): continue
        rows.append(dict(
            event_id=r.event_id,
            det_x=r.det_x_for_residual, det_y=r.det_y_for_residual,
            ref_x=r.ref_x_mm, ref_y=r.ref_y_mm,
            res_x=r.residual_x_mm, res_y=r.residual_y_mm,
            nstrip_x=r.x_fit.n_strips, nstrip_y=r.y_fit.n_strips,
            rchi2_x=r.x_fit.red_chi2, rchi2_y=r.y_fit.red_chi2,
            dur_x=r.x_fit.cluster_duration_ns, dur_y=r.y_fit.cluster_duration_ns))
    d = pd.DataFrame(rows)

    # ---- raw multiplicity / amplitude per event ----
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel', 'amplitude'], library='pd')
    raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    mult = raw.groupby('eventId').agg(total_hits=('channel', 'size'),
                                      mean_amp=('amplitude', 'mean')).reset_index()
    d = d.merge(mult, left_on='event_id', right_on='eventId', how='left')

    # ---- robust correlation-line translation refinement ----
    summary = [f'Correlation/outlier analysis — {CFG.DET_NAME} {CFG.RUN}/{CFG.SUB_RUN}',
               f'loaded alignment: {params}', f'matched X+Y events: {len(d)}']
    ax_, bx_, nx_ = robust_line(d['det_x'], d['ref_x'])
    ay_, by_, ny_ = robust_line(d['det_y'], d['ref_y'])
    summary.append(f'robust line ref_x = {ax_:.3f}*det_x + {bx_:+.2f}  (slope~1 if z ok; intercept=offset, n={nx_})')
    summary.append(f'robust line ref_y = {ay_:.3f}*det_y + {by_:+.2f}  (n={ny_})')
    dx_shift = sigma_clipped_median(d['res_x']); dy_shift = sigma_clipped_median(d['res_y'])
    summary.append(f'robust residual offset to remove: dX={dx_shift:+.2f} mm, dY={dy_shift:+.2f} mm')

    # apply the robust shift and recompute residuals
    d['res_x_c'] = d['res_x'] - dx_shift
    d['res_y_c'] = d['res_y'] - dy_shift
    d['rad_c'] = np.hypot(d['res_x_c'], d['res_y_c'])
    for lbl, col in [('before', 'res_x'), ('after', 'res_x_c')]:
        core = d[np.abs(d[col]) < 50]
        summary.append(f'  X residual {lbl}: median {d[col].median():+.2f}, '
                       f'core(<50mm) std {core[col].std():.2f} mm')
    for lbl, col in [('before', 'res_y'), ('after', 'res_y_c')]:
        core = d[np.abs(d[col]) < 50]
        summary.append(f'  Y residual {lbl}: median {d[col].median():+.2f}, '
                       f'core(<50mm) std {core[col].std():.2f} mm')

    # ---- correlation plot with robust line + diagonal ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, dc, rc_, a, b, lbl in [(axes[0], 'det_x', 'ref_x', ax_, bx_, 'X'),
                                   (axes[1], 'det_y', 'ref_y', ay_, by_, 'Y')]:
        ax.scatter(d[dc], d[rc_], s=3, alpha=0.25, color='steelblue', linewidths=0)
        xs = np.array([d[dc].min(), d[dc].max()])
        ax.plot(xs, a * xs + b, 'r-', lw=1.5, label=f'robust fit: {a:.2f}x{b:+.1f}')
        lims = [min(d[dc].min(), d[rc_].min()), max(d[dc].max(), d[rc_].max())]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.6, label='y=x')
        ax.set_xlabel(f'detector {lbl} (aligned) [mm]'); ax.set_ylabel(f'reference {lbl} [mm]')
        ax.set_title(f'{lbl} correlation'); ax.legend(fontsize=8); ax.set_aspect('equal')
    fig.suptitle(f'{CFG.DET_NAME} position correlation + robust line — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout(); fig.savefig(f'{out_dir}/correlation_fit.png', dpi=150, bbox_inches='tight')

    # ---- outliers ----
    d['outlier'] = d['rad_c'] > OUTLIER_MM
    n_out = int(d['outlier'].sum()); n = len(d)
    summary.append(f'\nOutliers (radial residual > {OUTLIER_MM} mm after offset fix): '
                   f'{n_out}/{n} ({100*n_out/n:.1f}%)')
    # spark association
    for thr in (20, 50):
        spark = d['total_hits'] > thr
        out_and_spark = (d['outlier'] & spark).sum()
        summary.append(f'  of outliers, fraction with >{thr} total hits (spark-like): '
                       f'{100*out_and_spark/max(n_out,1):.1f}%  '
                       f'| inliers spark-like: {100*(~d["outlier"]&spark).sum()/max(n-n_out,1):.1f}%')
    # feature medians
    summary.append('  feature medians (inlier vs outlier):')
    for f in ['total_hits', 'nstrip_x', 'nstrip_y', 'rchi2_x', 'rchi2_y', 'dur_x', 'dur_y', 'mean_amp']:
        mi = d.loc[~d['outlier'], f].median(); mo = d.loc[d['outlier'], f].median()
        summary.append(f'    {f:11s}: inlier {mi:8.2f}   outlier {mo:8.2f}')

    # residual vs multiplicity
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(d['total_hits'], d['rad_c'], s=4, alpha=0.3,
                    c=d['outlier'].map({True: 'red', False: 'steelblue'}))
    ax.axhline(OUTLIER_MM, color='k', ls='--', lw=1, label=f'outlier cut {OUTLIER_MM} mm')
    ax.set_xscale('log'); ax.set_xlabel('total hits in event (multiplicity)')
    ax.set_ylabel('radial residual [mm]'); ax.legend()
    ax.set_title(f'{CFG.DET_NAME} residual vs event multiplicity — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout(); fig.savefig(f'{out_dir}/residual_vs_multiplicity.png', dpi=150, bbox_inches='tight')

    # feature comparison hists
    feats = [('total_hits', True), ('nstrip_x', False), ('nstrip_y', False),
             ('rchi2_x', True), ('rchi2_y', True), ('mean_amp', False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axx, (f, logx) in zip(axes.ravel(), feats):
        ina = d.loc[~d['outlier'], f].replace([np.inf, -np.inf], np.nan).dropna()
        outa = d.loc[d['outlier'], f].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = np.nanpercentile(pd.concat([ina, outa]), [0.5, 99.5])
        if logx and lo > 0:
            bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 50); axx.set_xscale('log')
        else:
            bins = np.linspace(lo, hi, 50)
        axx.hist(ina, bins=bins, density=True, histtype='step', lw=1.6, color='steelblue', label='inlier')
        axx.hist(outa, bins=bins, density=True, histtype='step', lw=1.6, color='red', label='outlier')
        axx.set_xlabel(f); axx.set_ylabel('norm.'); axx.legend(fontsize=8)
    fig.suptitle(f'{CFG.DET_NAME} inlier vs outlier features — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout(); fig.savefig(f'{out_dir}/outlier_feature_comparison.png', dpi=150, bbox_inches='tight')

    txt = '\n'.join(summary); print(txt)
    open(f'{out_dir}/summary.txt', 'w').write(txt + '\n')
    print(f'\nWritten to: {out_dir}')


if __name__ == '__main__':
    main()
