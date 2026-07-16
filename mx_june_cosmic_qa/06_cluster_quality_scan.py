#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_cluster_quality_scan.py

Iterate on cluster-quality cuts to maximise agreement with the M3 reference.
Uses the already-aligned (z, theta fixed) per-event results from a prior
  03_alignment_and_tpc.py <key> --flipy   run.

Strategy:
  - Spark veto first (raw hits/event > SPARK_MAX), sparks tagged not deleted.
  - For each candidate cut, re-center the translation (robust median residual)
    on the survivors and measure: core residual sigma (X,Y, robust) and the
    fraction of survivors within 5/15 mm of the reference, vs retained fraction.
  - The good cut lowers sigma / raises frac-within-5mm for small event loss.

Usage: python 06_cluster_quality_scan.py ovn_det1 --flipy
Products (output/<run>/<det>/cluster_quality/): tradeoff.png, summary.txt
"""
import os, sys, pickle
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

FLIPY = '--flipy' in sys.argv
tag = '_flipy' if FLIPY else ''
SPARK_MAX = 50      # raw hits/event above this = spark, vetoed for alignment


def rmed(v, nsig=3, it=5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v)
        k = np.abs(v - m) <= nsig * s
        if k.all() or k.sum() < 10: break
        v = v[k]
    return float(np.median(v))


def rstd(v, nsig=3, it=5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v)
        k = np.abs(v - m) <= nsig * s
        if k.all() or k.sum() < 10: break
        v = v[k]
    return float(np.std(v))


def metrics(sub):
    rx = sub['res_x'].values - rmed(sub['res_x']); ry = sub['res_y'].values - rmed(sub['res_y'])
    rad = np.hypot(rx, ry)
    return dict(n=len(sub), sig_x=rstd(rx), sig_y=rstd(ry),
                f5=float(np.mean(rad < 5)), f15=float(np.mean(rad < 15)))


def main():
    out_dir = CFG.out_dir('cluster_quality')
    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', f'event_results{tag}.pkl'), 'rb'))
    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json'))
    # intentionally loose (not qa_config.M3_CHI2_CUT): this script scans cluster-quality
    # cuts against the M3 reference itself, so the reference must not be pre-cut.
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=20)
    xa, ya, an = get_xy_angles(rays.ray_data); xa = -np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)

    rows = []
    for r in res:
        if not r.has_both or np.isnan(r.ref_x_mm) or np.isnan(r.ref_y_mm): continue
        rows.append(dict(event_id=r.event_id, res_x=r.residual_x_mm, res_y=r.residual_y_mm,
                         nstrip=r.x_fit.n_strips + r.y_fit.n_strips,
                         ndrop=r.x_fit.n_dropped + r.y_fit.n_dropped,
                         amp=0.0))
    d = pd.DataFrame(rows)
    d['domfrac'] = d['nstrip'] / (d['nstrip'] + d['ndrop']).clip(lower=1)

    # raw multiplicity + amplitude per event
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel', 'amplitude'], library='pd')
    raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    mult = raw.groupby('eventId').agg(total_hits=('channel', 'size'),
                                      mean_amp=('amplitude', 'mean')).reset_index()
    d = d.merge(mult, left_on='event_id', right_on='eventId', how='left')

    summary = [f'Cluster-quality scan — {CFG.DET_NAME} {CFG.RUN}/{CFG.SUB_RUN}',
               f'matched X+Y events with ref: {len(d)}']

    # --- spark population (study, then veto) ---
    spark = d['total_hits'] > SPARK_MAX
    msp = metrics(d[spark]); mno = metrics(d[~spark])
    summary.append(f'sparks (>{SPARK_MAX} hits): {spark.sum()} ({100*spark.mean():.1f}%) '
                   f'| spark f15={msp["f15"]:.2f}  non-spark f15={mno["f15"]:.2f}')
    base = d[~spark].copy()
    b = metrics(base)
    summary.append(f'BASELINE (spark-vetoed, no quality cut): n={b["n"]} '
                   f'sig_x={b["sig_x"]:.2f} sig_y={b["sig_y"]:.2f} f5={b["f5"]:.2f} f15={b["f15"]:.2f}')

    # --- scans ---
    scans = {
        'n_dropped<=D': ('ndrop', [0, 1, 2, 3, 5, 8, 12, 20, 1e9], 'le'),
        'domfrac>=f':   ('domfrac', [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0], 'ge'),
        'mean_amp>=a':  ('mean_amp', list(np.percentile(base['mean_amp'].dropna(), [0,10,20,30,40,50,60,70])), 'ge'),
    }
    curves = {}
    for name, (col, thrs, op) in scans.items():
        pts = []
        summary.append(f'\n  {name}:')
        for t in thrs:
            sub = base[base[col] <= t] if op == 'le' else base[base[col] >= t]
            if len(sub) < 50: continue
            m = metrics(sub); ret = m['n'] / b['n']
            pts.append((ret, m['sig_x'], m['sig_y'], m['f5'], m['f15'], t))
            summary.append(f'    {col} {op} {t:8.3g}: keep {100*ret:5.1f}%  '
                           f'sig_x={m["sig_x"]:5.2f} sig_y={m["sig_y"]:5.2f} '
                           f'f5={m["f5"]:.2f} f15={m["f15"]:.2f}')
        curves[name] = np.array(pts)

    # --- trade-off plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in curves.items():
        if len(p) == 0: continue
        axes[0].plot(p[:, 0] * 100, p[:, 1], 'o-', label=f'{name} (σx)', ms=4)
        axes[1].plot(p[:, 0] * 100, p[:, 3] * 100, 'o-', label=name, ms=4)
    axes[0].axhline(b['sig_x'], color='gray', ls=':', label='baseline σx')
    axes[0].set_xlabel('retained fraction [%]'); axes[0].set_ylabel('core σ_x [mm]')
    axes[0].set_title('resolution vs retention'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(b['f5'] * 100, color='gray', ls=':', label='baseline')
    axes[1].set_xlabel('retained fraction [%]'); axes[1].set_ylabel('% within 5 mm')
    axes[1].set_title('precision vs retention'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle(f'{CFG.DET_NAME} cluster-quality cut trade-offs — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout(); fig.savefig(f'{out_dir}/tradeoff.png', dpi=150, bbox_inches='tight')

    txt = '\n'.join(summary); print(txt)
    open(f'{out_dir}/summary.txt', 'w').write(txt + '\n')
    print(f'\nWritten to: {out_dir}')


if __name__ == '__main__':
    main()
