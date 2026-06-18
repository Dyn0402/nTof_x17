#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_refit_z_clean.py

Final robust-alignment demonstration for det1:
  spark veto (raw hits > SPARK_MAX) + cluster-quality cut (n_dropped <= MAXDROP),
  then re-fit z_x and z_y on the CLEAN sample (rotation/offset from the base
  alignment held fixed) by minimising the sigma-clipped residual width.

This isolates the result the upstream iterative z-scan missed because sparks /
competing-cluster outliers distorted its variance landscape.

Usage: python 07_refit_z_clean.py ovn_det1 --flipy
"""
import os, sys, pickle
import matplotlib; matplotlib.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions

FLIPY = '--flipy' in sys.argv
tag = '_flipy' if FLIPY else ''
SPARK_MAX = 50
MAXDROP = 2


def rstd(v, ns=3, it=5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v); k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10: break
        v = v[k]
    return float(np.std(v))


def main():
    out_dir = CFG.out_dir('final_alignment')
    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', f'event_results{tag}.pkl'), 'rb'))
    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json'))
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=20)
    xa, ya, an = get_xy_angles(rays.ray_data); xa = -np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)

    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel'], library='pd')
    raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    th = raw.groupby('eventId').size()
    spark = set(th[th > SPARK_MAX].index)

    # clean sample: not spark, both axes, has ref, cluster-quality cut
    ev = {}
    for r in res:
        if not r.has_both or np.isnan(r.ref_x_mm) or np.isnan(r.ref_y_mm): continue
        if r.event_id in spark: continue
        if (r.x_fit.n_dropped + r.y_fit.n_dropped) > MAXDROP: continue
        ev[r.event_id] = (r.det_x_aligned_mm, r.det_y_aligned_mm)
    print(f'clean events (no spark, n_dropped<={MAXDROP}): {len(ev)}')

    def scan(axis):
        zs = np.arange(CFG.DET_PLANE_Z - 80, CFG.DET_PLANE_Z + 120, 4)
        best = (None, 1e9)
        for z in zs:
            xr, yr, evn = get_xy_positions(rays.ray_data, z)
            ref = {e: (-xr[i] if axis == 'x' else yr[i]) for i, e in enumerate(evn)}
            d, rf = [], []
            for e, (dax, day) in ev.items():
                if e in ref:
                    d.append(dax if axis == 'x' else day); rf.append(ref[e])
            d, rf = np.array(d), np.array(rf); resid = rf - d
            s = rstd(resid - np.median(resid))
            if s < best[1]: best = (z, s)
        return best

    zx, sx = scan('x'); zy, sy = scan('y')
    print(f'BEST z_x={zx} -> sig_x={sx:.2f} mm ;  z_y={zy} -> sig_y={sy:.2f} mm')

    # build final residuals & correlation arrays at best z
    xr, yr, evn = get_xy_positions(rays.ray_data, zx); refx = {e: -xr[i] for i, e in enumerate(evn)}
    xr2, yr2, evn2 = get_xy_positions(rays.ray_data, zy); refy = {e: yr2[i] for i, e in enumerate(evn2)}
    DX, RX, DY, RY = [], [], [], []
    for e, (dax, day) in ev.items():
        if e in refx and e in refy:
            DX.append(dax); RX.append(refx[e]); DY.append(day); RY.append(refy[e])
    DX, RX, DY, RY = map(np.array, (DX, RX, DY, RY))
    rx = RX - np.median(RX - DX); ry = RY - np.median(RY - DY)
    rad = np.hypot(rx - DX, ry - DY)
    print(f'final: sig_x={rstd(rx-DX):.2f} sig_y={rstd(ry-DY):.2f} mm | '
          f'within 2mm={100*np.mean(rad<2):.0f}% 5mm={100*np.mean(rad<5):.0f}% 15mm={100*np.mean(rad<15):.0f}%')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, D, R, lbl, z in [(axes[0], DX, rx, 'X', zx), (axes[1], DY, ry, 'Y', zy)]:
        ax.scatter(D, R, s=4, alpha=0.3, color='steelblue', linewidths=0)
        lims = [min(D.min(), R.min()), max(D.max(), R.max())]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.7, label='y=x')
        ax.set_xlabel(f'detector {lbl} (aligned) [mm]'); ax.set_ylabel(f'reference {lbl} [mm]')
        ax.set_title(f'{lbl}  z={z} mm,  σ={rstd((R-D)):.2f} mm'); ax.legend(fontsize=8); ax.set_aspect('equal')
    fig.suptitle(f'{CFG.DET_NAME} FINAL alignment (spark veto + n_dropped<={MAXDROP} + z refit) — {CFG.RUN}/{CFG.SUB_RUN}\n'
                 f'{len(DX)} events')
    fig.tight_layout(); fig.savefig(f'{out_dir}/final_correlation.png', dpi=150, bbox_inches='tight')
    print(f'Written to: {out_dir}')


if __name__ == '__main__':
    main()
