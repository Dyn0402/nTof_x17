#!/usr/bin/env python3
"""
12_edge_sharpness.py — discrete broken strips, or a smoothly varying gap?

The band edges measured on all tracks are ~10 mm wide, but that number is blurred
by the muon itself: a track crossing the 30 mm drift gap at tan θ moves 30·tan θ
in X between entering the gap and reaching the mesh, so at a typical cosmic
tan θ ≈ 0.2 each "ray position" is smeared over ~6 mm.

Selecting near-vertical tracks removes most of that blur:

  discrete broken resistive strips -> the boundary is a strip pitch (0.78 mm) and
      the edge should sharpen towards ~1-2 mm as the tracks get vertical;
  a smoothly varying amplification gap -> the edge width is a property of the
      chamber and stays ~10 mm however vertical the tracks are.

Measures the 10-90 % transition width of the ≥3-strips-both-planes profile in
bins of |tan θ| of the reference track.

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/12_edge_sharpness.py
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import matplotlib                                            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import uproot                                                # noqa: E402
import pandas as pd                                          # noqa: E402
import cosmic_micro_tpc_analysis as cm                       # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from wft.seed import SPARK_VETO_HITS                         # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE              # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                          # noqa: E402
_uni = import_module('01_uniformity')

GAP_MM = 30.0


def build(key, amp_thr=60.0):
    cfg = get_config(key)
    params = cm.load_alignment(os.path.join(cfg.OUT_BASE, 'wft', 'alignment',
                                            'alignment.json'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    tan_by_ev = {int(e): (float(params.ref_x_sign * tx), float(ty))
                 for e, tx, ty in zip(an, xa, ya)}
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    lx, ly = _uni.ref_to_det(px, np.array(yr), params)
    ev = [int(v) for v in evn]
    # the excursion that matters is along local X; with theta ~ 90 deg that is the
    # reference-Y tangent. Take the larger of the two as a conservative proxy and
    # also keep both so the selection can be made on the right one.
    tx = np.array([tan_by_ev.get(e, (np.nan, np.nan))[0] for e in ev])
    ty = np.array([tan_by_ev.get(e, (np.nan, np.nan))[1] for e in ev])
    ray = pd.DataFrame({'eventId': ev, 'lx': lx, 'ly': ly, 'tx': tx, 'ty': ty})

    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel', 'amplitude'],
                             library='pd')
    det = raw[raw['feu'].isin(cfg.MX17_FEUS)]
    lo, hi = int(det['eventId'].min()), int(det['eventId'].max())
    big = det[det['amplitude'] >= amp_thr]
    mult = big.groupby('eventId').size()
    spark = set(mult[mult > SPARK_VETO_HITS].index)
    fx, fy = cfg.MX17_FEUS
    nx = big[big.feu == fx].groupby('eventId').size()
    ny = big[big.feu == fy].groupby('eventId').size()
    ray = ray[(ray.eventId >= lo) & (ray.eventId <= hi) & ~ray.eventId.isin(spark)]
    ray['ok3'] = ((ray.eventId.map(nx).fillna(0) >= 3)
                  & (ray.eventId.map(ny).fillna(0) >= 3))
    ax0, ax1 = TRUE_ACTIVE['x']
    ay0, ay1 = TRUE_ACTIVE['y']
    ray = ray[(ray.lx >= ax0) & (ray.lx <= ax1)
              & (ray.ly >= ay0) & (ray.ly <= ay1)]
    return ray[np.isfinite(ray.tx) & np.isfinite(ray.ty)], cfg


def edge_widths(x, ok, binw, xlo, xhi, minn=6):
    e = np.arange(xlo, xhi + binw, binw)
    c = 0.5 * (e[:-1] + e[1:])
    i = np.digitize(x, e) - 1
    p = np.array([ok[i == k].mean() if (i == k).sum() >= minn else np.nan
                  for k in range(len(c))])
    n = np.array([(i == k).sum() for k in range(len(c))])
    # 10-90 widths on every monotone rise/fall between a <=0.1 point and a >=0.9 point
    widths = []
    good = np.isfinite(p)
    cc, pp = c[good], p[good]
    k = 0
    while k < len(pp) - 1:
        if pp[k] <= 0.15:
            j = k
            while j + 1 < len(pp) and pp[j + 1] > pp[j] - 0.05:
                j += 1
                if pp[j] >= 0.85:
                    widths.append(cc[j] - cc[k])
                    break
            k = max(j, k + 1)
        else:
            k += 1
    return c, p, n, np.array(widths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='g_det4')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    ray, cfg = build(args.key)
    ax0, ax1 = TRUE_ACTIVE['x']
    # local X is the coordinate the stripes live in; with theta ~ 90 deg the
    # reference-Y tangent is what moves the track along it
    tan = np.abs(ray.ty.to_numpy())
    x = ray.lx.to_numpy()
    ok = ray.ok3.to_numpy().astype(float)

    rep = dict(run_key=args.key, gap_mm=GAP_MM, bins=[])
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    cuts = [(0.00, 0.05), (0.05, 0.12), (0.12, 0.25), (0.25, 0.60)]
    cols = ['#0072b2', '#009e73', '#e69f00', '#d55e00']
    for (lo, hi), col in zip(cuts, cols):
        m = (tan >= lo) & (tan < hi)
        if m.sum() < 400:
            continue
        binw = 2.0
        c, p, n, w = edge_widths(x[m], ok[m], binw, ax0, ax1)
        excursion = GAP_MM * 0.5 * (lo + hi)
        lab = (f'|tan θ| {lo:.2f}-{hi:.2f}  (n={m.sum():,}, '
               f'X excursion {excursion:.1f} mm)')
        axs[0].plot(c, p, lw=1.1, color=col, label=lab)
        rep['bins'].append(dict(
            tan_lo=lo, tan_hi=hi, n_rays=int(m.sum()),
            mean_x_excursion_mm=float(excursion),
            n_edges=int(len(w)),
            edge_width_median_mm=float(np.median(w)) if len(w) else None,
            edge_width_p25_mm=float(np.percentile(w, 25)) if len(w) else None,
            edge_width_p75_mm=float(np.percentile(w, 75)) if len(w) else None,
            contrast=float(np.nanmax(p) / max(np.nanmin(p), 1e-3))))
    axs[0].set_ylim(0, 1.05)
    axs[0].set_xlabel('detector-local X [mm]')
    axs[0].set_ylabel('fraction with ≥3 strips >60 ADC, both planes')
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=.3)
    axs[0].set_title(f'{args.key} — band profile vs reference track inclination '
                     f'(2 mm bins)')

    b = [r for r in rep['bins'] if r['edge_width_median_mm']]
    if b:
        axs[1].errorbar([r['mean_x_excursion_mm'] for r in b],
                        [r['edge_width_median_mm'] for r in b],
                        yerr=[[r['edge_width_median_mm'] - r['edge_width_p25_mm'] for r in b],
                              [r['edge_width_p75_mm'] - r['edge_width_median_mm'] for r in b]],
                        fmt='o-', capsize=4, color='k')
        xs = np.linspace(0, 12, 50)
        axs[1].plot(xs, np.hypot(xs, 1.0), ls=':', color='#0072b2',
                    label='blur-limited, 1 mm intrinsic (discrete strips)')
        axs[1].plot(xs, np.hypot(xs, 5.0), ls='--', color='#d55e00',
                    label='blur-limited, 5 mm intrinsic (smooth gap)')
        axs[1].set_ylim(0, 18)
        axs[1].set_xlabel('mean track excursion in X across the 30 mm gap [mm]')
        axs[1].set_ylabel('10-90 % band-edge width [mm]')
        axs[1].legend(fontsize=8)
        axs[1].grid(alpha=.3)
        axs[1].set_title('extrapolating to vertical tracks gives the chamber\'s '
                         'own edge width')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'edge_sharpness_{args.key}.png'), dpi=115)
    with open(os.path.join(args.out, f'edge_sharpness_{args.key}.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
