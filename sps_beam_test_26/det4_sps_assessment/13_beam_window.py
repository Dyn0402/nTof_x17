#!/usr/bin/env python3
"""
13_beam_window.py — is det4's best band a usable beam target?

The widest live band sits at detector-local X 177-215 mm (reference Y -10 to
+30 mm on the sliding map, since this run's alignment is theta ~ 90 deg). This
script asks the three questions a beam test actually needs answered:

  1. Inside that band, how does det4 perform, and is it uniform over the full
     360 mm of Y? A band that is only good over part of its length is a much
     smaller target than it looks.
  2. An 8 cm beam spot is wider than the band. How much of the spot lands on
     live chamber for the best placement, and what does the whole spot average?
  3. Tracks that are inclined *across* the stripes wander in X by 30*tan(theta)
     over the drift gap and can leave the band. Tilting the chamber so the
     inclination is along the stripes (large theta_Y, small theta_X) should keep
     them inside. Both configurations already exist in the cosmic sample, so the
     gain from tilting can be measured rather than assumed.

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/13_beam_window.py
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
import cosmic_micro_tpc_analysis as cm                       # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE              # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                          # noqa: E402
_st = import_module('04_stripe_metrics')

GAP_MM = 30.0
BAND = (177.0, 215.0)          # the widest live band, detector-local X [mm]
BEAM_D = 80.0                  # beam spot diameter [mm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='g_det4')
    ap.add_argument('--band', nargs=2, type=float, default=list(BAND))
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    b0, b1 = args.band

    ray, cfg = _st.build(args.key)
    ax0, ax1 = TRUE_ACTIVE['x']
    inband = (ray.lx >= b0) & (ray.lx <= b1)
    rep = dict(run_key=args.key, band_mm=[b0, b1], band_width_mm=b1 - b0)

    # ---- 1. performance in the band, and along Y ---------------------------
    rep['band'] = _st.score(ray[inband], f'band {b0:.0f}-{b1:.0f} mm')
    rep['whole_chamber'] = _st.score(ray, 'whole active area')

    ye = np.arange(TRUE_ACTIVE['y'][0], TRUE_ACTIVE['y'][1] + 30, 30.0)
    yc = 0.5 * (ye[:-1] + ye[1:])
    sub = ray[inband]
    iy = np.digitize(sub.ly.to_numpy(), ye) - 1
    near = np.nan_to_num(sub.near.to_numpy(), nan=0.0)
    effy = np.array([near[iy == k].mean() if (iy == k).sum() >= 15 else np.nan
                     for k in range(len(yc))])
    ny = np.array([(iy == k).sum() for k in range(len(yc))])
    rep['along_Y'] = dict(
        y_centres=[float(v) for v in yc],
        efficiency=[None if not np.isfinite(v) else float(v) for v in effy],
        n=[int(v) for v in ny],
        median=float(np.nanmedian(effy)),
        min=float(np.nanmin(effy)), max=float(np.nanmax(effy)),
        rms=float(np.nanstd(effy)))

    # ---- 2. what an 8 cm beam spot sees ------------------------------------
    # fine efficiency profile in X, then slide a BEAM_D-wide window over it
    e = np.arange(ax0, ax1 + 2, 2.0)
    c = 0.5 * (e[:-1] + e[1:])
    ix = np.digitize(ray.lx.to_numpy(), e) - 1
    nr = np.nan_to_num(ray.near.to_numpy(), nan=0.0)
    prof = np.array([nr[ix == k].mean() if (ix == k).sum() >= 8 else np.nan
                     for k in range(len(c))])
    nwin = int(BEAM_D / 2)
    # circular spot: weight each X slice by the chord length of a disc
    off = (np.arange(nwin) - (nwin - 1) / 2) * 2.0
    w = np.sqrt(np.clip((BEAM_D / 2) ** 2 - off ** 2, 0, None))
    w = w / w.sum()
    means = np.full(len(c) - nwin, np.nan)
    for i in range(len(means)):
        seg, ww = prof[i:i + nwin], w.copy()
        m = np.isfinite(seg)
        if m.sum() > nwin * 0.8:
            means[i] = float(np.sum(seg[m] * ww[m]) / ww[m].sum())
    ibest = int(np.nanargmax(means))
    rep['beam_spot'] = dict(
        diameter_mm=BEAM_D,
        best_centre_local_x_mm=float(c[ibest] + BEAM_D / 2),
        best_spot_mean_efficiency=float(means[ibest]),
        spot_centred_on_band=float(np.nanmax(means)),
        chamber_average=float(np.nanmean(prof)))

    # ---- 3. tilt along the stripes vs across them ---------------------------
    tanx = np.abs(ray.tan_x.to_numpy()) if 'tan_x' in ray else None
    rep['tilt'] = {}
    if 'tx' in ray.columns:
        pass
    # reference tangents are attached by 04's build via dthx/dthy only, so
    # recompute the reference tangents here from the M3 rays
    from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa
    params = cm.load_alignment(os.path.join(cfg.OUT_BASE, 'wft', 'alignment',
                                            'alignment.json'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    tanmap = {int(ev): (float(params.ref_x_sign * tx), float(ty))
              for ev, tx, ty in zip(an, xa, ya)}
    # with theta ~ 90 deg the reference-Y tangent moves the track along local X
    t_across = np.array([abs(tanmap.get(e, (np.nan, np.nan))[1])
                         for e in ray.eventId])     # across the stripes
    t_along = np.array([abs(tanmap.get(e, (np.nan, np.nan))[0])
                        for e in ray.eventId])      # along the stripes
    ray = ray.assign(t_across=t_across, t_along=t_along)
    sub = ray[inband & np.isfinite(t_across) & np.isfinite(t_along)]

    def band_score(sel, label):
        s = _st.score(sub[sel], label)
        s['n_sel'] = int(sel.sum())
        return s

    bins = [(0.0, 0.05), (0.05, 0.12), (0.12, 0.25), (0.25, 0.60)]
    rep['tilt']['across_stripes'] = [
        dict(tan_lo=lo, tan_hi=hi,
             excursion_mm=GAP_MM * 0.5 * (lo + hi),
             **band_score((sub.t_across >= lo) & (sub.t_across < hi),
                          f'across {lo}-{hi}'))
        for lo, hi in bins]
    rep['tilt']['along_stripes'] = [
        dict(tan_lo=lo, tan_hi=hi,
             **band_score((sub.t_along >= lo) & (sub.t_along < hi)
                          & (sub.t_across < 0.10), f'along {lo}-{hi}'))
        for lo, hi in bins]

    with open(os.path.join(args.out, f'beam_window_{args.key}.json'), 'w') as f:
        json.dump(rep, f, indent=1)

    # ------------------------------- figure ---------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    a = axs[0, 0]
    a.plot(c, prof, 'k-', lw=1)
    a.axvspan(b0, b1, color='#009e73', alpha=.2, label=f'band {b0:.0f}-{b1:.0f} mm')
    cen = rep['beam_spot']['best_centre_local_x_mm']
    a.axvspan(cen - BEAM_D / 2, cen + BEAM_D / 2, color='#0072b2', alpha=.12,
              label=f'best 80 mm spot ({rep["beam_spot"]["best_spot_mean_efficiency"]:.2f})')
    a.set_xlabel('detector-local X [mm]')
    a.set_ylabel('efficiency within 5 mm')
    a.set_ylim(0, 1.02)
    a.legend(fontsize=8)
    a.grid(alpha=.3)
    a.set_title(f'{args.key} — where to put the beam')

    a = axs[0, 1]
    a.errorbar(yc, effy, yerr=np.sqrt(np.clip(effy * (1 - effy), 0, None)
                                      / np.clip(ny, 1, None)),
               fmt='o-', capsize=3, color='#009e73')
    a.set_ylim(0, 1.02)
    a.set_xlabel('detector-local Y [mm]')
    a.set_ylabel('efficiency within 5 mm')
    a.grid(alpha=.3)
    a.set_title(f'inside the band, along its length '
                f'(median {rep["along_Y"]["median"]:.2f}, '
                f'rms {rep["along_Y"]["rms"]:.3f})')

    for a, key_, lab in [(axs[1, 0], 'across_stripes', 'inclined ACROSS the stripes'),
                         (axs[1, 1], 'along_stripes', 'inclined ALONG the stripes')]:
        rows = [r for r in rep['tilt'][key_] if r.get('n_rays', 0) > 60]
        xs = [0.5 * (r['tan_lo'] + r['tan_hi']) for r in rows]
        a.plot(xs, [r['within_5mm'] for r in rows], 'o-', color='#d55e00',
               label='efficiency within 5 mm')
        a.plot(xs, [r['reco_at_all'] for r in rows], 'o--', color='#0072b2',
               label='reconstructed at all')
        for r, x in zip(rows, xs):
            a.annotate(f'n={r["n_rays"]}', (x, r['within_5mm']), fontsize=7,
                       textcoords='offset points', xytext=(0, 8), ha='center')
        a.set_ylim(0, 1.02)
        a.set_xlabel('|tan θ| of the reference track')
        a.set_ylabel('fraction')
        a.legend(fontsize=8)
        a.grid(alpha=.3)
        a.set_title(f'in-band performance, {lab}')
    fig.suptitle(f'{args.key} — the X {b0:.0f}-{b1:.0f} mm band as a beam target')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'beam_window_{args.key}.png'), dpi=115)

    print(json.dumps({k: v for k, v in rep.items() if k != 'tilt'}, indent=1))
    for key_ in ('across_stripes', 'along_stripes'):
        print(f'\n{key_}:')
        print(f'{"|tan|":>12}{"n":>7}{"within5":>9}{"reco":>8}{"core σ":>9}'
              f'{"σθ_X":>8}{"σθ_Y":>8}')
        for r in rep['tilt'][key_]:
            print(f'{r["tan_lo"]:5.2f}-{r["tan_hi"]:<6.2f}{r.get("n_rays",0):7d}'
                  f'{r.get("within_5mm",float("nan")):9.3f}'
                  f'{r.get("reco_at_all",float("nan")):8.3f}'
                  f'{r.get("core_sigma_mm",float("nan")):9.3f}'
                  f'{r.get("sigma_theta_x",float("nan")):8.2f}'
                  f'{r.get("sigma_theta_y",float("nan")):8.2f}')


if __name__ == '__main__':
    main()
