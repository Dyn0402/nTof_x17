#!/usr/bin/env python3
"""
11_strip_break_test.py — do the bands die *partway along Y*, like a broken strip?

A resistive strip is fed its HV through a resistor at the end(s) of the strip. If
one breaks partway along its length, the segment on the far side of the break
floats and stops amplifying: the chamber goes dead over that strip's X, but only
from the break to the far edge in Y. The prediction is specific and testable:

  * dead-in-X columns should not be uniformly dead in Y — many should be live up
    to some Y and dead beyond it;
  * if the resistive strips are fed from one end only, the surviving segments
    should consistently be on the *same* side in Y;
  * the transition in Y should be sharp and at a different Y for different X
    (each strip breaks where it breaks).

A uniform-in-Y column, by contrast, is what a mesh/gap defect running the whole
length of the chamber gives.

Second job: the efficiency map is built on hits selected by *significance*, and
FEU 6 connectors 7-8 carry a stale pedestal on this run (see 10_pedestals.py),
so their significance is inflated. This script therefore also recomputes the
reconstructability profile with a fixed ADC amplitude threshold, which does not
use the pedestal at all.

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/11_strip_break_test.py
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
from M3RefTracking import M3RefTracking, get_xy_positions    # noqa: E402
from wft.seed import SPARK_VETO_HITS                         # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE              # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                          # noqa: E402
_uni = import_module('01_uniformity')


def amp_threshold_table(key, amp_thr=60.0):
    """Per-ray ≥3-strips-on-both-planes using a fixed ADC cut (pedestal-free)."""
    cfg = get_config(key)
    params = cm.load_alignment(os.path.join(cfg.OUT_BASE, 'wft', 'alignment',
                                            'alignment.json'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    lx, ly = _uni.ref_to_det(px, np.array(yr), params)
    ray = pd.DataFrame({'eventId': [int(v) for v in evn], 'lx': lx, 'ly': ly})

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
    ray = ray[(ray.eventId >= lo) & (ray.eventId <= hi)]
    ray = ray[~ray.eventId.isin(spark)]
    ray['nx'] = ray.eventId.map(nx).fillna(0)
    ray['ny'] = ray.eventId.map(ny).fillna(0)
    ray['ok3'] = (ray.nx >= 3) & (ray.ny >= 3)
    ax0, ax1 = TRUE_ACTIVE['x']
    ay0, ay1 = TRUE_ACTIVE['y']
    return ray[(ray.lx >= ax0) & (ray.lx <= ax1)
               & (ray.ly >= ay0) & (ray.ly <= ay1)], cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='g_det4')
    ap.add_argument('--amp-thr', type=float, default=60.0)
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    key = args.key

    rec, cfg = _uni.categorise(key)
    ax0, ax1 = TRUE_ACTIVE['x']
    ay0, ay1 = TRUE_ACTIVE['y']
    m = ((rec['x'] >= ax0) & (rec['x'] <= ax1)
         & (rec['y'] >= ay0) & (rec['y'] <= ay1))
    x, y, c = rec['x'][m], rec['y'][m], rec['cat'][m]
    near = (c == 4).astype(float)

    # --- Y structure column by column ---------------------------------------
    xe = np.arange(ax0, ax1 + 6, 6.0)
    ye = np.arange(ay0, ay1 + 30, 30.0)
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    ix, iy = np.digitize(x, xe) - 1, np.digitize(y, ye) - 1
    eff = np.full((len(xc), len(yc)), np.nan)
    for i in range(len(xc)):
        for j in range(len(yc)):
            s = (ix == i) & (iy == j)
            if s.sum() >= 10:
                eff[i, j] = near[s].mean()

    cols = []
    for i in range(len(xc)):
        v = eff[i]
        ok = np.isfinite(v)
        if ok.sum() < 5 or np.nanmax(v) < 0.35:
            continue                    # only columns that are live somewhere
        # best split point in Y: maximises |mean(low Y) - mean(high Y)|
        best = (0.0, None)
        for j in range(2, len(yc) - 1):
            a, b = np.nanmean(v[:j]), np.nanmean(v[j:])
            if np.isfinite(a) and np.isfinite(b) and abs(a - b) > best[0]:
                best = (abs(a - b), j, a, b)
        if best[1] is None:
            continue
        step, j, a, b = best
        cols.append(dict(x=float(xc[i]), step=float(step), y_split=float(ye[j]),
                         low_side=float(a), high_side=float(b),
                         dead_side=('high_Y' if b < a else 'low_Y'),
                         span=float(np.nanmax(v) - np.nanmin(v))))
    steps = np.array([c_['step'] for c_ in cols])
    sides = [c_['dead_side'] for c_ in cols if c_['step'] > 0.30]
    splits = np.array([c_['y_split'] for c_ in cols if c_['step'] > 0.30])

    # --- pedestal-free amplitude-threshold profile --------------------------
    ray, _ = amp_threshold_table(key, args.amp_thr)
    e2 = np.arange(ax0, ax1 + 4, 4.0)
    c2 = 0.5 * (e2[:-1] + e2[1:])
    i2 = np.digitize(ray.lx.to_numpy(), e2) - 1
    ok3 = ray.ok3.to_numpy().astype(float)
    prof = np.array([ok3[i2 == k].mean() if (i2 == k).sum() >= 10 else np.nan
                     for k in range(len(c2))])

    rep = dict(
        run_key=key, amp_threshold_adc=args.amp_thr,
        n_columns_live_somewhere=len(cols),
        y_step=dict(median=float(np.median(steps)), p90=float(np.percentile(steps, 90)),
                    n_above_0p30=int((steps > 0.30).sum()),
                    frac_above_0p30=float((steps > 0.30).mean())),
        dead_side_counts={s: sides.count(s) for s in set(sides)} if sides else {},
        y_split_positions=dict(
            median=float(np.median(splits)) if len(splits) else np.nan,
            iqr=[float(np.percentile(splits, 25)), float(np.percentile(splits, 75))]
            if len(splits) else [np.nan, np.nan]),
        amp_profile=dict(
            n_rays=int(len(ray)),
            min=float(np.nanmin(prof)), max=float(np.nanmax(prof)),
            contrast=float(np.nanmax(prof) / max(np.nanmin(prof), 1e-3)),
            frac_bins_below_0p2=float(np.nanmean(prof < 0.2))),
    )
    with open(os.path.join(args.out, f'strip_break_{key}.json'), 'w') as f:
        json.dump(dict(rep, columns=cols), f, indent=1)

    fig, axs = plt.subplots(3, 1, figsize=(14, 11))
    im = axs[0].imshow(eff.T, origin='lower', aspect='auto', vmin=0, vmax=1,
                       extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap='viridis',
                       interpolation='nearest')
    axs[0].set_ylabel('detector-local Y [mm]')
    axs[0].set_title(f'{key} — efficiency, 6 x 30 mm cells (columns are the '
                     f'candidate strips)')
    fig.colorbar(im, ax=axs[0], fraction=.025, pad=.01)

    axs[1].plot([c_['x'] for c_ in cols], steps, 'o', ms=4, color='#0072b2')
    axs[1].axhline(0.30, color='r', ls='--', lw=1,
                   label='step that would mean a break partway along Y')
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('best |low-Y − high-Y| efficiency step')
    axs[1].set_xlabel('detector-local X [mm]')
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=.3)
    axs[1].set_title('per-column Y asymmetry — flat and small means the columns '
                     'are dead (or live) along their whole length')

    axs[2].plot(c2, prof, 'k-', lw=1.2)
    axs[2].set_ylim(0, 1.02)
    axs[2].set_xlabel('detector-local X [mm]')
    axs[2].set_ylabel(f'fraction of rays with ≥3 strips >{args.amp_thr:.0f} ADC '
                      f'on both planes')
    axs[2].grid(alpha=.3)
    axs[2].set_title('pedestal-free cross-check: same bands with a fixed ADC cut '
                     '(no significance, no pedestal)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'strip_break_{key}.png'), dpi=115)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
