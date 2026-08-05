#!/usr/bin/env python3
"""
03_angles.py — angular resolution and angle bias against the M3 reference.

Two things are measured, and the second is the one that matters:

1. **Resolution/bias**: per-event reconstructed angle minus reference angle,
   per plane. The physics floor is ~1 deg (diffusion and charge granularity,
   measured by toy closure in WAVEFORM_FIRST_THREADING.md §12) — do not read a
   result below that as an improvement.
2. **Implied-v flatness**: median (w / tan_ref) in bins of |tan_ref|. A
   reconstruction that is geometrically honest gives the *same* drift velocity
   at every angle. The hits ladder does not: it falls 56 -> 39 um/ns across the
   angle range, which is the compression signature. This is the test that
   catches a chain that has quietly reacquired the old bias.

Planes with |tan| < 0.08 carry no slope information (``slope_reliable``) and are
excluded from both — including them is how a bias sneaks back in.

    ../.venv/bin/python mx_june_wft/03_angles.py <run_key>
Outputs: <OUT_BASE>/wft/angles/{angular_resolution.json, angles.png}
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import matplotlib                                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
import cosmic_micro_tpc_analysis as cm                   # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles   # noqa: E402
from wft import compat                                   # noqa: E402

ANGLE_BINS = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]


def robust_sigma(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--table', default=None)
    ap.add_argument('--alignment', default=None)
    ap.add_argument('--out', default=None,
                    help='output dir (default: the standard wft/angles)')
    args = ap.parse_args()

    cfg = get_config(args.run_key)
    table = args.table or os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    align_path = args.alignment or os.path.join(cfg.OUT_BASE, 'wft', 'alignment',
                                                'alignment.json')
    out_dir = args.out or cfg.out_dir('wft', 'angles')
    os.makedirs(out_dir, exist_ok=True)

    params = cm.load_alignment(align_path)
    df = compat.load_table(table)
    meta = compat.table_meta(table)
    v_cal = meta.get('bundle', {}).get('v_drift', np.nan)

    results = compat.as_event_results(df)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)

    # rotate the reference tangents into the detector's raw strip frame, exactly
    # as the hits chain does (the alignment rotation is ~90 deg here)
    ref = {}
    for r in results:
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, params)
        ref[int(r.event_id)] = (tx, ty)

    idx = df.set_index('event_id')
    summary = {'run_key': args.run_key, 'v_cal_um_ns': v_cal,
               'basis': 'waveform-first (wft)', 'planes': {}}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for i, plane in enumerate(('x', 'y')):
        eids = [e for e in idx.index if e in ref and idx.loc[e, f'{plane}_ok']]
        tan_ref = np.array([ref[e][0 if plane == 'x' else 1] for e in eids])
        tan_fit = idx.loc[eids, f'{plane}_tan_theta'].to_numpy()
        rel = idx.loc[eids, f'{plane}_slope_reliable'].to_numpy().astype(bool)
        w = idx.loc[eids, f'{plane}_w'].to_numpy()

        use = rel & np.isfinite(tan_fit) & np.isfinite(tan_ref)
        dth = (np.degrees(np.arctan(tan_fit[use]))
               - np.degrees(np.arctan(tan_ref[use])))
        sig, med = robust_sigma(dth), float(np.median(dth))
        s68 = float(np.percentile(np.abs(dth - med), 68)) if len(dth) else np.nan

        ax = axs[0, i]
        ax.hist(dth, bins=np.linspace(-8, 8, 100), histtype='step', lw=2,
                label=f'median {med:+.2f}, sigma {sig:.2f}, s68 {s68:.2f} deg')
        ax.set_xlabel('reconstructed - reference angle [deg]')
        ax.set_title(f'{plane}: per-event angle residual (n={use.sum():,})')
        ax.legend(fontsize=8)
        ax.axvline(0, color='gray', lw=0.8)

        # implied v vs angle — flat means geometrically honest
        at = np.abs(tan_ref)
        vimp = w * 1e3 / tan_ref
        ctr, medv, errv = [], [], []
        for lo, hi in ANGLE_BINS:
            m = use & (at >= lo) & (at < hi)
            ctr.append(0.5 * (lo + hi))
            medv.append(float(np.nanmedian(vimp[m])) if m.sum() else np.nan)
            errv.append(robust_sigma(vimp[m]) / max(np.sqrt(m.sum()), 1)
                        if m.sum() else np.nan)
        ax = axs[1, i]
        ax.errorbar(ctr, medv, yerr=errv, fmt='o-', capsize=3,
                    label='waveform-first')
        if np.isfinite(v_cal):
            ax.axhline(v_cal, color='k', ls='--', lw=1,
                       label=f'calibration v = {v_cal:.1f}')
        ax.set_xlabel('|tan(theta)| reference')
        ax.set_ylabel('median w / tan_ref [um/ns]')
        ax.set_title(f'{plane}: implied drift velocity vs angle (flat = physical)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        spread = (float(np.nanmax(medv) - np.nanmin(medv))
                  if np.isfinite(medv).any() else np.nan)
        summary['planes'][plane] = dict(
            n=int(use.sum()), bias_deg=med, sigma_deg=sig, s68_deg=s68,
            implied_v=medv, implied_v_bins=[list(b) for b in ANGLE_BINS],
            implied_v_spread=spread,
            frac_slope_reliable=float(np.mean(rel)) if len(rel) else np.nan)
        print(f'{plane}: n={use.sum():,}  bias {med:+.2f} deg  sigma {sig:.2f} deg  '
              f's68 {s68:.2f}  implied-v spread {spread:.2f} um/ns  '
              f'(slope_reliable {100*np.mean(rel):.0f} %)')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'angles.png'), dpi=110)
    with open(os.path.join(out_dir, 'angular_resolution.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nwrote {out_dir}')


if __name__ == '__main__':
    main()
