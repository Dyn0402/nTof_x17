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

Coverage (changed 2026-08-13): the residual/bias accounting uses EVERY ok
plane — the old ``slope_reliable`` gate (|tan| >= 0.08) was a hits-chain
inheritance (the time-ladder angle has no lever arm head-on; June filled that
band with the 33/34 signature-hybrid). The forward fit measures the head-on
band natively: on det3-golden it is unbiased (|bias| <= 0.15 deg) at the same
sigma68 as the inclined bands with 88-97 % sign fidelity, while the gate was
masking 37-44 % of reconstructed planes (JUNE_CONTINUITY_2026-08-13.md §5b).
The JSON also reports the |theta| < 5 deg band (``s68_lt5_deg`` — the June
hybrid's headline convention) and the old gated numbers (``*_relonly``) for
continuity. **Implied-v keeps the |tan_ref| >= 0.08 bins**: it divides by
tan_ref, so the head-on band genuinely carries no velocity information there
— that part of the old caveat stands.

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

    # duplicate event ids (multi-datrun collisions) crash .loc lookups and are
    # untrustworthy rows anyway — keep neither copy
    dup = df['event_id'].duplicated(keep=False)
    if dup.any():
        print(f'WARNING: dropping {int(dup.sum())} rows with duplicate '
              f'event ids ({df.loc[dup, "event_id"].nunique()} ids)')
        df = df[~dup]
    idx = df.set_index('event_id')
    summary = {'run_key': args.run_key, 'v_cal_um_ns': v_cal,
               'basis': 'waveform-first (wft)',
               'coverage': 'full — slope_reliable not gated (2026-08-13); '
                           'implied-v unchanged (|tan_ref| >= 0.08 bins)',
               'planes': {}}
    LT5 = 0.0875                              # tan(5 deg)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for i, plane in enumerate(('x', 'y')):
        eids = [e for e in idx.index if e in ref and idx.loc[e, f'{plane}_ok']]
        tan_ref = np.array([ref[e][0 if plane == 'x' else 1] for e in eids])
        tan_fit = idx.loc[eids, f'{plane}_tan_theta'].to_numpy()
        rel = idx.loc[eids, f'{plane}_slope_reliable'].to_numpy().astype(bool)
        w = idx.loc[eids, f'{plane}_w'].to_numpy()

        finite = np.isfinite(tan_fit) & np.isfinite(tan_ref)
        use = finite                              # full coverage (see docstring)
        use_rel = rel & finite                    # old gated selection
        dth_all = (np.degrees(np.arctan(tan_fit[use]))
                   - np.degrees(np.arctan(tan_ref[use])))
        sig, med = robust_sigma(dth_all), float(np.median(dth_all))
        s68 = (float(np.percentile(np.abs(dth_all - med), 68))
               if len(dth_all) else np.nan)
        dth_rel = (np.degrees(np.arctan(tan_fit[use_rel]))
                   - np.degrees(np.arctan(tan_ref[use_rel])))
        sig_rel = robust_sigma(dth_rel)
        med_rel = float(np.median(dth_rel)) if len(dth_rel) else np.nan
        lt5 = use & (np.abs(tan_ref) < LT5)
        dth5 = (np.degrees(np.arctan(tan_fit[lt5]))
                - np.degrees(np.arctan(tan_ref[lt5])))
        med5 = float(np.median(dth5)) if len(dth5) else np.nan
        s68_5 = (float(np.percentile(np.abs(dth5 - med5), 68))
                 if len(dth5) else np.nan)

        ax = axs[0, i]
        ax.hist(dth_all, bins=np.linspace(-8, 8, 100), histtype='step', lw=2,
                label=f'full coverage: median {med:+.2f}, sigma {sig:.2f}, '
                      f's68 {s68:.2f} deg')
        ax.hist(dth5, bins=np.linspace(-8, 8, 100), histtype='step', lw=1.4,
                ls='--',
                label=f'|theta|<5 deg: s68 {s68_5:.2f} deg (n={lt5.sum():,})')
        ax.set_xlabel('reconstructed - reference angle [deg]')
        ax.set_title(f'{plane}: per-event angle residual (n={use.sum():,}, '
                     f'full coverage)')
        ax.legend(fontsize=8)
        ax.axvline(0, color='gray', lw=0.8)

        # implied v vs angle — flat means geometrically honest
        at = np.abs(tan_ref)
        vimp = w * 1e3 / tan_ref
        ctr, medv, errv = [], [], []
        for lo, hi in ANGLE_BINS:
            m = use_rel & (at >= lo) & (at < hi)   # implied-v: old gated basis
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
            n_lt5=int(lt5.sum()), bias_lt5_deg=med5, s68_lt5_deg=s68_5,
            n_relonly=int(use_rel.sum()), bias_deg_relonly=med_rel,
            sigma_deg_relonly=sig_rel,
            implied_v=medv, implied_v_bins=[list(b) for b in ANGLE_BINS],
            implied_v_spread=spread,
            frac_slope_reliable=float(np.mean(rel)) if len(rel) else np.nan)
        print(f'{plane}: n={use.sum():,}  bias {med:+.2f} deg  sigma {sig:.2f} deg  '
              f's68 {s68:.2f}  |t|<5deg s68 {s68_5:.2f} (n={lt5.sum():,})  '
              f'implied-v spread {spread:.2f} um/ns  '
              f'(gated would keep {100*np.mean(rel):.0f} %: '
              f'bias {med_rel:+.2f} sigma {sig_rel:.2f})')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'angles.png'), dpi=110)
    with open(os.path.join(out_dir, 'angular_resolution.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nwrote {out_dir}')


if __name__ == '__main__':
    main()
