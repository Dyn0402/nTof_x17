#!/usr/bin/env python3
"""
01_alignment.py — align the detector to the M3 reference using the
waveform-first reconstruction.

Same geometry procedure as the hits chain (03_alignment_and_tpc.py): iterate
z -> in-plane rotation -> translation, on the clean-cluster subset, then attach
reference positions and measure residuals. What changed is the *input*: the
detector's position in each event is now the fitted track position at the mesh
(wft), not the earliest-hit strip.

The alignment scans themselves are reused from cosmic_micro_tpc_analysis — they
act on positions, never on hit times (see RECONSTRUCTION_BASIS.md).

    ../.venv/bin/python mx_june_wft/01_alignment.py <run_key> [--table PATH]
                                                    [--maxdrop N] [--rot0 DEG]
Outputs: <OUT_BASE>/wft/alignment/{alignment.json, *.png}
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import cosmic_micro_tpc_analysis as cm                    # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles    # noqa: E402
from common.Mx17StripMap import RunConfig                 # noqa: E402
from wft import compat                                    # noqa: E402

Z_SCAN = np.arange(600.0, 820.0, 2.0)
CENTRE_XY = 200.0
REF_X_SIGN = +1.0
N_ITER = 3


def default_table(cfg):
    return os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--table', default=None)
    ap.add_argument('--maxdrop', type=int, default=2,
                    help='max strips in competing clusters for alignment events')
    ap.add_argument('--rot0', type=float, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = get_config(args.run_key)
    table = args.table or default_table(cfg)
    out_dir = args.out or cfg.out_dir('wft', 'alignment')

    df = compat.load_table(table)
    meta = compat.table_meta(table)
    print(f'{len(df):,} reconstructed events from {table}')
    print(f'  calibration: {meta.get("bundle", {}).get("v_drift", "?")} um/ns, '
          f'{meta.get("bundle", {}).get("provenance", {}).get("code_commit", "?")}')

    results = compat.as_event_results(df)
    n_both = sum(r.has_both for r in results)
    print(f'  {n_both:,} events with both planes reconstructed')

    rc = RunConfig(cfg.run_config_path, cfg.MAP_CSV_PATH)
    det = rc.get_detector(cfg.DET_NAME)
    rot0 = args.rot0 if args.rot0 is not None else float(
        det.orientation.get('z', 0.0) or 0.0)
    print(f'  base rotation {rot0:.2f} deg')

    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xang, yang, evn = get_xy_angles(rays.ray_data)
    xang = REF_X_SIGN * np.array(xang)

    align = [r for r in results if r.has_both and
             (r.x_fit.n_dropped + r.y_fit.n_dropped) <= args.maxdrop]
    print(f'  {len(align):,} clean events used to determine the alignment')

    initial = cm.AlignmentParams(z_x=cfg.DET_PLANE_Z, z_y=cfg.DET_PLANE_Z,
                                 theta_deg=rot0, centre_x=CENTRE_XY,
                                 centre_y=CENTRE_XY, ref_x_sign=REF_X_SIGN)
    best = cm.run_alignment(align, rays, initial_params=initial,
                            n_iterations=N_ITER, z_values=Z_SCAN,
                            theta_values=np.linspace(rot0 - 2.0, rot0 + 2.0, 81),
                            plot_each=False, plot_final=True,
                            mask_to_active_region=False, out_dir=out_dir)
    cm.save_alignment(best, os.path.join(out_dir, 'alignment.json'))

    cm.attach_reference_positions(results, rays, best, xang, evn)
    cm.plot_position_correlation(results, out_dir=out_dir)
    cm.plot_radial_residuals(results, radius_cut_mm=10.0, out_dir=out_dir)
    fit_x, fit_y = cm.plot_residuals(results, out_dir=out_dir)
    if fit_x:
        print(f'X resolution: {fit_x.resolution:.3f} +/- {fit_x.resolution_err:.3f} mm')
    if fit_y:
        print(f'Y resolution: {fit_y.resolution:.3f} +/- {fit_y.resolution_err:.3f} mm')

    # the reconstruction already knows v; report the alignment-implied one as a
    # cross-check only (it is not used to define the angles any more)
    print(f'\nAlignment written to {out_dir}/alignment.json')


if __name__ == '__main__':
    main()
