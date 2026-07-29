#!/usr/bin/env python3
"""
04_maps.py — efficiency and resolution maps on the waveform-first reconstruction.

Reuses the map builders from cosmic_micro_tpc_analysis (they consume positions,
not hit times). The sliding-kernel resolution map is the one that feeds the
`sliding_within` number in the baseline digest.

    ../.venv/bin/python mx_june_wft/04_maps.py <run_key>
Outputs: <OUT_BASE>/wft/maps/
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
import cosmic_micro_tpc_analysis as cm                   # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles   # noqa: E402
from common.Mx17StripMap import RunConfig                # noqa: E402
from wft import compat                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--table', default=None)
    ap.add_argument('--alignment', default=None)
    ap.add_argument('--grid', type=int, default=60,
                    help='sliding-map grid points per axis (hits chain used 100)')
    args = ap.parse_args()

    cfg = get_config(args.run_key)
    table = args.table or os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    align_path = args.alignment or os.path.join(cfg.OUT_BASE, 'wft', 'alignment',
                                                'alignment.json')
    out_dir = cfg.out_dir('wft', 'maps')

    params = cm.load_alignment(align_path)
    df = compat.load_table(table)
    results = compat.as_event_results(df)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, _ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)

    rc = RunConfig(cfg.run_config_path, cfg.MAP_CSV_PATH)
    det = rc.get_detector(cfg.DET_NAME)
    (xmn, xmx), (ymn, ymx) = cm.get_active_det_bounds(det, cfg.MAP_CSV_PATH)
    cx, cy = cm._det_to_ref(np.array([xmn, xmx, xmn, xmx]),
                            np.array([ymn, ymn, ymx, ymx]), params)
    active = (float(cx.min()), float(cx.max()), float(cy.min()), float(cy.max()))

    csv_dir = cfg.out_dir('wft', 'maps', 'Plot_Data')
    for rc_cut, name, title in [
            (None, 'efficiency_no_cut.csv', 'Efficiency (any hit) — waveform-first'),
            (10.0, 'efficiency_r10mm_cut.csv', 'Efficiency (r<10 mm) — waveform-first')]:
        cm.plot_efficiency_map(results, rays, params, bins=40,
                               min_tracks_per_bin=5, radius_cut_mm=rc_cut,
                               title=title,
                               csv_out_path=os.path.join(csv_dir, name),
                               active_region=active, out_dir=out_dir,
                               det_name=cfg.DET_NAME)
    cm.plot_resolution_map(results, rays, params, bins=20, min_hits_per_bin=20,
                           radius_cut_mm=None, out_dir=out_dir,
                           det_name=cfg.DET_NAME)
    # 60x60 rather than the hits chain's 100x100: this stage is serial and cost
    # ~20 min per detector at 100, which is most of a fleet chain's wall clock.
    # The kernel is 50 mm, so a 60-point grid still oversamples it; only the
    # picture is coarser. Use --grid 100 if you need the old sampling.
    cm.plot_resolution_map_sliding(results, grid_points=args.grid,
                                   kernel_radius_mm=50.0, min_hits=50,
                                   out_dir=out_dir, sigma_vmax=1.0,
                                   params=params, det_name=cfg.DET_NAME)

    # the sliding map writes its own json next to the figure if available
    for cand in ('efficiency_map_sliding.json', 'resolution_map_sliding.json'):
        p = os.path.join(out_dir, cand)
        if os.path.exists(p):
            print(f'{cand}: {json.load(open(p))}')
    print(f'wrote {out_dir}')


if __name__ == '__main__':
    main()
