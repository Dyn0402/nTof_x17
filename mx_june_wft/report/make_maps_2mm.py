#!/usr/bin/env python3
"""
make_maps_2mm.py — spatial efficiency maps with the tight r < 2 mm success
criterion, on the campaign waveform-first reconstruction.

The standard 04_maps.py writes any-hit and r < 10 mm maps; 2 mm is the cut
that isolates the core (fleet core sigma is 0.43-0.62 mm, so 2 mm accepts the
core and rejects the tail), and is the map the MPGD26 report leads with.
Binned 40x40 like 04_maps.py (~12 mm pitch) — a literal 2 mm *kernel* would
hold ~0.05 rays and be pure noise at cosmic statistics.

    ../../.venv/bin/python mx_june_wft/report/make_maps_2mm.py [keys...]

Outputs <OUT_BASE>/wft/maps/efficiency_r_2_mm_waveform_first.png (+ CSV in
Plot_Data/) per key, same conventions as 04_maps.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis'),
                os.path.join(REPO, 'mx_june_wft')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import matplotlib                                        # noqa: E402
matplotlib.use('Agg')
import cosmic_micro_tpc_analysis as cm                   # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles   # noqa: E402
from common.Mx17StripMap import RunConfig                # noqa: E402
from wft import compat                                   # noqa: E402
from fleet_state import FLEET                            # noqa: E402


def one(key):
    cfg = get_config(key)
    table = os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    align_path = os.path.join(cfg.OUT_BASE, 'wft', 'alignment',
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
    cm.plot_efficiency_map(results, rays, params, bins=40,
                           min_tracks_per_bin=5, radius_cut_mm=2.0,
                           title='Efficiency (r<2 mm) — waveform-first',
                           csv_out_path=os.path.join(csv_dir,
                                                     'efficiency_r2mm_cut.csv'),
                           active_region=active, out_dir=out_dir,
                           det_name=cfg.DET_NAME)
    print(f'{key}: wrote {out_dir}/efficiency_r_2_mm_waveform_first.png')


def main():
    keys = sys.argv[1:] or FLEET
    for key in keys:
        one(key)


if __name__ == '__main__':
    main()
