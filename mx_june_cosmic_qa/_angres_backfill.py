#!/usr/bin/env python3
"""Backfill alignment_tpc_veto50/angular_resolution.json for a key whose 03 --full
ran before the angular-resolution save was added to plot_angle_correlation.
Loads the (already-regenerated) veto50 cache + v2 rays + alignment and re-runs the
micro-TPC angle correlation, which now writes the JSON. Usage: _angres_backfill.py <key>"""
import os, sys, pickle
import numpy as np
sys.argv = ['x', sys.argv[1]]
from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

out_dir = CFG.out_dir('alignment_tpc_veto50')
cache = os.path.join(CFG.out_dir('cache'), 'event_results_veto50.pkl')
results = pickle.load(open(cache, 'rb'))
best = cm.load_alignment(os.path.join(out_dir, 'alignment.json'))
rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=5.0)
xa, ya, evn = get_xy_angles(rays.ray_data)
xa = best.ref_x_sign * np.array(xa)
cm.attach_reference_positions(results, rays, best, xa, evn)
cm.plot_angle_correlation(results, residual_cut_mm=10.0, min_strips=4, max_red_chi2=None,
                          v_scan_min=25.0, v_scan_max=50.0, v_scan_steps=51,
                          out_dir=out_dir, params=best)
print('angular_resolution.json ->', os.path.join(out_dir, 'angular_resolution.json'))
