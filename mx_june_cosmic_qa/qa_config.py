#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_config.py

Central configuration + import-path setup for the June cosmic-bench QA.

Run registry
------------
Multiple runs are registered in RUNS, keyed by a short name.  Scripts pick a
run with `cfg = config_from_argv()` (first CLI arg = run key, default below) or
`get_config('<key>')`.  The heavy lifting (strip mapping, M3 tracking,
micro-TPC analysis) is reused directly from ``cosmic_bench_analysis/`` — we only
reconfigure it.

Registered runs (all on det_3 bench area):
  det3_ariso   mx17_det3_ArIso_Test_6-16-26 / run        ZERO-SUPPRESSED (tpc).
               FEU 7=X, 8=Y, det z=702.  M3 tracking is poor here (ZS eats the
               M3 reference-detector signals) → alignment fails.  See memory.
  zs_initial   zs_compression_scan_4_6-6-26 / initial_run  NOT zero-suppressed.
               FEU 3=X, 4=Y, det z=232.
  long_run     mx17_det3_long_run_5-6-26 / long_run         NOT zero-suppressed.
               FEU 3=X, 4=Y, det z=232.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
CBA_DIR = os.path.join(REPO_ROOT, 'cosmic_bench_analysis')

DEFAULT_RUN = 'det3_ariso'


def setup_paths() -> None:
    """Put the repo root and cosmic_bench_analysis/ on sys.path for imports."""
    for p in (REPO_ROOT, CBA_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


class _Config:
    def __init__(self, key, run, sub_run, feus, det_z,
                 base_path='/home/dylan/x17/cosmic_bench/det_3/',
                 det_name='mx17_1', zero_suppressed=False):
        self.KEY = key
        self.BASE_PATH = base_path
        self.RUN = run
        self.SUB_RUN = sub_run
        self.DET_NAME = det_name
        self.MX17_FEUS = list(feus)              # [X_feu, Y_feu]
        self.MX17_FEU_X, self.MX17_FEU_Y = feus
        self.DET_PLANE_Z = det_z                 # mx17_1 det_center z [mm]
        self.ZERO_SUPPRESSED = zero_suppressed
        self.MAP_CSV_PATH = os.path.join(REPO_ROOT, 'mx17_m1_map.csv')
        # All QA output goes to a single top-level Analysis/ tree under cosmic_bench
        # (kept separate from the data, which lives under the per-area dirs), keyed by
        # run + sub_run + detector so subruns of the same run don't collide.
        # cosmic_bench root = parent of base_path (e.g. .../cosmic_bench/det1_det2/ ->
        # .../cosmic_bench).
        cosmic_bench_root = os.path.dirname(base_path.rstrip('/'))
        self.OUT_BASE = os.path.join(cosmic_bench_root, 'Analysis', run, sub_run, det_name)

    @property
    def run_config_path(self):
        return f'{self.BASE_PATH}{self.RUN}/run_config.json'

    @property
    def combined_hits_dir(self):
        return f'{self.BASE_PATH}{self.RUN}/{self.SUB_RUN}/combined_hits_root/'

    @property
    def m3_tracking_dir(self):
        return f'{self.BASE_PATH}{self.RUN}/{self.SUB_RUN}/m3_tracking_root/'

    def out_dir(self, *parts):
        d = os.path.join(self.OUT_BASE, *parts)
        os.makedirs(d, exist_ok=True)
        return d


RUNS = {
    'det3_ariso': _Config('det3_ariso', 'mx17_det3_ArIso_Test_6-16-26', 'run',
                          feus=[7, 8], det_z=702.0, zero_suppressed=True),
    'zs_initial': _Config('zs_initial', 'zs_compression_scan_4_6-6-26', 'initial_run',
                          feus=[3, 4], det_z=232.0, zero_suppressed=False),
    'long_run':   _Config('long_run', 'mx17_det3_long_run_5-6-26', 'long_run',
                          feus=[3, 4], det_z=232.0, zero_suppressed=False),
    # Two-detector overnight run (6-17-26), NOT zero-suppressed. Subrun longer_run.
    # det1 = noisy (huge occupancy), det2 = expected inefficient.
    'ovn_det1':   _Config('ovn_det1', 'mx17_det1_det2_overnight_6-17-26', 'longer_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_1', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det1_det2/'),
    'ovn_det2':   _Config('ovn_det2', 'mx17_det1_det2_overnight_6-17-26', 'longer_run',
                          feus=[7, 8], det_z=702.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det1_det2/'),
    # Same overnight run, the longer 'long_run' subrun (~10x stats vs longer_run).
    'long_det1':  _Config('long_det1', 'mx17_det1_det2_overnight_6-17-26', 'long_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_1', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det1_det2/'),
    'long_det2':  _Config('long_det2', 'mx17_det1_det2_overnight_6-17-26', 'long_run',
                          feus=[7, 8], det_z=702.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det1_det2/'),
    # New short run (6-18-26), subrun short_run. Same det layout, Ar/Iso, non-ZS.
    'short_det1': _Config('short_det1', 'mx17_det1_det2_short_6-18-26', 'short_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_1', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det1_det2/'),
    'short_det2': _Config('short_det2', 'mx17_det1_det2_short_6-18-26', 'short_run',
                          feus=[7, 8], det_z=702.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det1_det2/'),
    # Daytime det1 single-detector run (1-28-26), NOT zero-suppressed. Subrun overnight_run.
    # FEU 4=X, 6=Y (non-consecutive). Ar/Iso 95/5. X/Y hits well balanced.
    # NB: run_config nominal det_center z = 411, but the M3-frame alignment optimum is
    # ~251 mm (this bench's M3 stations sit at different z than the det_3 runs). Using 251
    # so the default +/-60 z-scan brackets it; alignment -> sub-mm residual (sigma~0.82 mm).
    'day_det1':   _Config('day_det1', 'mx17_det1_daytime_run_1-28-26', 'overnight_run',
                          feus=[4, 6], det_z=251.0, det_name='mx17_1', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det_0/'),
}


def get_config(key):
    if key not in RUNS:
        raise KeyError(f'Unknown run key {key!r}. Options: {list(RUNS)}')
    return RUNS[key]


def config_from_argv():
    """First non-flag CLI arg selects the run key; otherwise DEFAULT_RUN."""
    key = next((a for a in sys.argv[1:] if not a.startswith('-')), DEFAULT_RUN)
    return get_config(key)


# Backwards-compatible default
CFG = RUNS[DEFAULT_RUN]
