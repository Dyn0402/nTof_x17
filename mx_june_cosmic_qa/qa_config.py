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
    # Weekend det2/det3 run (6-19-26), subrun short_run. Ar/Iso 95/5, non-ZS,
    # det_orientation.z=90. NB detector NAMES are mx17_2 (FEU 3/4, bottom z=232) and
    # mx17_3 (FEU 7/8, top z=702) -- different physical dets than the det1/det2 runs.
    # Also has a fine resist_<NNN>V_drift_1000V HV scan (450-525V in 5V steps).
    'wknd_det2':  _Config('wknd_det2', 'mx17_det2_det3_weekend_6-19-26', 'short_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    'wknd_det3':  _Config('wknd_det3', 'mx17_det2_det3_weekend_6-19-26', 'short_run',
                          feus=[7, 8], det_z=702.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    # Same weekend run, the longer 'longer_run' subrun (more stats than short_run).
    'wknd_long_det2': _Config('wknd_long_det2', 'mx17_det2_det3_weekend_6-19-26', 'longer_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    'wknd_long_det3': _Config('wknd_long_det3', 'mx17_det2_det3_weekend_6-19-26', 'longer_run',
                          feus=[7, 8], det_z=702.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    # Overnight det2/det3 run (6-22-26), Ar/Iso 95/5, non-ZS, det_orientation.z=90.
    # NB the detector slots moved again: mx17_3 = FEU 3(X)/4(Y) bottom (z=232), and
    # mx17_2 = FEU 6(X)/8(Y) top (z=702) -- X FEU is 6 here (NOT 7 as in the weekend run).
    # short_run and longer_run share these parameters (combine stats once both processed).
    # Also a resist_525V_drift_1000V pedthr HV point.
    'o22_det3':   _Config('o22_det3', 'mx17_det2_det3_overnight_6-22-26', 'short_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    'o22_det2':   _Config('o22_det2', 'mx17_det2_det3_overnight_6-22-26', 'short_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    'o22_long_det3': _Config('o22_long_det3', 'mx17_det2_det3_overnight_6-22-26', 'longer_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    'o22_long_det2': _Config('o22_long_det2', 'mx17_det2_det3_overnight_6-22-26', 'longer_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    # FLAT-THRESHOLD reprocessing of the 6-22 run: hit threshold = 3 × (per-FEU median
    # pedestal RMS) instead of 5 × per-channel RMS, to recover micro-TPC signal that the
    # noisy/spark-inflated pedestals were suppressing. Reprocessed locally with the
    # WFA_THRESHOLD_SIGMA=3 + WFA_FLAT_SIGMA=<median> build; rays reused. Same det layout
    # as o22_* (mx17_3=FEU3/4 z232, mx17_2=FEU6/8 z702).
    'f3_det3':      _Config('f3_det3', 'mx17_det2_det3_overnight_6-22-26', 'short_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/_flat3_reproc/'),
    'f3_det2':      _Config('f3_det2', 'mx17_det2_det3_overnight_6-22-26', 'short_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/_flat3_reproc/'),
    'f3_long_det3': _Config('f3_long_det3', 'mx17_det2_det3_overnight_6-22-26', 'longer_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/_flat3_reproc/'),
    'f3_long_det2': _Config('f3_long_det2', 'mx17_det2_det3_overnight_6-22-26', 'longer_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/_flat3_reproc/'),
    # Pedestal flat-sigma reprocessing TEST (det2 longer_run file 003 only): nominal vs
    # flat10 reprocessing, to compare efficiency. See _pedestal_test/README.md.
    'pedtest_nom':    _Config('pedtest_nom', 'nominal', 'f003',
                          feus=[3, 4], det_z=232.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/_pedestal_test/qa/'),
    'pedtest_flat10': _Config('pedtest_flat10', 'flat10', 'f003',
                          feus=[3, 4], det_z=232.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/_pedestal_test/qa/'),
    # Daytime det1 single-detector run (1-28-26), NOT zero-suppressed. Subrun overnight_run.
    # FEU 4=X, 6=Y (non-consecutive). Ar/Iso 95/5. X/Y hits well balanced.
    # NB: run_config nominal det_center z = 411, but the M3-frame alignment optimum is
    # ~251 mm (this bench's M3 stations sit at different z than the det_3 runs). Using 251
    # so the default +/-60 z-scan brackets it; alignment -> sub-mm residual (sigma~0.82 mm).
    'day_det1':   _Config('day_det1', 'mx17_det1_daytime_run_1-28-26', 'overnight_run',
                          feus=[4, 6], det_z=251.0, det_name='mx17_1', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det_0/'),

    # 6-23-26 det3_det4 overnight run (Ar/Iso 95/5, non-ZS, det_orientation.z=90).
    # Detector slots (re-read from run_config detectors[]):
    #   mx17_3 = FEU 3/4 (X=3, Y=4) bottom z=232, resist HV 495 V
    #   mx17_4 = FEU 6/8 (X=6, Y=8) top    z=702, resist HV 455 V
    # longer_run analysed first (only subrun processed at pull time; indices 000+001,
    # the pairs with both combined_hits and rays). Has a 465->525 V resist HV scan +
    # 48 h long_run still in the backlog.
    'o23_long_det3': _Config('o23_long_det3', 'mx17_det3_det4_overnight_6-23-26', 'longer_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det3_det4/'),
    'o23_long_det4': _Config('o23_long_det4', 'mx17_det3_det4_overnight_6-23-26', 'longer_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_4', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det3_det4/'),

    # --- GRAND JUNE COMPILATION (6-26 session) -------------------------------
    # One best high-stats decoded subrun per physical detector (2,3,4,6,7; no 5),
    # for the final per-detector overview PDF. Slot mapping is consistent across all
    # June runs: bottom = FEU 3(X)/4(Y) z=232, top = FEU 6(X)/8(Y) z=702.
    #
    # det2 & det3 share the 6-22 overnight 'long_run' (16 file-pairs, ~53 GB, most
    # stats; M3 reference healthy on this run -- the short/longer subruns gave clean
    # ~42% efficiency). One pull of long_run feeds BOTH detectors.
    'g_det2': _Config('g_det2', 'mx17_det2_det3_overnight_6-22-26', 'long_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_2', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    'g_det3': _Config('g_det3', 'mx17_det2_det3_overnight_6-22-26', 'long_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det2_det3/'),
    # Weekend det3 (6-28) — highest-stats / best det3 (top slot, FEU7/8, z702), pulled from
    # lxplus. Lives in mx17_det3_p2_det1_overnight_6-27-26 next to a non-mx17 "P2_1" detector
    # (ignored). Same DET_NAME (mx17_3) as g_det3, so build_final_pdf.select_keys picks the
    # one with more rays -> this weekend run (52,995 rays) wins the Detector-3 page.
    'g_det3_wknd': _Config('g_det3_wknd', 'mx17_det3_p2_det1_overnight_6-27-26',
                          'long_run_p2_det1_sanity_check',
                          feus=[7, 8], det_z=702.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det3/'),
    # Saturday det3 scan run (6-27): ~7 h long run at operating point (resist 490 V /
    # drift 1000 V) + drift scan (100-1100 V, 15 min each) + two resist scans. det3 in
    # the TOP slot, FEU 7(X)/8(Y), z=702. The long run seeds the alignment for the scans.
    'sat_det3': _Config('sat_det3', 'mx17_det3_saturday_scan_6-27-26',
                          'long_run_resist_490V_drift_1000V',
                          feus=[7, 8], det_z=702.0, det_name='mx17_3', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det3/'),
    # det4: dedicated 6-24 daytime run 'long_run' (8 file-pairs, ~26 GB). Avoids the
    # 6-23 det3_det4 run whose M3 reference is degraded (~4% clean tracks).
    'g_det4': _Config('g_det4', 'mx17_det4_day_6-24-26', 'long_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_4', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det4_day/'),
    # det6 & det7: 6-26 overnight run. NOT decoded at session start (raw_daq_data only);
    # the orchestrator polls rays for combined_hits_root/m3_tracking_root and processes
    # once decoding lands. short_run is the subrun being recorded; larger subruns, if
    # they appear, can be swapped in.
    'g_det6': _Config('g_det6', 'mx17_det6_det7_overnight_6-26-26', 'short_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_6', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det6_det7/'),
    'g_det7': _Config('g_det7', 'mx17_det6_det7_overnight_6-26-26', 'short_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_7', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det6_det7/'),
    # det6/det7 additional subruns of the same 6-26 run, analysed as each one finishes
    # decoding overnight. Same slot map. The PDF auto-picks the best-stats subrun per
    # detector, so all of these can be fed to build_final_pdf.py together.
    'g_det6_longer': _Config('g_det6_longer', 'mx17_det6_det7_overnight_6-26-26', 'longer_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_6', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det6_det7/'),
    'g_det7_longer': _Config('g_det7_longer', 'mx17_det6_det7_overnight_6-26-26', 'longer_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_7', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det6_det7/'),
    'g_det6_long': _Config('g_det6_long', 'mx17_det6_det7_overnight_6-26-26', 'long_run',
                          feus=[3, 4], det_z=232.0, det_name='mx17_6', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det6_det7/'),
    'g_det7_long': _Config('g_det7_long', 'mx17_det6_det7_overnight_6-26-26', 'long_run',
                          feus=[6, 8], det_z=702.0, det_name='mx17_7', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det6_det7/'),

    # Dedicated 6-26 det6+det7 resist-HV scan (mx17_det6_det7_hv_scan_6-26-26): det6 AND
    # det7 stepped TOGETHER at the SAME voltage, 400->500 V in 5 V steps (drift 700 V) --
    # the lower-range re-scan (the overnight scan started too high). This run has NO
    # long_run subrun, so 10_hv_scan_efficiency.py must be run with
    #   --seed=<6-26 OVERNIGHT long_run alignment for the same det>
    # (same detectors/bench geometry). sub_run is unused by the HV-scan script.
    'g_det6_hv': _Config('g_det6_hv', 'mx17_det6_det7_hv_scan_6-26-26', 'hv_scan',
                          feus=[3, 4], det_z=232.0, det_name='mx17_6', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det6_det7/'),
    'g_det7_hv': _Config('g_det7_hv', 'mx17_det6_det7_hv_scan_6-26-26', 'hv_scan',
                          feus=[6, 8], det_z=702.0, det_name='mx17_7', zero_suppressed=False,
                          base_path='/home/dylan/x17/cosmic_bench/det6_det7/'),
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
