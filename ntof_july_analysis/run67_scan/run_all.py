#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver for the whole run_67 analysis, in dependency order. Assumes the per-event
reco cache is already built (process.py). Runs:

  1. feu_presence.build()   per-event FEU readout / liveness flags
  2. flash_timing           PART 1: time-since-flash vs IPC spectrum per threshold
  3. analyze_tracks         PART 2: per_cell_stats CSVs + yield-vs-HV / recovery
  4. detA_2d                PART 2: Det A raw 2-D efficiency (the preferred view)
  5. compare_thresholds     PART 2: threshold comparison + throughput + best pts
  6. slide_plots            PART 3: boxcar-smoothed efficiency vs time-since-flash

NOTE --force-feu: the presence table is derived from combined_hits, so it is
stale whenever the runs are re-decoded (as on 2026-07-24), not just when more
sub-runs are added. attach() left-joins it and fills misses with
readout_*=False, which silently empties efficiency denominators. After ANY
reprocessing or re-reco, run this with --force-feu.

Run: .venv/bin/python ntof_july_analysis/run67_scan/run_all.py [--force-feu]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def main():
    import feu_presence
    print('=== 1/6 feu_presence.build ===', flush=True)
    feu_presence.build(force='--force-feu' in sys.argv)

    print('\n=== 2/6 flash_timing (PART 1) ===', flush=True)
    import flash_timing
    flash_timing.main()

    print('\n=== 3/6 analyze_tracks (PART 2) ===', flush=True)
    import analyze_tracks
    analyze_tracks.main()

    print('\n=== 4/6 detA_2d (PART 2, preferred view) ===', flush=True)
    import detA_2d
    detA_2d.main()

    print('\n=== 5/6 compare_thresholds (PART 2 synthesis) ===', flush=True)
    import compare_thresholds
    compare_thresholds.main()

    print('\n=== 6/6 slide_plots (PART 3, boxcar vs time-since-flash) ===',
          flush=True)
    import slide_plots
    slide_plots.main([])

    # Outputs live under the DATA tree, not the repo — take the path from
    # scan_lib rather than rebuilding it from _HERE (which pointed at a
    # non-existent 'July_HV_Scan' inside the repo).
    import scan_lib
    print('\nDONE. Outputs under', scan_lib.OUT_BASE)


if __name__ == '__main__':
    main()
