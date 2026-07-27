#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run30_flash_compare.py

run_30 flash blocks: mesh-injection ON vs OFF, interleaved at every HV point
(2026-07-11). Thin wrapper around compare_scans.py with the coarse time
windows REALIGNED for run_30's readout timing (400 smp x 20 ns, latency 60,
PS/flash trigger): the flash rise sits at ~1.4-1.6 us in the window (measured
from the combined hits), not at ~0.8 us as in runs 19-22.

Windows: one pre-flash baseline window, one fully containing the
injection+flash complex, then three post-flash recovery windows.

Output -> July_HV_Scan/run30_flash_mesh/  (flask "Analysis" tab).

Run:  ~/PycharmProjects/nTof_x17/.venv/bin/python ntof_july_analysis/run30_flash_compare.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import july_hv_scan as jhs  # noqa: E402

# Realign the coarse windows (ns) BEFORE anything computes metrics. In-place so
# every module that did `from july_hv_scan import TIME_WINDOWS` sees the change.
RUN30_EDGES_NS = [0, 1200, 2600, 4400, 6200, 8000]
#                  ^pre    ^flash ^post1 ^post2 ^post3
jhs.TIME_WINDOWS[:] = list(zip(RUN30_EDGES_NS[:-1], RUN30_EDGES_NS[1:]))

import compare_scans as cs  # noqa: E402  (binds the same TIME_WINDOWS list)

cs.SERIES[:] = [
    {'label': 'run_30 flashOn (mesh injection ON)',  'run': 'run_30', 'match': r'^flashOn_A'},
    {'label': 'run_30 flashOff (mesh injection OFF)', 'run': 'run_30', 'match': r'^flashOff_A'},
]
cs.OUT_LABEL = 'run30_flash_mesh'
cs.MID_WINDOW = (2600, 4400)      # first post-flash window = the turn-off row

if __name__ == '__main__':
    cs.main()
