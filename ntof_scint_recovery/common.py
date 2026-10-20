"""Shared constants for the scintillator flash-recovery study."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths  # noqa: E402

RUN = 224572
#: raw stream1 heads pulled 2026-07-29 (FINDINGS_2026-07-28_psa_optimization.md),
#: 450 MB each, ~2 complete bunches per file
RAW_DIR = Path(os.environ.get('X17_RAW_224572', '/media/dylan/data/x17/ntof_raw_224572'))
RAW_FILES = [RAW_DIR / f'head_{n}.bin' for n in (8, 20, 40, 60, 80, 100, 120)]
#: our v12 PSA reprocessing of the same run, every hit of all 3018 bunches
PSA_DIR = Path(os.environ.get('X17_PSA_224572', '/media/dylan/data/x17/ntof_reproc/v12_liqpileup'))

MV_PER_ADC = 0.0306      # every scintillator channel, ntof_scint_stack.extract
#: bunches above this PulseIntensity are dedicated (~8.5e12), below parasitic (~4.1e12)
DEDICATED_PPP = 6e12
FAMILIES = ('WAL', 'PSS', 'LIQ')
DETS = tuple(f'{f}{a}' for f in FAMILIES for a in 'ABCD')

#: t_flash(det) = tof_PKUP + C.  Walls from the divert-off calibration
#: (flash_timing/), plastics and liquids are run 224572's own measured offsets
#: (flash_timing/data/plastic_liq_flash_by_run.csv).  Good to ~10 ns, which is
#: far below anything this study resolves.
_FT = REPO / 'ntof_processing/flash_timing/data'
C_NS = {k: v['C_ns'] for k, v in json.loads(
    (_FT / 'flash_timing_calibration.json').read_text())['constants'].items()
    if k.startswith('WAL')}
C_NS.update({'PSSA': -1679.49, 'PSSB': -1690.64, 'PSSC': -1682.14, 'PSSD': -1682.68,
             'LIQA': -1707.91, 'LIQB': -1710.19, 'LIQC': -1695.53, 'LIQD': -1701.43})

#: the production coincidence window (ntof_scint_stack.extract.WINDOWS['on']),
#: used to turn a hit rate into "chance a random hit sits in your window"
WINDOW_US = 0.160
#: the emulated hardware-trigger thresholds, mV (ntof_scint_stack.extract)
WALL_THR = {'A': 26.0, 'B': 35.0, 'C': 38.0, 'D': 37.0}
PLAS_THR = {'A': 112.0, 'B': 136.0, 'C': 151.0, 'D': 130.0}


def out_dir(*parts) -> Path:
    d = paths.spell('scint', 'recovery', *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
