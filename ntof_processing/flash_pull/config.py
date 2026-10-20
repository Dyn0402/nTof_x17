"""What the flash pull keeps, and where it puts it.

The raw-format conventions (PRE_SAMPLES, signed int16, the fill code) are the
waveform pull's and are imported from there, not restated.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..waveform_pull import config as WC

SCHEMA = 'flash_pull/1'

#: tof windows kept for every bunch and every channel, ns, [lo, hi).  The flash
#: sits at tof ~11.6 us.  0-10 ms covers the walls' undershoot (gone by ~1 ms),
#: the plastics' baseline (<= 1.6 ms) and their occupancy-limited tail; 30-32 ms
#: is a same-bunch reference far from the flash, for the "full amplitude"
#: comparison.  Measured on 224572 (ntof_scint_recovery/): 0-10 ms is 75 MB
#: (parasitic) to 120 MB (dedicated) of samples per bunch, the reference 0.7 MB.
WINDOWS_NS = ((0, 10_000_000), (30_000_000, 32_000_000))

#: detectors kept: the twelve scintillators and PKUP (the flash time base)
DETS = WC.SCINT_DETS + (WC.PKUP_DET,)
#: `det` code in the product: the slim's 0-11, then PKUP
DET_CODE = dict(WC.DET_CODE, **{WC.PKUP_DET: len(WC.SCINT_DETS)})

#: raw files per run: spread evenly through the run, ~20 bunches each
FILES_PER_RUN = 2
#: runs worth more: the only undiverted-wall runs, and the run the first look used
FILES_PER_RUN_SPECIAL = 6
SPECIAL_RUNS = {224356, 224357, 224358, 224359, 224360, 224464, 224466, 224572}

#: the official processed files: per-bunch PulseIntensity and the PKUP flash tof
OFFICIAL = Path(os.environ.get('X17_NTOF_OFFICIAL', '/eos/experiment/ntof/processing/official/done'))

#: where the product goes (experiment area: this is experiment data)
DEST = os.environ.get('X17_FLASH_DEST',
                      'root://eosuser.cern.ch//eos/experiment/ntof/data/x17/flash_tail')


def out_name(run: int, idx: int) -> str:
    return f'flash_{run}_{idx:04d}.root'
