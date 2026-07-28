#!/usr/bin/env python3
"""Pre-calibrated per-tree gamma-flash time offsets.

WHY THIS EXISTS
---------------
The PSA has no input for a fixed or externally supplied flash time: every
G-FLASH OPTION (0 = first threshold crossing, 1 = first saturation,
2 = oscillatory) locates the flash *in the waveform*.  So a detector whose
waveform does not contain a faithful copy of the flash cannot be given the
right `tflash` by the processing, no matter how the UserInput is tuned.

That is exactly the SiPM walls' situation: their signal is intentionally
diverted for ~1 us around the flash, so what the digitiser records there is
the protection circuit's behaviour, not the flash.  The only faithful thing in
that window is a heavily attenuated, clamped copy of the flash leaking through
at ~11.60 us (see FLASH_TIME_BASE.md).

Therefore the flash time base is set in TWO places:

  1. the UserInput chooses WHICH waveform feature the PSA times
     (`ntof_processing/userinputs/*/UserInput.h`), and
  2. a per-tree constant, applied HERE, converts that feature's time into the
     physical flash arrival:

         t_since_flash = tof - (tflash_stored + offset[tree])

This module is that second place -- the single point where pre-calibrated
numbers are plugged in.  Constants live in `flash_calibration.json` so they can
be replaced by measured values without touching code.

USE
---
    from ntof_processing.flash_calibration import offsets
    off = offsets(224572)            # {'WALA': 0.0, 'PSSA': -362.3, ...}

and in ntof_dream_merge.tflash_repair.corrected_tflash pass
`offsets_source='calib'` to use these instead of the per-run coincidence fit.
"""
import json
from pathlib import Path

JSON_PATH = Path(__file__).with_name('flash_calibration.json')

TREES = ([f'WAL{a}' for a in 'ABCD'] + [f'PSS{a}' for a in 'ABCD']
         + [f'LIQ{a}' for a in 'ABCD'] + ['PKUP'])


def _load():
    with open(JSON_PATH) as f:
        return json.load(f)


def entries():
    """All calibration entries, newest first."""
    return sorted(_load()['calibrations'], key=lambda e: e['valid_from'],
                  reverse=True)


def offsets(run: int, strict: bool = False) -> dict:
    """Per-tree offset in ns to ADD to the stored tflash, for `run`.

    Picks the calibration entry whose [valid_from, valid_to] run range contains
    `run`.  With strict=True a missing entry raises; otherwise all-zero offsets
    are returned (i.e. "trust the stored tflash"), which is the right fallback
    for detectors and periods we have not calibrated.
    """
    for e in entries():
        if e['valid_from'] <= run <= e.get('valid_to', 10 ** 9):
            return {t: float(e['offsets_ns'].get(t, 0.0)) for t in TREES}
    if strict:
        raise KeyError(f'no flash calibration covering run {run}')
    return {t: 0.0 for t in TREES}


def describe(run: int) -> str:
    for e in entries():
        if e['valid_from'] <= run <= e.get('valid_to', 10 ** 9):
            return (f"{e['name']}  ({e['status']}, {e['method']})\n"
                    f"  runs {e['valid_from']}-{e.get('valid_to', '...')}"
                    f"  reference: {e['reference_tree']}\n"
                    f"  {e['note']}")
    return f'no calibration covering run {run}'


if __name__ == '__main__':
    import sys
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 224572
    print(describe(run))
    off = offsets(run)
    for t in TREES:
        print(f'  {t}: {off[t]:+8.1f} ns')
