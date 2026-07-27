#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beam_july_paths.py -- where the July beam data tree lives, on whichever machine.

The July analysis grew up on the DAQ machine (mx17), where everything sits under
/mnt/data/x17/beam_july, and most scripts hardcode that. The same tree is mirrored
onto the laptop under ~/x17/beam_july (a subset: combined_hits_root, the staged
n_TOF run, slow_control, analysis caches -- no raw_daq_data or decoded_root), so
the hardcoded path makes those scripts unrunnable off the DAQ box.

Resolution order:
  1. $X17_BEAM_JULY, if set  -- explicit override, wins everywhere.
  2. /mnt/data/x17/beam_july -- the DAQ machine.
  3. ~/x17/beam_july         -- the laptop mirror.

Nothing here creates directories or validates the tree beyond existence; a script
that needs decoded_root on a machine that has no decoded_root should fail on the
missing subdirectory, with its own message, not here.
"""
from __future__ import annotations

import os
from pathlib import Path

_CANDIDATES = (Path('/mnt/data/x17/beam_july'), Path.home() / 'x17' / 'beam_july')


def beam_july_base() -> Path:
    """Root of the July beam data tree on this machine."""
    env = os.environ.get('X17_BEAM_JULY')
    if env:
        return Path(env).expanduser()
    for p in _CANDIDATES:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        'July beam data tree not found. Looked for $X17_BEAM_JULY, then '
        + ', '.join(str(p) for p in _CANDIDATES))


BASE = beam_july_base()

RUNS_DIR = BASE / 'runs'
ANALYSIS_DIR = BASE / 'analysis'
SLOW_CONTROL_DIR = BASE / 'slow_control'
BEAM_LOG_DIR = SLOW_CONTROL_DIR / 'beam_intensity'
NTOF_DATA_DIR = BASE / 'ntof_data'


if __name__ == '__main__':
    print(f'BASE            {BASE}')
    for name in ('runs', 'analysis', 'slow_control', 'ntof_data'):
        p = BASE / name
        print(f'  {name:<14} {"OK " if p.is_dir() else "MISSING"} {p}')
