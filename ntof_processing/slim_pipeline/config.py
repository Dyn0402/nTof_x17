#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py -- where things are, and the numbers the slim is defined by.

Every constant that a reader might otherwise have to guess at lives here, with
the measurement it came from. See `../SLIM_FEASIBILITY_2026-08-08.md`.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- the windows
# Final accept, after the per-bunch clock fit. Measured: efficiency 95.84 % at a
# 0.049 % accidental rate; tighter costs 3 points at +-15 ns, wider buys 0.17
# points for 7x the background (DREAM_NTOF_CALIBRATION.md section 4).
ACCEPT_NS = 25.0

# What the slim keeps. Six times the accept window.
#
# The coincidence itself needs far less: measured 2026-08-08 on
# run_79/stat090_0000, the background-subtracted excess over the +100 us control
# is contained to
#     WAL  94 % within +-25 ns, 99.1 % within +-100      (it IS the trigger)
#     LIQ  55 % within +-25 ns, 88 % within +-100, 92 % within +-150
#     PSS  a real LATE tail, still 4x the floor at +240 ns -- see below
# and the published liquid same-arm diagonal is identical, to 0.001 per event,
# whether the slim is +-100 or +-250 (0.135/0.119/0.012/0.075 either way). An
# earlier version of this file claimed +-100 clipped it; that was wrong -- it
# clipped the ACCIDENTAL FLOOR inside liq_coincidence's +-100 ns integration
# window, which cancels in the subtraction.
#
# +-150 is chosen for two reasons that are not the coincidence width:
#   1. it holds 92 % of the liquid excess rather than 88 %, i.e. the late tail
#      that the peak-centred metric never integrates but a total yield would;
#   2. it leaves ~6x the accept window of flat sideband, so a segment whose
#      clock went wrong is visible in the file itself.
# For (2) note the primary alarm is NOT the window: `qa.json` carries the
# efficiency and `events.residual_ns` the nearest-candidate residual out to the
# 400 ns fit search, so a shifted clock shows up there whatever the slim width.
SLIM_NS = 150.0

# The accidental control. NOT a local sideband -- the singles rate varies far
# too much across the 80 ms. This is the shift the measured 0.049 % comes from.
CONTROL_SHIFT_NS = 100_000.0

# ------------------------------------------------------------------ the trees
SCINT_TREES = ('WALA', 'WALB', 'WALC', 'WALD',
               'PSSA', 'PSSB', 'PSSC', 'PSSD',
               'LIQA', 'LIQB', 'LIQC', 'LIQD')

# Per-hit branches carried into the slim. `area` is deliberately absent: with
# AMPLITUDE OPTION=2 it is amp x integral(shape) by construction and carries
# nothing `amp` does not. `amp_0`/`area_0` are the MEASURED pair and are what a
# real integral or a saturation-independent amplitude needs.
HIT_BRANCHES = ('BunchNumber', 'detn', 'tof', 'amp', 'amp_0', 'area_0',
                'fwhm', 'risetime', 'chi2', 'satuflag', 'pileup1', 'pulseshape')

# -------------------------------------------------------------------- outputs
# EOS only -- not the DAQ machine. One directory per DREAM sub-run, mirroring
# the DREAM tree, so the slim sits beside the data it belongs to.
EOS_JULY = Path(os.environ.get(
    'X17_EOS_JULY', '/eos/experiment/ntof/data/x17/july_beam'))


def out_dir(dream_run: str, dream_subrun: str, base: Path | None = None) -> Path:
    """<base>/runs/<run>/<subrun>/ntof_hits/"""
    return (Path(base) if base else EOS_JULY) / 'runs' / dream_run / \
        dream_subrun / 'ntof_hits'


# ------------------------------------------------------------------- n_TOF in
# The reprocessed campaign. Verified 2026-08-08 to carry our v12 UserInput on
# all 14 detector rows and all 26 templates, under the name
# UserInput_2026_EAR2_X17_v4.h (n_TOF's version counter, not ours).
NTOF_DONE = Path(os.environ.get(
    'X17_NTOF_DONE', '/eos/experiment/ntof/processing/official/done'))
NTOF_PROCESSING = 'official_done_v4_eq_v12_liqpileup'

# Local staging, for the reference pair and for testing off-CERN.
LOCAL_V12 = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')


def ntof_files(run: int, source: Path | None = None) -> list:
    """The run's file(s). NEVER hadd a run -- read the partials in order."""
    src = Path(source) if source else NTOF_DONE
    if src.is_file():
        return [src]
    parts = sorted(src.glob(f'run{run}_[0-9]*.root'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if parts:
        return parts
    single = src / f'run{run}.root'
    if single.exists():
        return [single]
    chained = sorted((src / f'run{run}.parts').glob(f'run{run}_[0-9]*.root'),
                     key=lambda p: int(p.stem.split('_')[-1]))
    if chained:
        return chained
    raise FileNotFoundError(f'no n_TOF files for run {run} under {src}')


# ------------------------------------------------------------------- DREAM in
DREAM_RUNS = Path(os.environ.get(
    'X17_DREAM_RUNS', '/media/dylan/data/x17/beam_july/runs'))

# --------------------------------------------------------------------- caches
# `ntof_io` puts its bunch index under <beam_july>/analysis/. On a condor worker
# `$X17_BEAM_JULY` points at EOS, and a worker must not write there -- set this
# to node-local scratch instead. Empty means "use ntof_io's own default".
CACHE_BASE = os.environ.get('X17_SLIM_CACHE', '')
