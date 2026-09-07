"""Paths, constants and conventions for the waveform pull.

Everything that is a CONVENTION of the raw stream1 format lives here with the
measurement that established it, because none of it is derivable from the file.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------- raw sample time base
# 1 GS/s: one sample = one nanosecond.  A block's `start` is the zero-suppression
# TRIGGER sample and the payload begins PRE_SAMPLES earlier, so sample j of a
# block sits at
#       tof = start + j - PRE_SAMPLES        (start > 0)
#       tof = j                              (start == 0, the mandatory flash block)
# Measured -258.7 ns on LIQA over 135/135 pulses, spread 1.1 ns (DAQ repo,
# ntof_raw.parse_acqc).  This is the same `tof` the PSA writes.
PRE_SAMPLES = 259
NS_PER_SAMPLE = 1

# int16_t in ntoflib's ReaderStructACQC.h; the S014/ADQ14 cards are 16 bit and
# every channel is parked near the rail opposite its pulse direction, so a full
# pulse crosses zero.  Reading these unsigned is silently wrong -- see
# SIGNED_DECODE_FIX_NOTE.md in the DAQ repo.
SAMPLE_DTYPE = '<i2'
BYTES_PER_SAMPLE = 2

# The zero-suppression fill code, bit-identical to the negative rail.  A filled
# gap and a genuine clip differ only by context.
ZS_FILL_CODE = -32768

# ------------------------------------------------------------------ what to pull
# The twelve scintillator detectors, in the slim's `det` code order.  `detn` in
# the slim IS the raw ACQC channel id (verified 2026-08-12 on 224572: identical
# 1-based ranges, 8/8/8/8/2/2/2/2/1/1/1/1).
SCINT_DETS = ('WALA', 'WALB', 'WALC', 'WALD',
              'PSSA', 'PSSB', 'PSSC', 'PSSD',
              'LIQA', 'LIQB', 'LIQC', 'LIQD')
DET_CODE = {d: i for i, d in enumerate(SCINT_DETS)}

# Channels per detector, 1-based, as MODH declares them (verified on 224572).
# Needed because a DEAD channel emits no hit anywhere and so appears in no
# tflash table -- it has to be enumerated from the configuration, not inferred
# from the data, or the pull would silently have no window for exactly the
# channels worth looking at.
DET_NCHAN = {'WALA': 8, 'WALB': 8, 'WALC': 8, 'WALD': 8,
             'PSSA': 2, 'PSSB': 2, 'PSSC': 2, 'PSSD': 2,
             'LIQA': 1, 'LIQB': 1, 'LIQC': 1, 'LIQD': 1,
             'PKUP': 1}

# PKUP carries the proton-pulse pickup and is the flash time base's own witness
# (t_flash = tof_PKUP + C, C ~ -1719 ns/channel -- see ../flash_timing/).  It is
# one channel and costs nothing, so it comes along unless switched off.
PKUP_DET = 'PKUP'

# --------------------------------------------------------------------- windows
# Default half-width around each corrected DREAM prediction.  This is the WIDE
# choice and it is deliberate: the recall from tape is the expensive, one-shot
# step, and widening the cut is nearly free against it.  +-5 us holds 93 % of the
# background-subtracted PSS excess (../SLIM_FEASIBILITY_2026-08-08.md, the
# pss_tail_probe row) against 80 % at +-1 us.
WINDOW_NS = 5000.0

# The accidental control, identical to the slim's.  NOT a local sideband.
CONTROL_SHIFT_NS = 100_000.0

# ---------------------------------------------------------------------- paths
# Raw stream1, EOS disk staging area.  Holds data for ~2 WEEKS after the run,
# then only the CTA copy survives.
NTOF_RAW = Path(os.environ.get(
    'X17_NTOF_RAW', '/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement'))

# The tape archive and its xrootd endpoint.
CTA_XRD = os.environ.get('X17_CTA_XRD', 'root://eosctapublicdisk.cern.ch/')
CTA_BASE = os.environ.get(
    'X17_CTA_BASE', '/eos/ctapublicdisk/archive/ntof/2026/EAR2/X17_measurement')

# Where the slim products live, and where the waveforms go beside them.
EOS_JULY = Path(os.environ.get(
    'X17_EOS_JULY', '/eos/experiment/ntof/data/x17/july_beam'))


def raw_name(run: int, idx: int) -> str:
    return f'run{run}_{idx}_s1.raw'


def cta_url(run: int, idx: int) -> str:
    """Tape copies carry a `.finished` suffix the disk copies do not."""
    return f'{CTA_XRD}{CTA_BASE}/{run}/stream1/{raw_name(run, idx)}.finished'


def slim_dir(dream_run: str, dream_subrun: str, base: Path | None = None) -> Path:
    return (Path(base) if base else EOS_JULY) / 'runs' / dream_run / \
        dream_subrun / 'ntof_hits'


def out_dir(dream_run: str, dream_subrun: str, base: Path | None = None) -> Path:
    """Beside the slim, one directory per DREAM sub-run."""
    return (Path(base) if base else EOS_JULY) / 'runs' / dream_run / \
        dream_subrun / 'ntof_wf'


def out_name(dream_run: str, dream_subrun: str, ntof_run: int) -> str:
    return f'ntof_wf_{dream_run}_{dream_subrun}_{ntof_run}.root'
