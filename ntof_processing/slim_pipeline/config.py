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

# A pulse's DREAM triggers that must show a same-arm wall AND plastic hit inside
# ACCEPT_NS for the pulse to count as matched. THE SINGLE SOURCE: clock_qa reads
# it for its per-pulse check, coincidence_arbiter for its lock decision, and the
# dashboard and pulse ledger for their tables -- it lived in two modules until
# 2026-08-13 and a threshold with two homes drifts.
#
# 0.80 sits in the empty gap between the two populations: at the right lock the
# median pulse runs 87-96 %, at a wrong one ~0-2 %. Three orders of magnitude,
# so its exact value cannot matter.
PULSE_MIN_FRAC = 0.80

# ------------------------------------------------------------- the beam record
# A PS pulse below this delivered NO PROTONS, and the bunch is dropped from the
# slim before anything is fitted or read.
#
# Measured 2026-08-10 over the whole first campaign
# (`../FINDINGS_2026-08-10_unfitted_bunches.md`): the beam is bimodal at
# ~413e10 (parasitic, 47.8 %) and ~851e10 (dedicated, 52.2 %), and 1,658 of
# 96,206 bunches sit at ZERO -- intensity < 1e10 on 97.5 % of them and
# `tflash == 0` on all of them, i.e. the n_TOF flash finder found no gamma
# flash either. Nothing separates the populations by less than two orders of
# magnitude, so this is a classifier and not a tuning knob.
#
# What those bunches contain: 0-19 DREAM triggers against 46-139 in a beam
# bunch, at ~22 Hz against ~1.2 kHz, because DREAM's gate opens on the PS
# timing whether or not protons arrive. Their hits are SiPM dark counts (WAL
# 2.80 per trigger, PSS 0.017, LIQ 0.000) and NONE of them matches an n_TOF
# candidate -- `bunch_join` labels the first background trigger of the gate
# `is_flash`, so their time base is referenced to nothing.
EMPTY_PULSE_E10 = 10.0

# How many triggers a dropped (no-beam) pulse may hold, relative to the median
# beam bunch, before the drop stops being a beam statement and starts being a
# join failure. Healthy: 1-2 triggers against ~92, i.e. ~0.02. Broken:
# run_116/stat090_0013 x 224636, a 13 %-overlap proposal that fitted a
# -1,324 s burst-to-pulse offset, put 66-108 triggers in each "empty" bunch --
# ratio ~1.0. Nothing real lives between these.
EMPTY_TRIGGER_RATIO_WARN = 0.25
EMPTY_TRIGGER_RATIO_FAIL = 0.50

# Splits the two beam families, for reporting only -- nothing is cut on it.
# Per-segment match efficiency correlates with the parasitic fraction at
# r = -0.82 (97.7 % dedicated vs 91.2 % parasitic), which is most of the
# campaign's 93.6-97.3 % spread, so a segment's mix is worth carrying.
PARASITIC_E10 = 600.0

# Fewest DREAM physics events a segment needs before it is worth fitting a clock
# to. Checked straight after the join, BEFORE the candidate pass, so a proposal
# that did not pan out costs seconds instead of minutes: the wall-clock overlap
# is an estimate, and a sub-run can be proposed against an n_TOF run it barely
# touches (3 % overlap -> 0 events joined, which used to crash on an empty
# concatenate after a full pass).
MIN_EVENTS = 500

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
# WIDENED TO +-1 us on 2026-08-09, after measuring the plastic tail properly.
#
# +-150 was set believing the tail was a curiosity. `slim_study/pss_tail_probe.py`
# slimmed 150 bunches of the reference pair at +-10 us and measured what it
# actually is. Background-subtracted against the +100 us control:
#
#   family   early (< -150 ns)   late (> +150 ns)   ratio   core (|dt| < 25)
#   WAL                   -572              3,136       -             32,026
#   PSS                  3,147             69,199     22x             17,490
#   LIQ                  3,701              2,224    0.6x                879
#
# So: the tail is real, it is PLASTIC ONLY, and it is one-sided late. The LIQ
# "tail" is symmetric, i.e. subtraction noise -- an earlier reading of the
# integral scan called it real and that was wrong; liquids are contained at
# +-150. WAL is the trigger and is contained at +-25.
#
# EXPLAINED, later on 2026-08-09, by `../pss_ringing/` -- the tail is REAL
# AFTER-PULSING in the plastics, not a fitter artifact. Every large plastic pulse
# is followed by a train of genuine secondary pulses in the raw stream1 trace,
# ~4.4 extra PSA hits per pulse over 18-1000 ns against 0.007 on the SiPM walls
# in the same run. Established against an event-mixed accidental control, a
# time-reversal control (4.13 forward vs 0.90 backward), the walls, and the raw
# traces one event at a time. Full account: `../pss_ringing/report.html`.
#
# Two corrections to what this comment used to say:
#   * "no discrete echoes" was a binning artifact. At 1 ns there is a 2 ns-wide
#     echo at 81-82 ns, identical on all four plastics -- a reflection. In the
#     DREAM residual it smears into a bump at 70-90 ns.
#   * it is not the template fitter splitting a long pulse. The amplitude-
#     normalised MEDIAN trace decays smoothly, and the walls -- 3x wider pulse,
#     fatter tail, same PSA -- show no excess at all.
#
# WHAT TO DO WITH IT, measured in `../pss_ringing/report_veto.html`: flag a hit
# when `amp_0 < 0.05 * max(amp_0 on the same channel in the previous 1000 ns)`.
# That removes 99.5 % of the 150-1000 ns excess and 94.8 % of the 25-150 ns
# excess for 10.4 % of the core, all of it small-amplitude. It must be computed
# on the FULL n_TOF stream with a full 1 us of lookback -- an after-pulse whose
# parent sits just outside this window is exactly what a slim-only
# recomputation gets wrong.
#
# IMPLEMENTED 2026-08-09 in pass 2 (`slim.pass2_hits`), which sees the full
# per-bunch stream before the window cut: every kept hit carries
#   shadow_amp   largest amp_0 on the same (bunch, channel) in the previous
#                SHADOW_HOLD_NS (0 = nothing there)
#   shadow_dt    ns since that largest hit (-1 = nothing there)
# stored as floats, NOT a boolean, so `ratio` and any t_hold <= SHADOW_HOLD_NS
# stay re-tunable per analysis: the adopted flag is
#   amp_0 < SHADOW_RATIO * shadow_amp  (and shadow_dt <= t_hold if tightening).
# Files slimmed before this date lack the branches; `clock_qa.py` falls back
# to an in-window recomputation, which is complete for the LATE tail (an
# after-pulse at +dt always has its parent inside the window) and measures
# 100.6 % of the late excess removed on the reference segment.
#
# QA: clock_qa checks 'PSS late tail is ringing' (the flag explains the late
# excess) and 'plastic primary within accept' (per matched trigger, the
# LARGEST plastic pulse on the trigger's own arm lands within +-25 ns -- 92.0 %
# on the reference; "earliest" gives 31 % and must not be used).
SHADOW_HOLD_NS = 1000.0
SHADOW_RATIO = 0.05
#
# So the plastic hit yield is now quotable, but only against one of those cuts:
# the yield inside +-1 us is ~6.1x the coincident core (288 k against 47 k).
#
# PSS cumulative capture: 46 % at +-150, 57 % at +-250, 71 % at +-500,
# 80 % at +-1000, 86 % at +-2000, 93 % at +-5000 ns.
#
# +-1000 captures 93 % of the excess lying within +-2 us at 2.24x the hits
# (~72 MB per segment, ~14 GB for the campaign). Past ~2 us each extra
# microsecond adds 1-2.4 k counts against an early-side noise level of ~1 k,
# which is not worth paying for. Widening is the cheap direction: the source is
# 21 TB and re-reading it is the expensive one, and an analysis can always cut
# tighter than the slim but can never recover what the slim threw away.
SLIM_NS = 1000.0

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

# The per-job partials the pass leaves behind BEFORE the merge node runs. For 28
# of the 41 runs we asked n_TOF to reprocess, the processing had in fact already
# succeeded and only the merge is missing, so `done/` has no file while this
# directory holds a complete, contiguous, v12 partial set -- 77 % of the beam
# time we thought was unprocessed. Verified 2026-08-10: part count equals
# ceil(raw files / 4), indices 1..N with no gap, and the `history` string is
# byte-identical (md5 e51a3ef3dc0b32c1803e59ad18639a7c) to the reference
# done/run224572.root. See ../skip_diagnosis/README.md.
NTOF_COMPLETED = Path(os.environ.get(
    'X17_NTOF_COMPLETED', '/eos/experiment/ntof/processing/official/completed'))

# Local staging, for the reference pair and for testing off-CERN.
LOCAL_V12 = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')


def _merged_ok(p: Path) -> bool:
    """A merged run file that is actually worth opening.

    A FAILED merge leaves a zero-byte `run<run>.root` behind rather than nothing
    -- `done/run224405.root` and `done/run224667.root` are both 0 bytes, dated
    2026-08-05. `exists()` says yes to those, so anything keyed on existence
    picks an empty file over a complete partial set sitting in completed/<run>/.
    """
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def ntof_files(run: int, source: Path | None = None) -> list:
    """The run's file(s). NEVER hadd a run -- read the partials in order.

    A merged file in `done/` is preferred when one exists AND is non-empty; the
    unmerged partial set under `completed/<run>/` is the fallback, and it is the
    SAME processing, not a lesser one -- the merge is a separate DAG node that
    adds no content.
    """
    src = Path(source) if source else NTOF_DONE
    if src.is_file():
        return [src]
    parts = sorted(src.glob(f'run{run}_[0-9]*.root'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if parts:
        return parts
    single = src / f'run{run}.root'
    if _merged_ok(single):
        return [single]
    chained = sorted((src / f'run{run}.parts').glob(f'run{run}_[0-9]*.root'),
                     key=lambda p: int(p.stem.split('_')[-1]))
    if chained:
        return chained
    if source is None:
        unmerged = sorted((NTOF_COMPLETED / str(run)).glob(f'run{run}_[0-9]*.root'),
                          key=lambda p: int(p.stem.split('_')[-1]))
        if unmerged:
            _require_complete(run, unmerged)
            return unmerged
    raise FileNotFoundError(f'no n_TOF files for run {run} under {src}')


# The raw the pass ran over, needed to know how many partials a run SHOULD have.
NTOF_RAW = Path(os.environ.get(
    'X17_NTOF_RAW', '/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement'))
FILES_PER_JOB = 4


def _require_complete(run: int, parts: list) -> None:
    """Refuse a partial set that is still being written.

    On 2026-08-11 n_TOF began REPROCESSING 24 runs: each `completed/<run>/` was
    emptied and is refilling one partial at a time. Reading such a directory
    gives a silently short run -- the slim would fit a clock on a third of the
    bunches and report a healthy efficiency for it. So a fallback to the
    unmerged set is only allowed when the set is contiguous 1..N AND N matches
    ceil(raw stream1 files / 4), the split RunProcessing uses.
    """
    idx = [int(p.stem.split('_')[-1]) for p in parts]
    if idx != list(range(1, len(idx) + 1)):
        raise FileNotFoundError(
            f'run {run}: unmerged partial set is not contiguous '
            f'({len(idx)} files, indices {idx[0]}..{idx[-1]}) -- it is probably '
            f'being rewritten; do not read it')
    try:
        n_raw = len(os.listdir(NTOF_RAW / str(run) / 'stream1'))
    except OSError:
        return          # raw aged off disk: contiguity is all we can check
    want = -(-n_raw // FILES_PER_JOB)
    if want and len(idx) != want:
        raise FileNotFoundError(
            f'run {run}: unmerged partial set has {len(idx)} of {want} partials '
            f'-- incomplete or mid-reprocessing; do not read it')


# ------------------------------------------------------------------- DREAM in
DREAM_RUNS = Path(os.environ.get(
    'X17_DREAM_RUNS', '/media/dylan/data/x17/beam_july/runs'))

# --------------------------------------------------------------------- caches
# `ntof_io` puts its bunch index under <beam_july>/analysis/. On a condor worker
# `$X17_BEAM_JULY` points at EOS, and a worker must not write there -- set this
# to node-local scratch instead. Empty means "use ntof_io's own default".
CACHE_BASE = os.environ.get('X17_SLIM_CACHE', '')
