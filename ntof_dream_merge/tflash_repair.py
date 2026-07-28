#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tflash_repair.py -- fix the broken per-bunch `tflash` of the PSS trees.

THE BUG (found 2026-07-28, and it was the whole "missing plastic" mystery).
Every hit in the official trees carries its bunch's `tflash`, and the analysis
time base is t_since_flash = tof - tflash. That is only as good as the PSA's
gamma-flash identification, and on the PLASTIC trees it is broken: measured per
bunch over all 3018 bunches of run224572 against the mode of each tree,

    WALA 1.7 %   WALB 1.1 %   WALC 0.3 %   WALD 0.0 %      |tflash err| > 150 ns
    PSSA 84.5 %  PSSB 65.4 %  PSSC 36.8 %  PSSD 80.6 %
    LIQA-D 0.0 %                PKUP 0.0 %

The PSS failures are large (up to the full 11.6 us -- the finder tags a pulse
near the window start instead of the flash) and hit essentially EVERY parasitic
(half-intensity) pulse plus an arm-dependent fraction of dedicated ones. The raw
stream1 waveforms show the plastic flash is present and rails the ADC high in
every pulse (parasitic included), so this is purely a reconstruction fault, not
a detector one. Confirmed per event: for DREAM events whose plastic partner was
"missing", the nearest-plastic residual equals the bunch's tflash error, and the
raw waveform contains the plastic pulse within ~30 ns of the predicted position.

Repairing the time base lifts the plastic partner fraction of wall-matched DREAM
events from 48.9 % to 99.7 % (uniform 99.4-99.9 % across arms) on the reference
pair -- i.e. the hardware wall AND plastic trigger is fully accounted for.

THE REPAIR. The true flash time of tree T in bunch b is

    tflash_true(T, b) = mode_T + jitter(b)

where mode_T is the tree's per-run modal tflash (a cable constant, 10 ns bins)
and jitter(b) is the bunch-common part, estimated as the median over the STABLE
trees (WALB-D, LIQA-D, PKUP -- WALA is excluded for its 1.7 % +374 ns
population) of (tflash_R(b) - mode_R). The median makes single-tree glitches
irrelevant; the jitter itself is small (p1-p99 within about +-15 ns).

This repairs every tree, including WALA's own glitches. Do NOT use the stored
tflash of a PSS tree for anything.

Tables are built once per run (one pass over BunchNumber+tflash of each tree,
~10 min for a 26 GB file) and cached next to the bunch indexes.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.beam_july_paths import ANALYSIS_DIR   # noqa: E402

CACHE_DIR = ANALYSIS_DIR / 'ntof_dream_merge' / 'cache'

ALL_TREES = tuple(f'{k}{a}' for k in ('WAL', 'PSS', 'LIQ') for a in 'ABCD') + ('PKUP',)
# Trees whose flash-finding is trusted for the bunch-common jitter estimate.
REF_TREES = ('WALB', 'WALC', 'WALD',
             'LIQA', 'LIQB', 'LIQC', 'LIQD', 'PKUP')
MAX_BUNCH = 4000


def tflash_tables(run: int, rebuild: bool = False) -> dict:
    """Per-bunch stored tflash for every tree: {tree: array[MAX_BUNCH] (ns, NaN
    where the tree has no hits in that bunch)}. Built once per run, cached."""
    from ntof_dream_merge.ntof_io import ntof_path
    cache = CACHE_DIR / f'tflash_table_{run}.npz'
    if cache.exists() and not rebuild:
        with np.load(cache) as z:
            return {t: z[t] for t in z.files}
    tables = {}
    with uproot.open(ntof_path(run)) as f:
        for tree in ALL_TREES:
            tf = np.full(MAX_BUNCH, np.nan)
            seen = np.zeros(MAX_BUNCH, bool)
            for chunk in f[tree].iterate(['BunchNumber', 'tflash'], library='np',
                                         step_size='200 MB'):
                bn, fl = chunk['BunchNumber'], chunk['tflash']
                for b, i in zip(*np.unique(bn, return_index=True)):
                    if not seen[b]:
                        tf[b] = fl[i]
                        seen[b] = True
            tables[tree] = tf
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **tables)
    return tables


def _mode(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    h, e = np.histogram(v, bins=np.arange(0.0, 20000.0, 10.0))
    return float(e[h.argmax()] + 5.0)


def _coinc_offsets(run: int) -> dict:
    """Per-tree residual time offset vs the same arm's wall, measured from data.

    The per-tree tflash modes are NOT mutually consistent: the PSA times the
    flash on a different waveform feature per detector (WALB/PSS/LIQ all sit at
    ~11.6 us while WALA/C/D sit ~350-400 ns earlier), so after removing each
    tree's mode a large-amplitude plastic hit still sits at an arm-dependent
    -375/+25/-325/-325 ns from its wall partner. Measure that peak (amp > 1000
    ADC, late hits, prompt coincidences) and fold it into the corrected tflash,
    so that a true coincidence reconstructs at dt ~ 0 in every arm.

    Walls are the per-arm reference (offset 0 by construction); PSS and LIQ get
    one constant each. Cached per run.
    """
    cache = CACHE_DIR / f'tflash_offsets_{run}.npz'
    if cache.exists():
        with np.load(cache) as z:
            return {t: float(z[t]) for t in z.files}
    from ntof_dream_merge.ntof_io import read_bunches, bunch_edges
    # calibrate on ~100 bunches with data in every tree
    nb = np.diff(bunch_edges(run, 'WALA'))
    good = np.flatnonzero(nb > 0) + 1
    bl = good[len(good) // 3:len(good) // 3 + 100]
    base = corrected_tflash(run, _with_offsets=False)
    out = {}
    for arm in 'ABCD':
        w = read_bunches(run, f'WAL{arm}', bl, branches=('BunchNumber', 'tof'),
                         repair_tflash=False)
        wt_fix = base[f'WAL{arm}'][w['BunchNumber'].astype(np.int64)]
        for tree, amp_min in ((f'PSS{arm}', 1000.0), (f'LIQ{arm}', 0.0)):
            h = read_bunches(run, tree, bl, branches=('BunchNumber', 'tof', 'amp'),
                             repair_tflash=False)
            ht_fix = base[tree][h['BunchNumber'].astype(np.int64)]
            dts = []
            for b in np.unique(w['BunchNumber']):
                mw = w['BunchNumber'] == b
                tw = np.sort(w['tof'][mw] - wt_fix[mw])
                tw = tw[tw > 20e6]
                mh = (h['BunchNumber'] == b) & (h['amp'] > amp_min)
                tp = h['tof'][mh] - ht_fix[mh]
                tp = tp[tp > 20e6]
                if tw.size == 0 or tp.size == 0:
                    continue
                j = np.searchsorted(tw, tp)
                j0 = np.clip(j - 1, 0, tw.size - 1)
                j1 = np.clip(j, 0, tw.size - 1)
                d0, d1 = tp - tw[j0], tp - tw[j1]
                dts.append(np.where(np.abs(d0) <= np.abs(d1), d0, d1))
            d = np.concatenate(dts) if dts else np.array([0.0])
            d = d[np.abs(d) < 1000]
            if d.size < 50:
                out[tree] = 0.0
                continue
            hist, e = np.histogram(d, bins=200, range=(-1000, 1000))
            c = 0.5 * (e[1:] + e[:-1])
            pk = float(c[hist.argmax()])
            core = d[np.abs(d - pk) < 30]
            out[tree] = float(np.median(core))
        out[f'WAL{arm}'] = 0.0
    out['PKUP'] = 0.0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def corrected_tflash(run: int, _with_offsets: bool = True,
                     offsets_source: str = 'fit') -> dict:
    """{tree: array[MAX_BUNCH]} of repaired tflash =
    mode_tree + jitter(bunch) + offset_tree.

    The mode fixes the per-bunch mis-tags; the offset fixes the ~350 ns
    per-tree inconsistency in WHICH waveform feature the PSA timed.

    `offsets_source` selects where that offset comes from:
      'fit'   -- measure it per run from prompt wall/plastic coincidences
                 (default; self-calibrating, relative to the walls)
      'calib' -- take pre-calibrated constants from
                 ntof_processing/flash_calibration.json. Use this once the
                 dedicated flash pre-calibration exists, and for any run whose
                 statistics are too thin for the in-situ fit.
      'none'  -- no offset (mode + jitter only)

    Diagnostics in ['_err_frac'] (fraction of bunches whose STORED tflash
    deviates >150 ns from mode+jitter), ['_modes'], ['_offsets'].
    """
    tab = tflash_tables(run)
    modes = {t: _mode(tab[t]) for t in tab}
    dev = np.stack([tab[t] - modes[t] for t in REF_TREES])
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN = empty bunch
        jitter = np.nanmedian(dev, axis=0)
    jitter[~np.isfinite(jitter)] = 0.0
    if not _with_offsets or offsets_source == 'none':
        offs = {}
    elif offsets_source == 'calib':
        from ntof_processing.flash_calibration import offsets as _calib
        offs = _calib(run)
    elif offsets_source == 'fit':
        offs = _coinc_offsets(run)
    else:
        raise ValueError(f'unknown offsets_source {offsets_source!r}')
    out = {t: modes[t] + jitter + offs.get(t, 0.0) for t in tab}
    out['_err_frac'] = {
        t: float(np.mean(np.abs((tab[t] - modes[t] - jitter)[np.isfinite(tab[t])])
                         > 150.0))
        for t in tab}
    out['_modes'] = modes
    out['_offsets'] = offs
    return out


if __name__ == '__main__':
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 224572
    c = corrected_tflash(run)
    print(f'run{run}: per-tree modal tflash and stored-tflash failure rate')
    for t in ALL_TREES:
        print(f'  {t}: mode {c["_modes"][t]:9.1f} ns   '
              f'stored bad (>150 ns): {c["_err_frac"][t]:6.1%}')
