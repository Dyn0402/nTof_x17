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


def corrected_tflash(run: int) -> dict:
    """{tree: array[MAX_BUNCH]} of repaired tflash = mode_tree + jitter(bunch).

    Also carries diagnostics in ['_err_frac'][tree]: the fraction of bunches
    whose STORED tflash deviates >150 ns from the repaired one.
    """
    tab = tflash_tables(run)
    modes = {t: _mode(tab[t]) for t in tab}
    dev = np.stack([tab[t] - modes[t] for t in REF_TREES])
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN = empty bunch
        jitter = np.nanmedian(dev, axis=0)
    jitter[~np.isfinite(jitter)] = 0.0
    out = {t: modes[t] + jitter for t in tab}
    out['_err_frac'] = {
        t: float(np.mean(np.abs((tab[t] - out[t])[np.isfinite(tab[t])]) > 150.0))
        for t in tab}
    out['_modes'] = modes
    return out


if __name__ == '__main__':
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 224572
    c = corrected_tflash(run)
    print(f'run{run}: per-tree modal tflash and stored-tflash failure rate')
    for t in ALL_TREES:
        print(f'  {t}: mode {c["_modes"][t]:9.1f} ns   '
              f'stored bad (>150 ns): {c["_err_frac"][t]:6.1%}')
