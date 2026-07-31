#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared setup for the DREAM<->n_TOF matching study.

Every script here reads a CANDIDATE processing (v12_liqpileup by default), never
the official file, and must therefore point `ntof_io` at those partials and give
them a cache of their own -- the bunch index and the tflash cache are keyed by
run number only, so an official and a reprocessed run224572 sharing a cache
directory would silently mix (REVIEW.md section 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

STUDY = Path(__file__).resolve().parents[1]
DATA = STUDY / 'data'
FIGS = STUDY / 'figures'

V12 = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
NTOF_RUN = 224572
DREAM_RUN = 'run_79'
SUBRUNS = ('stat090_0000', 'stat090_0001')

# DREAM -> n_TOF time map, as calibrated in match_window/dream_regression:
# the DREAM clock runs 108.9 ppm slow against n_TOF's, plus a fixed offset.
K, T0 = 1.089e-4, -197.5

# The accept bands in use up to now, measured on late times in match_window.py.
BANDS = ((-150.0, 150.0), (250.0, 450.0))

# Control shift: far enough that no real coincidence can survive, small enough
# that the local singles rate is unchanged.
SHIFT_NS = 100_000.0


def use_variant(path: Path = V12, run: int = NTOF_RUN):
    """Point ntof_io at a candidate processing; return its file list."""
    import ntof_dream_merge.ntof_io as ntof_io
    import ntof_dream_merge.tflash_repair as rep

    p = Path(path).resolve()
    files = (sorted(p.glob(f'run{run}_[0-9]*.root'),
                    key=lambda q: int(q.stem.split('_')[-1]))
             if p.is_dir() else [p])
    if not files:
        raise SystemExit(f'no run{run}_NNNN.root partials under {p}')
    ntof_io.ntof_paths = lambda r: files          # type: ignore
    ntof_io.ntof_path = lambda r: files[0]        # type: ignore
    rep.CACHE_DIR = ntof_io.CACHE_DIR = ntof_io.variant_cache(p, files)
    ntof_io._TFLASH_FIX_CACHE.clear()
    return files


def dream_events(sub: str, run: str = DREAM_RUN, ntof_run: int = NTOF_RUN,
                 nb: int | None = None):
    """(events DataFrame without flash triggers, bunch list) for one sub-run.

    The DREAM<->bunch join runs off the PKUP and index trees, i.e. the beam
    record, and is unaffected by the PSA settings -- so it is built once and is
    the same for every candidate processing.
    """
    from ntof_dream_merge.bunch_join import dream_event_to_bunch
    ev = dream_event_to_bunch(run, sub, ntof_run)
    bunches = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())
    if nb is not None:
        bunches = bunches[:nb]
    sel = ev[(ev['BunchNumber'].isin(bunches)) & (~ev['is_flash'])]
    return sel.reset_index(drop=True), bunches


def predicted_time(ets: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """Where in the n_TOF time base a DREAM event at `ets` should be found."""
    return ets + K * ets + T0 + shift
