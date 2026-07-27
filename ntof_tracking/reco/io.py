#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io.py — beam combined_hits loading + strip-position mapping for reco.

Differences from ntof_july_analysis.july_hv_scan.load_hits:
  * loads the FULL hits schema (time_over_threshold, integral, saturated, ...)
    — the feature/quality machinery needs it;
  * vectorized (feu, channel) -> (det, plane, pos_mm) mapping via lookup
    tables (the per-row map_hit loop is ~100x too slow at 2M rows/file);
  * one tidy DataFrame out, with det/plane/pos_mm columns attached.
"""
from __future__ import annotations

import os
import re
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

BASE_PATH = '/mnt/data/x17/beam_july/runs/'
ANALYSIS_DIR = '/mnt/data/x17/beam_july/analysis/'
MAP_CSV_PATH = os.path.join(_REPO, 'mx17_m1_map.csv')
MX17_DETECTORS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
PEDESTAL_NAME_TOKEN = '_pedestals_'

HIT_COLUMNS = ['eventId', 'feu', 'channel', 'amplitude', 'time',
               'time_over_threshold', 'integral', 'saturated', 'sample',
               'local_baseline']

# ---- unphysical-hit sanity cut (added 2026-07-25) ----
# A few per cent of hits carry a diverged pulse-time fit: `sample` values of
# +/- 5e6 on a 32-sample waveform and `time` out to +/- 3e8 ns (0.3 s), versus a
# real DAQ window of ~2 us. run_67/m090 measured 3.7 % such hits after the
# 2026-07-24 reprocessing (2.9 % before it -- the reprocessing made them more
# frequent, it did not invent them). They are failed fits, not small pulses.
#
# They are not merely noise, they are actively destructive: noise._band_intervals
# bins each plane's time range at 30 ns, so ONE hit at 3e8 ns turns a ~70-bin
# histogram into a 2-million-bin one, which crashed the run_67 re-reco outright
# (IndexError at noise.py:58) and, short of crashing, silently wrecks the
# clustering and drift spectra.
#
# The cut is deliberately a SANITY bound on `sample`, not a physics window on
# `time`: it is independent of how many samples a given run digitized (32 here,
# other runs differ), so it removes only impossible values and never trims a
# real waveform. Real hits sit at sample -10..35 here; anything past +/-1000 is
# arithmetic garbage under any run configuration.
SAMPLE_SANITY_MAX = 1000.0

# strip map spans 0..398.58 mm (512 x 0.78 mm) on both axes; detector-local
# coordinates are centred on the plane middle so that local (0,0) = the
# det_center_coords point of run_config.json.
STRIP_SPAN_MM = 398.58
STRIP_MID_MM = STRIP_SPAN_MM / 2.0


def load_run_config(run: str, base_path: str = BASE_PATH) -> dict:
    with open(os.path.join(base_path, run, 'run_config.json')) as f:
        return json.load(f)


def build_channel_lut(cfg: dict) -> pd.DataFrame:
    """One row per (feu, channel): det, plane ('x'/'y'), pos_mm (centred).

    Uses common.Mx17StripMap.Detector.map_hit once per channel (512/feu) —
    cheap — then everything downstream is a vectorized merge.
    """
    import sys
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from common.Mx17StripMap import Mx17StripMap, Detector

    smap = Mx17StripMap(MAP_CSV_PATH)
    rows = []
    for det_cfg in cfg.get('detectors', []):
        name = det_cfg['name']
        if name not in MX17_DETECTORS:
            continue
        det = Detector(name=name, det_cfg=det_cfg, strip_map=smap)
        # which plane does each feu serve? read from dream_feus keys
        feu_axis: Dict[int, str] = {}
        for det_key, (feu_id, _conn) in det.dream_feus.items():
            feu_axis[feu_id] = det_key[0]
        for feu_id, axis in feu_axis.items():
            for ch in range(512):
                pos = det.map_hit(feu_id, ch)
                if pos is None:
                    continue
                x, y = pos
                p = x if axis == 'x' else y
                if p is None:
                    continue
                rows.append((feu_id, ch, name, axis, p - STRIP_MID_MM))
    lut = pd.DataFrame(rows, columns=['feu', 'channel', 'det', 'plane', 'pos_mm'])
    return lut


def _real_files(directory: str, suffix: str) -> List[str]:
    """Real acquisition files (the shared pedestal copy is excluded)."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(suffix) and '_datrun_' in f
        and PEDESTAL_NAME_TOKEN not in f
    )


def load_subrun_hits(run: str, subrun: str, lut: pd.DataFrame,
                     base_path: str = BASE_PATH,
                     columns: Optional[List[str]] = None,
                     verbose: bool = True) -> Optional[pd.DataFrame]:
    """All real combined hits of one subrun, mapped: adds det/plane/pos_mm.

    Rows whose (feu, channel) is not in the strip map (unconnected channels)
    are dropped. Duplicate rows from ROOT re-cycles are removed.
    """
    hits_dir = os.path.join(base_path, run, subrun, 'combined_hits_root')
    sources = _real_files(hits_dir, '.root')
    good = []
    for s in sources:
        try:
            with uproot.open(s) as f:
                if 'hits' in f:
                    good.append(s)
        except Exception:
            continue
    if not good:
        return None
    cols = columns or HIT_COLUMNS
    df = uproot.concatenate([f'{s}:hits' for s in good], cols, library='pd')
    df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
    df = drop_unphysical(df, tag=f'{run}/{subrun}', verbose=verbose)
    df = df.merge(lut, on=['feu', 'channel'], how='inner')
    return df.sort_values(['eventId', 'det', 'plane', 'time']).reset_index(drop=True)


def drop_unphysical(df: pd.DataFrame, tag: str = '', verbose: bool = True,
                    sample_max: float = SAMPLE_SANITY_MAX) -> pd.DataFrame:
    """Remove hits whose pulse-time fit diverged (see SAMPLE_SANITY_MAX).

    Reports the fraction dropped rather than doing it silently: if this ever
    climbs above the ~3-4 % seen in run_67 it is a decoding regression, not a
    detector effect, and the number should be looked at before the physics is.
    """
    if 'sample' not in df.columns:
        return df
    s = df['sample'].to_numpy(float)
    bad = ~np.isfinite(s) | (np.abs(s) > sample_max)
    if 'time' in df.columns:
        bad |= ~np.isfinite(df['time'].to_numpy(float))
    n_bad = int(bad.sum())
    if n_bad and verbose:
        print(f'    {tag}: dropped {n_bad} unphysical hits '
              f'({100.0 * n_bad / max(len(df), 1):.2f} %) — diverged pulse-time '
              f'fits (|sample| > {sample_max:g})', flush=True)
    return df[~bad] if n_bad else df


def parse_drift_hv(subrun: str) -> Optional[float]:
    """Drift HV [V] from subrun names like 'scintd_dr800_A460_D440_00'."""
    m = re.search(r'_dr(\d+)_', subrun)
    return float(m.group(1)) if m else None
