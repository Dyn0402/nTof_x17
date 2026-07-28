"""
Bridge from the wft event table to the analysis machinery that is still valid.

The alignment scans, the reference attachment, the efficiency/resolution maps
and the residual fits in ``cosmic_bench_analysis.cosmic_micro_tpc_analysis``
act on *positions*, never on hit times, so they are unaffected by the reason
the hits chain was retired — reusing them is not reusing the hits basis. They
expect ``EventResult``/``StripFitResult`` objects, so this module builds those
from the table.

What is carried over, and what it means now:

    mesh_position_mm  -> p0, the fitted track position at the mesh (was: the
                         earliest-hit strip, quantised to the pitch and pulled
                         by shared charge)
    slope_ns_per_mm   -> 1 / w, from the fitted transverse speed (was: the
                         amplitude-weighted strip-time ladder, compressed
                         20-30 %)
    earliest_time_ns  -> t0, the fitted arrival time of charge from the mesh
    latest_time_ns    -> t0 + the fitted charge column duration
    red_chi2          -> chi2/dof of the waveform fit (a much larger number
                         than the ladder's: it counts every sample, and the
                         model is not perfect at the percent level)
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

_CBA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'cosmic_bench_analysis')
if _CBA not in sys.path:
    sys.path.insert(0, _CBA)

import cosmic_micro_tpc_analysis as cm       # noqa: E402


def _strip_fit(row, plane, require_slope=False):
    if not row[f'{plane}_ok']:
        return None
    w = row[f'{plane}_w']
    if require_slope and not row[f'{plane}_slope_reliable']:
        return None
    slope = 1.0 / w if np.isfinite(w) and abs(w) > 1e-9 else np.nan
    dur = row[f'{plane}_q_uend']
    t0 = row[f'{plane}_t0']
    return cm.StripFitResult(
        slope_ns_per_mm=float(slope),
        mesh_position_mm=float(row[f'{plane}_p0']),
        earliest_time_ns=float(t0),
        latest_time_ns=float(t0 + (dur if np.isfinite(dur) else 0.0)),
        n_strips=int(row[f'{plane}_n_strips']),
        n_dropped=int(row[f'{plane}_n_dropped']),
        red_chi2=float(row[f'{plane}_chi2'] / max(row[f'{plane}_dof'], 1)))


def as_event_results(df: pd.DataFrame, quality_only: bool = True,
                     require_slope: bool = False) -> List[cm.EventResult]:
    """Table -> list of EventResult, for the position-side analysis machinery."""
    out = []
    for row in df.to_dict('records'):
        r = cm.EventResult(event_id=int(row['event_id']))
        for plane in ('x', 'y'):
            if quality_only and not row.get(f'{plane}_quality_ok', False):
                continue
            fit = _strip_fit(row, plane, require_slope=require_slope)
            setattr(r, f'{plane}_fit', fit)
        out.append(r)
    return out


def load_table(path: str, quality_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if quality_only:
        for p in ('x', 'y'):
            bad = df[f'{p}_ok'] & ~df[f'{p}_quality_ok']
            df.loc[bad, f'{p}_ok'] = False
    return df


def table_meta(path: str) -> dict:
    import json
    meta = path.replace('.parquet', '.meta.json')
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f)
    return {}
