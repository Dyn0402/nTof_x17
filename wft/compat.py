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


def as_event_results(df: pd.DataFrame, quality_only: bool = False,
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


MAX_DROPPED = 2      # strips allowed in competing clusters, per plane


def apply_cluster_quality(df: pd.DataFrame, max_dropped: int = MAX_DROPPED
                          ) -> pd.DataFrame:
    """Reject plane-fits whose seed cluster competes with another cluster.

    ``n_dropped`` counts strips in *other* clusters of the same plane. When the
    muon's cluster is not the largest one, the seed — and therefore the fit
    window — lands on the wrong charge: measured on det3's failures, the
    reference sits a median of 37 mm outside the fit window, against 1.9 mm for
    good fits, and the far events carry a median n_dropped of 4 against 0.

    The forward fit converges on whatever charge it is given, so unlike the hits
    chain (whose line fit simply failed on junk clusters, quietly removing them)
    it needs this stated explicitly. The threshold is the one the hits chain
    already uses for its alignment subset (``--maxdrop`` in
    03_alignment_and_tpc.py), so the two chains reject the same events.

    An event rejected here is *not* a detection failure: it fired strips, we
    just decline to trust the point. Efficiency accounting must therefore count
    it as 'hit, no reco', which is what 02_efficiency.py does.
    """
    if max_dropped is None:
        return df
    for p in ('x', 'y'):
        bad = df[f'{p}_ok'] & (df[f'{p}_n_dropped'] > max_dropped)
        df.loc[bad, f'{p}_ok'] = False
    return df


def load_table(path: str, quality_only: bool = False,
               max_dropped: int | None = MAX_DROPPED) -> pd.DataFrame:
    """Load a reco table.

    ``quality_only`` is OFF by default and should stay off for anything the
    hits chain is compared against: the hits chain applies no equivalent cut,
    and chi2/dof here is large by construction (every sample counts, and the
    model is imperfect at the percent level -- the median is ~110 on X and
    ~180 on Y for det3). CHI2DOF_BAD is a flag for finding showers and
    multi-track events, not a reconstruction filter.
    """
    df = pd.read_parquet(path)
    df = apply_cluster_quality(df, max_dropped)
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
