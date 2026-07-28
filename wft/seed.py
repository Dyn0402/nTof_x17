"""
Seeding: the one place hits are allowed in.

Hits answer 'which events and which strips carry a track' — a question about
detection, which is what the analyzer's trigger is for. They do not answer
'where and at what angle', which is what the waveform fit is for. Everything
this module returns is a *set of channels*; no hit time crosses the boundary.

The clustering reproduces the production selection so that detection semantics
(and therefore efficiency) stay comparable with the hits chain:
per-plane relative significance floor, then spatial clustering with the
production gap threshold, largest cluster kept
(``cosmic_micro_tpc_analysis._fit_single_axis``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

SIG_REL_FLOOR = 0.10       # per-plane, relative to that plane's strongest strip
GAP_THRESHOLD_MM = 12.0    # production spatial clustering gap
MIN_STRIPS = 3
SPARK_VETO_HITS = 50       # full-detector discharge: drop the event


@dataclass
class Seed:
    """Candidate cluster for one plane of one event."""
    channels: np.ndarray
    n_strips: int
    n_dropped: int          # strips in competing clusters
    amp_sum: float
    n_raw: int              # strips before the significance floor


def apply_significance_floor(df: pd.DataFrame, rel: float = SIG_REL_FLOOR) -> pd.DataFrame:
    """Keep strips with significance >= rel x that plane's strongest strip in
    that event (the 2026-07-25 fix: coherent noise otherwise inflates
    multiplicity, steals cluster membership and fakes the spark veto)."""
    if not rel or 'significance' not in df.columns:
        return df
    sig = df['significance'].to_numpy()
    if not np.isfinite(sig).any():
        return df
    mx = df.groupby(['eventId', 'feu'])['significance'].transform('max')
    return df[sig >= rel * mx.to_numpy()].copy()


def seed_plane(pos: np.ndarray, channels: np.ndarray, amps: np.ndarray,
               gap_mm: float = GAP_THRESHOLD_MM,
               min_strips: int = MIN_STRIPS) -> Optional[Seed]:
    """Largest spatial cluster of one plane, from already-floored hits."""
    good = np.isfinite(pos)
    pos, channels, amps = pos[good], channels[good], amps[good]
    if len(pos) < min_strips:
        return None
    o = np.argsort(pos)
    pos, channels, amps = pos[o], channels[o], amps[o]
    lab = np.concatenate([[0], np.cumsum(np.diff(pos) > gap_mm)])
    big = np.bincount(lab).argmax()
    m = lab == big
    if m.sum() < min_strips:
        return None
    return Seed(channels=channels[m].astype(np.int64), n_strips=int(m.sum()),
                n_dropped=int((~m).sum()), amp_sum=float(amps[m].sum()),
                n_raw=int(len(pos)))


def seeds_from_hits(df_hits: pd.DataFrame, pos_maps: Dict[int, np.ndarray],
                    feu_x: int, feu_y: int, rel_floor: float = SIG_REL_FLOOR,
                    spark_veto: Optional[int] = SPARK_VETO_HITS) -> Dict[int, dict]:
    """Build per-event seeds for both planes from a combined hits DataFrame.

    Returns {eventId: {'x': Seed|None, 'y': Seed|None, 'n_hits': int,
                       'spark': bool}}.
    """
    df = df_hits[df_hits['feu'].isin((feu_x, feu_y))]
    df = apply_significance_floor(df, rel_floor)
    out: Dict[int, dict] = {}
    if len(df) == 0:
        return out
    counts = df.groupby('eventId').size()
    for eid, g in df.groupby('eventId'):
        n_hits = int(counts.loc[eid])
        rec = {'x': None, 'y': None, 'n_hits': n_hits,
               'spark': bool(spark_veto is not None and n_hits > spark_veto)}
        if not rec['spark']:
            for plane, feu in (('x', feu_x), ('y', feu_y)):
                gp = g[g['feu'] == feu]
                if len(gp) == 0:
                    continue
                ch = gp['channel'].to_numpy().astype(int)
                rec[plane] = seed_plane(pos_maps[feu][ch], ch,
                                        gp['amplitude'].to_numpy())
        out[int(eid)] = rec
    return out
