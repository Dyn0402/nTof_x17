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
N_CANDIDATES = 3           # clusters offered to the fit per plane


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


def seed_candidates(pos: np.ndarray, channels: np.ndarray, amps: np.ndarray,
                    gap_mm: float = GAP_THRESHOLD_MM,
                    min_strips: int = MIN_STRIPS,
                    n_candidates: int = 1, hot=None) -> list:
    """Spatial clusters of one plane, ranked by CLEAN (non-``hot``) strip
    count.

    ``n_candidates > 1`` returns the runners-up as well, so the caller can let
    the waveform fit decide which cluster is the muon. That matters: "largest
    cluster wins" picks the wrong charge in ~5 % of events, and when it does,
    the true track sits a median of 37 mm outside the fit window (measured on
    det3 — see mx_june_wft/DET3_GATE_2026-07-29.md).

    ``hot`` (HANDOFF_D_NOISY_CHANNELS.md's wildcard spec, item 1: "never seed
    on a flagged channel") is a set/array of channel numbers that must not, by
    themselves, qualify a cluster as a seed. Ranking and the ``min_strips``
    admission test both use the CLEAN count only, so a cluster made entirely
    of hot strips (a noise column self-clustering) cannot outrank or displace
    a real candidate and cannot become a seed on its own. Cluster MEMBERSHIP
    is untouched: a hot strip inside a real track's cluster stays in
    ``Seed.channels`` and still reaches the fit -- gap-clustering still
    bridges across it, so a track is never split or lost for having crossed
    one. It arrives at the fit down-weighted instead (``wft.model.prep_plane``
    reads the same bundle's ``hot`` list), which is where its influence is
    actually capped (item 3).
    """
    good = np.isfinite(pos)
    pos, channels, amps = pos[good], channels[good], amps[good]
    if len(pos) < min_strips:
        return []
    o = np.argsort(pos)
    pos, channels, amps = pos[o], channels[o], amps[o]
    lab = np.concatenate([[0], np.cumsum(np.diff(pos) > gap_mm)])
    n_labels = int(lab.max()) + 1 if len(lab) else 0
    hot_mask = (np.isin(channels, list(hot)) if hot is not None and len(hot)
               else np.zeros(len(channels), bool))
    clean_counts = np.bincount(lab[~hot_mask], minlength=n_labels)
    order = np.argsort(clean_counts)[::-1][:max(1, n_candidates)]
    out = []
    for c in order:
        if clean_counts[c] < min_strips:
            continue
        m = lab == c
        out.append(Seed(channels=channels[m].astype(np.int64),
                        n_strips=int(m.sum()), n_dropped=int((~m).sum()),
                        amp_sum=float(amps[m].sum()), n_raw=int(len(pos))))
    return out


def seed_plane(pos: np.ndarray, channels: np.ndarray, amps: np.ndarray,
               gap_mm: float = GAP_THRESHOLD_MM,
               min_strips: int = MIN_STRIPS, hot=None) -> Optional[Seed]:
    """Largest (clean-strip-ranked) spatial cluster of one plane, from
    already-floored hits."""
    c = seed_candidates(pos, channels, amps, gap_mm, min_strips, 1, hot=hot)
    return c[0] if c else None


def seeds_from_hits(df_hits: pd.DataFrame, pos_maps: Dict[int, np.ndarray],
                    feu_x: int, feu_y: int, rel_floor: float = SIG_REL_FLOOR,
                    spark_veto: Optional[int] = SPARK_VETO_HITS,
                    n_candidates: int = N_CANDIDATES,
                    hot: Optional[Dict[str, object]] = None) -> Dict[int, dict]:
    """Build per-event seeds for both planes from a combined hits DataFrame.

    ``hot``: optional ``{'x': [...], 'y': [...]}`` channel numbers, passed to
    ``seed_candidates`` (see there — "never seed on a flagged channel").
    Typically ``cal.hot`` off the calibration bundle in use.

    Returns {eventId: {'x': [Seed], 'y': [Seed], 'n_hits': int,
                       'spark': bool}} — a list per plane, ranked by (clean)
    strip count, for the waveform fit to choose between (see
    wft.reco.fit_plane_candidates).
    """
    df = df_hits[df_hits['feu'].isin((feu_x, feu_y))]
    df = apply_significance_floor(df, rel_floor)
    out: Dict[int, dict] = {}
    if len(df) == 0:
        return out
    hot = hot or {}
    counts = df.groupby('eventId').size()
    for eid, g in df.groupby('eventId'):
        n_hits = int(counts.loc[eid])
        rec = {'x': [], 'y': [], 'n_hits': n_hits,
               'spark': bool(spark_veto is not None and n_hits > spark_veto)}
        if not rec['spark']:
            for plane, feu in (('x', feu_x), ('y', feu_y)):
                gp = g[g['feu'] == feu]
                if len(gp) == 0:
                    continue
                ch = gp['channel'].to_numpy().astype(int)
                rec[plane] = seed_candidates(pos_maps[feu][ch], ch,
                                             gp['amplitude'].to_numpy(),
                                             n_candidates=n_candidates,
                                             hot=hot.get(plane))
        out[int(eid)] = rec
    return out
