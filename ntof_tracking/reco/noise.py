#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise.py — post-CM residual noise identification for beam Micromegas hits.

The dominant residual after common-mode subtraction is a ~1.2-1.3 MHz
whole-plane oscillation (mesh_daq_survival study): it produces COHERENT TIME
BANDS — one narrow time slice (~10-30 ns) in which 50-400 strips of a plane
fire together at low amplitude, repeating every ~0.77 us. A real micro-TPC
track instead progresses through time with position. Strategy:

  1. per (event, plane): find time slices whose distinct-strip count exceeds
     BAND_MIN_STRIPS -> coherent band intervals;
  2. hits inside a band are flagged noise UNLESS their amplitude clearly
     towers over the band's own level (rescues track hits that happen to
     cross the band time);
  3. isolated-hit removal on what remains (no neighbour within LINK window).

All functions take/return the tidy hits DataFrame of reco.io and only ADD
boolean columns — nothing is dropped here, so displays can show what was
flagged and why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# coherent-band finder
BAND_TIME_BIN_NS = 30.0      # histogram bin for coincidence counting
BAND_MIN_STRIPS = 40         # >= this many distinct strips in one bin = band
BAND_RESCUE_FACTOR = 3.0     # in-band hit kept if amp > factor * band median
# isolated-hit removal
ISO_POS_MM = 5.0
ISO_TIME_NS = 250.0


def _band_intervals(t: np.ndarray, ch: np.ndarray,
                    bin_ns: float = BAND_TIME_BIN_NS,
                    min_strips: int = BAND_MIN_STRIPS):
    """Merged [t0, t1] intervals whose strip-coincidence count is band-like.

    Two offset binnings (0 and bin/2 phase) so a band straddling a bin edge
    is not diluted below threshold.
    """
    if len(t) == 0:
        return []
    lo, hi = t.min(), t.max() + 1e-6
    flagged = []
    for phase in (0.0, 0.5 * bin_ns):
        edges = np.arange(lo - phase, hi + bin_ns, bin_ns)
        if len(edges) < 2:
            continue
        # np.digitize returns len(edges) for any t >= edges[-1], so idx can
        # reach len(edges)-1 and edges[b+1] then runs off the end. In principle
        # arange puts the last edge above t.max(); in practice, once the range
        # is wide enough for float error (or a diverged hit time -- see
        # io.drop_unphysical) that guarantee fails, and this raised IndexError
        # mid-re-reco. Clip so a hit at the top edge joins the last real bin.
        idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
        # distinct strips per bin
        df = pd.DataFrame({'b': idx, 'ch': ch})
        n = df.groupby('b')['ch'].nunique()
        for b in n[n >= min_strips].index:
            flagged.append((edges[b], edges[b + 1]))
    if not flagged:
        return []
    flagged.sort()
    merged = [list(flagged[0])]
    for a, b in flagged[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(m) for m in merged]


def flag_coherent_bands(g: pd.DataFrame) -> pd.Series:
    """Boolean 'in_band' for one (event, det, plane) group (index-aligned).

    A hit inside a band interval is flagged unless its amplitude exceeds
    BAND_RESCUE_FACTOR x the median amplitude of the band's hits.
    """
    t = g['time'].to_numpy()
    ch = g['channel'].to_numpy()
    amp = g['amplitude'].to_numpy()
    out = np.zeros(len(g), bool)
    for t0, t1 in _band_intervals(t, ch):
        m = (t >= t0) & (t <= t1)
        if not m.any():
            continue
        med = np.median(amp[m])
        out |= m & (amp < BAND_RESCUE_FACTOR * med)
    return pd.Series(out, index=g.index)


def flag_isolated(g: pd.DataFrame, in_band: pd.Series,
                  pos_mm: float = ISO_POS_MM,
                  time_ns: float = ISO_TIME_NS) -> pd.Series:
    """Boolean 'isolated' for the non-band hits of one (event, plane) group:
    no other non-band hit within pos_mm AND time_ns."""
    keep = ~in_band.to_numpy()
    p = g['pos_mm'].to_numpy()[keep]
    t = g['time'].to_numpy()[keep]
    iso_sub = np.ones(len(p), bool)
    if len(p) > 1:
        order = np.argsort(p)
        ps, ts = p[order], t[order]
        for i in range(len(ps)):
            j = i - 1
            near = False
            while j >= 0 and ps[i] - ps[j] <= pos_mm:
                if abs(ts[i] - ts[j]) <= time_ns:
                    near = True
                    break
                j -= 1
            j = i + 1
            while not near and j < len(ps) and ps[j] - ps[i] <= pos_mm:
                if abs(ts[j] - ts[i]) <= time_ns:
                    near = True
                    break
                j += 1
            iso_sub[order[i]] = not near
    out = np.zeros(len(g), bool)
    out[keep] = iso_sub
    return pd.Series(out, index=g.index)


def flag_noise(hits: pd.DataFrame) -> pd.DataFrame:
    """Add 'in_band', 'isolated', 'clean' columns to a hits DataFrame
    (any number of events/detectors/planes)."""
    hits = hits.reset_index(drop=True)
    in_band = np.zeros(len(hits), bool)
    isolated = np.zeros(len(hits), bool)
    for _, g in hits.groupby(['eventId', 'det', 'plane'], sort=False):
        ib = flag_coherent_bands(g)
        iso = flag_isolated(g, ib)
        in_band[g.index] = ib.to_numpy()
        isolated[g.index] = iso.to_numpy()
    hits['in_band'] = in_band
    hits['isolated'] = isolated
    hits['clean'] = ~hits['in_band'] & ~hits['isolated']
    return hits


def hot_channels(hits: pd.DataFrame, frac: float = 0.25) -> pd.DataFrame:
    """Channels firing in more than `frac` of events (per det/plane) — stuck
    or noisy strips to mask in the search. Returns det/plane/feu/channel/rate."""
    n_ev = hits['eventId'].nunique()
    r = (hits.groupby(['det', 'plane', 'feu', 'channel'])['eventId']
             .nunique().rename('n_ev').reset_index())
    r['rate'] = r['n_ev'] / max(n_ev, 1)
    return r[r['rate'] > frac].sort_values('rate', ascending=False)
