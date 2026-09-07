#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clusters.py -- vectorised strip clustering on the July beam `combined_hits`.

This is a *detection* / occupancy layer, not a reconstruction one: it never
touches hit times, so it stays on the right side of RECONSTRUCTION_BASIS.md.
What it produces is "which strips took part in a particle-like cluster", which
is exactly the observable an active-area measurement needs.

A cluster is a run of contiguous strips on one plane of one chamber in one
event, each strip above a matched-filter significance cut, allowing at most
`MAX_GAP` missing strips inside the run.  Noise is overwhelmingly isolated
single strips; a minimum-ionising track spreads over several.

FEU -> (chamber, plane) comes from the run config: every July chamber is wired
with one FEU per plane and detector connector i on FEU connector i, all eight
connectors "inverted", so the strip index is

    strip = (channel // 64) * 64 + (63 - channel % 64)          [0 .. 511]
    position_mm = strip * 0.78

`plane 'x'` is the chamber's tangential coordinate u, `plane 'y'` is the one
along the beam, v (ntof_tracking/run79_merge_prelim.track_frame).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot

PITCH_MM = 0.78
N_STRIPS = 512
STRIP_MAX_MM = 398.58

# FEU id -> (chamber, plane).  run_79/run_config.json `dream_feus`.
FEU2DET = {3: ('A', 'x'), 4: ('A', 'y'),
           5: ('B', 'x'), 6: ('B', 'y'),
           7: ('C', 'x'), 8: ('C', 'y'),
           1: ('D', 'x'), 2: ('D', 'y')}
CHAMBERS = ('A', 'B', 'C', 'D')
PLANES = ('x', 'y')

# columns of the per-chamber `pairs` array returned by scan_subrun()
PAIR_COLS = ('u_cent', 'v_cent', 'q_u', 'q_v', 'w_u', 'w_v',
             'u_first', 'u_last', 'v_first', 'v_last')
N_PAIR_COLS = len(PAIR_COLS)

# June bench alias for each n_TOF chamber (for the cross-check against the
# M3-telescope measurement in common/mx17_active_area.py).
BENCH_ALIAS = {'A': 'mx17_3', 'B': 'mx17_2', 'C': 'mx17_6', 'D': 'mx17_7'}


@dataclass(frozen=True)
class ClusterCuts:
    sig: float = 5.0        # matched-filter significance per strip
    min_strips: int = 3     # strips in the contiguous run
    max_gap: int = 1        # dead strips tolerated inside a run


def channel_to_strip(channel: np.ndarray) -> np.ndarray:
    """FEU channel [0..511] -> detector strip index [0..511] (all connectors
    inverted, detector connector i on FEU connector i)."""
    return (channel // 64) * 64 + (63 - channel % 64)


def cluster_hits(ev, feu, strip, amp, cuts: ClusterCuts):
    """Cluster a flat hit list.  Inputs must already be significance-filtered.

    Returns a dict of per-cluster arrays: event, feu, size, charge, centroid
    (strip units), first/last strip, plus `strip_index`/`cluster_of` giving the
    strip membership of every kept cluster.
    """
    order = np.lexsort((strip, feu, ev))
    ev, feu, strip, amp = ev[order], feu[order], strip[order], amp[order]
    if ev.size == 0:
        empty = np.zeros(0)
        return dict(event=empty, feu=empty, size=empty, charge=empty,
                    centroid=empty, first=empty, last=empty,
                    strip_index=empty, cluster_of=empty)

    new = np.empty(ev.size, bool)
    new[0] = True
    new[1:] = ((ev[1:] != ev[:-1]) | (feu[1:] != feu[:-1])
               | (np.diff(strip) > cuts.max_gap + 1))
    cid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    size = np.diff(np.append(starts, ev.size))

    charge = np.add.reduceat(amp, starts)
    moment = np.add.reduceat(amp * strip, starts)
    keep = (size >= cuts.min_strips) & (charge > 0)

    # strip membership of kept clusters
    memb = keep[cid]
    remap = np.full(starts.size, -1, np.int64)
    remap[keep] = np.arange(keep.sum())

    return dict(event=ev[starts][keep], feu=feu[starts][keep],
                size=size[keep], charge=charge[keep],
                centroid=(moment[keep] / charge[keep]),
                first=strip[starts][keep],
                last=strip[np.append(starts, ev.size)[1:] - 1][keep],
                strip_index=strip[memb], cluster_of=remap[cid[memb]])


def subrun_files(base: Path | str, run: str, sub_run: str) -> list[str]:
    return sorted(glob.glob(str(Path(base) / 'runs' / run / sub_run
                                / 'combined_hits_root' / '*.root')))


def scan_subrun(files, cuts: ClusterCuts, want_pairs: bool = True):
    """Accumulate per-plane strip occupancy and (optionally) single-cluster x/y
    pairs over a list of combined_hits files.

    Returns (occ, cent, pairs, n_events) where
      occ[(chamber, plane)]  -- 512-vector, times a strip was in a kept cluster
      cent[(chamber, plane)] -- 512-vector of cluster charge centroids
      pairs[chamber]         -- (n, 10) array for events with exactly one
                               cluster on each plane:
                               u_cent, v_cent, q_u, q_v, w_u, w_v,
                               u_first, u_last, v_first, v_last  (strip units)
    """
    keys = sorted(set(FEU2DET.values()))
    occ = {k: np.zeros(N_STRIPS) for k in keys}
    cent = {k: np.zeros(N_STRIPS) for k in keys}
    pairs = {c: [] for c in CHAMBERS}
    n_events = 0

    for fn in files:
        with uproot.open(fn) as f:
            a = f['hits'].arrays(['eventId', 'channel', 'feu', 'amplitude',
                                  'significance'], library='np')
        ev = a['eventId'].astype(np.int64)
        n_events += np.unique(ev).size
        g = (a['significance'] > cuts.sig) & (a['amplitude'] > 0)
        cl = cluster_hits(ev[g], a['feu'][g].astype(np.int32),
                          channel_to_strip(a['channel'][g].astype(np.int32)),
                          a['amplitude'][g], cuts)
        if cl['event'].size == 0:
            continue

        for fid, key in FEU2DET.items():
            m = cl['feu'] == fid
            if not m.any():
                continue
            occ[key] += np.bincount(cl['strip_index'][np.isin(cl['cluster_of'],
                                                              np.flatnonzero(m))],
                                    minlength=N_STRIPS)
            c = np.clip(np.rint(cl['centroid'][m]).astype(int), 0, N_STRIPS - 1)
            cent[key] += np.bincount(c, minlength=N_STRIPS)

        if want_pairs:
            _collect_pairs(cl, pairs)
    out_pairs = {c: (np.concatenate(v) if v else np.zeros((0, N_PAIR_COLS)))
                 for c, v in pairs.items()}
    return occ, cent, out_pairs, n_events


def _collect_pairs(cl, pairs):
    """Events with exactly one kept cluster on each plane of a chamber."""
    for chamber in CHAMBERS:
        fx = next(f for f, k in FEU2DET.items() if k == (chamber, 'x'))
        fy = next(f for f, k in FEU2DET.items() if k == (chamber, 'y'))
        mx, my = cl['feu'] == fx, cl['feu'] == fy
        if not (mx.any() and my.any()):
            continue
        ex, ey = cl['event'][mx], cl['event'][my]
        # unique-per-event on each plane
        ux, cx = np.unique(ex, return_counts=True)
        uy, cy = np.unique(ey, return_counts=True)
        solo_x, solo_y = ux[cx == 1], uy[cy == 1]
        common = np.intersect1d(solo_x, solo_y)
        if common.size == 0:
            continue
        ix = np.flatnonzero(mx)[np.isin(ex, common)]
        iy = np.flatnonzero(my)[np.isin(ey, common)]
        ox, oy = np.argsort(cl['event'][ix]), np.argsort(cl['event'][iy])
        ix, iy = ix[ox], iy[oy]
        pairs[chamber].append(np.column_stack([
            cl['centroid'][ix], cl['centroid'][iy],
            cl['charge'][ix], cl['charge'][iy],
            cl['size'][ix], cl['size'][iy],
            cl['first'][ix], cl['last'][ix],
            cl['first'][iy], cl['last'][iy]]))
