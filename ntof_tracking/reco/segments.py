#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segments.py — per-plane spatio-temporal clustering, robust micro-TPC line
fits, and cluster classification.

A "segment" is a per-plane track candidate: a cluster of clean hits whose
position progresses coherently with drift time (the micro-TPC diagonal).
This is deliberately stricter than any hit-count selection:

  cluster  ->  robust line fit pos(t)  ->  class in
      {track, point, band_fragment, blob}

Classification logic:
  * band_fragment — isochronous (tiny time span) but spatially extended:
    the residue of a coherent noise band that survived the noise flags;
  * point — compact in BOTH position and time: a real but point-like
    ionisation deposit (most nTOF energy deposits look like this);
  * track — extended in time AND position with a good robust line
    (r2, inlier fraction, residual rms all pass);
  * blob — everything else (pile-up, delta blobs, unresolved overlaps).

The bench-grade measurement chain (anchored time fit + hits6 features from
ntof_tracking.microtpc_lib) is applied to every 'track' for downstream
angle work — pattern recognition here, measurement there.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .. import microtpc_lib as mtpc

# clustering: hits are linked when close in BOTH scaled coordinates
LINK_POS_MM = 4.0            # ~5 strips
LINK_TIME_NS = 250.0
MIN_CLUSTER_HITS = 3

# track classification
TRK_MIN_STRIPS = 5           # distinct strips
TRK_MIN_PSPAN_MM = 4.0
TRK_MIN_TSPAN_NS = 120.0     # 2 samples at 60 ns
TRK_MIN_R2 = 0.65
TRK_MIN_INLIER_FRAC = 0.65
TRK_MAX_RES_MM = 3.0         # robust rms of position residuals
TRK_MIN_OCCUPANCY = 0.30     # n_strips / (pspan/pitch + 1): tracks are
                             # (near-)contiguous strip runs; guards against
                             # disjoint point deposits merging into a fake line
PNT_MAX_PSPAN_MM = 4.0
PNT_MAX_TSPAN_NS = 150.0
BANDF_MAX_TSPAN_NS = 60.0    # spatially long but isochronous

# fragment merging (noise bands / dead strips punch holes in real tracks)
MERGE_MAX_TIME_GAP_NS = 500.0
MERGE_MIN_R2 = 0.70
MERGE_MIN_INLIER_FRAC = 0.75


def cluster_plane(pos: np.ndarray, time: np.ndarray,
                  link_pos: float = LINK_POS_MM,
                  link_time: float = LINK_TIME_NS) -> np.ndarray:
    """Connected-components clustering with Chebyshev linking in scaled
    (pos/link_pos, time/link_time) space. Returns integer labels (-1 never)."""
    n = len(pos)
    if n == 0:
        return np.array([], int)
    pts = np.c_[pos / link_pos, time / link_time]
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1.0, p=np.inf, output_type='ndarray')
    if len(pairs) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def robust_line_fit(t: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None,
                    iters: int = 5, clip: float = 3.0):
    """Iteratively re-weighted deg-1 fit p(t) with MAD outlier clipping.
    Returns dict(slope, intercept, r2, inlier_frac, res_rms, inliers)."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    w0 = np.ones(len(t)) if w is None else np.asarray(w, float)
    # Weights are amplitudes, and amplitudes CAN be <= 0 (baseline-subtracted
    # undershoot; ~0.1 % of hits). sqrt() of those is NaN, np.polyfit then
    # returns NaN coefficients, and the caller's `not isfinite(slope)` check
    # threw the WHOLE cluster away — so one undershooting strip silently
    # destroyed an otherwise good track segment. This bites hardest on the
    # low-amplitude hits recovered by the 2026-07-24 reprocessing, i.e. exactly
    # the population the small-pulse re-reco is meant to gain. Clamp instead:
    # a non-positive-amplitude hit gets zero weight (ignored), not NaN.
    w0 = np.where(np.isfinite(w0), np.clip(w0, 0.0, None), 0.0)
    if not np.any(w0 > 0):
        w0 = np.ones(len(t))     # degenerate: fall back to unweighted
    keep = np.ones(len(t), bool)
    slope = inter = np.nan
    for _ in range(iters):
        if keep.sum() < 3 or np.ptp(t[keep]) == 0:
            break
        wk = np.sqrt(w0[keep])
        if not np.any(wk > 0):
            wk = None            # all kept hits zero-weight -> unweighted fit
        coef = np.polyfit(t[keep], p[keep], 1, w=wk)
        slope, inter = coef
        r = p - np.polyval(coef, t)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
        new = np.abs(r - np.median(r[keep])) < clip * 1.4826 * mad
        if new.sum() < 3 or new.sum() == keep.sum():
            keep = new if new.sum() >= 3 else keep
            break
        keep = new
    if not np.isfinite(slope):
        return None
    r = p[keep] - (slope * t[keep] + inter)
    ss_tot = np.sum((p[keep] - p[keep].mean()) ** 2)
    r2 = 1.0 - np.sum(r ** 2) / ss_tot if ss_tot > 0 else 0.0
    return dict(slope_mm_ns=float(slope), intercept_mm=float(inter),
                r2=float(r2), inlier_frac=float(keep.mean()),
                res_rms_mm=float(np.std(r)), inliers=keep)


def occupancy(n_strips: int, pspan: float) -> float:
    return n_strips / (pspan / mtpc.PITCH_MM + 1.0)


def classify(n_strips, pspan, tspan, fit) -> str:
    if pspan > 2 * PNT_MAX_PSPAN_MM and tspan < BANDF_MAX_TSPAN_NS:
        return 'band_fragment'
    if pspan <= PNT_MAX_PSPAN_MM and tspan <= PNT_MAX_TSPAN_NS:
        return 'point'
    if (fit is not None and n_strips >= TRK_MIN_STRIPS
            and pspan >= TRK_MIN_PSPAN_MM and tspan >= TRK_MIN_TSPAN_NS
            and fit['r2'] >= TRK_MIN_R2
            and fit['inlier_frac'] >= TRK_MIN_INLIER_FRAC
            and fit['res_rms_mm'] <= TRK_MAX_RES_MM
            and occupancy(n_strips, pspan) >= TRK_MIN_OCCUPANCY):
        return 'track'
    return 'blob'


def merge_fragments(labels: np.ndarray, pos: np.ndarray, time: np.ndarray,
                    amp: np.ndarray) -> np.ndarray:
    """Guided fragment merge: noise bands and dead strips punch time/position
    holes in real tracks wider than the clustering link scale, splitting one
    diagonal into fragments. Greedily merge cluster pairs whose UNION still
    robust-fits as one coherent line (r2, inlier fraction, residuals all at
    track quality) and whose time gap is bridgeable. Iterates to closure.
    """
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        labs = [l for l in np.unique(labels)
                if (labels == l).sum() >= 2]
        spans = {l: (time[labels == l].min(), time[labels == l].max())
                 for l in labs}
        for i, li in enumerate(labs):
            for lj in labs[i + 1:]:
                t0i, t1i = spans[li]
                t0j, t1j = spans[lj]
                gap = max(t0i, t0j) - min(t1i, t1j)   # <0 if overlapping
                if gap > MERGE_MAX_TIME_GAP_NS:
                    continue
                m = (labels == li) | (labels == lj)
                if m.sum() < TRK_MIN_STRIPS:
                    continue
                # only rescue TRACK fragments: the union must have real time
                # extent and near-contiguous strips, not two glued points
                if np.ptp(time[m]) < TRK_MIN_TSPAN_NS:
                    continue
                n_str = len(np.unique(pos[m]))
                if occupancy(n_str, float(np.ptp(pos[m]))) < TRK_MIN_OCCUPANCY:
                    continue
                fit = robust_line_fit(time[m], pos[m], w=amp[m])
                if fit is None:
                    continue
                inl = fit['inliers']
                mi = labels[m] == li
                frac_i = inl[mi].mean() if mi.any() else 0.0
                frac_j = inl[~mi].mean() if (~mi).any() else 0.0
                if (fit['r2'] >= MERGE_MIN_R2
                        and fit['inlier_frac'] >= MERGE_MIN_INLIER_FRAC
                        and fit['res_rms_mm'] <= TRK_MAX_RES_MM
                        and min(frac_i, frac_j) >= 0.6):
                    labels[labels == lj] = li
                    changed = True
                    break
            if changed:
                break
    return labels


def find_segments(g: pd.DataFrame, det: str, plane: str,
                  eventId: int) -> List[dict]:
    """Cluster + fit + classify the CLEAN hits of one (event, det, plane).

    g: hits DataFrame slice (needs pos_mm, time, amplitude,
    time_over_threshold, clean). Returns a list of segment dicts (one per
    cluster with >= MIN_CLUSTER_HITS hits), each carrying its class, the
    robust fit, bench measurement (anchored fit) and the hit index list.
    """
    gc = g[g['clean']]
    if len(gc) < MIN_CLUSTER_HITS:
        return []
    pos = gc['pos_mm'].to_numpy()
    tim = gc['time'].to_numpy()
    amp = gc['amplitude'].to_numpy()
    tot = gc['time_over_threshold'].to_numpy()
    labels = cluster_plane(pos, tim)
    labels = merge_fragments(labels, pos, tim, amp)
    out = []
    for lab in np.unique(labels):
        m = labels == lab
        if m.sum() < MIN_CLUSTER_HITS:
            continue
        p, t, a, q = pos[m], tim[m], amp[m], tot[m]
        idx = gc.index.to_numpy()[m]
        n_strips = len(np.unique(p))
        pspan = float(np.ptp(p))
        tspan = float(np.ptp(t))
        fit = robust_line_fit(t, p, w=a)
        cls = classify(n_strips, pspan, tspan, fit)
        seg = dict(
            eventId=int(eventId), det=det, plane=plane, cls=cls,
            n_hits=int(m.sum()), n_strips=int(n_strips),
            pspan_mm=pspan, tspan_ns=tspan,
            t0_ns=float(t.min()), t1_ns=float(t.max()),
            pos_lo_mm=float(p.min()), pos_hi_mm=float(p.max()),
            q_sum=float(a.sum()), a_max=float(a.max()),
            a_med=float(np.median(a)),
            hit_index=idx,
        )
        if fit is not None:
            seg.update({k: fit[k] for k in
                        ('slope_mm_ns', 'intercept_mm', 'r2',
                         'inlier_frac', 'res_rms_mm')})
        if cls == 'track':
            anch = mtpc.anchored_time_fit(p, t, a)
            if anch:
                seg.update(anchor_pos_mm=anch['mesh_position_mm'],
                           anchor_t_ns=anch['earliest_time_ns'],
                           duration_ns=anch['duration_ns'],
                           extent_mm=anch['extent_mm'],
                           red_chi2=anch['red_chi2'])
            feats = mtpc.hit_features(p, a, t, q)
            if feats:
                seg.update({f'f_{k}': v for k, v in feats.items()})
        out.append(seg)
    return out


def segments_for_event(hits_ev: pd.DataFrame) -> List[dict]:
    """All per-plane segments of one event's (noise-flagged) hits."""
    segs: List[dict] = []
    for (det, plane), g in hits_ev.groupby(['det', 'plane'], sort=False):
        ev = int(hits_ev['eventId'].iloc[0])
        segs.extend(find_segments(g, det, plane, ev))
    return segs
