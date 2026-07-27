#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtrack_lib.py — Detector-A DOUBLE-TRACK finder (2026-07-21).

The frozen reco chain (ntof_tracking.reco.segments) fits ONE robust line per
connected-component cluster. That already separates two SPATIALLY DISJOINT
tracks (they land in different components), but it CANNOT split two tracks that
cross or share strips near a common vertex — those merge into one cluster and
are fit as a single line (or rejected as a blob). The e+e- pair opening from
the target is exactly that merged/crossing topology.

This module adds an intra-cluster multi-line extractor (sequential weighted
RANSAC) on top of the existing clustering, so per plane we recover a LIST of
track-grade lines whether the two tracks are separated OR crossing. An event is
a Det-A double-track candidate when BOTH planes carry >= 2 distinct track-grade
lines (the user's chosen definition). We then pair the X/Y lines into 3-D
micro-TPC segments (reusing the bench charge-balance + time-IoU matcher) and
tag the topology (separated vs converging/vertex).

Everything here is READ-ONLY w.r.t. ntof_tracking.reco: we import its noise
flagger, robust line fit, clustering and pairing primitives and never mutate
them.
"""
from __future__ import annotations

import os
import sys
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from ntof_tracking.reco import io, noise, geometry as geo          # noqa: E402
from ntof_tracking.reco import segments as S                       # noqa: E402
from ntof_tracking import microtpc_lib as mtpc                     # noqa: E402

DET_A = 'mx17_A'

# ---- track-grade gate for ONE extracted line -----------------------------
# A split line must be at least as convincing as a single-cluster 'track'.
# We keep segments.py's thresholds but tighten r2/res a little: a fake split
# (chopping one real track in two, or gluing noise) should not survive.
LINE_MIN_HITS = 5
LINE_MIN_STRIPS = 5
LINE_MIN_PSPAN_MM = 4.0
LINE_MIN_TSPAN_NS = 120.0
LINE_MIN_R2 = 0.70
LINE_MIN_INLIER_FRAC = 0.60
LINE_MAX_RES_MM = 2.5
LINE_MIN_OCC = 0.30

# ---- RANSAC line extraction ----------------------------------------------
RANSAC_RES_MM = 2.0          # inlier half-width around a candidate line
RANSAC_ITERS = 300           # random model draws per line
RANSAC_MIN_DT_NS = 120.0     # the two seed hits must span >= this in time
MAX_LINES_PER_PLANE = 4

# ---- distinctness: when are two lines really two DIFFERENT tracks? --------
# The failure mode to kill is OVER-SPLITTING: one compact deposit fit by two
# cospatial lines whose slopes merely differ a little. Two lines are distinct
# tracks only if their predicted positions separate by >= DISTINCT_SEP_MM
# somewhere over their UNION time span -- i.e. they are either spatially offset
# (two parallel tracks) or they fan/cross apart (a vertex). Cospatial
# over-splits stay close everywhere and are rejected. Evaluating over the union
# (not the shared) span is what lets a crossing pair, whose arms barely overlap
# in time, still register as separated at the far ends.
DISTINCT_SEP_MM = 14.0       # ~18 strips: real doubles open far wider than this

_RNG = np.random.default_rng(17)


def _line_is_track_grade(p, t, a, fit) -> bool:
    if fit is None:
        return False
    n_strips = len(np.unique(p))
    pspan = float(np.ptp(p))
    tspan = float(np.ptp(t))
    return (len(p) >= LINE_MIN_HITS and n_strips >= LINE_MIN_STRIPS
            and pspan >= LINE_MIN_PSPAN_MM and tspan >= LINE_MIN_TSPAN_NS
            and fit['r2'] >= LINE_MIN_R2
            and fit['inlier_frac'] >= LINE_MIN_INLIER_FRAC
            and fit['res_rms_mm'] <= LINE_MAX_RES_MM
            and S.occupancy(n_strips, pspan) >= LINE_MIN_OCC)


def _ransac_best_line(pos, time, amp, res=RANSAC_RES_MM, iters=RANSAC_ITERS):
    """Weighted RANSAC: return (inlier_mask, slope, inter) of the line with the
    largest amplitude-weighted inlier support, or None. Models are drawn from
    pairs of hits separated by >= RANSAC_MIN_DT_NS in time (so the seed defines
    a real slope, not a vertical pileup)."""
    n = len(pos)
    if n < LINE_MIN_HITS:
        return None
    t = time.astype(float)
    p = pos.astype(float)
    w = amp.astype(float)
    best_score = -1.0
    best_mask = None
    # candidate seed pairs
    for _ in range(iters):
        i, j = _RNG.integers(0, n, size=2)
        if abs(t[i] - t[j]) < RANSAC_MIN_DT_NS:
            continue
        slope = (p[i] - p[j]) / (t[i] - t[j])
        inter = p[i] - slope * t[i]
        resid = np.abs(p - (slope * t + inter))
        mask = resid <= res
        score = w[mask].sum()
        if score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None or best_mask.sum() < LINE_MIN_HITS:
        return None
    return best_mask


def extract_lines(pos, time, amp, tot=None,
                  max_lines=MAX_LINES_PER_PLANE) -> List[Dict]:
    """Sequential weighted RANSAC over ONE point set (a plane's clean hits, or
    one cluster's hits). Repeatedly pull out the best-supported line, refine it
    with the robust IRLS fit, keep it only if track-grade, remove its hits, and
    continue on the remainder. Returns a list of line dicts."""
    pos = np.asarray(pos, float)
    time = np.asarray(time, float)
    amp = np.asarray(amp, float)
    idx_all = np.arange(len(pos))
    remaining = np.ones(len(pos), bool)
    lines: List[Dict] = []
    for _ in range(max_lines):
        if remaining.sum() < LINE_MIN_HITS:
            break
        sub = idx_all[remaining]
        rr = _ransac_best_line(pos[sub], time[sub], amp[sub])
        if rr is None:
            break
        seed_idx = sub[rr]
        # refine with the robust IRLS fit on the RANSAC inliers
        fit = S.robust_line_fit(time[seed_idx], pos[seed_idx], w=amp[seed_idx])
        if fit is None:
            # cannot fit -> drop these hits to avoid an infinite loop
            remaining[seed_idx] = False
            continue
        keep = fit['inliers']
        gidx = seed_idx[keep]
        if not _line_is_track_grade(pos[gidx], time[gidx], amp[gidx], fit):
            # dominant remaining structure is not a track -> stop cleanly
            break
        p, t, a = pos[gidx], time[gidx], amp[gidx]
        lines.append(dict(
            slope_mm_ns=fit['slope_mm_ns'], intercept_mm=fit['intercept_mm'],
            r2=fit['r2'], inlier_frac=fit['inlier_frac'],
            res_rms_mm=fit['res_rms_mm'],
            n_hits=int(len(gidx)), n_strips=int(len(np.unique(p))),
            pspan_mm=float(np.ptp(p)), tspan_ns=float(np.ptp(t)),
            t0_ns=float(t.min()), t1_ns=float(t.max()),
            pos_lo_mm=float(p.min()), pos_hi_mm=float(p.max()),
            q_sum=float(a.sum()), a_max=float(a.max()),
            idx=gidx,
        ))
        remaining[gidx] = False
    return lines


def _max_union_sep(l1: Dict, l2: Dict) -> float:
    """Largest position gap between two lines over their UNION time span."""
    t_lo = min(l1['t0_ns'], l2['t0_ns'])
    t_hi = max(l1['t1_ns'], l2['t1_ns'])
    ts = np.linspace(t_lo, t_hi, 9)
    sep = np.abs((l1['slope_mm_ns'] * ts + l1['intercept_mm'])
                 - (l2['slope_mm_ns'] * ts + l2['intercept_mm']))
    return float(sep.max())


def _distinct(l1: Dict, l2: Dict) -> bool:
    """Two lines are different tracks -- judged WITHOUT extrapolation.

    Extrapolating two fits far beyond their measured time ranges turns a tiny
    slope difference into a fake separation. That produced a whole class of
    false doubles: ONE track running along the drift direction (constant
    position, long in time) chopped in two by a coherent noise band, whose
    fragments then "separated" only under extrapolation (e.g. run_61 ev2522).
    So:
      * lines that COEXIST in time -> compare positions only inside the shared
        window (a real pair is offset there, or has already fanned apart);
      * lines DISJOINT in time -> demand their MEASURED position bands be
        clearly apart; two fragments of one track sit at the same position and
        are correctly rejected.
    """
    t_ov = min(l1['t1_ns'], l2['t1_ns']) - max(l1['t0_ns'], l2['t0_ns'])
    if t_ov > 0:
        ts = np.linspace(max(l1['t0_ns'], l2['t0_ns']),
                         min(l1['t1_ns'], l2['t1_ns']), 9)
        sep = np.abs((l1['slope_mm_ns'] * ts + l1['intercept_mm'])
                     - (l2['slope_mm_ns'] * ts + l2['intercept_mm']))
        return bool(sep.max() >= DISTINCT_SEP_MM)
    # time-disjoint: gap between the measured position bands (>0 = disjoint)
    gap = max(l1['pos_lo_mm'], l2['pos_lo_mm']) - min(l1['pos_hi_mm'],
                                                      l2['pos_hi_mm'])
    return bool(gap >= DISTINCT_SEP_MM)


def distinct_lines(lines: List[Dict]) -> List[Dict]:
    """Greedily drop lines that are collinear with an already-kept (higher-q)
    line — those are fragments of one track, not a second track."""
    order = sorted(range(len(lines)), key=lambda k: -lines[k]['q_sum'])
    kept: List[Dict] = []
    for k in order:
        if all(_distinct(lines[k], kj) for kj in kept):
            kept.append(lines[k])
    return kept


def plane_lines(gc: pd.DataFrame) -> List[Dict]:
    """All distinct track-grade lines in one plane's CLEAN hits. Clusters first
    (so far-apart tracks are handled independently and cheaply), then RANSAC-
    splits each cluster, then de-duplicates collinear fragments across the
    plane."""
    if len(gc) < LINE_MIN_HITS:
        return []
    pos = gc['pos_mm'].to_numpy(float)
    tim = gc['time'].to_numpy(float)
    amp = gc['amplitude'].to_numpy(float)
    labels = S.cluster_plane(pos, tim)
    all_lines: List[Dict] = []
    for lab in np.unique(labels):
        m = labels == lab
        if m.sum() < LINE_MIN_HITS:
            continue
        sub = np.flatnonzero(m)
        for ln in extract_lines(pos[sub], tim[sub], amp[sub]):
            ln['idx'] = sub[ln['idx']]          # remap to plane-local indices
            all_lines.append(ln)
    return distinct_lines(all_lines)


def _pair_lines(xl: List[Dict], yl: List[Dict],
                drift: geo.DriftModel) -> List[Dict]:
    """Pair X and Y lines into 3-D local micro-TPC segments (bench time-IoU +
    charge balance). Mirrors reco.pairing.pair_xy_3d but on our line dicts."""
    if not xl or not yl:
        return []
    x_c = [dict(t0=s['t0_ns'], t1=s['t1_ns'], q=s['q_sum']) for s in xl]
    y_c = [dict(t0=s['t0_ns'], t1=s['t1_ns'], q=s['q_sum']) for s in yl]
    matches = mtpc.pair_planes(x_c, y_c, f_med=0.50, f_s68=0.09, min_iou=0.20)
    v = drift.v_mm_ns
    t0d = drift.t0_ns
    out = []
    for ix, iy, iou, pull in matches:
        sx, sy = xl[ix], yl[iy]
        t_lo = max(sx['t0_ns'], sy['t0_ns'])
        t_hi = min(sx['t1_ns'], sy['t1_ns'])
        line = lambda s, tt: s['slope_mm_ns'] * tt + s['intercept_mm']
        w_lo, w_hi = (t_lo - t0d) * v, (t_hi - t0d) * v
        p_lo = np.array([line(sx, t_lo), line(sy, t_lo), w_lo])
        p_hi = np.array([line(sx, t_hi), line(sy, t_hi), w_hi])
        out.append(dict(
            ix=ix, iy=iy, iou=float(iou), bal_pull=float(pull),
            dxdw=sx['slope_mm_ns'] / v, dydw=sy['slope_mm_ns'] / v,
            tan_theta=float(np.hypot(sx['slope_mm_ns'], sy['slope_mm_ns']) / v),
            q_x=sx['q_sum'], q_y=sy['q_sum'],
            p_lo_local=p_lo, p_hi_local=p_hi,
            t_lo_ns=t_lo, t_hi_ns=t_hi,
        ))
    return out


def _convergence(l1: Dict, l2: Dict):
    """Topology of two lines in a plane: where (time) do they cross, and do
    they open from a common point? Returns (t_cross_ns, sep_at_early_mm)."""
    ds = l1['slope_mm_ns'] - l2['slope_mm_ns']
    di = l1['intercept_mm'] - l2['intercept_mm']
    t_cross = -di / ds if abs(ds) > 1e-9 else np.nan
    t_early = min(l1['t0_ns'], l2['t0_ns'])
    sep_early = abs((l1['slope_mm_ns'] * t_early + l1['intercept_mm'])
                    - (l2['slope_mm_ns'] * t_early + l2['intercept_mm']))
    return float(t_cross), float(sep_early)


def analyze_event(g_ev: pd.DataFrame, drift: geo.DriftModel,
                  det: str = DET_A) -> Optional[Dict]:
    """Run the double-track finder on ONE event's Det-A hits (already
    noise-flagged: needs the 'clean' column). Returns a summary dict or None
    if the detector has too few clean hits to bother."""
    gd = g_ev[(g_ev['det'] == det)]
    if gd['clean'].sum() < 2 * LINE_MIN_HITS:
        return None
    # reseed RANSAC from the eventId so the result is reproducible and
    # independent of how many events were processed before this one.
    global _RNG
    _RNG = np.random.default_rng(int(g_ev['eventId'].iloc[0]) + 17)
    xl = plane_lines(gd[(gd['plane'] == 'x') & gd['clean']])
    yl = plane_lines(gd[(gd['plane'] == 'y') & gd['clean']])
    pairs = _pair_lines(xl, yl, drift)
    nx, ny = len(xl), len(yl)
    is_double = (nx >= 2 and ny >= 2)
    # topology from the two highest-charge X lines and two highest-charge Y
    topo = {}
    if is_double:
        xt = sorted(xl, key=lambda s: -s['q_sum'])[:2]
        yt = sorted(yl, key=lambda s: -s['q_sum'])[:2]
        tcx, sepx = _convergence(*xt)
        tcy, sepy = _convergence(*yt)
        # 'crossing/vertex' if the two lines cross INSIDE the drift window and
        # emerge close (small early separation) in at least one plane
        w_lo, w_hi = -600.0, 4200.0
        cross_in = [(w_lo < tc < w_hi) for tc in (tcx, tcy)]
        topo = dict(t_cross_x=tcx, t_cross_y=tcy,
                    sep_early_x=sepx, sep_early_y=sepy,
                    tag=('converging' if any(cross_in) else 'separated'))
    return dict(
        eventId=int(g_ev['eventId'].iloc[0]), det=det,
        n_xline=nx, n_yline=ny, n_pair=len(pairs),
        is_double=bool(is_double),
        q_lines=float(sum(l['q_sum'] for l in xl + yl)),
        min_r2=float(min([l['r2'] for l in xl + yl], default=np.nan)),
        xlines=xl, ylines=yl, pairs=pairs, topo=topo,
    )
