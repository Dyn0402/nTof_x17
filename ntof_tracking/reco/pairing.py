#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairing.py — X/Y plane pairing into per-chamber 3D micro-TPC segments.

Within one Micromegas, the X and Y strip planes sample the SAME ionisation
column, so a real 3D track produces one 'track' segment per plane with
near-identical time spans and balanced charge (bench PLAN_38: f = qx/(qx+qy)
narrow). Pairing reuses microtpc_lib.pair_planes (time IoU + charge balance).

A paired (x, y) segment gives a full local 3D line: positions from the two
strip coordinates, depth from drift time,
      w(t) = (t - t0_daq) * v_drift        [mm from the strip plane, inward]
so the local direction is  d ∝ (sx*v, sy*v, -v) ... expressed here as the
two transverse slopes dx/dw, dy/dw against drift depth. t0_daq (the DAQ time
at which charge born AT the strips arrives) and v_drift come from
geometry.DriftModel; both are calibration parameters, not truths.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .. import microtpc_lib as mtpc

MIN_IOU = 0.20
F_MED_DEFAULT, F_S68_DEFAULT = 0.50, 0.09


def pair_xy_3d(segs: List[dict], drift, f_balance: dict = None) -> List[dict]:
    """Pair per-plane 'track' segments detector-by-detector into 3D segments.

    segs: output of segments.segments_for_event (one event).
    drift: geometry.DriftModel (v_um_ns, t0_ns).
    f_balance: {det: (med, s68)} charge-balance priors (bench or in-situ).
    Returns a list of 3D-segment dicts in DETECTOR-LOCAL coordinates
    (x/y centred mm, w = drift depth mm from strip plane, inward positive).
    """
    f_balance = f_balance or {}
    out = []
    dets = sorted({s['det'] for s in segs})
    for det in dets:
        xs = [s for s in segs if s['det'] == det and s['plane'] == 'x'
              and s['cls'] == 'track']
        ys = [s for s in segs if s['det'] == det and s['plane'] == 'y'
              and s['cls'] == 'track']
        if not xs or not ys:
            continue
        f_med, f_s68 = f_balance.get(det, (F_MED_DEFAULT, F_S68_DEFAULT))
        x_c = [dict(t0=s['t0_ns'], t1=s['t1_ns'], q=s['q_sum']) for s in xs]
        y_c = [dict(t0=s['t0_ns'], t1=s['t1_ns'], q=s['q_sum']) for s in ys]
        pairs = mtpc.pair_planes(x_c, y_c, f_med=f_med, f_s68=f_s68,
                                 min_iou=MIN_IOU)
        for ix, iy, iou, pull in pairs:
            sx, sy = xs[ix], ys[iy]
            v = drift.v_mm_ns              # mm / ns
            t0d = drift.t0_ns
            # evaluate each plane's line at the common time span
            t_lo = max(sx['t0_ns'], sy['t0_ns'])
            t_hi = min(sx['t1_ns'], sy['t1_ns'])
            def line(s, t):
                return s['slope_mm_ns'] * t + s['intercept_mm']
            w_lo = (t_lo - t0d) * v        # drift depth, mm from strips
            w_hi = (t_hi - t0d) * v
            p_lo = np.array([line(sx, t_lo), line(sy, t_lo), w_lo])
            p_hi = np.array([line(sx, t_hi), line(sy, t_hi), w_hi])
            # transverse slopes per unit drift depth (dimensionless)
            dxdw = sx['slope_mm_ns'] / v
            dydw = sy['slope_mm_ns'] / v
            out.append(dict(
                eventId=sx['eventId'], det=det, iou=iou, bal_pull=pull,
                t_lo_ns=t_lo, t_hi_ns=t_hi,
                w_lo_mm=w_lo, w_hi_mm=w_hi,
                p_lo_local=p_lo, p_hi_local=p_hi,
                dxdw=dxdw, dydw=dydw,
                tan_theta=float(np.hypot(dxdw, dydw)),
                q_x=sx['q_sum'], q_y=sy['q_sum'],
                r2_x=sx.get('r2', np.nan), r2_y=sy.get('r2', np.nan),
                n_strips_x=sx['n_strips'], n_strips_y=sy['n_strips'],
                seg_x=sx, seg_y=sy,
            ))
    return out
