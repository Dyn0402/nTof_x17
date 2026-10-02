#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scint_stack_ana.py -- the scintillator stack, calibrated from the tracks that
point at it.

Reads the per-track tables `scint_stack` wrote (every gated track on all four
arms, with every wall end, both plastic bars and the liquid read out in a
prompt and a same-width pre-trigger window) and turns them into:

  1. **the extrapolation scale**, per arm and per layer, fitted from the
     scintillators' own fixed boundaries -- the wall's four read-out groups,
     the plastic's L/R gap and its two ends in v.  A pointing estimator: it
     uses only how the apparent boundary moves with the track's slope.
  2. **how far the MM tracks can be trusted** -- the fraction of tracks that
     no layer confirms, net of accidentals, as a function of every
     track-quality variable and of where the track sits on the chamber.
  3. **efficiency maps** of each layer, by tag and probe, net of accidentals,
     on the events ANOTHER arm triggered (the trigger is wall AND plastic in
     any arm, so the arm that triggered has its own wall and plastic lit by
     construction -- both samples are carried so the bias is visible).
  4. **gain maps** -- the wall's MIP response (geometric mean of the two ends,
     path-length corrected), the plastic's and the liquid's response in keVee.
  5. **the position along the wall bars** from the two ends: amplitude ratio
     and time difference against the MM-predicted v.
  6. **whether to demand both wall ends**, with the cost and the gain measured.
  7. **stability**, run by run.

    python -m sept26_prelim_analysis.scint_stack_ana            # all arms
    python -m sept26_prelim_analysis.scint_stack_ana --arms A
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.campaign_imaging import (  # noqa: E402
    in_block, run_number)

SCHEMA = 'sept26_prelim/scint_stack_ana/1'
ARMS = ('A', 'B', 'C', 'D')
WINS = ('on', 'off')

# --- geometry (identical on all four arms and all runs; `scint_stack`
# asserts it, and `geometry.csv` carries the per-arm numbers used below) ----
#: Wall read-out group edges in u, mm: four groups of four 25 mm bars.
WALL_EDGES = np.array([-225.0, -125.0, -25.0, 75.0, 175.0])
WALL_HALF_V = 250.0
PLAS_HALF_U, PLAS_HALF_V = 100.0, 150.0
LS_HALF_U, LS_HALF_V = 225.6, 225.3
#: The chamber's active area about its own centre, mm.  Outside it in v the
#: fitted y position RAILS (det_a_scint, 2026-09-10) -- carried as a flag.
MM_HALF_U, MM_HALF_V = 190.0, 170.0

#: Map binning on each layer, mm.
BIN = {'wall': 25.0, 'plas': 25.0, 'ls': 50.0, 'mm': 20.0}
MIN_CELL = 30
#: Before this, the n_TOF rate is high enough that the accidental floor of a
#: plastic bar reaches 10-20 % per track and half the tags are themselves
#: accidental (measured on arm A: 0.5-10 ms against <= 1 % past 10 ms).
LATE_MS = 10.0
#: Margin from the plastic's OUTER edges (its u ends and its v ends), mm.  The
#: L/R gap between the two bars is handled by a tolerant match instead (a
#: track within 1.5 widths of it may light either bar) -- see `predict`.
PLAS_MARGIN = 25.0
#: A chamber cell is masked when its in-time tracks are confirmed (wall OR
#: plastic, net) at less than this fraction of the arm's median cell.
MASK_FRAC = 0.5

#: Energy scale: the 2026-07-28 two-source calibration, mV per keVee.
SRCCAL = Path(REPO) / 'mx_july_beam_qa' / 'calib' / 'srccal_energy_calib.json'

TRACK_COLS = ['run', 'subrun', 'event_id', 'arm', 'n_trk', 'u_mm', 'v_mm',
              'tan_raw_x', 'tan_raw_y', 'k_arm', 'x_slope_reliable',
              'y_slope_reliable', 'chi2dof_x', 'chi2dof_y', 'x_n_strips',
              'y_n_strips', 'q_total', 'dca_axis_mm', 'drift_railed',
              't_since_flash_ns', 'other_trig', 'n_coinc_arms',
              'x_t0', 'y_t0', 'q_per_len', 'other_hw',
              'hw_A', 'hw_B', 'hw_C', 'hw_D']


def out_dir() -> Path:
    return paths.out('scint_stack')


def load_arm(arm: str) -> pd.DataFrame:
    """Every track of one arm, campaign-wide, with the columns used here."""
    cols = list(TRACK_COLS)
    for w in WINS:
        for n in range(1, 9):
            cols += [f'w{n}_amp_{w}', f'w{n}_dt_{w}', f'w{n}_sat_{w}']
        for p in ('p1', 'p2'):
            cols += [f'{p}_amp_{w}', f'{p}_dt_{w}', f'{p}_sat_{w}']
        cols += [f'l_amp_{w}', f'l_dt_{w}', f'l_sat_{w}', f'l_area_{w}']
    out = []
    for f in sorted(glob.glob(str(out_dir() / 'tracks' / 'stack_run_*.parquet'))):
        out.append(pd.read_parquet(f, columns=cols,
                                   filters=[('arm', '==', arm)]))
    d = pd.concat(out, ignore_index=True)
    for c in ('run', 'subrun', 'arm'):
        d[c] = d[c].astype(str)
    # the track's own time: where its drift times put the particle, ns from
    # the trigger.  Mean of the two planes where both have one.
    d['t0'] = d[['x_t0', 'y_t0']].mean(axis=1)
    # THE UNBIASED SAMPLE.  DREAM fires on (wall-sum AND plastic) in any arm;
    # on an event this arm triggered, its own wall and plastic fired by
    # construction, and a dead patch cannot show as a hole because tracks
    # through it never trigger.  `other_hw` = another arm satisfied the
    # EMULATED hardware trigger (`scint_stack.WALL_THR/PLAS_THR`), so the event
    # did not need this arm.  Restricted to late times, where it is clean.
    d['late'] = d.t_since_flash_ns.to_numpy() > LATE_MS * 1e6
    d['unb'] = d.other_hw.to_numpy() & d.late.to_numpy()
    d['self_hw'] = d[f'hw_{arm}'].to_numpy()
    return d


def geometry(arm: str) -> dict:
    G = pd.read_csv(out_dir() / 'geometry.csv')
    g = G[G.arm == arm].iloc[0].to_dict()
    # the plastic L/R gap: between bar 1's high edge and bar 2's low edge
    g['plas_gap'] = 0.5 * ((g['plas_u_1'] + PLAS_HALF_U)
                           + (g['plas_u_2'] - PLAS_HALF_U))
    g['plas_lo'] = g['plas_u_1'] - PLAS_HALF_U
    g['plas_hi'] = g['plas_u_2'] + PLAS_HALF_U
    return g


def energy_scale() -> dict:
    """{channel: (mv_per_kevee, offset_mv)}; the through-origin slope where a
    channel has only one fitted edge (the liquids).  LIQC has none."""
    J = json.loads(SRCCAL.read_text())['channels']
    out = {}
    for ch, c in J.items():
        if 'mv_per_kevee' in c:
            out[ch] = (c['mv_per_kevee'], c.get('offset_mv', 0.0))
        elif 'mv_per_kevee_origin' in c:
            out[ch] = (c['mv_per_kevee_origin'], 0.0)
    return out


# --------------------------------------------------------------------------- #
# 1. the extrapolation scale, from the scintillators' boundaries
# --------------------------------------------------------------------------- #
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nll_step(p, x0, tn, y, bpos, L):
    """Rising edges at ``bpos`` (one per row): P(high side) vs crossing."""
    s, lsig, alo, ahi = p[:4]
    dl = p[4:]
    x = x0 + L * s * tn - bpos - dl[0]
    P = _sig(alo) + (_sig(ahi) - _sig(alo)) * ndtr(x / np.exp(lsig))
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return -np.sum(np.where(y, np.log(P), np.log(1 - P)))


def _nll_edges(p, x0, tn, y, bidx, bpos, L):
    s, lsig, alo, ahi = p[:4]
    dl = p[4:]
    x = x0 + L * s * tn - bpos[bidx] - dl[bidx]
    P = _sig(alo) + (_sig(ahi) - _sig(alo)) * ndtr(x / np.exp(lsig))
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return -np.sum(np.where(y, np.log(P), np.log(1 - P)))


def _nll_window(p, x0, tn, y, lo, hi, L):
    """A plateau between two falling ends: P(fired) vs crossing in v."""
    s, lsig, alo, ahi, d1, d2 = p
    x = x0 + L * s * tn
    sg = np.exp(lsig)
    P = _sig(alo) + (_sig(ahi) - _sig(alo)) * (
        ndtr((x - lo - d1) / sg) * ndtr((hi + d2 - x) / sg))
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return -np.sum(np.where(y, np.log(P), np.log(1 - P)))


def _fit(nll, p0, args, n_prof=0):
    r = minimize(nll, p0, args=args, method='L-BFGS-B')
    h = 0.01

    def prof(ds):
        return minimize(lambda q: nll(np.r_[r.x[0] + ds, q], *args),
                        r.x[1:], method='L-BFGS-B').fun
    c = (prof(-h) + prof(h) - 2 * r.fun) / h ** 2
    return r.x, (1 / np.sqrt(c) if c > 0 else np.nan)


def wall_groups_fired(d: pd.DataFrame, win: str = 'on') -> np.ndarray:
    """(n, 4) bool: group g lit at EITHER end."""
    return np.stack([(d[f'w{2 * g + 1}_amp_{win}'].notna()
                      | d[f'w{2 * g + 2}_amp_{win}'].notna()).to_numpy()
                     for g in range(4)], 1)


def fit_wall_u(t: pd.DataFrame, L: float, s0: float) -> dict:
    """Shared scale over the three INTERNAL group boundaries.

    Single-track events with exactly one group lit, each boundary fitted on
    the tracks whose lit group is one of the two it separates.  The two outer
    wall ends are left out: past them the 'other side' is no group at all.
    """
    F = wall_groups_fired(t)
    one = F.sum(1) == 1
    u, tn = t.u_mm.to_numpy()[one], t.tan_raw_x.to_numpy()[one]
    gf = np.argmax(F[one], 1)
    X, T, Y, B = [], [], [], []
    for b in range(3):
        ub = WALL_EDGES[b + 1]
        m = (np.isin(gf, [b, b + 1]) & np.isfinite(tn)
             & (np.abs(u + L * s0 * tn - ub) < 90))
        X.append(u[m]), T.append(tn[m]), Y.append(gf[m] == b + 1)
        B.append(np.full(m.sum(), b))
    X, T, Y, B = map(np.concatenate, (X, T, Y, B))
    p, e = _fit(_nll_edges, np.r_[s0, np.log(15), -3, 3, 1, 1, 1],
                (X, T, Y, B, WALL_EDGES[1:4], L))
    return dict(s=p[0], s_err=e, sigma=np.exp(p[1]), floor_lo=_sig(p[2]),
                floor_hi=_sig(p[3]), delta=json.dumps(np.round(p[4:], 2).tolist()),
                n=len(X))


def fit_plas_u(t: pd.DataFrame, g: dict, L: float, s0: float) -> dict:
    """The L/R gap between the two plastic bars; exactly one bar lit."""
    P1 = t.p1_amp_on.notna().to_numpy()
    P2 = t.p2_amp_on.notna().to_numpy()
    one = P1 ^ P2
    u, tn = t.u_mm.to_numpy()[one], t.tan_raw_x.to_numpy()[one]
    y = P2[one]
    m = np.isfinite(tn) & (np.abs(u + L * s0 * tn - g['plas_gap']) < 120)
    p, e = _fit(_nll_step, np.r_[s0, np.log(30), -3, 3, 1],
                (u[m], tn[m], y[m], g['plas_gap'], L))
    return dict(s=p[0], s_err=e, sigma=np.exp(p[1]), floor_lo=_sig(p[2]),
                floor_hi=_sig(p[3]), delta=json.dumps([round(p[4], 2)]),
                n=int(m.sum()))


def fit_v_from_wall(t: pd.DataFrame, g: dict, s_u: float) -> dict:
    """The v scale, checked against the wall's OWN position measurement.

    The plastic cannot fix it: its ends at +-150 mm sit where the chamber's
    fitted y is already degrading (active area +-170, a rail just outside),
    and a first fit against them returned a scale of the wrong sign.  The
    wall's two-end amplitude ratio, ln(a1/a2), measures v along the bar with
    no MM input at all.  So scan the scale s_v in v_wall = v + L s_v tan_y
    and take the one whose prediction ln(a1/a2) follows most tightly (robust
    residual, per-group line).  Expected: s_v = s_u, because the slope scale
    is the drift-time scale and the x and y planes share it.
    """
    one = t[t.n_trk == 1]
    uw = one.u_mm.to_numpy() + g['L_wall'] * s_u * one.tan_raw_x.to_numpy()
    grp = np.digitize(uw, WALL_EDGES) - 1
    d_edge = np.min(np.abs(uw[:, None] - WALL_EDGES[None, :]), 1)
    a1 = np.full(len(one), np.nan)
    a2 = a1.copy()
    for gg in range(4):
        m = grp == gg
        a1[m] = one[f'w{2 * gg + 1}_amp_on'].to_numpy()[m]
        a2[m] = one[f'w{2 * gg + 2}_amp_on'].to_numpy()[m]
    lr = np.log(a1 / a2)
    v, ty = one.v_mm.to_numpy(), one.tan_raw_y.to_numpy()
    ok = (np.isfinite(lr) & np.isfinite(ty) & (d_edge > 25)
          & (grp >= 0) & (grp < 4) & (np.abs(v) < MM_HALF_V - 5))
    grid = np.arange(-0.4, 1.81, 0.1)
    res = []
    for s in grid:
        vw = v + g['L_wall'] * s * ty
        r2 = []
        for gg in range(4):
            m = ok & (grp == gg)
            b, a, sd = _robust_line(vw[m], lr[m])
            r2.append(sd)
        res.append(np.mean(r2))
    res = np.array(res)
    i = int(np.argmin(res))
    lo, hi = max(i - 2, 0), min(i + 3, len(grid))
    c = np.polyfit(grid[lo:hi], res[lo:hi], 2)
    s_best = -c[1] / (2 * c[0]) if c[0] > 0 else grid[i]
    return dict(s=float(s_best), s_err=np.nan, sigma=np.nan,
                floor_lo=np.nan, floor_hi=np.nan,
                delta=json.dumps(dict(grid=np.round(grid, 2).tolist(),
                                      lr_rms=np.round(res, 5).tolist())),
                n=int(ok.sum()))


def fit_scales(t: pd.DataFrame, g: dict) -> pd.DataFrame:
    one = t[t.n_trk == 1]
    k = np.nanmedian(one.k_arm) if one.k_arm.notna().any() else np.nan
    s0 = 1 / k if np.isfinite(k) else 0.8
    rows = []
    # two passes: the pre-selection window is set by the scale, so the second
    # pass re-centres it on the first pass's answer.
    w = fit_wall_u(one, g['L_wall'], s0)
    w = fit_wall_u(one, g['L_wall'], w['s'])
    rows.append(dict(layer='wall', axis='u', lever=g['L_wall'], **w))
    p = fit_plas_u(one, g, g['L_plas'], s0)
    p = fit_plas_u(one, g, g['L_plas'], p['s'])
    rows.append(dict(layer='plas', axis='u', lever=g['L_plas'], **p))
    v = fit_v_from_wall(one, g, w['s'])
    rows.append(dict(layer='wall', axis='v', lever=g['L_wall'], **v))
    R = pd.DataFrame(rows)
    R['inv_k'] = 1 / k if np.isfinite(k) else np.nan
    return R


def scale_by_run(t: pd.DataFrame, g: dict, s_all: float) -> pd.DataFrame:
    """The wall's u scale fitted run by run, beside each run's own 1/k.

    The k-block (runs 128-147, STATUS 2026-09-10) is a 2-8 % rise in `k` on
    every arm at once.  If it were a real change in how the tracks point, the
    scintillators would see the same change; if it is a property of the
    target-imaging fit, they would not.
    """
    rows = []
    for run, r in t[t.n_trk == 1].groupby('run'):
        if len(r) < 20000:
            continue
        try:
            w = fit_wall_u(r, g['L_wall'], s_all)
        except Exception:
            continue
        k = np.nanmedian(r.k_arm) if r.k_arm.notna().any() else np.nan
        rows.append(dict(run=run, rn=run_number(run), k_block=in_block(run),
                         s=w['s'], s_err=w['s_err'], sigma=w['sigma'],
                         n=w['n'], inv_k=1 / k if np.isfinite(k) else np.nan))
    return pd.DataFrame(rows).sort_values('rn')


# --------------------------------------------------------------------------- #
# 2. prediction and match, per track
# --------------------------------------------------------------------------- #
def predict(t: pd.DataFrame, g: dict, sc: dict, sig: dict) -> pd.DataFrame:
    """Crossings on every layer, the predicted channel, and what fired."""
    tx, ty = t.tan_raw_x.to_numpy(), t.tan_raw_y.to_numpy()
    u, v = t.u_mm.to_numpy(), t.v_mm.to_numpy()
    # v uses each layer's u scale: the slope scale is the drift-time scale,
    # common to the two strip planes; `fit_v_from_wall` checks it.
    su_w, su_p = sc['wall_u'], sc['plas_u']
    sv_w, sv = su_w, su_p
    P = pd.DataFrame(index=t.index)
    P['u_w'] = u + g['L_wall'] * su_w * tx
    P['v_w'] = v + g['L_wall'] * sv_w * ty
    P['u_p'] = u + g['L_plas'] * su_p * tx
    P['v_p'] = v + g['L_plas'] * sv * ty
    # the liquid sits 58 mm past the plastic; the plastic's scale is the
    # nearest measured one, and at 451 mm across the cell it is not critical.
    P['u_l'] = u + g['L_ls'] * su_p * tx
    P['v_l'] = v + g['L_ls'] * sv * ty
    # path-length factor through each layer (cos of the angle to its normal)
    P['cos_w'] = 1 / np.sqrt(1 + (su_w * tx) ** 2 + (sv_w * ty) ** 2)
    P['cos_p'] = 1 / np.sqrt(1 + (su_p * tx) ** 2 + (sv * ty) ** 2)

    uw = P.u_w.to_numpy()
    grp = np.digitize(uw, WALL_EDGES) - 1
    grp[(uw < WALL_EDGES[0]) | (uw >= WALL_EDGES[-1]) | ~np.isfinite(uw)] = -1
    P['grp'] = grp
    P['d_edge_w'] = np.min(np.abs(uw[:, None] - WALL_EDGES[None, :]), 1)
    P['on_w'] = (grp >= 0) & (np.abs(P.v_w) <= WALL_HALF_V)

    up = P.u_p.to_numpy()
    bar = np.where(up < g['plas_gap'], 1, 2)
    on_p = ((up >= g['plas_lo']) & (up <= g['plas_hi'])
            & (np.abs(P.v_p) <= PLAS_HALF_V))
    P['bar'] = np.where(on_p, bar, -1)
    P['d_gap_p'] = np.abs(up - g['plas_gap'])
    # within 1.5 widths of the L/R gap the prediction does not choose a bar
    amb = P.d_gap_p.to_numpy() < 1.5 * sig['plas_u']
    P['amb_p'] = amb
    # distance to the plastic's OUTER boundary only
    P['d_edge_p'] = np.minimum.reduce([
        up - g['plas_lo'], g['plas_hi'] - up,
        PLAS_HALF_V - np.abs(P.v_p.to_numpy())])
    P['on_p'] = on_p
    ul, vl = P.u_l.to_numpy() - g['u_ls'], P.v_l.to_numpy() - g['v_ls']
    P['on_l'] = (np.abs(ul) <= LS_HALF_U) & (np.abs(vl) <= LS_HALF_V)
    P['d_edge_l'] = np.minimum(LS_HALF_U - np.abs(ul), LS_HALF_V - np.abs(vl))

    for w in WINS:
        e1 = np.zeros(len(t), bool)
        e2 = np.zeros(len(t), bool)
        oth = np.zeros(len(t), bool)
        for gg in range(4):
            f1 = t[f'w{2 * gg + 1}_amp_{w}'].notna().to_numpy()
            f2 = t[f'w{2 * gg + 2}_amp_{w}'].notna().to_numpy()
            m = grp == gg
            e1 |= m & f1
            e2 |= m & f2
            oth |= (grp != gg) & (f1 | f2)
        P[f'e1_{w}'], P[f'e2_{w}'] = e1, e2
        P[f'wany_{w}'] = e1 | e2
        P[f'wboth_{w}'] = e1 & e2
        P[f'woth_{w}'] = oth
        f1 = t[f'p1_amp_{w}'].notna().to_numpy()
        f2 = t[f'p2_amp_{w}'].notna().to_numpy()
        strict = np.where(bar == 1, f1, f2) & on_p
        other = np.where(bar == 1, f2, f1)
        P[f'pms_{w}'] = strict
        # tolerant: the predicted bar, or the other one when on the gap
        P[f'pm_{w}'] = strict | (on_p & amb & other)
        P[f'poth_{w}'] = other & ~amb
        P[f'lf_{w}'] = t[f'l_amp_{w}'].notna().to_numpy()

    # the predicted channels' own hits, prompt window only
    a1 = np.full(len(t), np.nan)
    a2, t1, t2, s1, s2 = (a1.copy() for _ in range(5))
    for gg in range(4):
        m = grp == gg
        for arr, col in ((a1, f'w{2 * gg + 1}_amp_on'),
                         (a2, f'w{2 * gg + 2}_amp_on'),
                         (t1, f'w{2 * gg + 1}_dt_on'),
                         (t2, f'w{2 * gg + 2}_dt_on'),
                         (s1, f'w{2 * gg + 1}_sat_on'),
                         (s2, f'w{2 * gg + 2}_sat_on')):
            arr[m] = t[col].to_numpy()[m]
    P['a1'], P['a2'], P['t1'], P['t2'] = a1, a2, t1, t2
    P['wsat'] = (np.nan_to_num(s1) > 0) | (np.nan_to_num(s2) > 0)
    P['pa'] = np.where(bar == 1, t.p1_amp_on, t.p2_amp_on)
    P['pt'] = np.where(bar == 1, t.p1_dt_on, t.p2_dt_on)
    P['psat'] = np.nan_to_num(np.where(bar == 1, t.p1_sat_on, t.p2_sat_on)) > 0
    P['la'], P['lt'] = t.l_amp_on.to_numpy(), t.l_dt_on.to_numpy()
    P['larea'] = t.l_area_on.to_numpy()
    P['lsat'] = np.nan_to_num(t.l_sat_on.to_numpy()) > 0
    return P


# --------------------------------------------------------------------------- #
# helpers: net rates and their errors
# --------------------------------------------------------------------------- #
def net(k_on, k_off, n, c=0.0, q=0.0):
    """Accidental-subtracted probability, and its binomial error.

    eps = (P_on - P_off) / (1 - P_off): P_off is the same probe in the
    same-width pre-trigger window on the same tracks -- the chance that the
    channel is lit by something unrelated to the track.

    ``c`` is the fraction of the TAGS that are themselves accidental and ``q``
    the probe's net rate on such a track; eps -> (eps - c q) / (1 - c).  Both
    are measured by `tag_probe` on the parent sample, before the tag is
    required.  Past LATE_MS c is a few per mille and this changes nothing; it
    is there so that it is measured rather than assumed.
    """
    n = np.asarray(n, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_on = np.asarray(k_on, float) / n
        p_off = np.asarray(k_off, float) / n
        e = (p_on - p_off) / (1 - p_off)
        e = (e - c * q) / (1 - c)
        err = np.sqrt(np.clip(p_on * (1 - p_on), 1e-6, None) / n)
    return e, err


def cell_index(x, y, bx, by, x0, y0):
    return (np.floor((x - x0) / bx).astype(int),
            np.floor((y - y0) / by).astype(int))


# --------------------------------------------------------------------------- #
# 3. how much can the MM track be trusted
# --------------------------------------------------------------------------- #
QUALITY = {
    'chi2dof_max': ('max of the two planes\' chi2/dof',
                    [0, 1, 2, 3, 5, 8, 15, 1e9]),
    'n_strips_min': ('fewer of the two planes\' strip counts',
                     [0, 4, 6, 8, 10, 14, 20, 1e9]),
    'q_total': ('total charge', None),
    'dca_axis_mm': ('distance of closest approach to the beam axis, mm',
                    [0, 10, 20, 30, 50, 80, 120, 200, 1e9]),
    'v_mm': ('fitted y at the strip plane, mm',
             [-1e9, -185, -170, -120, -60, 0, 60, 120, 170, 185, 1e9]),
    'n_trk': ('gated tracks in this arm', [0, 1, 2, 3, 5, 1e9]),
    'drift_railed': ('drift fit railed', [-0.5, 0.5, 1.5]),
    'abs_tan_x': ('|raw slope|, x plane', [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1e9]),
    'abs_tan_y': ('|raw slope|, y plane', [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1e9]),
    't0': ('track t0 from its drift times, ns',
           [-1e9, -300, -200, -150, -100, -50, 0, 50, 100, 150, 200, 300,
            600, 1e9]),
    'pointing_y': ('y slope against the source direction: tan_y * sign(v)',
                   [-1e9, -0.3, -0.1, -0.03, 0.03, 0.1, 0.3, 1e9]),
}


def confirmation(t: pd.DataFrame, P: pd.DataFrame, sig: dict) -> tuple:
    """(summary rows, quality table, MM-plane map).

    The test population is every track whose line crosses the wall AND the
    plastic away from any boundary, so that a real particle following it has
    one unambiguous channel to light on each.  For it, 'confirmed' = the
    predicted wall group OR the predicted plastic bar lit.  The accidental
    floor is the same test in the pre-trigger window.
    """
    inter = (P.on_w & P.on_p & (P.d_edge_w > 2 * sig['wall_u'])
             & (P.d_edge_p > PLAS_MARGIN)).to_numpy()
    rows = []
    one = (t.n_trk == 1).to_numpy()
    unb = t.unb.to_numpy()
    it, good = t.intime.to_numpy(), t.good.to_numpy()
    for samp, sm in (('all', np.ones(len(t), bool)),
                     ('unbiased', unb),
                     ('single', one),
                     ('single_unbiased', one & unb),
                     ('single_intime', one & it),
                     ('single_outoftime', one & ~it),
                     ('single_good', one & good),
                     ('single_good_unbiased', one & good & unb)):
        m = inter & sm
        r = dict(sample=samp, n=int(m.sum()),
                 frac_of_arm=float(m.sum() / max(inter.sum(), 1)))
        for w in WINS:
            W = P[f'wany_{w}'].to_numpy()[m]
            Pm = P[f'pm_{w}'].to_numpy()[m]
            r[f'wall_{w}'] = W.mean()
            r[f'plas_{w}'] = Pm.mean()
            r[f'both_{w}'] = (W & Pm).mean()
            r[f'either_{w}'] = (W | Pm).mean()
            r[f'neither_{w}'] = (~W & ~Pm).mean()
            r[f'liq_{w}'] = P[f'lf_{w}'].to_numpy()[m].mean()
        rows.append(r)
    S = pd.DataFrame(rows)

    # against each quality variable, on the unbiased single-track sample
    q = pd.DataFrame(dict(
        chi2dof_max=np.fmax(t.chi2dof_x, t.chi2dof_y),
        n_strips_min=np.fmin(t.x_n_strips, t.y_n_strips),
        q_total=t.q_total, dca_axis_mm=t.dca_axis_mm, v_mm=t.v_mm,
        n_trk=t.n_trk, drift_railed=t.drift_railed.astype(float),
        abs_tan_x=t.tan_raw_x.abs(), abs_tan_y=t.tan_raw_y.abs(),
        pointing_y=t.tan_raw_y * np.sign(t.v_mm), t0=t.t0))
    E_on = (P.wany_on | P.pm_on).to_numpy()
    E_off = (P.wany_off | P.pm_off).to_numpy()
    Q = []
    for samp, base in (('all', inter), ('unbiased', inter & unb),
                       ('intime', inter & it)):
        for var, (label, edges) in QUALITY.items():
            x = q[var].to_numpy(float)
            if edges is None:
                edges = np.r_[np.nanquantile(x[base], np.linspace(0, 1, 9))]
                edges[-1] += 1e-6
            idx = np.digitize(x, edges) - 1
            nb = int((base & one).sum()) if var != 'n_trk' else int(base.sum())
            for i in range(len(edges) - 1):
                m = base & (idx == i)
                if var != 'n_trk':
                    m &= one
                n = int(m.sum())
                if n < 50:
                    continue
                e, err = net(E_on[m].sum(), E_off[m].sum(), n)
                Q.append(dict(sample=samp, var=var, label=label, lo=edges[i],
                              hi=edges[i + 1], n=n, on=E_on[m].mean(),
                              off=E_off[m].mean(), net=float(e),
                              err=float(err), frac_of_tracks=n / max(nb, 1)))
    Q = pd.DataFrame(Q)

    # where on the CHAMBER the unconfirmed tracks sit -- every track that
    # reaches both layers, not only the interior ones, so the rails show
    on2 = (P.on_w & P.on_p).to_numpy() & it & one
    b = BIN['mm']
    iu, iv = cell_index(t.u_mm.to_numpy(), t.v_mm.to_numpy(), b, b, -260, -260)
    M = pd.DataFrame(dict(iu=iu[on2], iv=iv[on2], on=E_on[on2],
                          off=E_off[on2]))
    M = M.groupby(['iu', 'iv']).agg(n=('on', 'size'), k_on=('on', 'sum'),
                                    k_off=('off', 'sum')).reset_index()
    M['u'] = -260 + (M.iu + 0.5) * b
    M['v'] = -260 + (M.iv + 0.5) * b
    M['net'], M['err'] = net(M.k_on, M.k_off, M.n)
    # all tracks (not only those reaching both layers): the occupancy
    iu, iv = cell_index(t.u_mm.to_numpy(), t.v_mm.to_numpy(), b, b, -260, -260)
    occ = (pd.DataFrame(dict(iu=iu, iv=iv)).groupby(['iu', 'iv']).size()
           .rename('n_all').reset_index())
    M = M.merge(occ, on=['iu', 'iv'], how='outer')
    M['u'] = -260 + (M.iu + 0.5) * b
    M['v'] = -260 + (M.iv + 0.5) * b
    return S, Q, M


def t0_window(t: pd.DataFrame, P: pd.DataFrame, sig: dict) -> tuple:
    """(profile, (lo, hi)): confirmation against the track's own t0.

    The chamber integrates over its whole drift window, so it reconstructs
    real particles that crossed it at other times than the trigger -- and no
    scintillator in the prompt window can confirm those.  The window is the
    contiguous run of 25 ns bins around the peak whose net confirmation (wall
    OR plastic, single tracks reaching both layers away from their edges) is
    at least half the peak's.
    """
    inter = (P.on_w & P.on_p & (P.d_edge_w > 2 * sig['wall_u'])
             & (P.d_edge_p > PLAS_MARGIN)).to_numpy() & (t.n_trk == 1).to_numpy()
    E_on = (P.wany_on | P.pm_on).to_numpy()
    E_off = (P.wany_off | P.pm_off).to_numpy()
    edges = np.arange(-800, 1001, 25.0)
    x = t.t0.to_numpy()
    idx = np.digitize(x, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        for samp, sm in (('all', np.ones(len(t), bool)),
                         ('unbiased', t.unb.to_numpy())):
            m = inter & sm & (idx == i)
            n = int(m.sum())
            e, err = (net(E_on[m].sum(), E_off[m].sum(), n) if n
                      else (np.nan, np.nan))
            rows.append(dict(sample=samp, lo=edges[i], hi=edges[i + 1], n=n,
                             net=float(e), err=float(err)))
    T = pd.DataFrame(rows)
    a = T[(T['sample'] == 'all') & (T.n >= 200)].reset_index(drop=True)
    ip = int(a.net.idxmax())
    half = 0.5 * a.net.iloc[ip]
    lo = hi = ip
    while lo > 0 and a.net.iloc[lo - 1] >= half:
        lo -= 1
    while hi < len(a) - 1 and a.net.iloc[hi + 1] >= half:
        hi += 1
    return T, (float(a.lo.iloc[lo]), float(a.hi.iloc[hi]))


def chamber_mask(t: pd.DataFrame, P: pd.DataFrame, sig: dict) -> pd.DataFrame:
    """Chamber cells (20 mm, strip-plane u v) whose tracks are not real.

    Built from IN-TIME single tracks that reach both the wall and the plastic
    away from their edges, scored by whether the wall OR the plastic confirms
    them.  OR, not AND, and not either alone: a dead patch of one layer must
    not mask itself out of that layer's own efficiency map, and with a 97 mm
    lever a wall hole projects back onto nearly the same chamber cell.  A cell
    is masked below MASK_FRAC of the arm's median cell; cells with fewer than
    MIN_CELL such tracks are not judged.  Outside the active area in v the
    fitted y rails, and those cells are masked whatever their score.
    """
    m = ((P.on_w & P.on_p & (P.d_edge_w > 2 * sig['wall_u'])
          & (P.d_edge_p > PLAS_MARGIN)).to_numpy()
         & (t.n_trk == 1).to_numpy() & t.intime.to_numpy())
    E_on = (P.wany_on | P.pm_on).to_numpy()
    E_off = (P.wany_off | P.pm_off).to_numpy()
    b = BIN['mm']
    iu, iv = cell_index(t.u_mm.to_numpy(), t.v_mm.to_numpy(), b, b, -260, -260)
    D = pd.DataFrame(dict(iu=iu[m], iv=iv[m], on=E_on[m], off=E_off[m]))
    M = D.groupby(['iu', 'iv']).agg(n=('on', 'size'), k_on=('on', 'sum'),
                                    k_off=('off', 'sum')).reset_index()
    M['net'], M['err'] = net(M.k_on, M.k_off, M.n)
    judged = M.n >= MIN_CELL
    med = float(M.net[judged].median())
    M['u'] = -260 + (M.iu + 0.5) * b
    M['v'] = -260 + (M.iv + 0.5) * b
    M['masked'] = (judged & (M.net < MASK_FRAC * med)) | (M.v.abs() > MM_HALF_V)
    M['arm_median'] = med
    return M


def apply_mask(t: pd.DataFrame, M: pd.DataFrame) -> np.ndarray:
    b = BIN['mm']
    iu, iv = cell_index(t.u_mm.to_numpy(), t.v_mm.to_numpy(), b, b, -260, -260)
    bad = set(zip(M.iu[M.masked], M.iv[M.masked]))
    rail = np.abs(t.v_mm.to_numpy()) > MM_HALF_V
    return ~(np.fromiter(((a, c) in bad for a, c in zip(iu, iv)), bool, len(t))
             | rail)


def edge_profiles(t: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Which channel fired, against where the fitted line says it should.

    Single good late tracks.  WALL: events with exactly one group lit, the
    share of each group per 10 mm of predicted u.  PLASTIC: exactly one bar
    lit, the share of bar 2.  LIQUID: tracks confirmed by the wall (both
    ends) and the plastic, the share that light the liquid, against u and
    against v on the cell.  With the right scale every boundary sits where
    the survey puts it and is as sharp as the extrapolation allows.
    """
    one = ((t.n_trk == 1) & t.late).to_numpy()
    rows = []
    F = wall_groups_fired(t)
    m = one & (F.sum(1) == 1) & (np.abs(P.v_w.to_numpy()) < WALL_HALF_V)
    gf = np.argmax(F, 1)
    e = np.arange(-260, 261, 10.0)
    idx = np.digitize(P.u_w.to_numpy(), e) - 1
    for i in range(len(e) - 1):
        mm = m & (idx == i)
        n = int(mm.sum())
        if n < MIN_CELL:
            continue
        for gg in range(4):
            rows.append(dict(layer='wall', axis='u', x=e[i] + 5, n=n,
                             what=f'g{gg}', p=float((gf[mm] == gg).mean())))
    P1 = t.p1_amp_on.notna().to_numpy()
    P2 = t.p2_amp_on.notna().to_numpy()
    m = one & (P1 ^ P2) & (np.abs(P.v_p.to_numpy()) < PLAS_HALF_V)
    idx = np.digitize(P.u_p.to_numpy(), e) - 1
    for i in range(len(e) - 1):
        mm = m & (idx == i)
        n = int(mm.sum())
        if n >= MIN_CELL:
            rows.append(dict(layer='plas', axis='u', x=e[i] + 5, n=n,
                             what='bar2', p=float(P2[mm].mean())))
    m = (one & P.wboth_on.to_numpy() & P.pm_on.to_numpy())
    e2 = np.arange(-300, 301, 25.0)
    for ax, col, other in (('u', 'u_l', 'v_l'), ('v', 'v_l', 'u_l')):
        x = P[col].to_numpy() - (0 if ax == 'u' else 0)
        idx = np.digitize(x, e2) - 1
        for i in range(len(e2) - 1):
            mm = m & (idx == i)
            n = int(mm.sum())
            if n < MIN_CELL:
                continue
            on = P.lf_on.to_numpy()[mm].sum()
            off = P.lf_off.to_numpy()[mm].sum()
            ef, _ = net(on, off, n)
            rows.append(dict(layer='liq', axis=ax, x=e2[i] + 12.5, n=n,
                             what='lf', p=float(ef)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. efficiency maps, tag and probe
# --------------------------------------------------------------------------- #
def eff_map(x, y, probe_on, probe_off, b, x0, y0, c=0.0, q=0.0
            ) -> pd.DataFrame:
    ix, iy = cell_index(x, y, b, b, x0, y0)
    D = pd.DataFrame(dict(ix=ix, iy=iy, on=probe_on, off=probe_off))
    M = D.groupby(['ix', 'iy']).agg(n=('on', 'size'), k_on=('on', 'sum'),
                                    k_off=('off', 'sum')).reset_index()
    M['x'] = x0 + (M.ix + 0.5) * b
    M['y'] = y0 + (M.iy + 0.5) * b
    M['eff'], M['err'] = net(M.k_on, M.k_off, M.n, c, q)
    M.loc[M.n < MIN_CELL, ['eff', 'err']] = np.nan
    return M


def tag_probe(parent, tag_on, tag_off, probe_on, probe_off) -> dict:
    """One tag-and-probe measurement, contamination included.

    ``parent`` is the geometric selection BEFORE the tag; the measurement is
    on parent & tag_on.  c = P(tag in the pre-trigger window) / P(tag in the
    prompt window) on the parent: the share of prompt tags that are accidental.
    q = the probe's net rate on parent tracks WITHOUT a prompt tag, standing in
    for what the probe does when the tag lied.
    """
    m = parent & tag_on
    n_p = max(int(parent.sum()), 1)
    c = float(tag_off[parent].sum() / max(tag_on[parent].sum(), 1))
    u = parent & ~tag_on
    q, _ = net(probe_on[u].sum(), probe_off[u].sum(), max(int(u.sum()), 1))
    k_on, k_off, n = probe_on[m].sum(), probe_off[m].sum(), int(m.sum())
    e, err = net(k_on, k_off, max(n, 1), c, float(q))
    e0, _ = net(k_on, k_off, max(n, 1))
    return dict(n=n, n_parent=n_p, on=k_on / max(n, 1), off=k_off / max(n, 1),
                tag_rate=float(tag_on[parent].mean()) if n_p else np.nan,
                c=c, q=float(q), eff_raw=float(e0), eff=float(e),
                err=float(err)), m, c, float(q)


def efficiencies(t: pd.DataFrame, P: pd.DataFrame, sig: dict, g: dict
                 ) -> tuple:
    """Per layer: (summary rows, maps).

    WALL    probe: predicted group lit (either end).  tag: predicted plastic
            bar lit.  A particle that reached the plastic crossed the wall.
    PLASTIC probe: predicted bar lit.  tag: predicted wall group lit at BOTH
            ends.  Electrons below ~1 MeV stop in the wall's 3 mm and its
            wrapping, so this is a response probability given a particle that
            crossed the wall, not a pure detector efficiency.
    LIQUID  probe: the cell lit.  tag: wall (both ends) AND plastic.  The
            plastic is 20 mm of PVT, which stops electrons below ~4 MeV, so
            for the beam's few-MeV electrons this is mostly a PUNCH-THROUGH
            probability.  It is mapped anyway: a shadowed or dead region shows
            as a hole in a map whose punch-through part is smooth.

    Three samples.  ``unbiased``: another arm satisfied the (emulated)
    hardware trigger, past LATE_MS -- the number to quote.  ``all_late``: every
    late track, which includes the events this arm triggered itself, so its
    own wall and plastic fired by construction; it is carried because it has
    ~10x the statistics and its difference from ``unbiased`` IS the trigger
    bias.  ``all``: no time cut, accidental-dominated below 10 ms.
    """
    single = (t.n_trk == 1).to_numpy()
    rows, maps = [], []
    A = lambda c: P[c].to_numpy()  # noqa: E731
    in_w = A('on_w') & (A('d_edge_w') > 2 * sig['wall_u'])
    in_p = A('on_p') & (A('d_edge_p') > PLAS_MARGIN)
    in_l = A('on_l') & (A('d_edge_l') > 40)
    for samp, sm in (('unbiased', t.unb.to_numpy() & single),
                     ('all_late', t.late.to_numpy() & single),
                     ('all', single)):
        # the unbiased sample is a few per cent of the tracks: coarser cells
        fb = 2.0 if samp == 'unbiased' else 1.0
        # --- wall, tagged by the plastic
        parent = sm & in_w & in_p
        for probe in ('wany', 'wboth'):
            r, m, c, q = tag_probe(parent, A('pm_on'), A('pm_off'),
                                   A(f'{probe}_on'), A(f'{probe}_off'))
            rows.append(dict(layer='wall', probe=probe, sample=samp, **r))
            if probe == 'wany':
                M = eff_map(A('u_w')[m], A('v_w')[m], A('wany_on')[m],
                            A('wany_off')[m], fb * BIN['wall'], -250, -275,
                            c, q)
                maps.append(M.assign(layer='wall', sample=samp))
        for gg in range(4):
            r, *_ = tag_probe(parent & (A('grp') == gg), A('pm_on'),
                              A('pm_off'), A('wany_on'), A('wany_off'))
            rows.append(dict(layer='wall', probe=f'wany_g{gg}', sample=samp,
                             **r))
        # --- plastic, tagged by the wall at both ends
        parent = sm & in_p & in_w
        r, m, c, q = tag_probe(parent, A('wboth_on'), A('wboth_off'),
                               A('pm_on'), A('pm_off'))
        rows.append(dict(layer='plas', probe='pm', sample=samp, **r))
        M = eff_map(A('u_p')[m], A('v_p')[m], A('pm_on')[m], A('pm_off')[m],
                    fb * BIN['plas'], -250, -175, c, q)
        maps.append(M.assign(layer='plas', sample=samp))
        for bar in (1, 2):
            r, *_ = tag_probe(parent & (A('bar') == bar) & ~A('amb_p'),
                              A('wboth_on'), A('wboth_off'), A('pms_on'),
                              A('pms_off'))
            rows.append(dict(layer='plas', probe=f'pm_bar{bar}', sample=samp,
                             **r))
        # --- liquid, tagged by wall (both ends) AND plastic
        parent = sm & in_l & in_w & in_p
        t_on = A('wboth_on') & A('pm_on')
        t_off = A('wboth_off') & A('pm_off')
        r, m, c, q = tag_probe(parent, t_on, t_off, A('lf_on'), A('lf_off'))
        rows.append(dict(layer='liq', probe='lf', sample=samp, **r))
        M = eff_map(A('u_l')[m] - g['u_ls'], A('v_l')[m] - g['v_ls'],
                    A('lf_on')[m], A('lf_off')[m], fb * BIN['ls'], -250, -250,
                    c, q)
        maps.append(M.assign(layer='liq', sample=samp))
        # the liquid's two halves, behind plastic bar 1 and bar 2 -- the
        # July source runs saw LIQA and LIQD answer to one bar only
        for bar in (1, 2):
            r, *_ = tag_probe(parent & (A('bar') == bar) & ~A('amb_p'),
                              t_on, t_off, A('lf_on'), A('lf_off'))
            rows.append(dict(layer='liq', probe=f'lf_behind_bar{bar}',
                             sample=samp, **r))
    return pd.DataFrame(rows), pd.concat(maps, ignore_index=True)


def liquid_vs_plastic(t, P, sig, ecal, arm) -> pd.DataFrame:
    """Liquid response against the energy the particle left in the plastic.

    A through-going minimum-ionising particle leaves ~4 MeV in 20 mm of PVT;
    an electron that stops there leaves all of its energy.  If the liquid
    answers mostly where the plastic saw a through-going deposit, the liquid
    is seeing punch-through, and its efficiency is that of the cell.
    """
    m = ((t.n_trk == 1) & P.on_l & (P.d_edge_l > 40) & P.wboth_on & P.pm_on
         & (P.d_edge_p > PLAS_MARGIN) & ~P.psat & ~P.amb_p).to_numpy()
    E = plastic_kevee(P, ecal, arm)
    rows = []
    edges = [0, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 8000,
             12000, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = m & (E >= lo) & (E < hi)
        n = int(mm.sum())
        if n < 50:
            continue
        on, off = P.lf_on.to_numpy()[mm], P.lf_off.to_numpy()[mm]
        e, err = net(on.sum(), off.sum(), n)
        for samp, sm in (('all_late', t.late.to_numpy()),
                         ('unbiased', t.unb.to_numpy())):
            ms = mm & sm
            if ms.sum() < 50:
                continue
            on, off = P.lf_on.to_numpy()[ms], P.lf_off.to_numpy()[ms]
            e, err = net(on.sum(), off.sum(), ms.sum())
            rows.append(dict(sample=samp, e_lo=lo, e_hi=hi, n=int(ms.sum()),
                             on=on.mean(), off=off.mean(), eff=float(e),
                             err=float(err)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. gain maps
# --------------------------------------------------------------------------- #
def plastic_kevee(P, ecal, arm):
    mv = P.pa.to_numpy() * P.cos_p.to_numpy()
    out = np.full(len(P), np.nan)
    for bar in (1, 2):
        k, off = ecal.get(f'PSS{arm}{bar}', (np.nan, 0.0))
        m = P.bar.to_numpy() == bar
        out[m] = (mv[m] - off) / k
    return out


def liquid_kevee(P, ecal, arm):
    k, off = ecal.get(f'LIQ{arm}1', (np.nan, 0.0))
    return (P.la.to_numpy() - off) / k


def _median_map(x, y, val, b, x0, y0):
    ix, iy = cell_index(x, y, b, b, x0, y0)
    D = pd.DataFrame(dict(ix=ix, iy=iy, val=val))
    M = D.groupby(['ix', 'iy']).val.agg(['size', 'median']).reset_index()
    M.columns = ['ix', 'iy', 'n', 'med']
    M['x'] = x0 + (M.ix + 0.5) * b
    M['y'] = y0 + (M.iy + 0.5) * b
    M.loc[M.n < MIN_CELL, 'med'] = np.nan
    return M


def gains(t, P, sig, g, ecal, arm) -> tuple:
    """(maps, wall attenuation, summary).

    WALL    the geometric mean of the two ends, sqrt(a1 a2), times cos(theta):
            for an exponentially attenuating bar the product of the two ends
            does not depend on where along the bar the light was made, so this
            is the MIP response of the scintillator itself.  Both ends lit,
            neither saturated, track interior to its group.
    PLASTIC the predicted bar's amplitude times cos(theta), in keVee.
    LIQUID  the cell's amplitude in keVee (no path correction: what reaches it
            is not a straight-through track).
    Medians, not means: the wall's MIP response is a Landau.

    Two samples, and the difference matters most for the PLASTIC: the trigger
    needs a plastic bar above ~0.9 MIP (118-157 mV), so on the events an arm
    triggered itself its plastic spectrum is cut just below its own median.
    ``unbiased`` is free of that; ``all_late`` has the statistics.
    """
    single = (t.n_trk == 1).to_numpy()
    A_ = lambda c: P[c].to_numpy()  # noqa: E731
    maps, rows = [], []
    G = np.sqrt(A_('a1') * A_('a2')) * A_('cos_w')
    E = plastic_kevee(P, ecal, arm)
    El = liquid_kevee(P, ecal, arm)
    lunit = 'keVee' if np.isfinite(El).any() else 'mV'
    if lunit == 'mV':
        El = A_('la')
    for samp, sm in (('unbiased', single & t.unb.to_numpy()),
                     ('all_late', single & t.late.to_numpy())):
        fb = 2.0 if samp == 'unbiased' else 1.0
        # wall
        mw = (sm & A_('wboth_on') & ~A_('wsat')
              & (A_('d_edge_w') > 2 * sig['wall_u']) & A_('pm_on'))
        M = _median_map(A_('u_w')[mw], A_('v_w')[mw], G[mw],
                        fb * BIN['wall'], -250, -275)
        maps.append(M.assign(layer='wall', quantity='wall_gm', unit='mV',
                             sample=samp))
        for gg in range(4):
            mg = mw & (A_('grp') == gg)
            rows.append(dict(sample=samp, layer='wall', channel=f'g{gg}',
                             n=int(mg.sum()), unit='mV',
                             median=float(np.median(G[mg])) if mg.any()
                             else np.nan))
        # plastic: the bar must be unambiguous, since the energy is its own
        mp = (sm & A_('pms_on') & ~A_('psat') & A_('wboth_on')
              & (A_('d_edge_p') > PLAS_MARGIN) & ~A_('amb_p'))
        M = _median_map(A_('u_p')[mp], A_('v_p')[mp], E[mp],
                        fb * BIN['plas'], -250, -175)
        maps.append(M.assign(layer='plas', quantity='plas_kevee',
                             unit='keVee', sample=samp))
        mt = mp & A_('lf_on')
        M = _median_map(A_('u_p')[mt], A_('v_p')[mt], E[mt], 50.0, -250, -175)
        maps.append(M.assign(layer='plas', quantity='plas_kevee_through',
                             unit='keVee', sample=samp))
        for bar in (1, 2):
            mb = mp & (A_('bar') == bar)
            sat_base = sm & A_('pms_on') & (A_('bar') == bar)
            rows.append(dict(sample=samp, layer='plas', channel=f'bar{bar}',
                             n=int(mb.sum()), unit='keVee',
                             median=float(np.nanmedian(E[mb])) if mb.any()
                             else np.nan,
                             median_mv=float(np.nanmedian(
                                 A_('pa')[mb] * A_('cos_p')[mb]))
                             if mb.any() else np.nan,
                             frac_sat=float(A_('psat')[sat_base].mean())
                             if sat_base.any() else np.nan))
            mbt = mb & A_('lf_on')
            rows.append(dict(sample=samp, layer='plas',
                             channel=f'bar{bar}_through', n=int(mbt.sum()),
                             unit='keVee',
                             median=float(np.nanmedian(E[mbt])) if mbt.any()
                             else np.nan))
        # liquid
        ml = (sm & A_('lf_on') & ~A_('lsat') & A_('wboth_on') & A_('pm_on')
              & A_('on_l'))
        M = _median_map(A_('u_l')[ml] - g['u_ls'], A_('v_l')[ml] - g['v_ls'],
                        El[ml], fb * BIN['ls'], -250, -250)
        maps.append(M.assign(layer='liq', quantity='liq_' + lunit.lower(),
                             unit=lunit, sample=samp))
        sat_base = sm & A_('lf_on')
        rows.append(dict(sample=samp, layer='liq', channel='cell',
                         n=int(ml.sum()), unit=lunit,
                         median=float(np.nanmedian(El[ml])) if ml.any()
                         else np.nan,
                         frac_sat=float(A_('lsat')[sat_base].mean())
                         if sat_base.any() else np.nan))

    # attenuation along the bar, each end separately (late tracks; the ratio
    # of the two ends does not care about the trigger threshold on their sum
    # except at the very ends, which the report shows)
    mw = (single & t.late.to_numpy() & A_('wboth_on') & ~A_('wsat')
          & (A_('d_edge_w') > 2 * sig['wall_u']) & A_('pm_on')
          & (np.abs(t.v_mm.to_numpy()) < MM_HALF_V - 5))
    vw = A_('v_w')
    At = []
    for gg in range(4):
        mg = mw & (A_('grp') == gg)
        for end, arr in ((1, A_('a1') * A_('cos_w')),
                         (2, A_('a2') * A_('cos_w'))):
            for vlo in np.arange(-250, 250, 25.0):
                mm = mg & (vw >= vlo) & (vw < vlo + 25)
                if mm.sum() < MIN_CELL:
                    continue
                At.append(dict(grp=gg, end=end, v=vlo + 12.5, n=int(mm.sum()),
                               median=float(np.median(arr[mm]))))
    return pd.concat(maps, ignore_index=True), pd.DataFrame(At), pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. position along the wall bars, from the two ends
# --------------------------------------------------------------------------- #
def _robust_line(x, y, n_iter=10, k=2.5):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    keep = np.ones(len(x), bool)
    for _ in range(n_iter):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = y - (a + b * x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) < k * s
        if (new == keep).all():
            break
        keep = new
    return b, a, s


def rsig(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def wall_position(t, P, sig) -> tuple:
    """(per-group calibration, resolution table, residual sample).

    For a track interior to its group with both ends lit and unsaturated, the
    MM line predicts v on the bar.  Fit, per group,

        ln(a1/a2) = a_lr + b_lr v,      t1 - t2 = a_dt + b_dt v

    -- 2/|b_lr| is the bar's effective attenuation length and 2/|b_dt| the
    effective speed of light along it -- then invert each to a position and
    compare it with the prediction.  The residual is the wall's position
    resolution PLUS the MM's prediction error at the wall, so it is an UPPER
    LIMIT on the wall's; the MM part is estimated from the plastic's v edge
    width scaled to the wall's lever and taken off in quadrature, labelled.

    Calibrated on the tracks with |v_mm| < 165, i.e. not railed, and on
    alternating events (even event_id) so the residuals are on the other half.
    """
    m = ((t.n_trk == 1) & t.late & P.wboth_on & ~P.wsat
         & (P.d_edge_w > 2 * sig['wall_u'])
         & (np.abs(t.v_mm) < MM_HALF_V - 5) & P.pm_on).to_numpy()
    lr = np.log(P.a1.to_numpy() / P.a2.to_numpy())
    dt = P.t1.to_numpy() - P.t2.to_numpy()
    vw = P.v_w.to_numpy()
    grp = P.grp.to_numpy()
    even = (t.event_id.to_numpy() % 2) == 0
    C, R = [], []
    vrec = {k: np.full(len(t), np.nan) for k in ('lr', 'dt')}
    for gg in range(4):
        mg = m & (grp == gg)
        rec = dict(grp=gg, n=int(mg.sum()))
        for est, y in (('lr', lr), ('dt', dt)):
            b, a, s = _robust_line(vw[mg & even], y[mg & even])
            rec[f'{est}_slope'], rec[f'{est}_icpt'], rec[f'{est}_rms'] = b, a, s
            vrec[est][mg] = (y[mg] - a) / b
        rec['atten_mm'] = 2 / abs(rec['lr_slope'])
        rec['c_eff_mm_ns'] = 2 / abs(rec['dt_slope'])
        C.append(rec)
    C = pd.DataFrame(C)
    test = m & ~even
    res = {k: vrec[k] - vw for k in vrec}
    s_lr, s_dt = rsig(res['lr'][test]), rsig(res['dt'][test])
    w_lr, w_dt = 1 / s_lr ** 2, 1 / s_dt ** 2
    comb = (w_lr * vrec['lr'] + w_dt * vrec['dt']) / (w_lr + w_dt)
    res['comb'] = comb - vw
    G = np.sqrt(P.a1.to_numpy() * P.a2.to_numpy())
    for est in ('lr', 'dt', 'comb'):
        R.append(dict(est=est, by='all', lo=np.nan, hi=np.nan,
                      n=int(test.sum()), sigma=rsig(res[est][test]),
                      bias=float(np.nanmedian(res[est][test]))))
        for lo, hi in ((-150, -75), (-75, 0), (0, 75), (75, 150)):
            mm = test & (vw >= lo) & (vw < hi)
            R.append(dict(est=est, by='v_wall', lo=lo, hi=hi, n=int(mm.sum()),
                          sigma=rsig(res[est][mm]),
                          bias=float(np.nanmedian(res[est][mm]))))
        q = np.nanquantile(G[test], np.linspace(0, 1, 6))
        for lo, hi in zip(q[:-1], q[1:]):
            mm = test & (G >= lo) & (G < hi)
            R.append(dict(est=est, by='amp_gm_mV', lo=lo, hi=hi,
                          n=int(mm.sum()), sigma=rsig(res[est][mm]),
                          bias=float(np.nanmedian(res[est][mm]))))
    R = pd.DataFrame(R)
    rng = np.random.default_rng(1)
    idx = np.flatnonzero(test)
    idx = rng.choice(idx, min(len(idx), 60000), replace=False)
    S = pd.DataFrame(dict(grp=grp[idx], v_pred=vw[idx], lr=lr[idx],
                          dt=dt[idx], v_lr=vrec['lr'][idx],
                          v_dt=vrec['dt'][idx], v_comb=comb[idx],
                          gm=G[idx]))
    return C, R, S


# --------------------------------------------------------------------------- #
# 7. both ends?
# --------------------------------------------------------------------------- #
def both_ends(t, P, sig, cal: pd.DataFrame) -> tuple:
    """(summary, profile along the bar).

    The question: a group counts as lit today if EITHER end fired -- in the
    hardware (an analog sum of the two ends is discriminated) and offline
    (stage 1, `efficiency`, `det_a_scint`).  Requiring both, and optionally
    a time coincidence between them, would cost the real hits that light only
    one end and save the accidentals that do.  Both are measured here:

      * REAL: tracks confirmed by the plastic, interior to their group, in the
        prompt window, net of the same in the pre-trigger window.
      * ACCIDENTAL: the identical test in the pre-trigger window -- whatever
        lights the predicted group there is unrelated to the track.

    The time cut is |(t1 - t2) - predicted(v)| < 3 sigma, the prediction from
    `wall_position`'s per-group line.
    """
    base = ((t.n_trk == 1) & P.on_w & (P.d_edge_w > 2 * sig['wall_u'])
            & P.pm_on).to_numpy()
    grp = P.grp.to_numpy()
    vw = P.v_w.to_numpy()
    # time coincidence on the prompt hits
    dt = P.t1.to_numpy() - P.t2.to_numpy()
    pred = np.full(len(t), np.nan)
    rms = np.full(len(t), np.nan)
    for _, r in cal.iterrows():
        mg = grp == r.grp
        pred[mg] = r.dt_icpt + r.dt_slope * vw[mg]
        rms[mg] = r.dt_rms
    tcoinc = np.abs(dt - pred) < 3 * rms
    rows = []
    for samp, sm in (('unbiased', t.unb.to_numpy()),
                     ('all_late', t.late.to_numpy()),
                     ('all', np.ones(len(t), bool))):
        m = base & sm
        n = int(m.sum())
        r = dict(sample=samp, n=n)
        for w in WINS:
            e1 = P[f'e1_{w}'].to_numpy()[m]
            e2 = P[f'e2_{w}'].to_numpy()[m]
            r[f'any_{w}'] = (e1 | e2).mean()
            r[f'both_{w}'] = (e1 & e2).mean()
            r[f'only1_{w}'] = (e1 & ~e2).mean()
            r[f'only2_{w}'] = (~e1 & e2).mean()
        r['both_tcoinc_on'] = (P.wboth_on.to_numpy() & tcoinc)[m].mean()
        r['eff_any'], _ = net(r['any_on'] * n, r['any_off'] * n, n)
        r['eff_both'], _ = net(r['both_on'] * n, r['both_off'] * n, n)
        r['keep_real_both'] = r['eff_both'] / r['eff_any']
        r['keep_acc_both'] = r['both_off'] / r['any_off'] if r['any_off'] else np.nan
        rows.append(r)
    S = pd.DataFrame(rows)
    # along the bar: where do the one-ended real hits sit?
    m = base & t.late.to_numpy()
    Pr = []
    for vlo in np.arange(-250, 250, 25.0):
        mm = m & (vw >= vlo) & (vw < vlo + 25)
        if mm.sum() < MIN_CELL:
            continue
        e1, e2 = P.e1_on.to_numpy()[mm], P.e2_on.to_numpy()[mm]
        Pr.append(dict(v=vlo + 12.5, n=int(mm.sum()), any=(e1 | e2).mean(),
                       both=(e1 & e2).mean(), only1=(e1 & ~e2).mean(),
                       only2=(~e1 & e2).mean()))
    return S, pd.DataFrame(Pr)


def accidental_rates(t: pd.DataFrame) -> pd.DataFrame:
    """Per group: how often the pre-trigger window lights one end vs both.

    Every track's event, whatever it points at: this is the channel's own
    background, and whether it comes in pairs (a real particle, which fires
    both ends) or singly (dark counts, noise, a hit near one end).
    """
    rows = []
    for gg in range(4):
        f1 = t[f'w{2 * gg + 1}_amp_off'].notna().to_numpy()
        f2 = t[f'w{2 * gg + 2}_amp_off'].notna().to_numpy()
        ev = ~t.duplicated(['run', 'subrun', 'event_id']).to_numpy()
        f1, f2 = f1[ev], f2[ev]
        rows.append(dict(grp=gg, n_events=int(ev.sum()),
                         p_any=(f1 | f2).mean(), p_both=(f1 & f2).mean(),
                         p_only1=(f1 & ~f2).mean(), p_only2=(~f1 & f2).mean()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 8. run by run
# --------------------------------------------------------------------------- #
def by_run(t, P, sig, ecal, arm) -> pd.DataFrame:
    # late tracks only: below LATE_MS the accidentals move every median
    single = ((t.n_trk == 1) & t.late).to_numpy()
    Gm = np.sqrt(P.a1.to_numpy() * P.a2.to_numpy()) * P.cos_w.to_numpy()
    E = plastic_kevee(P, ecal, arm)
    mw = (single & P.wboth_on.to_numpy() & ~P.wsat.to_numpy()
          & (P.d_edge_w > 2 * sig['wall_u']).to_numpy() & P.pm_on.to_numpy())
    mp = (single & P.pms_on.to_numpy() & ~P.psat.to_numpy()
          & ~P.amb_p.to_numpy() & P.wboth_on.to_numpy())
    ml = mp & P.lf_on.to_numpy()
    mi = (single & t.unb.to_numpy() & P.on_w.to_numpy()
          & (P.d_edge_w > 2 * sig['wall_u']).to_numpy()
          & P.pm_on.to_numpy())
    rows = []
    run = t.run.to_numpy()
    for r in pd.unique(run):
        m = run == r
        n_i = int((mi & m).sum())
        e, err = net(P.wany_on.to_numpy()[mi & m].sum(),
                     P.wany_off.to_numpy()[mi & m].sum(), max(n_i, 1))
        rows.append(dict(run=r, rn=run_number(r), k_block=in_block(r),
                         n_tracks=int(m.sum()),
                         wall_gm_mV=float(np.median(Gm[mw & m]))
                         if (mw & m).sum() > 100 else np.nan,
                         plas_kevee=float(np.nanmedian(E[mp & m]))
                         if (mp & m).sum() > 100 else np.nan,
                         liq_amp_mV=float(np.nanmedian(P.la.to_numpy()[ml & m]))
                         if (ml & m).sum() > 50 else np.nan,
                         wall_eff=float(e) if n_i > 100 else np.nan,
                         wall_eff_err=float(err),
                         liq_given_plas=float(P.lf_on.to_numpy()[mp & m].mean()
                                              - P.lf_off.to_numpy()[mp & m].mean())
                         if (mp & m).sum() > 100 else np.nan))
    return pd.DataFrame(rows).sort_values('rn')


# --------------------------------------------------------------------------- #
def _scales(S):
    sc = {'wall_u': float(S[(S.layer == 'wall') & (S.axis == 'u')].s.iloc[0]),
          'plas_u': float(S[(S.layer == 'plas') & (S.axis == 'u')].s.iloc[0]),
          'wall_v_check': float(S[(S.layer == 'wall')
                                  & (S.axis == 'v')].s.iloc[0])}
    sig = {'wall_u': float(S[(S.layer == 'wall') & (S.axis == 'u')].sigma.iloc[0]),
           'plas_u': float(S[(S.layer == 'plas') & (S.axis == 'u')].sigma.iloc[0])}
    return sc, sig


def run_arm(arm: str) -> dict:
    """Two passes.  The first fits the scales on every single track, which is
    enough to find the in-time window and the chamber cells whose tracks are
    not real; the second refits on the tracks that are left, and every product
    is built from that.  Both passes' scales are stored."""
    t0 = time.time()
    od = out_dir() / 'ana'
    od.mkdir(parents=True, exist_ok=True)
    g = geometry(arm)
    ecal = energy_scale()
    t = load_arm(arm)
    log = [f'{arm}: {len(t):,} tracks loaded [{time.time() - t0:.0f} s]']

    S1 = fit_scales(t, g).assign(arm=arm, fit_pass=1)
    sc, sig = _scales(S1)
    P = predict(t, g, sc, sig)
    TP, (lo, hi) = t0_window(t, P, sig)
    t['intime'] = (t.t0 >= lo) & (t.t0 < hi)
    M = chamber_mask(t, P, sig)
    t['mm_ok'] = apply_mask(t, M)
    t['good'] = t.intime & t.mm_ok
    log.append(f'   pass 1 scales {sc} widths {sig}; t0 window [{lo:.0f}, '
               f'{hi:.0f}) ns keeps {t.intime.mean():.1%}; mask keeps '
               f'{t.mm_ok.mean():.1%}; good {t.good.mean():.1%} '
               f'[{time.time() - t0:.0f} s]')

    S2 = fit_scales(t[t.good], g).assign(arm=arm, fit_pass=2)
    sc, sig = _scales(S2)
    log.append(f'   pass 2 scales {sc} widths {sig} [{time.time() - t0:.0f} s]')
    S = pd.concat([S1, S2], ignore_index=True)
    SR = scale_by_run(t[t.good], g, sc['wall_u']).assign(arm=arm)

    P = predict(t, g, sc, sig)
    CS, CQ, CM = confirmation(t, P, sig)
    G = t.good.to_numpy()
    EP = edge_profiles(t[G].reset_index(drop=True), P[G].reset_index(drop=True))
    tg, Pg = t[G].reset_index(drop=True), P[G].reset_index(drop=True)
    ES, EM = efficiencies(tg, Pg, sig, g)
    LV = liquid_vs_plastic(tg, Pg, sig, ecal, arm)
    GM, AT, GS = gains(tg, Pg, sig, g, ecal, arm)
    WC, WR, WS = wall_position(tg, Pg, sig)
    BS, BP = both_ends(tg, Pg, sig, WC)
    AR = accidental_rates(t)
    RR = by_run(tg, Pg, sig, ecal, arm)
    log.append(f'   done [{time.time() - t0:.0f} s]')

    tabs = dict(scales=S, scale_run=SR, confirm=CS, confirm_quality=CQ,
                confirm_map=CM, t0_profile=TP, mask=M, eff=ES, eff_map=EM,
                edge_profile=EP,
                liq_vs_plas=LV, gain_map=GM, wall_atten=AT, gain=GS,
                wallpos_cal=WC, wallpos_res=WR, both_ends=BS, both_ends_v=BP,
                accidental=AR, by_run=RR)
    for name, df in tabs.items():
        df = df.assign(arm=arm) if 'arm' not in df.columns else df
        df.to_parquet(od / f'{name}_{arm}.parquet', index=False)
    WS.assign(arm=arm).to_parquet(od / f'wallpos_sample_{arm}.parquet',
                                  index=False)
    return dict(arm=arm, log=log, n=len(t), scales=sc, sigmas=sig,
                t0_window=[lo, hi], frac_intime=float(t.intime.mean()),
                frac_mm_ok=float(t.mm_ok.mean()), frac_good=float(t.good.mean()))


def merge(arms) -> None:
    od = out_dir() / 'ana'
    names = sorted({p.name.rsplit('_', 1)[0] for p in od.glob('*_?.parquet')})
    for nm in names:
        parts = [od / f'{nm}_{a}.parquet' for a in arms
                 if (od / f'{nm}_{a}.parquet').exists()]
        if parts:
            pd.concat([pd.read_parquet(p) for p in parts],
                      ignore_index=True).to_parquet(od / f'{nm}.parquet',
                                                    index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--arms', default='ABCD')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    arms = list(a.arms)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(run_arm, arms))
    for r in res:
        print('\n'.join(r['log']), flush=True)
    merge(ARMS)
    (out_dir() / 'ana' / 'meta.json').write_text(json.dumps(dict(
        schema=SCHEMA, arms=arms,
        n_tracks={r['arm']: r['n'] for r in res},
        scales={r['arm']: r['scales'] for r in res},
        sigmas={r['arm']: r['sigmas'] for r in res},
        t0_window={r['arm']: r['t0_window'] for r in res},
        frac_intime={r['arm']: r['frac_intime'] for r in res},
        frac_mm_ok={r['arm']: r['frac_mm_ok'] for r in res},
        frac_good={r['arm']: r['frac_good'] for r in res}), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
