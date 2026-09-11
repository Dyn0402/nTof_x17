#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source_imaging.py -- where the He-3 capsule is, and which chambers disagree.

THE OBSERVABLE, AND WHY IT IS THE RIGHT ONE.  A track from a point source at
perpendicular distance ``d`` must cross the strip plane at ``tan = (u - u0)/d``,
so ``median(tan)`` against ``u`` is a straight line whose **zero crossing is the
source**.  That crossing is *scale-free*: multiplying every angle by k scales the
slope and the intercept together and leaves ``-intercept/slope`` untouched.  So
the alignment measured here survives every doubt in the drift velocity, the
angle scale and the bundle -- which is exactly what an alignment number has to
do (`k_arm.py`, and PLAN.md sec 2.3).

WHAT EACH CHAMBER CAN SEE.  Each measures ONE transverse coordinate, its own
``u_hat``:

    A  u_hat = +x    ->  global X          C  u_hat = -x    ->  global X
    B  u_hat = +z    ->  global Z          D  u_hat = -z    ->  global Z

so **X is measured twice, from opposite sides, and Z once** -- because chamber B
has no drift field.  For X that gives the thing an alignment needs and a single
chamber can never provide: the mean is the source and **the difference is a
relative in-plane offset between the two chambers**, since shifting one
chamber's strip origin by delta moves only that chamber's estimate.  For Z there
is no such check, and this module says so rather than quoting Z as if there
were.

THE THIRD COORDINATE HAS NO CROSSING.  The capsule is 10 mm across in the
transverse plane but 80 mm long in y, so ``tan_y`` has no zero to find.  y comes
instead from the *distribution* of where tracks pass closest to the beam axis,
compared against a forward model of the actual gas polycone through the actual
acceptance -- :func:`y_forward_model`.  Chamber-to-chamber differences are read
the same way as in X: as alignment, not as physics.

WHAT IS DELIBERATELY NOT DONE.  No fit that lets the source position and the
angle scale float together: k comes from `k_arm.py` and is an input here.  And
no attempt to split the common offset into "the target really is off axis" and
"the survey is wrong" -- nothing in this data separates them (PLAN.md D8).

    python -m sept26_prelim_analysis.source_imaging --run run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import k_arm as K  # noqa: E402

SCHEMA = 'sept26_prelim/source_imaging/1'
ARMS = ('A', 'B', 'C', 'D')
D_PERP_MM = 234.6
#: A dead run is >= this many consecutive occupancy bins below THRESH of the
#: plane's own median.  Both are the values the 2026-09-08 dead-channel scan
#: used; they are recomputed here rather than transcribed.
DEAD_MIN_BINS, DEAD_THRESH = 8, 0.20
DEAD_BIN_MM = 0.7784          # 398.58 mm / 512 strips


# --------------------------------------------------------------------------- #
# the band, and its zero crossing
# --------------------------------------------------------------------------- #
def _robust_line(u, t, n_iter=8):
    A = np.vstack([u, np.ones_like(u)]).T
    w = np.ones_like(u)
    c = np.array([0.0, 0.0])
    for _ in range(n_iter):
        c, *_ = np.linalg.lstsq(A * w[:, None], t * w, rcond=None)
        r = t - (c[0] * u + c[1])
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * max(s, 1e-4))) ** 2)
    return float(c[0]), float(c[1])


def crossing(xl, tx, window, mask=None, n_boot=200, seed=1) -> dict:
    """Zero crossing of the pointing band, in the chamber's own u, with error.

    ``window`` is the |lever| range the band is fitted over, measured from the
    perpendicular foot.  ``mask`` optionally drops u ranges (dead channels).
    The error is a bootstrap over tracks, which is the only error that means
    anything here: the fit residual is dominated by the angular spread of real
    tracks, not by measurement noise.
    """
    lev = xl - window['foot']
    m = ((np.abs(lev) > window['lo']) & (np.abs(lev) < window['hi'])
         & (np.abs(tx) > 1e-3))
    if mask is not None:
        m &= mask
    if m.sum() < 200:
        return dict(n=int(m.sum()), x0=float('nan'), err=float('nan'),
                    slope=float('nan'))
    u, t = lev[m], tx[m]
    sl, ic = _robust_line(u, t)
    x0 = window['foot'] - ic / sl
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(u), len(u))
        s2, i2 = _robust_line(u[i], t[i])
        bs.append(window['foot'] - i2 / s2)
    return dict(n=int(m.sum()), x0=float(x0), err=float(np.std(bs)),
                slope=float(sl))


def symmetric_crossing(xl, tx, foot, hi, mask=None, n_iter=4) -> dict:
    """The crossing refitted over a lever window SYMMETRIC about itself.

    Why it exists.  The default window is symmetric about the *surveyed* foot.
    If the chamber's acceptance is not -- and chamber D loses ~130 channels all
    on one side -- then more of the fitted band sits on one side of the crossing
    than the other, and a robust line fit through an asymmetric sample pulls the
    crossing toward the populated side.  Recentring the window on the fitted
    crossing and iterating removes that by construction.  If the answer does not
    move, the asymmetry was not the explanation, and that is worth knowing too.
    """
    out = crossing(xl, tx, dict(foot=foot, lo=30.0, hi=hi), mask)
    for _ in range(n_iter):
        if not np.isfinite(out['x0']):
            break
        nxt = crossing(xl, tx, dict(foot=out['x0'], lo=30.0, hi=hi), mask)
        if not np.isfinite(nxt['x0']) or abs(nxt['x0'] - out['x0']) < 0.05:
            out = nxt
            break
        out = nxt
    return out


def dead_ranges(x_p0: np.ndarray) -> list:
    """u ranges with no occupancy, found from the data rather than transcribed.

    Returns [(lo_mm, hi_mm), ...] in RAW strip-map coordinates, so the caller
    converts to its own local frame the same way it converts positions.

    Must be given the FULL occupancy of the plane -- every seeded event, no
    selection.  Run on a thousand-track band sample instead it finds nothing,
    because at 512 bins a thousand tracks are sparse everywhere and the median
    of the non-empty bins is 1.
    """
    edges = np.arange(0.0, 398.58 + DEAD_BIN_MM, DEAD_BIN_MM)
    h, _ = np.histogram(x_p0, bins=edges)
    med = np.median(h[h > 0]) if (h > 0).any() else 0.0
    if med <= 0:
        return []
    low = h < DEAD_THRESH * med
    out, i = [], 0
    while i < len(low):
        if not low[i]:
            i += 1
            continue
        j = i
        while j < len(low) and low[j]:
            j += 1
        if j - i >= DEAD_MIN_BINS:
            out.append((float(edges[i]), float(edges[j])))
        i = j
    return out


_TRS = {}


def transforms(run: str):
    """Per-arm detector transforms, from the run's own ``run_config.json``.

    Cached: the crossing is evaluated four ways on four arms on three sub-runs,
    and re-reading the config forty-eight times would be forty-seven times too
    many.
    """
    from ntof_tracking.reco import geometry as G
    if run not in _TRS:
        cfg = json.loads((paths.root('runs') / run
                          / 'run_config.json').read_text())
        _TRS[run] = G.detector_transforms(cfg)
    return _TRS[run]


def to_global(run: str, arm: str, x0: float) -> tuple:
    """(axis, value) -- the crossing placed in the global frame.

    ``src = centre + x0 * u_hat``, and the arm's u_hat picks out which global
    axis it is a statement about.  This is the one place the pinwheel offset
    and the in-plane sign both have to be right, and both come from the
    geometry module rather than from here.
    """
    tr = transforms(run)[f'mx17_{arm}']
    uhat = tr.R @ np.array([1.0, 0.0, 0.0])
    src = tr.center + x0 * uhat
    ax = 'X' if abs(uhat[0]) > 0.5 else 'Z'
    return ax, float(src[0] if ax == 'X' else src[2])


# --------------------------------------------------------------------------- #
# X and Z
# --------------------------------------------------------------------------- #
def plane_occupancy(run: str, subruns, arm: str, merged_dir: str) -> np.ndarray:
    """Every fitted x position on one plane, all sub-runs, no selection."""
    out = []
    for sub in subruns:
        p = paths.require(os.path.join(merged_dir, sub, f'mx17_{arm}',
                                       'events_prelim.parquet'),
                          f'merged full-pass table for {arm}/{sub}')
        out.append(pd.read_parquet(p, columns=['x_p0']).x_p0.to_numpy())
    v = np.concatenate(out)
    return v[np.isfinite(v)]


def dead_table(run: str, subruns, merged_dir: str,
               lo: float = 30.0, hi: float = 130.0) -> pd.DataFrame:
    """Where each plane is dead, and what that costs the band fit.

    A dead channel produces no tracks, so *masking* tracks in a dead range is
    close to a no-op -- there are none.  What dead channels actually do to a
    crossing is make the ACCEPTANCE asymmetric about it, so more of the fitted
    band sits on one side than the other.  This table is what says how much: the
    lever range lost on each side of the perpendicular foot, inside the window
    the band is fitted over.
    """
    from ntof_tracking import run145_target_imaging as TI
    rows = []
    for arm in ARMS:
        x = plane_occupancy(run, subruns, arm, merged_dir)
        foot = TI.PINWHEEL[arm]
        lost_neg = lost_pos = 0.0
        spans = []
        for raw_lo, raw_hi in dead_ranges(x):
            # raw -> local -> lever; IN_PLANE_SIGN flips the order, so sort
            a, b = sorted(TI.IN_PLANE_SIGN * (np.array([raw_lo, raw_hi])
                                              - TI.STRIP_MAP_HALF) - foot)
            spans.append((round(float(a), 1), round(float(b), 1)))
            for sgn in (-1, +1):
                w0, w1 = (sgn * lo, sgn * hi) if sgn > 0 else (-hi, -lo)
                w0, w1 = min(w0, w1), max(w0, w1)
                ov = max(0.0, min(b, w1) - max(a, w0))
                if sgn < 0:
                    lost_neg += ov
                else:
                    lost_pos += ov
        rows.append(dict(arm=arm, n_ranges=len(spans),
                         dead_mm=sum(b - a for a, b in spans),
                         dead_frac_plane=sum(b - a for a, b in spans) / 398.58,
                         lever_spans=str(spans),
                         lost_neg_mm=lost_neg, lost_pos_mm=lost_pos,
                         asymmetry_mm=lost_pos - lost_neg,
                         window_mm=hi - lo))
    return pd.DataFrame(rows)


def projection(VS: pd.DataFrame, run_scale: float = 50.0) -> pd.DataFrame:
    """How many pairs a vertex measurement needs, from what run_145 measured.

    Nothing here is a hope: the background fraction is the measured mixed-event
    rate, and the significance of an excess of fraction ``f`` over a background
    ``b`` in ``N`` pairs is (f-b) N / sqrt(bN), so N_needed scales as 1/(f-b)^2.
    Quoted for the excess run_145 can still accommodate at 95 % CL, which is
    the honest ceiling rather than an assumed signal.
    """
    rows = []
    for _, r in VS.iterrows():
        if not np.isfinite(r.get('frac_mixed', np.nan)):
            rows.append(dict(topology=r.topology))
            continue
        n, b = int(r.n_real), float(r.frac_mixed)
        # 95 % CL ceiling on the excess fraction, Gaussian
        ul = max(r.frac_real - b, 0.0) + 1.645 * np.sqrt(max(b, 1e-6) / n)
        need = {}
        for nsig in (3.0, 5.0):
            need[nsig] = (np.inf if ul <= 0 else
                          float(nsig ** 2 * b / ul ** 2))
        rows.append(dict(topology=r.topology, n_real=n, bkg_frac=b,
                         excess_ul95=float(ul),
                         n_pairs_for_3sigma=need[3.0],
                         n_pairs_for_5sigma=need[5.0],
                         campaign_pairs=n * run_scale,
                         reach=('3 sigma' if n * run_scale >= need[3.0]
                                else 'below 3 sigma')))
    return pd.DataFrame(rows)


def transverse(run: str, subruns, merged_dir: str) -> pd.DataFrame:
    """The crossing per (arm, sub-run), in four variants.

    baseline   the window k_arm uses, symmetric about the surveyed foot
    symmetric  the window recentred on the crossing itself, iterated
    masked     dead u ranges removed
    both       masked and recentred
    """
    from ntof_tracking import run145_target_imaging as TI
    # Dead channels are a property of the PLANE, not of one sub-run's band
    # sample, so they are found once on the full occupancy and reused.
    dead = {a: dead_ranges(plane_occupancy(run, subruns, a, merged_dir))
            for a in ARMS}
    rows = []
    for sub in subruns:
        for arm in ARMS:
            # The pointing-coincident sample, built by k_arm so the frame, the
            # in-plane sign, the coincidence geometry and the charge window are
            # the ones already measured -- and so a change there cannot leave
            # the alignment and the angle scale describing different samples.
            # No try/except: a missing product is a staging problem and must
            # stop the run, not become a silent NaN row.
            S = K.coincident_tracks(run, sub, arm, merged_dir)
            xl, tx = S['xl'], S['tx']
            foot = S['foot_x']
            raw = TI.IN_PLANE_SIGN * xl + TI.STRIP_MAP_HALF
            keep = np.ones(len(xl), bool)
            for lo, hi in dead[arm]:
                keep &= ~((raw >= lo) & (raw < hi))
            variants = {
                'baseline':  crossing(xl, tx, dict(foot=foot, lo=30., hi=130.)),
                'symmetric': symmetric_crossing(xl, tx, foot, 130.),
                'masked':    crossing(xl, tx, dict(foot=foot, lo=30., hi=130.),
                                      keep),
                'both':      symmetric_crossing(xl, tx, foot, 130., keep),
            }
            for name, v in variants.items():
                ax, mm = ((None, float('nan')) if not np.isfinite(v['x0'])
                          else to_global(run, arm, v['x0']))
                rows.append(dict(subrun=sub, arm=arm, variant=name,
                                 n=v['n'], x0_local=v['x0'], err=v['err'],
                                 slope=v['slope'], axis=ax, mm=mm,
                                 n_dead_ranges=len(dead[arm]),
                                 dead_frac=float(1 - keep.mean())))
    return pd.DataFrame(rows)


def combine_axis(T: pd.DataFrame, variant: str = 'baseline') -> pd.DataFrame:
    """Per (axis, arm): the crossing averaged over sub-runs, with two errors.

    ``err_stat`` is the bootstrap; ``err_repro`` is the spread between sub-runs,
    which is the honest one -- it contains everything that changes run to run
    and nothing that a bootstrap can see.
    """
    g = T[(T.variant == variant) & T.mm.notna()]
    rows = []
    for (ax, arm), h in g.groupby(['axis', 'arm']):
        rows.append(dict(axis=ax, arm=arm, n_subruns=len(h),
                         mm=float(h.mm.mean()),
                         err_stat=float(np.sqrt((h.err ** 2).sum()) / len(h)),
                         err_repro=float(h.mm.std(ddof=1)) if len(h) > 1
                         else float('nan'),
                         n=int(h.n.sum())))
    return pd.DataFrame(rows).sort_values(['axis', 'arm'], ignore_index=True)


def axis_verdict(C: pd.DataFrame) -> list:
    """One statement per transverse axis: the source, and the disagreement."""
    out = []
    for ax, g in C.groupby('axis'):
        arms = list(g.arm)
        val = float(g.mm.mean())
        # The chamber-to-chamber half-difference IS the alignment systematic.
        half = float((g.mm.max() - g.mm.min()) / 2) if len(g) > 1 else np.nan
        err = float(np.sqrt((g.err_stat ** 2).sum()) / len(g))
        out.append(dict(axis=ax, arms=arms, n_chambers=len(g),
                        source_mm=val, err_stat=err,
                        align_syst_mm=half,
                        cross_checked=len(g) > 1,
                        per_arm={r.arm: round(r.mm, 2) for r in g.itertuples()}))
    return out


# --------------------------------------------------------------------------- #
# y -- no crossing, so a forward model
# --------------------------------------------------------------------------- #
def y_forward_model(run: str, arm: str, n: int = 400_000,
                    y_shift: float = 0.0, seed: int = 3) -> np.ndarray:
    """Where tracks from the real gas polycone pass closest to the beam axis.

    Straight lines, no scattering.  Vertices are drawn from the He-3 gas
    profile (`geometry.HE3_GAS_Y/R`, a STEP-derived polycone) by rejection in
    r^2, so the sampling is by VOLUME and not by length.  A track counts if it
    crosses the chamber's strip plane inside the active area AND one of the two
    plastic bars behind it -- the plastic coincidence is the production
    trigger, and it is what limits the v acceptance to about +-85 mm.

    The returned quantity is the y at closest approach to the beam axis, which
    is what `build_tracks` writes as ``target_y_mm``, so the comparison is like
    for like.
    """
    from ntof_tracking.reco import geometry as G
    rng = np.random.default_rng(seed)

    ys, rs = G.HE3_GAS_Y, G.HE3_GAS_R
    yy = rng.uniform(ys.min(), ys.max(), n * 3)
    rmax = np.interp(yy, ys, rs)
    rr = G.HE3_R_MAX * np.sqrt(rng.uniform(0, 1, n * 3))
    ok = rr <= rmax
    yy, rr = yy[ok][:n], rr[ok][:n]
    if len(yy) < n:
        n = len(yy)
    ph = rng.uniform(0, 2 * np.pi, n)
    P = np.column_stack([rr * np.cos(ph), yy + y_shift, rr * np.sin(ph)])

    ct = rng.uniform(-1, 1, n)
    st = np.sqrt(1 - ct ** 2)
    az = rng.uniform(0, 2 * np.pi, n)
    D = np.column_stack([st * np.cos(az), ct, st * np.sin(az)])

    wh, uh = G.W_HAT[arm], G.U_HAT[arm]
    vh = G.V_HAT
    centre = transforms(run)[f'mx17_{arm}'].center

    def at_depth(depth):
        """(u, v) where each line crosses the plane `depth` mm out along w."""
        plane = centre @ wh + (depth - 0.0)
        s = (plane - P @ wh) / np.where(np.abs(D @ wh) < 1e-9, np.nan, D @ wh)
        X = P + s[:, None] * D
        return X @ uh - centre @ uh, X @ vh, s

    u0, v0, s0 = at_depth(0.0)
    # the plastic sits ~PLASTIC_W0 past the mylar front, i.e. W_STRIP behind it
    dp = G.PLASTIC_W0[arm] - G.W_STRIP + G.PLASTIC_THICK / 2
    up, vp, sp = at_depth(dp)

    ok = (np.isfinite(u0) & (s0 > 0)
          & (np.abs(u0) < G.MM_SIZE_U / 2) & (np.abs(v0) < G.MM_SIZE_V / 2)
          & (np.abs(vp) < G.PLASTIC_HALF_V)
          & (np.abs(np.abs(up) - G.PLASTIC_U_OFFSET) < G.PLASTIC_HALF_U))
    # y at closest approach to the beam axis, exactly as build_tracks defines it
    p = P[:, [0, 2]]
    d = D[:, [0, 2]]
    t = -np.einsum('ij,ij->i', p, d) / np.clip(
        np.einsum('ij,ij->i', d, d), 1e-12, None)
    return (P[:, 1] + t * D[:, 1])[ok]


def y_measured(run: str, subruns, dca_max: float = 30.0) -> pd.DataFrame:
    """``target_y_mm`` for gated, target-pointing, angle-calibrated tracks."""
    src = paths.out('stage3_fullpass')
    out = []
    for sub in subruns:
        p = paths.require(src / f'tracks_{run}_{sub}.parquet',
                          f'stage-3 tracks for {sub}')
        out.append(pd.read_parquet(p, columns=[
            'event_id', 'arm', 'gated', 'target_y_mm', 'dca_axis_mm',
            'angle_calibrated']).assign(subrun=sub))
    t = pd.concat(out, ignore_index=True)
    return t[t.gated & t.angle_calibrated & (t.dca_axis_mm < dca_max)
             & np.isfinite(t.target_y_mm)]


def y_compare(run: str, subruns, dca_max: float = 30.0) -> tuple:
    """Observed vs predicted y, per chamber -- offset, and implied resolution.

    The offset is the shift that best lines the observed distribution up with
    the prediction (matched on the median, which no tail can move).  The width
    ratio is the second number and it is not free: if the observed distribution
    is WIDER than the model, the excess is resolution; if it is NARROWER, the
    model's acceptance is wrong and the offset should not be trusted either.
    """
    obs = y_measured(run, subruns, dca_max)
    rows, curves = [], {}
    for arm in ARMS:
        g = obs[obs.arm == arm]
        if len(g) < 200:
            rows.append(dict(arm=arm, n=len(g)))
            continue
        pred = y_forward_model(run, arm)
        o = g.target_y_mm.to_numpy()
        o_med, p_med = float(np.median(o)), float(np.median(pred))
        o_iqr = float(np.subtract(*np.percentile(o, [75, 25])))
        p_iqr = float(np.subtract(*np.percentile(pred, [75, 25])))
        rows.append(dict(
            arm=arm, n=int(len(g)), obs_median=o_med, pred_median=p_med,
            offset_mm=o_med - p_med,
            obs_iqr=o_iqr, pred_iqr=p_iqr, width_ratio=o_iqr / p_iqr,
            # IQR -> Gaussian sigma is /1.349; the quadrature excess is the
            # resolution IF the model's acceptance is right.
            implied_sigma_mm=(float(np.sqrt(max(o_iqr ** 2 - p_iqr ** 2, 0.0))
                                    / 1.349))))
        curves[arm] = (o, pred)
    return pd.DataFrame(rows), curves


# --------------------------------------------------------------------------- #
# double tracks -- a vertex, and whether it means anything
# --------------------------------------------------------------------------- #
def _dca_two_lines(p1, d1, p2, d2):
    """Closest approach of two lines: (midpoint, distance)."""
    w0 = p1 - p2
    a = np.einsum('ij,ij->i', d1, d1)
    b = np.einsum('ij,ij->i', d1, d2)
    c = np.einsum('ij,ij->i', d2, d2)
    d = np.einsum('ij,ij->i', d1, w0)
    e = np.einsum('ij,ij->i', d2, w0)
    den = a * c - b * b
    den = np.where(np.abs(den) < 1e-12, np.nan, den)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    q1 = p1 + s[:, None] * d1
    q2 = p2 + t[:, None] * d2
    return 0.5 * (q1 + q2), np.linalg.norm(q1 - q2, axis=1)


def _track_table(run: str, subruns, dca_max: float, src=None) -> pd.DataFrame:
    """Gated, angle-calibrated tracks for one run.

    ``src`` defaults to ``<out>/stage3_fullpass`` -- where the run_145 pass
    wrote -- and the campaign pass passes ``<out>/stage3_campaign`` instead.
    Parameterised rather than repointed so the published run_145 products stay
    exactly where the published numbers were computed from.

    NOTE the ``angle_calibrated`` filter below: a run with no
    ``k_arm_<run>.json`` contributes NOTHING here, silently. That is why
    `k_arm` has to run before `build_tracks` in the campaign chain, not after.
    """
    src = paths.out('stage3_fullpass') if src is None else src
    cols = ['event_id', 'arm', 'gated', 'bunch', 'angle_calibrated',
            'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'dca_axis_mm']
    out = []
    for sub in subruns:
        p = paths.require(src / f'tracks_{run}_{sub}.parquet',
                          f'stage-3 tracks for {sub}')
        out.append(pd.read_parquet(p, columns=cols).assign(subrun=sub))
    t = pd.concat(out, ignore_index=True)
    t = t[t.gated & t.angle_calibrated & (t.dca_axis_mm < dca_max)].copy()
    t['key'] = t.subrun + ':' + t.event_id.astype(str)
    return t.reset_index(drop=True)


def _pairs_real(t: pd.DataFrame) -> pd.DataFrame:
    """Every unordered pair of distinct tracks inside one trigger."""
    import itertools
    L, R = [], []
    for _, g in t.groupby('key'):
        if len(g) < 2:
            continue
        for i, j in itertools.combinations(g.index, 2):
            L.append(i)
            R.append(j)
    # dtype pinned: an empty list gives an OBJECT column, and `_pairs_mixed`
    # then indexes a numpy array with it and raises.  A run whose k_arm never
    # certified has no calibrated tracks and so no pairs at all -- run_126 --
    # so the empty case is normal campaign-wide, not a staging failure.
    return pd.DataFrame(dict(i=np.asarray(L, dtype=np.int64),
                             j=np.asarray(R, dtype=np.int64)))


def _pairs_mixed(t: pd.DataFrame, real: pd.DataFrame, seed=5) -> pd.DataFrame:
    """The null: the SAME arm composition, drawn from DIFFERENT triggers.

    Getting this wrong is easy and silent.  Re-pairing every track ignores that
    a real pair needs a trigger that produced two tracks at all, which is a
    special population; shuffling one column leaves half the pairs intact.  So
    the mixing here reproduces the real sample pair by pair: for each real
    (arm1, arm2) it draws one track from the arm1 pool and one from the arm2
    pool, rejecting draws that land in the same trigger.
    """
    rng = np.random.default_rng(seed)
    # Draw from the tracks that ACTUALLY FORM REAL PAIRS, not from every
    # track.  A trigger that produced two tracks is a busier trigger, and its
    # tracks are not drawn from the same distribution as a lone one -- mixing
    # against the full pool makes the null look better than the data and the
    # lift come out below 1, which is what happened the first time.
    if real.empty:
        return pd.DataFrame(dict(i=np.zeros(0, np.int64),
                                 j=np.zeros(0, np.int64)))
    used = np.unique(np.concatenate([real.i.to_numpy(), real.j.to_numpy()]))
    tp = t.loc[used]
    pools = {a: g.index.to_numpy() for a, g in tp.groupby('arm')}
    keys = t.key.to_numpy()
    L, R = [], []
    for a1, a2 in zip(t.arm.to_numpy()[real.i.to_numpy()],
                      t.arm.to_numpy()[real.j.to_numpy()]):
        p1, p2 = pools.get(a1), pools.get(a2)
        if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
            continue
        for _ in range(20):
            i = int(rng.choice(p1))
            j = int(rng.choice(p2))
            if i != j and keys[i] != keys[j]:
                L.append(i)
                R.append(j)
                break
    return pd.DataFrame(dict(i=L, j=R))


#: The columns :func:`_vertex_frame` produces.  Named once so the EMPTY frame
#: carries them too: a run that yielded no pairs used to come back as a bare
#: ``DataFrame()``, and every consumer that filters on ``topology`` then died
#: on an AttributeError instead of seeing an empty sample.
VERTEX_COLUMNS = ('key', 'key1', 'key2', 'arm1', 'arm2', 'topology',
                  'vx', 'vy', 'vz', 'v_r', 'sep_mm', 'open_deg', 'mixed')


def _vertex_frame(t: pd.DataFrame, pr: pd.DataFrame, mixed: bool):
    if pr.empty:
        return pd.DataFrame({c: pd.Series(dtype='float64'
                                          if c in ('vx', 'vy', 'vz', 'v_r',
                                                   'sep_mm', 'open_deg')
                                          else 'object')
                             for c in VERTEX_COLUMNS})
    a = t.loc[pr.i.to_numpy()]
    b = t.loc[pr.j.to_numpy()]
    p1 = a[['p0_x', 'p0_y', 'p0_z']].to_numpy()
    d1 = a[['d_x', 'd_y', 'd_z']].to_numpy()
    p2 = b[['p0_x', 'p0_y', 'p0_z']].to_numpy()
    d2 = b[['d_x', 'd_y', 'd_z']].to_numpy()
    V, sep = _dca_two_lines(p1, d1, p2, d2)
    dot = np.einsum('ij,ij->i', d1, d2).clip(-1, 1)
    a1, a2 = a.arm.to_numpy(), b.arm.to_numpy()
    # BOTH keys, not just the first.  In the mixed sample the two tracks come
    # from DIFFERENT triggers, so a single `key` column silently claims the
    # second track belongs to the first one's event -- which is exactly the
    # trap a downstream timing study fell into on 2026-09-08.
    swap = a1 > a2
    k1 = np.where(swap, b.key.to_numpy(), a.key.to_numpy())
    k2 = np.where(swap, a.key.to_numpy(), b.key.to_numpy())
    return pd.DataFrame(dict(
        key=a.key.to_numpy(), key1=k1, key2=k2,
        arm1=np.minimum(a1, a2), arm2=np.maximum(a1, a2),
        topology=np.where(a1 == a2, 'intra', 'inter'),
        vx=V[:, 0], vy=V[:, 1], vz=V[:, 2],
        v_r=np.hypot(V[:, 0], V[:, 2]), sep_mm=sep,
        open_deg=np.degrees(np.arccos(dot)), mixed=mixed))


def vertices(run: str, subruns, dca_max: float = 30.0, seed: int = 5,
             src=None):
    """Two-track vertices, and the event-mixed null built to match them.

    Returns (real, mixed).  Both come from one track table and one pairing
    rule, so the only difference between them is whether the two tracks shared
    a trigger -- which is the whole point of a control.
    """
    t = _track_table(run, subruns, dca_max, src=src)
    real = _pairs_real(t)
    mix = _pairs_mixed(t, real, seed)
    return _vertex_frame(t, real, False), _vertex_frame(t, mix, True)


def vertex_summary(real: pd.DataFrame, mixed: pd.DataFrame,
                   r_cut: float = 20.0, sep_cut: float = 30.0) -> pd.DataFrame:
    """Does the vertex collapse onto the capsule more than chance says?

    The quantity is the fraction of pairs whose vertex sits inside the capsule
    bore and whose two lines actually approach each other.  Compared against
    the identical estimator on mixed events, which is the null.
    """
    rows = []
    for topo in ('intra', 'inter'):
        r, x = real[real.topology == topo], mixed[mixed.topology == topo]
        if len(r) < 20 or len(x) < 20:
            rows.append(dict(topology=topo, n_real=len(r), n_mixed=len(x)))
            continue
        f_r = float(((r.v_r < r_cut) & (r.sep_mm < sep_cut)).mean())
        f_x = float(((x.v_r < r_cut) & (x.sep_mm < sep_cut)).mean())
        k = int(((r.v_r < r_cut) & (r.sep_mm < sep_cut)).sum())
        err = float(np.sqrt(max(k, 1)) / len(r))
        rows.append(dict(topology=topo, n_real=int(len(r)),
                         n_mixed=int(len(x)),
                         frac_real=f_r, frac_mixed=f_x, err=err,
                         lift=f_r / f_x if f_x > 0 else np.nan,
                         excess_sigma=(f_r - f_x) / err if err > 0 else np.nan,
                         n_signal=int(round((f_r - f_x) * len(r))),
                         vy_median=float(r[(r.v_r < r_cut)
                                           & (r.sep_mm < sep_cut)].vy.median())
                         if k else float('nan')))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--dca', type=float, default=30.0)
    # The merged reco tree.  The default is the ALLOWLIST pass, which is where
    # the published run_145 numbers were computed and must stay; the condor
    # FULL pass is `<out>/reco_fullpass`, and `campaign_imaging.py` passes it.
    # Named rather than switched so a product can never be half one pass and
    # half the other without the meta sidecar saying so.
    ap.add_argument('--merged', default=None,
                    help='merged reco tree for this run; default '
                         '<out>/fullpass/<run> (the ALLOWLIST pass)')
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]
    merged = a.merged or str(paths.out('fullpass') / a.run)

    T = transverse(a.run, subs, merged)
    C = combine_axis(T, 'baseline')
    Csym = combine_axis(T, 'both')
    Y, curves = y_compare(a.run, subs, a.dca)
    real, mixed = vertices(a.run, subs, a.dca)
    VS = vertex_summary(real, mixed)
    DT = dead_table(a.run, subs, merged)
    PJ = projection(VS)

    od = paths.out('imaging')
    T.to_csv(od / f'crossings_{a.run}.csv', index=False)
    C.to_csv(od / f'transverse_{a.run}.csv', index=False)
    Csym.to_csv(od / f'transverse_robust_{a.run}.csv', index=False)
    Y.to_csv(od / f'y_compare_{a.run}.csv', index=False)
    VS.to_csv(od / f'vertex_summary_{a.run}.csv', index=False)
    DT.to_csv(od / f'dead_{a.run}.csv', index=False)
    PJ.to_csv(od / f'projection_{a.run}.csv', index=False)
    real.to_parquet(od / f'vertices_{a.run}.parquet', index=False)
    mixed.to_parquet(od / f'vertices_mixed_{a.run}.parquet', index=False)
    np.savez_compressed(od / f'y_curves_{a.run}.npz',
                        **{f'{k}_{w}': v[i] for k, v in curves.items()
                           for i, w in enumerate(('obs', 'pred'))})
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs, dca_max=a.dca,
                   merged=merged,
                   verdict=axis_verdict(C),
                   verdict_robust=axis_verdict(Csym)),
              open(od / f'imaging_{a.run}.meta.json', 'w'), indent=1,
              default=float)

    print('CROSSINGS, per sub-run and variant')
    print(T[['subrun', 'arm', 'variant', 'n', 'axis', 'mm', 'err',
             'dead_frac']].to_string(index=False))
    print('\nTRANSVERSE, combined (baseline)')
    print(C.to_string(index=False))
    print('\nTRANSVERSE, combined (dead-masked + self-centred window)')
    print(Csym.to_string(index=False))
    print('\nVERDICT')
    print(json.dumps(axis_verdict(C), indent=1, default=float))
    print('\nY, observed against the polycone forward model')
    print(Y.to_string(index=False))
    print('\nDEAD CHANNELS, and what they cost the band fit')
    print(DT.to_string(index=False))
    print('\nDOUBLE-TRACK VERTICES, against event mixing')
    print(VS.to_string(index=False))
    print('\nWHAT A VERTEX MEASUREMENT WOULD NEED')
    print(PJ.to_string(index=False))
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
