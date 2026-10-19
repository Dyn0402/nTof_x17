#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xy_t0.py -- what the two planes of one chamber actually agree on, and what
``dt_xy`` is.

THE QUESTION, from ``HANDOFF_T0_PRIOR.md``.  That handoff measured ``x_t0 -
y_t0`` on the published sample, found a 66-77 ns half-width and a > 60 ns tail
on ~43 % of tracks, and read it as **the 60 ns depth-bin degeneracy** of
``wft.model.chi2_plane`` "landing in different minima on the two planes".  It
also found that ``wft.reco.select_pair``'s ``dt_xy`` lookup can never hit on
arms A and C, by parity of its ``ftst_diff`` keys, and asked for four things:
count the fallback, measure ``dt_xy`` in situ, build a t0 prior, and judge it
on that residual.

This module does the first two, and in doing them finds that **the third and
fourth rest on a misreading, and the diagnosis is wrong**:

  1. ``fallback_census``   the parity argument is exact, not approximate.  The
     measured ``dt_xy`` is used on **0 of 454 180 A tracks, 0 of 360 869 B, 0
     of 632 629 C** and 34.6 % of D -- **10.9 % of the campaign**.  Everything
     else runs on the hardcoded -18.8 ns.
  2. ``insitu_dt``         ``dt_xy`` is not a per-chamber constant to be looked
     up.  The whole ``x_t0 - y_t0`` distribution **translates linearly with
     ``ftst_diff``**, by -7.7 to -8.8 ns per unit on all four arms alike.  That
     is the FEU fine-timestamp quantum: ``ftst`` has 6 phases and ``sample_ns``
     is 60, so one unit is 10 ns, and the estimator is biased low by the
     accidental pedestal it cannot fully subtract.  **It needs no bench.**  The
     bench measured two ``ftst_diff`` classes per arm and neither occurs in
     beam data on A, B or C.
  3. ``degeneracy_test``   **there is no 60 ns structure in the residual.**  A
     periodicity test that detects the 5 ns ``T0_STEP`` snap at 8.3-18.9 sigma
     finds 60 ns at +1.0, -0.5, -0.9 sigma against a null of arbitrary periods.
     Two near-degenerate minima one bin apart would put a comb there.  They do
     not.  So the handoff's mechanism is not what is happening.
  4. ``geometry_test``     nor is it the x/y plane separation, the handoff's own
     "first thing to check".  The residual half-width does grow with inclination
     -- A 65 -> 76 ns across the full ``tan theta`` range -- but the **zero-
     inclination floor is already 65-70 ns**.  Geometry is a ~10 ns effect on a
     66-77 ns spread.  It explains almost none of it.
  5. ``discrimination``    what it IS: the per-plane t0 is smoothly uncertain at
     the one-depth-bin scale.  Background-subtracted, the x/y agreement peak has
     a **78-85 ns half-width** -- ~55-60 ns per plane -- against a fitted
     ``x_t0_err`` that medians 3.8-10.6 ns over the whole population and 2.6-7.4
     on the selected one.  **The reported error is low by 5x to 21x**, depending
     on chamber and sample.  Same scale as the handoff said; a smooth
     uncertainty, not a two-minimum ambiguity, and a different fix.
  6. ``circularity``       **the handoff's proposed acceptance metric is
     circular.**  ``gated`` IS this cut: ``select_tracks`` gates on
     ``|(t0x - t0y) - dt| <= 120 ns`` and 0 of 2.11 M gated tracks fall outside
     it.  Measuring the x/y residual on the gated sample measures a distribution
     that was truncated at +-120 ns around a **wrong centre**, and a better
     prior would change which tracks are in the sample.  The Sec. 1 numbers are
     not wrong, they are a truncated view: unselected, the half-widths are
     107/136/138 ns, not 66/76/77.

WHAT THIS DOES NOT SETTLE.  It does not build the in-situ t0 prior (step 3 of
the handoff) -- but it removes the reason to expect one to help in the way that
was hoped, and it supplies the ``dt_xy`` law such a prior would need.  It does
not touch the waveform path: every number here is read from the stage-3 track
table, so nothing is re-reconstructed and nothing here can change a fit.

    python ntof_athens_26/xy_t0/xy_t0.py
    X17_ROOT=D:/x17 python ntof_athens_26/xy_t0/xy_t0.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                          # noqa: E402

SCHEMA = 'ntof_athens_26/xy_t0/1'

ARMS = ('A', 'B', 'C', 'D')

#: ``wft.reco.select_pair``/``select_tracks`` fall back to this when the
#: bundle's ``dt_xy`` has no entry for the event's ``ftst_diff``.  Not flagged,
#: not counted, not in any provenance -- which is why it took a census to find
#: that it is what almost the whole campaign ran on.
FALLBACK_DT = -18.8

#: ``wft.reco.DT_XY_TOL_NS`` -- the half-width of the x/y coincidence window.
#: ``select_tracks`` sets ``gated`` from ``coincident AND both plausible``, so
#: this constant is half of the published selection.
TOL_NS = 120.0

#: ``wft.model.T0_STEP`` -- fitted t0 is snapped to this.  Its appearance in
#: ``degeneracy_test`` is the positive control: a test that cannot see a known
#: 5 ns comb has no business reporting the absence of a 60 ns one.
T0_STEP = 5.0

#: ``wft.model.DT`` -- one charge/depth bin.  The periodicity the handoff
#: predicted.
DEPTH_BIN_NS = 60.0

#: ``ftst`` is a 6-phase fine timestamp on a ``sample_ns`` = 60 ns sample, so
#: one unit of ``ftst_diff`` is this many ns of x-vs-y readout phase.
FTST_PHASES = 6
SAMPLE_NS = 60.0
FTST_QUANTUM_NS = SAMPLE_NS / FTST_PHASES

#: `source_imaging._track_table`'s selection, reproduced so this module measures
#: the sample the handoff measured rather than a similar one.
DCA_MAX = 30.0

#: Histogram of ``x_t0 - y_t0`` used by every shift and shape estimator here.
HIST_BW = 5.0
HIST_EDGES = np.arange(-400.0, 400.0 + HIST_BW, HIST_BW)
HIST_CENTRES = 0.5 * (HIST_EDGES[1:] + HIST_EDGES[:-1])

#: Beyond this the true and scrambled pairings are indistinguishable, so the
#: accidental pedestal can be normalised there.
PEDESTAL_NS = 250.0

TRACK_COLS = [
    'arm', 'run', 'condition', 'subrun', 'event_id',
    'x_t0', 'y_t0', 'x_ftst', 'y_ftst', 'x_t0_err', 'y_t0_err',
    'n_cand_x', 'n_cand_y', 'x_tan_theta', 'y_tan_theta',
    'drift_len_mm', 'chi2dof_x', 'chi2dof_y', 'x_n_strips',
    'gated', 'angle_calibrated', 'dca_axis_mm',
]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def bundle_dt_xy(bundle_root: Path) -> dict:
    """``{arm: {ftst_diff: dt_ns}}`` from the bundles the campaign ran.

    Read from disk rather than quoted, because the point of the census below is
    that these keys and the beam data's ``ftst_diff`` values do not intersect,
    and a table copied into a docstring cannot establish that.
    """
    out = {}
    for arm in ARMS:
        hits = sorted(glob.glob(str(bundle_root / f'mx17_{arm}' /
                                    'calib_bundle_*' / 'bundle.json')))
        if not hits:
            out[arm] = {}
            continue
        meta = json.load(open(hits[0]))
        out[arm] = {int(k): float(v) for k, v in (meta.get('dt_xy') or {}).items()}
    return out


def load_tracks(src: Path, dt_xy: dict) -> pd.DataFrame:
    """The stage-3 campaign track table with the x/y timing columns derived.

    ``raw`` is the quantity ``dt_xy`` is supposed to be; ``dt_used`` is what the
    reconstruction actually subtracted (bundle entry or fallback); ``res_ship``
    is the handoff's residual.  Nothing is cut here -- the selection is applied
    where it is needed, because half the point is that applying it changes what
    the residual looks like.
    """
    d = pd.read_parquet(src, columns=TRACK_COLS)
    d['ftst_diff'] = d.x_ftst - d.y_ftst
    d['raw'] = d.x_t0 - d.y_t0
    d['dt_used'] = [dt_xy.get(a, {}).get(int(k), FALLBACK_DT)
                    for a, k in zip(d.arm, d.ftst_diff)]
    d['dt_hit'] = [int(k) in dt_xy.get(a, {}) for a, k in zip(d.arm, d.ftst_diff)]
    d['res_ship'] = d.raw - d.dt_used
    d['selected'] = d.gated & d.angle_calibrated & (d.dca_axis_mm < DCA_MAX)
    return d


def published_sample(d: pd.DataFrame) -> pd.DataFrame:
    """The handoff's Sec. 1 population: selected, and a leg of a real pair."""
    s = d[d.selected].copy()
    key = s.subrun.astype(str) + ':' + s.event_id.astype(str)
    return s[key.map(key.value_counts()) >= 2].copy()


# --------------------------------------------------------------------------- #
# 1 - the census the handoff asked for first
# --------------------------------------------------------------------------- #
def fallback_census(d: pd.DataFrame, dt_xy: dict) -> pd.DataFrame:
    """How often the measured ``dt_xy`` was used, per arm -- exactly.

    The handoff expected "~100 % miss on A and C from the parity argument" and
    asked for it to be confirmed rather than trusted.  It is not ~100 %, it is
    100.000 %: A, B and C carry only even ``ftst_diff`` and their bundle keys
    are odd, so the intersection is empty by construction and no statistics are
    involved.  D's keys are +-3, which do occur.
    """
    rows = []
    for arm, g in d.groupby('arm'):
        keys = sorted(dt_xy.get(arm, {}))
        seen = sorted(g.ftst_diff.unique())
        rows.append(dict(
            arm=arm, n=len(g), n_hit=int(g.dt_hit.sum()),
            hit_frac=float(g.dt_hit.mean()),
            bundle_keys=','.join(str(k) for k in keys),
            ftst_diff_seen=','.join(str(k) for k in seen),
            bundle_parity='odd' if all(k % 2 for k in keys) else 'mixed/even',
            data_parity='even' if all(k % 2 == 0 for k in seen) else 'odd',
        ))
    out = pd.DataFrame(rows)
    tot = out.n.sum()
    out.attrs['campaign_hit_frac'] = float(out.n_hit.sum() / tot) if tot else np.nan
    return out


# --------------------------------------------------------------------------- #
# 2 - dt_xy from the beam data itself
# --------------------------------------------------------------------------- #
def _hist(x: np.ndarray) -> np.ndarray:
    n, _ = np.histogram(x, HIST_EDGES)
    s = n.sum()
    return n / s if s else n.astype(float)


def _scrambled_hist(x_t0, y_t0, rng, n_rep: int = 6) -> np.ndarray:
    """The accidental pedestal: the same x against a *different* track's y.

    This is the null the coincidence test is supposed to beat.  Permuting
    within the (arm, ftst_diff) class holds everything except the pairing.
    """
    h = np.zeros(len(HIST_EDGES) - 1)
    for _ in range(n_rep):
        h += np.histogram(x_t0 - rng.permutation(y_t0), HIST_EDGES)[0]
    s = h.sum()
    return h / s if s else h


def _peak_lag(h: np.ndarray, ref: np.ndarray, max_lag: int = 40) -> float:
    """Shift of ``h`` relative to ``ref`` in ns, by cross-correlation.

    A mode would do if the distribution had a mode; it does not -- it is a
    ~200 ns plateau whose argmax moves 20 ns between neighbouring ``ftst``
    classes for no reason (measured).  Cross-correlating the whole shape uses
    every bin instead of the single tallest one, and a parabolic refinement on
    the correlation peak takes it below the 5 ns bin.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    c = np.array([float(np.sum(np.roll(h, -l) * ref)) for l in lags])
    i = int(np.argmax(c))
    sub = 0.0
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        if den:
            sub = 0.5 * (c[i - 1] - c[i + 1]) / den
    return float((lags[i] + sub) * HIST_BW)


def insitu_dt(d: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    """``dt_xy`` per (arm, ftst_diff), measured from the beam data.

    On **unambiguous, ungated** tracks: one candidate in each plane, so
    ``select_pair`` never chose and the measurement cannot inherit its ``dt``;
    and no gate, because the gate is a +-120 ns cut on this very quantity
    (``circularity``) and measuring inside it truncates the answer.

    The accidental pedestal is subtracted before the shift is taken -- normalised
    on ``|raw| > PEDESTAL_NS``, where the true and scrambled pairings agree -- so
    the shift is the signal's, not the mixture's.  What is left is still not a
    narrow peak (``discrimination`` reports its width); the shift is nonetheless
    well determined, because the whole distribution translates.

    **``shift_ns`` and ``dt_insitu`` are not equally trustworthy.**  ``shift_ns``
    is a difference between two classes and is what the straight line in
    ``insitu_law`` is fitted to; it is good to a few ns.  ``dt_insitu`` adds an
    absolute anchor -- the reference class's median -- and a median is a poor
    centre for a 200 ns plateau sitting on an accidental pedestal, so the whole
    column can be off by ~10 ns together.  Use the SHAPE (linear, -8.3 ns per
    ftst unit) as the result; treat the absolute level as provisional until
    something with a real time reference pins it.  That is also why no
    acceptance or efficiency number is re-derived here from ``dt_insitu``: at
    this anchor precision it would not mean anything.
    """
    rng = np.random.default_rng(seed)
    un = d[(d.n_cand_x == 1) & (d.n_cand_y == 1) & np.isfinite(d.raw)]
    tail = np.abs(HIST_CENTRES) > PEDESTAL_NS
    rows = []
    for arm, g in un.groupby('arm'):
        sig, meta = {}, {}
        for k, gk in g.groupby('ftst_diff'):
            ht = _hist(gk.raw.values)
            hb = _scrambled_hist(gk.x_t0.values, gk.y_t0.values, rng)
            f = ht[tail].sum() / hb[tail].sum() if hb[tail].sum() else 0.0
            sig[int(k)] = np.clip(ht - f * hb, 0.0, None)
            meta[int(k)] = (len(gk), float(np.median(gk.raw)), float(ht.sum() - f * hb.sum()))
        ks = sorted(sig)
        kref = 0 if 0 in ks else ks[len(ks) // 2]
        med_ref = meta[kref][1]
        for k in ks:
            shift = _peak_lag(sig[k], sig[kref])
            n, med, frac = meta[k]
            rows.append(dict(arm=arm, ftst_diff=k, n=n, shift_ns=shift,
                             dt_insitu=med_ref + shift, median_raw=med,
                             signal_frac=frac))
    return pd.DataFrame(rows)


def insitu_law(t: pd.DataFrame) -> pd.DataFrame:
    """The per-arm straight line through ``insitu_dt``, against the 10 ns quantum.

    ``ftst`` is a phase counter with ``FTST_PHASES`` states over one
    ``SAMPLE_NS`` sample, so a slope of -10 ns per unit is what one plane being
    read one phase later than the other must produce.  The measurement lands at
    -7.7 to -8.8 on four arms independently.  It is biased **low** and in a known
    direction: whatever accidental pedestal survives subtraction does not shift
    with ``ftst``, and pulls the correlation peak toward zero lag.  So the
    measurement is consistent with the quantum without proving equality, and the
    honest reading is that ``dt_xy`` is a readout-clock effect with no chamber
    physics in it at all.
    """
    rows = []
    for arm, g in t.groupby('arm'):
        slope, icept = np.polyfit(g.ftst_diff, g.shift_ns, 1)
        resid = g.shift_ns - (slope * g.ftst_diff + icept)
        rows.append(dict(arm=arm, slope_ns_per_unit=float(slope),
                         intercept_ns=float(icept),
                         max_abs_resid_ns=float(np.abs(resid).max()),
                         expected_ns_per_unit=-FTST_QUANTUM_NS,
                         frac_of_quantum=float(abs(slope) / FTST_QUANTUM_NS),
                         n_classes=len(g)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3 - the handoff's Sec. 1 table, and what moves it
# --------------------------------------------------------------------------- #
def _spread(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, halfwidth=np.nan, f30=np.nan, f60=np.nan, median=np.nan)
    return dict(n=len(x),
                halfwidth=float((np.percentile(x, 84) - np.percentile(x, 16)) / 2),
                f30=float((np.abs(x) > 30).mean()),
                f60=float((np.abs(x) > 60).mean()),
                median=float(np.median(x)))


def residual_table(d: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Sec. 1 reproduced, then re-measured with the in-situ ``dt_xy``.

    Reproduction first, because a re-measurement that cannot reproduce the
    number it is correcting is not a correction.  It lands on 66.1/76.4/77.4 ns
    and n = 16 877/15 063/13 481 against the handoff's 66/76/77 and the same
    three counts.

    Then the point: **subtracting the right ``dt_xy`` barely moves it** (66.1 ->
    66.3, 76.4 -> 76.2, 77.4 -> 75.6).  The wrong constant scatters the residual
    by 25 ns sd across ``ftst`` classes, which is real and worth fixing, and it
    is small beside a 66-77 ns spread.  Whatever the handoff found, a wrong
    ``dt_xy`` is not it.
    """
    lut = {(r.arm, r.ftst_diff): r.dt_insitu for r in t.itertuples()}
    d = d.copy()
    d['dt_insitu'] = [lut.get((a, int(k)), np.nan)
                      for a, k in zip(d.arm, d.ftst_diff)]
    d['res_insitu'] = d.raw - d.dt_insitu
    s = published_sample(d)
    rows = []
    for arm, g in s.groupby('arm'):
        if arm == 'B':
            continue
        for label, col in (('as shipped', 'res_ship'), ('in-situ dt', 'res_insitu')):
            r = dict(arm=arm, dt=label, **_spread(g[col].values))
            r['x_t0_err_med'] = float(g.x_t0_err.median())
            r['y_t0_err_med'] = float(g.y_t0_err.median())
            r['dt_error_sd'] = float((g.dt_insitu - g.dt_used).std())
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4 - the mechanism tests
# --------------------------------------------------------------------------- #
def degeneracy_test(d: pd.DataFrame, t: pd.DataFrame,
                    null_periods=(37, 41, 43, 47, 53, 71, 73, 79, 83, 89)) -> pd.DataFrame:
    """Is the residual combed at 60 ns, as two minima one depth bin apart would be?

    Rayleigh statistic |<exp(2 pi i r / T)>| at the depth bin, at ``T0_STEP``
    (the positive control -- a known 5 ns snap that MUST show), and at ten
    arbitrary periods that stand for "no structure".  The control comes in at
    0.06-0.17 and 60 ns at +1.3/-0.5/+0.9 sigma over the null.  **A test with
    demonstrated sensitivity finds nothing at the predicted period.**
    """
    lut = {(r.arm, r.ftst_diff): r.dt_insitu for r in t.itertuples()}
    s = published_sample(d)
    s = s.assign(r=s.raw - [lut.get((a, int(k)), np.nan)
                            for a, k in zip(s.arm, s.ftst_diff)])
    def amp(x, T):
        ph = 2 * np.pi * x / T
        return float(np.hypot(np.cos(ph).mean(), np.sin(ph).mean()))
    rows = []
    for arm, g in s.groupby('arm'):
        if arm == 'B':
            continue
        r = g.r.values
        r = r[np.isfinite(r)]
        null = np.array([amp(r, T) for T in null_periods])
        a60, a5 = amp(r, DEPTH_BIN_NS), amp(r, T0_STEP)
        rows.append(dict(arm=arm, n=len(r), amp_60ns=a60, amp_5ns_control=a5,
                         null_mean=float(null.mean()), null_sd=float(null.std()),
                         sigma_60=float((a60 - null.mean()) / null.std()),
                         sigma_5_control=float((a5 - null.mean()) / null.std())))
    return pd.DataFrame(rows)


def geometry_test(d: pd.DataFrame, t: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """The handoff's own first check: does the residual grow with inclination?

    An x/y plane separation of a few mm crossed by an inclined track is the
    stated alternative to a fit failure, and it predicts a residual that
    **vanishes at normal incidence**.  It does not: the lowest ``tan theta``
    quintile already carries 65-70 ns of half-width, and the full range adds
    ~10.  Geometry is present and is not the explanation.
    """
    lut = {(r.arm, r.ftst_diff): r.dt_insitu for r in t.itertuples()}
    s = published_sample(d)
    s = s.assign(r=s.raw - [lut.get((a, int(k)), np.nan)
                            for a, k in zip(s.arm, s.ftst_diff)])
    s = s.assign(tan=np.hypot(s.x_tan_theta, s.y_tan_theta))
    rows = []
    for arm, g in s.groupby('arm'):
        if arm == 'B':
            continue
        g = g[np.isfinite(g.tan) & np.isfinite(g.r)]
        if len(g) < 10 * n_bins:
            continue
        g = g.assign(q=pd.qcut(g.tan, n_bins, labels=False, duplicates='drop'))
        for q, gq in g.groupby('q'):
            rows.append(dict(arm=arm, quantile=int(q), tan_med=float(gq.tan.median()),
                             **_spread(gq.r.values)))
    return pd.DataFrame(rows)


def circularity(d: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """``gated`` is the +-120 ns cut on the residual -- so Sec. 1 measures a
    truncated distribution, and the handoff's proposed metric cannot be used.

    Two numbers per arm.  First, that the identity holds: every gated track is
    inside ``|res_ship| <= TOL_NS`` and none outside, so the gate is literally
    this cut (AND the plausibility flags, which remove some tracks inside it).
    Second, what the truncation costs: the unselected half-width is 107-138 ns
    against the gated sample's 66-78.

    The consequence for ``HANDOFF_T0_PRIOR.md`` Sec. 3 step 4 is direct.  A
    prior that moved t0 would move which tracks pass this cut, so the residual
    measured on the survivors would improve **partly by construction**.  Declare
    the target on the unselected population, or on a sample selected some other
    way, and the test means something again.
    """
    lut = {(r.arm, r.ftst_diff): r.dt_insitu for r in t.itertuples()}
    d = d.assign(res_insitu=d.raw - [lut.get((a, int(k)), np.nan)
                                     for a, k in zip(d.arm, d.ftst_diff)])
    rows = []
    for arm, g in d.groupby('arm'):
        inside = np.abs(g.res_ship) <= TOL_NS
        ok = np.isfinite(g.res_ship)
        rows.append(dict(
            arm=arm, n=len(g),
            frac_inside_window=float(inside[ok].mean()),
            frac_gated=float(g.gated[ok].mean()),
            gated_outside_window=int((g.gated & ~inside & ok).sum()),
            hw_gated=_spread(g.loc[g.gated, 'res_insitu'].values)['halfwidth'],
            hw_all=_spread(g.res_insitu.values)['halfwidth'],
            hw_ungated=_spread(g.loc[~g.gated, 'res_insitu'].values)['halfwidth'],
            frac_within_20ns_of_cut=float((np.abs(g.loc[g.gated, 'res_ship']) > TOL_NS - 20).mean()),
        ))
    return pd.DataFrame(rows)


def discrimination(d: pd.DataFrame, seed: int = 17) -> pd.DataFrame:
    """What the coincidence test is worth, and what the planes really agree to.

    Against the scrambled pairing (same arm, same ``ftst`` class, someone else's
    ``y_t0``) the +-120 ns window keeps 67-73 % of true pairs and 29-38 % of
    accidental ones.  That is a **2x** enhancement bought for 30 % of the real
    tracks -- not nothing, and not the discriminator ``select_pair``'s docstring
    describes when it says it uses information "that single-plane selection
    cannot use".

    ``signal_hw_ns`` is the half-width of the pedestal-subtracted agreement
    peak: 80-105 ns on the x-y difference, so ~57-74 ns per plane.  Beside a
    fitted ``x_t0_err`` of 2.6-7.4 ns that is a factor of 10-25.  **This, not a
    two-minimum ambiguity, is what Sec. 1 was looking at** -- and it is the same
    scale as one depth bin, which is why the bin was a tempting explanation.
    """
    rng = np.random.default_rng(seed)
    tail = np.abs(HIST_CENTRES) > PEDESTAL_NS
    rows = []
    for arm, g in d[np.isfinite(d.raw)].groupby('arm'):
        n_true = n_fake = n_tot = 0
        sig = np.zeros(len(HIST_CENTRES))
        for k, gk in g.groupby('ftst_diff'):
            dt = gk.dt_used.iloc[0]
            x, y = gk.x_t0.values, gk.y_t0.values
            n_true += int((np.abs(gk.raw - dt) <= TOL_NS).sum())
            n_fake += int((np.abs(x - rng.permutation(y) - dt) <= TOL_NS).sum())
            n_tot += len(gk)
            ht = _hist(gk.raw.values)
            hb = _scrambled_hist(x, y, rng)
            f = ht[tail].sum() / hb[tail].sum() if hb[tail].sum() else 0.0
            sig += np.clip(ht - f * hb, 0.0, None) * len(gk)
        tot = sig.sum()
        cum = np.cumsum(sig) / tot if tot else np.zeros_like(sig)
        hw = float((HIST_CENTRES[np.searchsorted(cum, 0.84)] -
                    HIST_CENTRES[np.searchsorted(cum, 0.16)]) / 2) if tot else np.nan
        rows.append(dict(arm=arm, n=n_tot,
                         acc_true=n_true / n_tot, acc_scrambled=n_fake / n_tot,
                         enhancement=(n_true / n_fake) if n_fake else np.nan,
                         signal_frac=float(tot / n_tot) if n_tot else np.nan,
                         signal_hw_ns=hw, per_plane_sigma_ns=hw / np.sqrt(2),
                         x_t0_err_med=float(g.x_t0_err.median()),
                         err_understated_by=hw / np.sqrt(2) / float(g.x_t0_err.median())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracks', default=None,
                    help='stage-3 campaign track parquet (default: resolved)')
    ap.add_argument('--bundles', default=None,
                    help='bundle root holding mx17_A..D (default: resolved)')
    ap.add_argument('--out', default=None, help='output directory')
    a = ap.parse_args()

    tracks = Path(a.tracks) if a.tracks else paths.spell(
        'out', 'stage3_campaign', 'tracks_campaign.parquet')
    broot = Path(a.bundles) if a.bundles else paths.spell(
        'x17', 'sept26_fullpass', 'bundles')
    out = Path(a.out) if a.out else paths.out('xy_t0')
    out.mkdir(parents=True, exist_ok=True)
    paths.require(tracks, 'stage-3 campaign track table')

    dt_xy = bundle_dt_xy(broot)
    print(f'[xy_t0] bundles: ' + '  '.join(
        f'{k}={sorted(v) or "none"}' for k, v in dt_xy.items()))
    d = load_tracks(tracks, dt_xy)
    print(f'[xy_t0] {len(d):,} tracks')

    census = fallback_census(d, dt_xy)
    insitu = insitu_dt(d)
    law = insitu_law(insitu)
    tables = {
        'fallback_census': census,
        'insitu_dt': insitu,
        'insitu_law': law,
        'residual_table': residual_table(d, insitu),
        'degeneracy_test': degeneracy_test(d, insitu),
        'geometry_test': geometry_test(d, insitu),
        'circularity': circularity(d, insitu),
        'discrimination': discrimination(d),
    }
    for name, t in tables.items():
        t.to_csv(out / f'{name}.csv', index=False)
        print(f'\n=== {name}')
        print(t.to_string(index=False, max_colwidth=40))

    meta = dict(schema=SCHEMA, tracks=str(tracks), bundles=str(broot),
                n_tracks=int(len(d)), fallback_dt=FALLBACK_DT, tol_ns=TOL_NS,
                t0_step=T0_STEP, depth_bin_ns=DEPTH_BIN_NS,
                ftst_quantum_ns=FTST_QUANTUM_NS,
                ftst_phases=FTST_PHASES, sample_ns=SAMPLE_NS,
                campaign_dt_hit_frac=float(census.attrs['campaign_hit_frac']),
                bundle_dt_xy={k: {str(i): v for i, v in m.items()}
                              for k, m in dt_xy.items()})
    (out / 'xy_t0.meta.json').write_text(json.dumps(meta, indent=1))
    print(f'\n[xy_t0] wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
