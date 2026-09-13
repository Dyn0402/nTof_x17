#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chi2_shape.py -- why chamber A's chi2/dof is double-humped and the others' is not.

THE QUESTION.  ``qa_chi2dof_worst`` (the pair-quality set) shows a clean two-bump
chi2/dof distribution wherever chamber A is a leg -- a sharp low peak near 1.5
and a broad one near 20 -- and a single broad blob everywhere else.  This module
takes that apart.

WHAT chi2/dof ACTUALLY IS HERE, because the answer turns on it.  This is not a
track-residual chi2.  ``wft.model.chi2_plane`` fits the forward model to the raw
WAVEFORM window -- every strip, every sample -- and

    dof = (~saturated).sum()                    == 20 * n_strips, verified below

so chi2/dof is the **mean squared residual per sample, in units of that strip's
own measured noise**.  ``wft.model.prep_plane`` takes the noise per strip from
the event itself (``P['noise']``, floored at 3 ADC, gain-corrected), so it is
self-calibrating per channel per chamber.  That fixes the scale absolutely:

    **chi2/dof = 1 is the noise floor, and it means the same thing on all four
    chambers.**  A chamber whose best tracks sit at 3 has a model that does not
    describe its data, not a units problem.

That is what makes "A is lower than C and D" a physics statement rather than a
normalisation one, and it is why this module never rescales a chamber to another.

FIRST, THE FIGURE THAT WAS ASKED ABOUT OVERSTATES IT.  ``pair_hist`` and
``chi2_axis`` measure this.  `make_pair_qa_figures._hist` returns
``n / tot / np.diff(edges)`` on ``np.geomspace`` edges, so every bin is divided
by a LINEAR width that grows with chi2/dof while the axis is log.  The plotted
height is the per-bin fraction divided by chi2/dof: the low end is lifted ~5x
and a 5.3 % shoulder is drawn as a mode.  Five of that set's nine panels are
log-scaled and all carry it.

BUT THE FEATURE IS REAL, and ``modality`` is the number that says so.  On the
honest log view -- log bins, per-bin fraction -- chamber A's SINGLE-TRACK
distribution has a mode at 1.45, a dip at 11.0 that is 5.4x below it, and a
second mode at 22.9.  Dip depth against the shallower mode: A 0.63, C 0.89,
D 0.76.  A's is the only convincing one.  (The log AXIS is not the issue; it is
the right axis for four decades.  ``chi2_axis`` also carries the linear view,
where the whole thing collapses to one spike and a tail -- and 31 % of chamber D
falls off the right-hand end.  That is not a check, it is the wrong axis.)

THE THREE ANSWERS, each with the table that carries it:

  1. ``chi2_vs_len``   the bimodality is **track length**.  chi2/dof climbs
     monotonically with ``x_n_strips`` on every chamber -- A 1.5 -> 21, C 2.8 ->
     26, D 3.1 -> 129 -- so the sample is a mixture of a short-track population
     at the floor and a long-track population an order of magnitude above it.
     At FIXED length every chamber is single-peaked (``chi2_hist``, the
     length-sliced columns): there is no second mechanism.
  2. ``chi2_grid``     at fixed length the driver is **pulse amplitude**.  In
     this selected sample the dependence is monotone: the model's residual is a
     fixed FRACTION of the pulse, so on a quiet track it is buried in the noise
     and on a loud one it outgrows it.  A's low-chi2 short tracks carry ~2.5x
     less charge than its high-chi2 ones at identical ``n_strips``.  (The
     unselected population also turns up at the very quiet end -- fits that
     explained nothing -- which these cuts mostly remove.)
  3. ``chi2_by_run``   the per-chamber OFFSET -- A 1.5 against C 2.8 and D 3.1
     on short tracks -- is constant across every run with enough statistics to
     measure and across both access conditions.  It is not a run, a period or a
     beam effect.

SO WHY ONLY A.  Both modes exist on every chamber.  A is the only chamber whose
short-track mode reaches the noise floor, which puts it 13x below its long-track
mode and leaves clear air between them.  C's short mode sits at 2.8 and D's at
3.1, not resolved from the long-track continuum, so the same two populations
read as one broad blob.  **The low bump is not missing on C and D -- it is not
separated.**  ``reweight`` shows the n_strips SPECTRUM is not the difference:
give A chamber C's length distribution and A still lands at 0.296 against C's
own 0.084.

WHAT THIS MODULE DOES NOT ESTABLISH.  It does not prove what causes the
per-chamber offset.  ``bundle_diff`` reports what the four BENCH bundles differ
in as a matter of record -- read its docstring before using it, because the
campaign ran ``calib_bundle_prelim`` derived from these, and the derivation
keeps the kernel hypers (including C's ``sigma_p0``/``Dp``, an order of
magnitude off the others') while replacing v_drift and dropping the t0 prior
**for every arm alike**.  No bundle at any stage carries a ``dead`` or ``hot``
channel mask.  Ranking these needs a refit with one knob moved at a time, which
is a separate job; see README.md section 5 and the two handoffs beside it.

    python ntof_athens_26/chi2_bimodality/chi2_shape.py
    X17_ROOT=D:/x17 python ntof_athens_26/chi2_bimodality/chi2_shape.py
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

SCHEMA = 'ntof_athens_26/chi2_bimodality/1'

ARMS = ('A', 'B', 'C', 'D')

#: Columns pulled from the stage-3 campaign track table.  One row per TRACK,
#: which is the level the question lives at -- the published figure is per PAIR
#: and takes the worst of four fits, so it compounds four draws of the same
#: single-track distribution and cannot show a mechanism.
TRACK_COLS = [
    'arm', 'run', 'condition', 'chi2dof_x', 'chi2dof_y',
    'x_chi2', 'x_dof', 'x_n_strips', 'y_n_strips', 'x_q_sum',
    'x_n_dropped', 'x_isochronous', 'drift_len_mm', 'drift_railed',
    'x_t0', 'x_w', 'angle_to_beam_deg',
    # the published selection, so this module can reproduce the exact
    # population the QA figure was drawn from rather than a similar one
    'gated', 'angle_calibrated', 'dca_axis_mm', 'event_id', 'subrun',
]

#: `source_imaging._track_table`'s selection, which every published pair is
#: built from: gated, angle-calibrated, pointing within ``DCA_MAX`` of the beam
#: axis.  Reproduced rather than approximated -- the double hump is a feature OF
#: THIS SAMPLE and is much weaker in the unselected stage-3 population, so an
#: investigation run on "roughly the same tracks" would be explaining a
#: different distribution from the one that was asked about.
DCA_MAX = 30.0

#: Track-length classes, in strips.  The two that matter are the first (the
#: short-track mode) and the last two (the long-track mode); the middle ones
#: are there so the crossover is visible rather than asserted.
LEN_BINS = [11, 14, 18, 24, 34, 60, 10_000]
LEN_LABELS = ['11-13', '14-17', '18-23', '24-33', '34-59', '60+']

#: The histogram the figures are drawn from, in log10(chi2/dof).
LOG_EDGES = np.arange(-0.6, 3.401, 0.08)

#: "At the noise floor" -- chi2/dof below this is a fit that describes the
#: waveform to within the strips' own measured noise.  1.5 not 1.0 because the
#: floor is a distribution, not a point, and a 12-strip window has 240 samples.
FLOOR = 1.5

#: q_sum above this is the NNLS charge blow-up, not a measurement: the profile
#: runs away into a near-null direction and returns 1e30-ish totals.  Excluded
#: from the charge tables and COUNTED, because the count is itself a finding.
Q_SANE_MAX = 1e6


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_tracks(src: Path, selected: bool = True) -> pd.DataFrame:
    """The stage-3 campaign track table, with the derived columns this needs.

    ``selected`` applies `source_imaging._track_table`'s cuts AND the
    two-tracks-in-one-trigger requirement that `_pairs_real` imposes, so the
    rows are the legs of the published pair sample.  Pass False for the whole
    stage-3 population, which is the generality check, not the subject.
    """
    d = pd.read_parquet(src, columns=TRACK_COLS)
    d['sample'] = 'stage3'
    if selected:
        d = d[d.gated & d.angle_calibrated & (d.dca_axis_mm < DCA_MAX)].copy()
        # a pair needs a trigger that made two tracks; a lone track never
        # reaches the published figure, and lone tracks are not drawn from the
        # same population as the busy ones (see `_pairs_mixed`'s docstring)
        key = d.subrun.astype(str) + ':' + d.event_id.astype(str)
        d = d[key.map(key.value_counts()) >= 2].copy()
        d['sample'] = 'paired'
    d['len_class'] = pd.cut(d.x_n_strips, LEN_BINS, right=False,
                            labels=LEN_LABELS)
    d['q_per_strip'] = d.x_q_sum / d.x_n_strips
    d['q_sane'] = np.isfinite(d.q_per_strip) & (d.q_per_strip > 0) & \
        (d.q_per_strip < Q_SANE_MAX)
    return d


def load_pairs(src: Path) -> pd.DataFrame:
    """The pair QA table -- only to reproduce the published figure's curve."""
    return pd.read_parquet(src, columns=['arm1', 'arm2', 'topology', 'mixed',
                                         'chi2dof_worst'])


def legs_crosscheck(src: Path) -> pd.DataFrame:
    """The same headline numbers, taken from the PAIR TABLE's own legs.

    `load_tracks(selected=True)` reproduces `_track_table`'s cuts from the
    merged stage-3 parquet; `pair_qa.py` applied them to the per-sub-run files
    and kept a few thousand legs this reconstruction does not (its
    ``--include-pre-access`` runs, and a handful of chamber-B legs that reach
    it by another route).  So the reproduction is close but not identical, and
    saying "the same sample" without checking would be a claim nobody had
    tested.  This takes the numbers straight from the published table instead:
    if the two disagree on the per-chamber ordering, the reconstruction is
    wrong and the conclusions go with it.
    """
    d = pd.read_parquet(src, columns=[
        'key1', 'key2', 'arm1', 'arm2', 'mixed',
        'chi2dof_x_1', 'chi2dof_x_2', 'x_n_strips_1', 'x_n_strips_2'])
    d = d[~d.mixed]
    legs = pd.concat([
        d[[f'key{i}', f'arm{i}', f'chi2dof_x_{i}', f'x_n_strips_{i}']]
        .rename(columns={f'key{i}': 'tkey', f'arm{i}': 'arm',
                         f'chi2dof_x_{i}': 'chi2dof_x',
                         f'x_n_strips_{i}': 'x_n_strips'})
        for i in (1, 2)], ignore_index=True).drop_duplicates('tkey')
    legs = legs[np.isfinite(legs.chi2dof_x)]
    short = legs[legs.x_n_strips.between(LEN_BINS[0], LEN_BINS[1] - 1)]
    g = legs.groupby('arm').chi2dof_x.agg(n='size', median_all='median')
    g['frac_at_floor'] = legs.groupby('arm').chi2dof_x.apply(
        lambda s: float((s < FLOOR).mean()))
    g['median_short'] = short.groupby('arm').chi2dof_x.median()
    g['n_short'] = short.groupby('arm').size()
    return g.reset_index()


# --------------------------------------------------------------------------- #
# 0 - the definition check
# --------------------------------------------------------------------------- #
def dof_check(d: pd.DataFrame) -> dict:
    """dof == 20 * n_strips, or the whole reading of chi2/dof is wrong.

    Asserted rather than assumed: everything downstream treats chi2/dof as a
    per-SAMPLE mean residual, which is only true if dof counts samples.
    """
    r = (d.x_dof / d.x_n_strips).replace([np.inf, -np.inf], np.nan).dropna()
    return dict(ratio_median=float(r.median()),
                ratio_p01=float(r.quantile(0.01)),
                ratio_p99=float(r.quantile(0.99)),
                frac_exactly_20=float((d.x_dof == 20 * d.x_n_strips).mean()),
                n=int(len(r)))


# --------------------------------------------------------------------------- #
# 1 - the histograms, whole and sliced by length
# --------------------------------------------------------------------------- #
def chi2_hist(d: pd.DataFrame) -> pd.DataFrame:
    """log10(chi2/dof) histograms: per arm, the whole sample and each length class.

    This is the figure that answers the question.  The 'all' column is what the
    published plot shows; the length columns are the same tracks split, and they
    show the two bumps ARE the two length populations.
    """
    rows = []
    ctr = 0.5 * (LOG_EDGES[:-1] + LOG_EDGES[1:])
    for arm in ARMS:
        g = d[(d.arm == arm) & np.isfinite(d.chi2dof_x) & (d.chi2dof_x > 0)]
        if not len(g):
            continue
        base = dict(arm=arm)
        h, _ = np.histogram(np.log10(g.chi2dof_x), bins=LOG_EDGES)
        tot = max(h.sum(), 1)
        for i, c in enumerate(ctr):
            rows.append({**base, 'log_chi2dof': round(float(c), 4),
                         'slice': 'all', 'n': int(h[i]),
                         'frac': float(h[i] / tot), 'frac_own': float(h[i] / tot)})
        for lab in LEN_LABELS:
            s = g[g.len_class == lab]
            if not len(s):
                continue
            hs, _ = np.histogram(np.log10(s.chi2dof_x), bins=LOG_EDGES)
            own = max(hs.sum(), 1)
            for i, c in enumerate(ctr):
                # two normalisations, because they answer different questions:
                # ``frac`` is of the whole arm, so the slices stack into 'all'
                # and the reader sees how much of the sample each mode is;
                # ``frac_own`` is of the slice, so the MODE POSITIONS can be
                # compared between slices that differ 30-fold in population.
                rows.append({**base, 'log_chi2dof': round(float(c), 4),
                             'slice': lab, 'n': int(hs[i]),
                             'frac': float(hs[i] / tot),
                             'frac_own': float(hs[i] / own)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2 - the length dependence
# --------------------------------------------------------------------------- #
#: The linear views.  ``fine`` is 0-10 at a 0.1 step, which is where the low
#: peak lives and where a unit-wide bin was hiding it -- chamber A puts 36 % of
#: its tracks in 1 < chi2/dof < 2 alone, so that one bin WAS the peak.  ``full``
#: is 0-60 at 0.5 for the context the zoom throws away.  Both are reported as a
#: DENSITY per unit chi2, which is what makes them the same curve at two
#: resolutions rather than two incomparable histograms -- and on a LINEAR axis
#: dividing by the bin width is simply correct, which is exactly what it is not
#: on the log axis (see ``pair_hist``).
LIN_FINE = np.arange(0.0, 10.0001, 0.1)
LIN_FULL = np.arange(0.0, 60.0001, 0.5)


def chi2_axis(d: pd.DataFrame) -> pd.DataFrame:
    """The same chi2/dof at three resolutions, linear and log.

    ``lin_fine``  linear, 0-10, step 0.1, density per unit chi2.  The low peak
                  resolved.  This is the view that shows chamber A's peak
                  sitting ON the noise floor rather than merely near it.
    ``lin_full``  linear, 0-60, step 0.5, same units -- the zoom's context.
    ``log``       log bins, per-bin FRACTION: the correctly normalised log
                  view, for the decades the linear axis cannot reach.

    ``dens`` is the column to plot for the two linear views and ``frac`` for
    the log one, and the reason is the whole point of this module's ``pair_hist``:
    dividing by a bin's linear width is right when the axis is linear and wrong
    when it is log.
    """
    rows = []
    lc = np.sqrt(10 ** LOG_EDGES[:-1] * 10 ** LOG_EDGES[1:])
    views = (('lin_fine', LIN_FINE), ('lin_full', LIN_FULL))
    for arm in ARMS:
        v = d.loc[(d.arm == arm) & np.isfinite(d.chi2dof_x) & (d.chi2dof_x > 0),
                  'chi2dof_x'].to_numpy()
        if len(v) < 200:
            continue
        tot = len(v)
        for name, e in views:
            h, _ = np.histogram(v, bins=e)
            w = np.diff(e)
            ctr = 0.5 * (e[:-1] + e[1:])
            for i, c in enumerate(ctr):
                rows.append(dict(arm=arm, view=name, chi2dof=float(c),
                                 n=int(h[i]), frac=float(h[i] / tot),
                                 dens=float(h[i] / tot / w[i]),
                                 frac_offaxis=float((v > e[-1]).mean())))
        h, _ = np.histogram(v, bins=10 ** LOG_EDGES)
        for i, c in enumerate(lc):
            rows.append(dict(arm=arm, view='log', chi2dof=float(c),
                             n=int(h[i]), frac=float(h[i] / tot),
                             dens=float(h[i] / tot),
                             frac_offaxis=float((v > 10 ** LOG_EDGES[-1]).mean())))
    return pd.DataFrame(rows)


def floor_peak(d: pd.DataFrame) -> pd.DataFrame:
    """Where each chamber's low peak actually sits, at 0.1 resolution.

    The headline of the whole investigation is "A reaches the noise floor and
    the others do not", and until now that rested on a median over a coarse
    length class.  This measures the mode directly on the fine linear view, so
    the claim is a located peak rather than a summary statistic.
    """
    rows = []
    ctr = 0.5 * (LIN_FINE[:-1] + LIN_FINE[1:])
    for arm in ARMS:
        v = d.loc[(d.arm == arm) & np.isfinite(d.chi2dof_x) & (d.chi2dof_x > 0),
                  'chi2dof_x'].to_numpy()
        if len(v) < 200:
            continue
        h, _ = np.histogram(v, bins=LIN_FINE)
        dens = h / len(v) / 0.1
        i = int(np.argmax(dens))
        # half-maximum width of that peak, walking out from the mode
        half = dens[i] / 2.0
        lo = i
        while lo > 0 and dens[lo] > half:
            lo -= 1
        hi = i
        while hi < len(dens) - 1 and dens[hi] > half:
            hi += 1
        rows.append(dict(
            arm=arm, n=int(len(v)), peak_at=float(ctr[i]),
            peak_dens=float(dens[i]),
            fwhm_lo=float(ctr[lo]), fwhm_hi=float(ctr[hi]),
            frac_below_1=float((v < 1.0).mean()),
            frac_1_to_2=float(((v >= 1.0) & (v < 2.0)).mean()),
            frac_below_10=float((v < 10.0).mean())))
    return pd.DataFrame(rows)


def modality(d: pd.DataFrame) -> pd.DataFrame:
    """Is the log-space bimodality real?  Peak, dip, peak -- measured, per arm.

    The claim "A is bimodal and C and D are not" should be a number, not an
    impression of a curve.  Low mode searched below chi2/dof 2.5, dip between
    2.5 and 12, high mode above 12, all on the per-bin fraction of the honest
    log view.  ``dip_depth`` is how far the dip sits below the SHALLOWER of the
    two modes: 1.0 is no dip at all and a genuine bimodality needs it well
    under 1.
    """
    rows = []
    lc = np.sqrt(10 ** LOG_EDGES[:-1] * 10 ** LOG_EDGES[1:])
    for arm in ARMS:
        v = d.loc[(d.arm == arm) & np.isfinite(d.chi2dof_x) & (d.chi2dof_x > 0),
                  'chi2dof_x'].to_numpy()
        if len(v) < 200:
            continue
        h, _ = np.histogram(v, bins=10 ** LOG_EDGES)
        h = h / len(v)
        lo, hi = np.searchsorted(lc, 2.5), np.searchsorted(lc, 12.0)
        if lo < 2 or hi >= len(h) - 1:
            continue
        i1 = int(np.argmax(h[:lo]))
        i2 = lo + int(np.argmin(h[lo:hi]))
        i3 = hi + int(np.argmax(h[hi:]))
        shallower = min(h[i1], h[i3])
        rows.append(dict(
            arm=arm, n=int(len(v)),
            low_mode_at=float(lc[i1]), low_mode=float(h[i1]),
            dip_at=float(lc[i2]), dip=float(h[i2]),
            high_mode_at=float(lc[i3]), high_mode=float(h[i3]),
            dip_depth=float(h[i2] / shallower) if shallower else np.nan))
    return pd.DataFrame(rows)


def chi2_vs_len(d: pd.DataFrame) -> pd.DataFrame:
    """chi2/dof against n_strips, per arm.  The universal rise, and the offset."""
    rows = []
    for arm in ARMS:
        g = d[(d.arm == arm) & np.isfinite(d.chi2dof_x)]
        for lab in LEN_LABELS:
            s = g[g.len_class == lab]
            if len(s) < 30:
                continue
            rows.append(dict(
                arm=arm, len_class=lab, n=int(len(s)),
                p25=float(s.chi2dof_x.quantile(.25)),
                median=float(s.chi2dof_x.median()),
                p75=float(s.chi2dof_x.quantile(.75)),
                frac_at_floor=float((s.chi2dof_x < FLOOR).mean()),
                median_q_per_strip=float(s.loc[s.q_sane, 'q_per_strip'].median()),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3 - the reweighting test
# --------------------------------------------------------------------------- #
def reweight(d: pd.DataFrame) -> pd.DataFrame:
    """Is the chamber difference just a different length spectrum?  No.

    Each arm's per-length-class frac_at_floor, re-averaged over every OTHER
    arm's length distribution.  If the spectrum were the explanation, an arm
    reweighted to another's spectrum would land on that other's raw value.
    """
    per = {}
    for arm in ARMS:
        g = d[(d.arm == arm) & np.isfinite(d.chi2dof_x)]
        per[arm] = g.groupby('len_class', observed=True).chi2dof_x.agg(
            n='size', flo=lambda s: float((s < FLOOR).mean()))
    rows = []
    for src in ARMS:
        if src not in per or not len(per[src]):
            continue
        own = per[src]
        rows.append(dict(arm=src, weights='own', value=float(
            (own.n * own.flo).sum() / own.n.sum())))
        for ref in ARMS:
            if ref == src or ref not in per or not len(per[ref]):
                continue
            j = own.join(per[ref].n.rename('nref'), how='inner').dropna()
            if not len(j) or j.nref.sum() == 0:
                continue
            rows.append(dict(arm=src, weights=ref, value=float(
                (j.nref * j.flo).sum() / j.nref.sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4 - the (length x amplitude) grid
# --------------------------------------------------------------------------- #
def chi2_grid(d: pd.DataFrame, n_q: int = 5) -> pd.DataFrame:
    """Median chi2/dof on a (length class x charge-per-strip quintile) grid.

    Quintiles are taken PER ARM, so a cell is "this chamber's quietest fifth",
    not a shared absolute charge -- the gains are not cross-calibrated and
    pretending otherwise would invent a comparison.  The edges are carried in
    the table so the reader can see what the bin actually is.
    """
    rows = []
    for arm in ARMS:
        g = d[(d.arm == arm) & np.isfinite(d.chi2dof_x) & d.q_sane].copy()
        if len(g) < 500:
            continue
        try:
            g['qb'], edges = pd.qcut(g.q_per_strip, n_q, labels=False,
                                     duplicates='drop', retbins=True)
        except ValueError:
            continue
        for lab in LEN_LABELS:
            for qb in range(len(edges) - 1):
                s = g[(g.len_class == lab) & (g.qb == qb)]
                if len(s) < 25:
                    continue
                rows.append(dict(
                    arm=arm, len_class=lab, q_bin=int(qb),
                    q_lo=float(edges[qb]), q_hi=float(edges[qb + 1]),
                    n=int(len(s)), median=float(s.chi2dof_x.median()),
                    frac_at_floor=float((s.chi2dof_x < FLOOR).mean())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5 - stability across runs and conditions
# --------------------------------------------------------------------------- #
def chi2_by_run(d: pd.DataFrame) -> pd.DataFrame:
    """Short-track median chi2/dof per run per arm.  The ordering never moves."""
    s = d[(d.len_class == LEN_LABELS[0]) & np.isfinite(d.chi2dof_x)]
    g = s.groupby(['run', 'arm'], observed=True).chi2dof_x.agg(
        ['size', 'median']).reset_index()
    g.columns = ['run', 'arm', 'n', 'median']
    cond = s.groupby('run', observed=True).condition.first()
    g['condition'] = g.run.map(cond)
    return g[g.n >= 25].sort_values(['run', 'arm'])


# --------------------------------------------------------------------------- #
# 6 - what the four bundles differ in
# --------------------------------------------------------------------------- #
#: Bundle fields that change what chi2 means.  Reported as a matter of record;
#: this module does not rank them -- that needs a refit (README section 5).
BUNDLE_SCALARS = ['detector', 'run_key', 'share_mode', 'v_drift',
                  't0_prior_sigma', 'sat_adc', 'n_depth_bins', 'pitch_mm']
BUNDLE_HYPER = ['c1', 'c2', 'c2_over_c1', 'kY', 'tau_s', 'sigma_s',
                'sigma_p0', 'Dp']


def bundle_diff(bundle_root: Path) -> pd.DataFrame:
    """The four per-detector BENCH bundles, field by field.

    ⚠ THESE ARE NOT THE BUNDLES THE CAMPAIGN RAN.  Stage-3 provenance names
    ``calib_bundle_prelim``, built on a condor worker by
    ``ntof_tracking.wft_beam.make_bundle`` from the bench bundle here.  That is
    a pure function, so reading the bench bundle is the right way to see what
    was carried in -- but only for the fields it carries:

      TRANSFERRED VERBATIM   the impulse template, and the sharing kernel and
                             its hypers (c1, c2/c2_over_c1, kY, tau_s,
                             sigma_s) -- hardware properties of the chambers.
                             ALSO sigma_p0 and Dp, which `make_bundle`'s own
                             docstring calls "the largest un-validated
                             assumption in this chain".
      REPLACED               v_drift (a shared Magboltz prior, 42.6 um/ns for
                             every arm), sat_adc, sample_ns, conditions.
      DROPPED FOR EVERY ARM  ``t0_abs`` and ``t0_prior_sigma``.

    So ``prior_active_on_bench`` below describes the BENCH configuration and
    is **not** a campaign difference between chambers: `make_bundle` zeroes the
    prior on all four, deliberately (RUN145_R06_2026-08-19.md §1, commit
    11ce347 -- the bench t0 is an absolute arrival time measured against the
    bench trigger and DAQ latency, which is a wrong answer for an n_TOF-
    triggered run, stated to +-5 ns and therefore effectively a hard pin).
    **The t0 prior is off for A, B, C and D alike in every product this
    investigation looks at, and cannot explain any chamber-to-chamber
    difference in chi2.**  The column is kept because the bench asymmetry is
    real and is worth knowing when bench work resumes -- not because it
    explains anything here.
    """
    rows = []
    for arm in ARMS:
        hits = sorted(glob.glob(str(bundle_root / f'mx17_{arm}' / '*' /
                                    'bundle.json')))
        if not hits:
            continue
        p = Path(hits[0])
        b = json.load(open(p))
        h = b.get('hyper') or {}
        t0a = b.get('t0_abs')
        sig = b.get('t0_prior_sigma')
        r = dict(arm=arm, bundle=p.parent.name)
        for k in BUNDLE_SCALARS:
            r[k] = b.get(k)
        for k in BUNDLE_HYPER:
            r[f'hyper_{k}'] = h.get(k)
        r['has_t0_abs'] = bool(t0a)
        # bench configuration only -- make_bundle zeroes this for every arm
        r['prior_active_on_bench'] = bool(sig) and bool(t0a)
        r['prior_active_in_campaign'] = False
        dead = b.get('dead')
        hot = b.get('hot')
        r['n_dead_masked'] = (sum(len(v) for v in dead.values())
                              if isinstance(dead, dict) else 0)
        r['n_hot_masked'] = (sum(len(v) for v in hot.values())
                             if isinstance(hot, dict) else 0)
        prov = b.get('provenance') or {}
        r['n_train'] = prov.get('n_train')
        r['fit_chi2'] = prov.get('chi2')
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7 - tie back to the published pair figure
# --------------------------------------------------------------------------- #
#: The published QA figure's own binning: `make_pair_qa_figures.SPECS`
#: ``chi2dof_worst`` is ``scale='log', lo=1.0, hi=3000.0, bins=44``.  Copied so
#: the two normalisations below are compared on the SAME bins as the figure
#: being questioned, not on a similar grid.
PUB_EDGES = np.geomspace(1.0, 3000.0, 45)


def pair_hist(p: pd.DataFrame) -> pd.DataFrame:
    """chi2dof_worst per arm pair, on the published figure's own bins, BOTH ways.

    THE PUBLISHED CURVE IS NOT THE DISTRIBUTION OF log chi2/dof.
    `make_pair_qa_figures._hist` returns ``n / tot / w`` with ``w =
    np.diff(edges)`` -- the bins' LINEAR widths -- while the edges are
    geometric and the axis is log.  A log-spaced bin's linear width grows in
    proportion to x, so the plotted height is the per-bin fraction divided by
    chi2/dof: the left-hand end of the axis is multiplied by roughly 5 relative
    to the peak, and a small low-chi2 shoulder is rendered as a bump of
    comparable height to the main mode.

    Both columns are written so the difference can be shown rather than
    asserted:

      ``dens_linear``  n/tot/dx  -- what the figure plots.
      ``frac``         n/tot     -- the fraction of pairs in the bin, which is
                       what a reader of a log axis reads off it.

    On a log axis the honest density is per unit log x (``frac`` divided by the
    constant log width, i.e. ``frac`` up to one overall factor), so ``frac`` is
    the curve the figure should have drawn.
    """
    rows = []
    lo = np.log10(PUB_EDGES[:-1])
    hi = np.log10(PUB_EDGES[1:])
    ctr = 0.5 * (lo + hi)
    dx = np.diff(PUB_EDGES)
    real = p[~p.mixed]
    for (a1, a2), g in real.groupby(['arm1', 'arm2'], observed=True):
        v = g.chi2dof_worst
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        n, _ = np.histogram(v, bins=PUB_EDGES)
        tot = len(v)                      # as published: every finite value
        for i, c in enumerate(ctr):
            rows.append(dict(pair=f'{a1}-{a2}',
                             topology=g.topology.iloc[0],
                             has_A=('A' in (a1, a2)),
                             log_chi2dof=round(float(c), 4),
                             chi2dof=float(10 ** c),
                             n=int(n[i]),
                             frac=float(n[i] / tot),
                             dens_linear=float(n[i] / tot / dx[i])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarise(d: pd.DataFrame, vlen: pd.DataFrame, rw: pd.DataFrame,
              bd: pd.DataFrame, dofc: dict) -> dict:
    short = LEN_LABELS[0]
    per_arm = {}
    for arm in ARMS:
        g = d[(d.arm == arm) & np.isfinite(d.chi2dof_x)]
        if not len(g):
            continue
        s = g[g.len_class == short]
        lon = g[g.len_class.isin(LEN_LABELS[-2:])]
        per_arm[arm] = dict(
            n_tracks=int(len(g)),
            median_all=float(g.chi2dof_x.median()),
            median_short=float(s.chi2dof_x.median()) if len(s) else None,
            median_long=float(lon.chi2dof_x.median()) if len(lon) else None,
            frac_at_floor=float((g.chi2dof_x < FLOOR).mean()),
            frac_at_floor_short=(float((s.chi2dof_x < FLOOR).mean())
                                 if len(s) else None),
            separation=(float(lon.chi2dof_x.median() / s.chi2dof_x.median())
                        if len(s) and len(lon) else None),
            frac_q_insane=float(1.0 - g.q_sane.mean()),
        )
    rwv = {f'{r.arm}<-{r.weights}': round(r.value, 4)
           for r in rw.itertuples()}
    return dict(
        schema=SCHEMA, floor=FLOOR, n_tracks=int(len(d)),
        n_runs=int(d.run.nunique()),
        sample=str(d['sample'].iloc[0]), dca_max=DCA_MAX,
        dof_check=dofc, per_arm=per_arm, reweight=rwv,
        bundles={r.arm: dict(bundle=r.bundle,
                             prior_active_on_bench=bool(r.prior_active_on_bench),
                             n_dead_masked=int(r.n_dead_masked),
                             n_hot_masked=int(r.n_hot_masked),
                             sigma_s=r.hyper_sigma_s, v_drift=r.v_drift)
                 for r in bd.itertuples()},
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracks', default=None,
                    help='stage-3 campaign track parquet (default: resolved)')
    ap.add_argument('--pairs', default=None,
                    help='pair QA parquet (default: resolved)')
    ap.add_argument('--bundles', default=None,
                    help='bundle root holding mx17_A..D (default: resolved)')
    ap.add_argument('--out', default=None, help='output directory')
    ap.add_argument('--all-tracks', action='store_true',
                    help='skip the published pair selection and use the whole '
                         'stage-3 population (the generality check)')
    a = ap.parse_args()

    tracks = Path(a.tracks) if a.tracks else paths.spell(
        'out', 'stage3_campaign', 'tracks_campaign.parquet')
    pairs = Path(a.pairs) if a.pairs else paths.spell(
        'out', 'pair_qa', 'pairs_qa_campaign.parquet')
    broot = Path(a.bundles) if a.bundles else \
        paths.spell('x17', 'sept26_fullpass', 'bundles')
    out = Path(a.out) if a.out else paths.out('chi2_bimodality')
    out.mkdir(parents=True, exist_ok=True)

    print(f'[chi2] tracks  {tracks}')
    if not tracks.exists():
        print('  !! missing; nothing to do', file=sys.stderr)
        return 2
    d = load_tracks(tracks, selected=not a.all_tracks)
    print(f'[chi2] sample={d["sample"].iloc[0]}  {len(d):,} tracks  ' +
          '  '.join(f'{k}={v:,}' for k, v in d.arm.value_counts().items()))

    dofc = dof_check(d)
    print(f'[chi2] dof/n_strips median {dofc["ratio_median"]:.3f}  '
          f'exactly 20x on {dofc["frac_exactly_20"]:.1%}')

    prod = {
        'chi2_hist': chi2_hist(d),
        'chi2_axis': chi2_axis(d),
        'floor_peak': floor_peak(d),
        'modality': modality(d),
        'chi2_vs_len': chi2_vs_len(d),
        'reweight': reweight(d),
        'chi2_grid': chi2_grid(d),
        'chi2_by_run': chi2_by_run(d),
        'bundle_diff': bundle_diff(broot),
    }
    if pairs.exists():
        prod['pair_hist'] = pair_hist(load_pairs(pairs))
        prod['legs_crosscheck'] = legs_crosscheck(pairs)
    else:
        print(f'  .. no pair table at {pairs}; skipping pair_hist')

    for name, t in prod.items():
        f = out / f'{name}.csv'
        t.to_csv(f, index=False)
        print(f'  -> {f.name}  ({len(t)} rows)')

    summ = summarise(d, prod['chi2_vs_len'], prod['reweight'],
                     prod['bundle_diff'], dofc)
    (out / 'summary.json').write_text(json.dumps(summ, indent=2))
    print(f'  -> summary.json')

    print('\n  arm  median(all)  short  long   sep   at-floor')
    for arm, s in summ['per_arm'].items():
        print(f'   {arm}   {s["median_all"]:9.2f}  {s["median_short"] or 0:6.2f} '
              f'{s["median_long"] or 0:6.1f}  {s["separation"] or 0:5.1f}x '
              f'{s["frac_at_floor"]:7.1%}')
    if len(prod['floor_peak']):
        print('\n  the low peak, at 0.1 resolution on a LINEAR axis')
        print('  arm  peak at   FWHM          density  frac<1  frac 1-2')
        for r in prod['floor_peak'].itertuples():
            print(f'   {r.arm}   {r.peak_at:6.2f}   {r.fwhm_lo:.1f}-{r.fwhm_hi:<5.1f} '
                  f'{r.peak_dens:8.4f} {r.frac_below_1:7.1%} {r.frac_1_to_2:8.1%}')
    if len(prod['modality']):
        print('\n  is the log-space bimodality real?  (dip_depth well under 1 = yes)')
        print('  arm  low mode      dip        high mode   dip_depth')
        for r in prod['modality'].itertuples():
            print(f'   {r.arm}   {r.low_mode:.4f}@{r.low_mode_at:<5.2f} '
                  f'{r.dip:.4f}@{r.dip_at:<5.2f} '
                  f'{r.high_mode:.4f}@{r.high_mode_at:<6.2f} {r.dip_depth:8.2f}')
    if 'legs_crosscheck' in prod:
        print('\n  cross-check, straight from the published pair table’s legs:')
        print('  arm  n       median(all)  short  at-floor')
        for r in prod['legs_crosscheck'].itertuples():
            print(f'   {r.arm}  {r.n:7,}  {r.median_all:9.2f}  '
                  f'{r.median_short:6.2f}  {r.frac_at_floor:7.1%}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
