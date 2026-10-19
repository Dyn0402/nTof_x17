#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking_qa.py -- the per-track tracking distributions, run by run and tag by tag.

WHY THIS EXISTS.  Every tracking number quoted so far in this analysis is a
median over the whole campaign: "arm D fits at chi2/dof 39".  A median cannot
tell a chamber that is uniformly mediocre from one that is fine for most of the
campaign and catastrophic for two hours on a Tuesday, and the second is the
case worth finding, because it is fixable and the first is not.  The angle
scale `k` is measured to move 13-17 % between runs with nothing in the
configuration to explain it (`STATUS.md`, 2026-09-10), so the question this
module exists to answer is: **does anything in the reconstruction itself drift,
run to run or hour to hour, and is any of it an outlier rather than a trend?**

WHAT IT PRODUCES.  Three levels of the same summary, all from one pass over the
track table:

  per arm            the campaign reference distribution -- what "normal" is
  per (run, arm)     36 runs; the level `k` is measured at
  per (tag, arm)     ~3 150 file tags, each ~2-6 minutes of beam; the finest
                     time slice that exists, and the only one that can resolve
                     a transient

Each is quantiles, never a mean: `q_total` reaches 1e34 in this table and
`x_tan_err` reaches 1e4, so a mean is a report on the worst track in the group
and nothing else.  Alongside the quantiles go the RATES -- the fraction gated,
railed, non-sane, non-reliable -- and the PATHOLOGY fractions, which count the
tracks whose fit did not merely fit badly but returned a number that cannot be
right.  Those are the ones a median hides completely.

WHAT IT DOES NOT DO.  It does not cut, correct or reweight anything, and it
does not read `k`.  It is a description of what the reconstruction produced.
An outlier flagged here is a lead, not a verdict: the MAD z-score at the bottom
is a way of ordering 36 runs by how far they sit from their own arm's centre,
and 3.5 is a threshold for "look at this one", not for "throw this one away".

THE TAG IS A CLOCK.  A file tag is `<YYMMDD>_<HH>H<MM>_<idx>`, so every row in
the per-tag table carries a real timestamp and the campaign can be read as a
time series.  That is the whole reason the tag level is here.

    python -m sept26_prelim_analysis.tracking_qa
    python -m sept26_prelim_analysis.tracking_qa --src <out>/stage3_fullpass/tracks_campaign.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

SCHEMA = 'sept26_prelim/tracking_qa/1'
ARMS = ('A', 'B', 'C', 'D')

#: The quantiles every distribution is reduced to.  p05/p95 rather than the
#: extremes: the extremes here are fit failures and carry no information about
#: the bulk, which is what a drift would move.
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

# --------------------------------------------------------------------------- #
# What gets profiled
# --------------------------------------------------------------------------- #
#: name -> (column, axis label, log x?, (lo, hi) display clip or None)
#:
#: The clip is for the FIGURES only -- every number in the tables is computed on
#: the unclipped column.  Without it a single 1e34 charge sets the axis and the
#: distribution collapses to one bin.
VARS = {
    'chi2dof_x':   ('chi2dof_x',   r'$\chi^2$/dof, x view',        True,  (0.3, 3e3)),
    'chi2dof_y':   ('chi2dof_y',   r'$\chi^2$/dof, y view',        True,  (0.3, 3e3)),
    'n_strips_x':  ('x_n_strips',  'strips in fit, x view',        False, (0, 220)),
    'n_strips_y':  ('y_n_strips',  'strips in fit, y view',        False, (0, 220)),
    'tan_err_x':   ('x_tan_err',   r'$\sigma$(tan $\theta$), x',   True,  (1e-2, 1e0)),
    'tan_err_y':   ('y_tan_err',   r'$\sigma$(tan $\theta$), y',   True,  (1e-2, 1e0)),
    'p0_err_x':    ('x_p0_err',    r'$\sigma(p_0)$, x   [strips]', True,  (0.2, 20)),
    'p0_err_y':    ('y_p0_err',    r'$\sigma(p_0)$, y   [strips]', True,  (0.2, 20)),
    't0_err_x':    ('x_t0_err',    r'$\sigma(t_0)$, x   [ns]',     True,  (1, 1e4)),
    'tan_x':       ('tanx',        r'tan $\theta_x$  (raw)',       False, (-3, 3)),
    'tan_y':       ('tany',        r'tan $\theta_y$  (raw)',       False, (-3, 3)),
    't0_x':        ('x_t0',        r'$t_0$, x view   [ns]',        False, (-400, 900)),
    't0_y':        ('y_t0',        r'$t_0$, y view   [ns]',        False, (-400, 900)),
    'drift_len':   ('drift_len_mm', 'drift span   [mm]',           False, (0, 40)),
    'q_total':     ('q_total',     'cluster charge   [ADC]',       True,  (10, 1e6)),
    'q_per_len':   ('q_per_len',   'charge / path length',         True,  (1, 1e4)),
    'q_u50_x':     ('x_q_u50',     'charge at mid-drift, x',       False, (0, 1100)),
    'n_dropped_x': ('x_n_dropped', 'strips dropped, x view',       False, (0, 160)),
    'n_cand_x':    ('n_cand_x',    'x-view candidates',            False, (0, 6)),
    'n_cand_y':    ('n_cand_y',    'y-view candidates',            False, (0, 6)),
}

#: Boolean columns reported as a fraction of the group.  These are the tracking
#: gates themselves, so a run that drifts on one of them has changed what the
#: downstream sample IS, not just how well it was measured.
FLAGS = ('gated', 'x_quality_ok', 'y_quality_ok', 'x_plausible', 'y_plausible',
         'x_slope_reliable', 'y_slope_reliable', 'tan_sane', 'drift_railed',
         'x_isochronous', 'y_isochronous')

#: name -> (column, test) -- fractions of tracks whose fit returned a value that
#: cannot be physically right, as opposed to one that is merely poor.  A median
#: is blind to every one of these; they are the reason this module reports them
#: separately rather than trusting a quantile to notice.
PATHOLOGY = {
    'frac_chi2_gt_100':  ('chi2dof_x',  lambda s: s > 100),
    'frac_chi2_gt_1000': ('chi2dof_x',  lambda s: s > 1000),
    'frac_tanerr_gt_0p1': ('x_tan_err', lambda s: s > 0.1),
    # 12-bit DREAM on ~330 ADC of pedestal: a cluster cannot hold 1e6 ADC.
    # It does in this table, up to 1e34, which is a fit blow-up and not charge.
    'frac_q_gt_1e6':     ('q_total',    lambda s: s > 1e6),
    'frac_q_nonfinite':  ('q_total',    lambda s: ~np.isfinite(s)),
    'frac_strips_ge_200': ('x_n_strips', lambda s: s >= 200),
}

#: Columns actually read.  Named rather than globbed so a table that loses a
#: column fails here with its name instead of three functions deeper.
#: `q_total` and `chi2dof_x` appear in both VARS and PATHOLOGY, so this is
#: de-duplicated -- a repeated name in `read_parquet(columns=...)` returns a
#: frame with two identical columns and every later `df[col]` becomes a frame.
NEEDED = list(dict.fromkeys(
    ['run', 'subrun', 'tag', 'arm', 'event_class', 'condition',
     # gap_check only: the applied scale and the depth-grid edge
     'k_arm', 'depth_grid_edge_ns']
    + sorted({c for c, *_ in VARS.values()})
    + list(FLAGS)
    + sorted({c for c, _ in PATHOLOGY.values()})))

TAG_RE = re.compile(r'^(\d{2})(\d{2})(\d{2})_(\d{2})H(\d{2})')


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def default_src() -> str:
    return str(paths.out('stage3_campaign') / 'tracks_campaign.parquet')


def load(src: str | None = None, gated_only: bool = True) -> pd.DataFrame:
    """The track table, with the columns this module profiles.

    ``gated_only`` keeps the 3D-gated tracks, which is the population every
    downstream stage uses.  Pass False to profile the ungated superset -- the
    gate rate itself is reported either way, from the ungated count, so that
    number does not depend on this switch.
    """
    p = paths.require(src or default_src(), 'campaign track table')
    import pyarrow.parquet as pq
    cols = set(pq.ParquetFile(p).schema.names)
    missing = [c for c in NEEDED if c not in cols]
    if missing:
        raise KeyError(f'{p}: track table is missing {missing} -- this is not '
                       f'a campaign_tracks output, or the schema moved.')
    df = pd.read_parquet(p, columns=NEEDED)
    for c in ('run', 'subrun', 'tag', 'arm', 'event_class', 'condition'):
        df[c] = df[c].astype('category')
    if gated_only:
        df = df[df['gated'].to_numpy()].copy()
    return df


def tag_time(tag: pd.Series) -> pd.Series:
    """`260805_14H06_000` -> a timestamp. The tag is the campaign's clock."""
    s = tag.astype(str)
    m = s.str.extract(TAG_RE)
    ok = m.notna().all(axis=1)
    out = pd.Series(pd.NaT, index=tag.index, dtype='datetime64[ns]')
    if ok.any():
        v = m[ok].astype(int)
        out.loc[ok] = pd.to_datetime(dict(
            year=2000 + v[0], month=v[1], day=v[2], hour=v[3], minute=v[4]))
    return out


# --------------------------------------------------------------------------- #
# The summaries
# --------------------------------------------------------------------------- #
def _q_table(g: pd.core.groupby.DataFrameGroupBy, col: str,
             name: str) -> pd.DataFrame:
    """Quantiles of one column per group, as `<name>_p05 ... <name>_p95`."""
    q = g[col].quantile(list(QUANTILES)).unstack()
    q.columns = [f'{name}_p{int(round(x * 100)):02d}' for x in q.columns]
    return q


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per group: n, quantiles of every VAR, every FLAG rate, every
    PATHOLOGY fraction.

    Everything is computed on the raw column.  The display clips in ``VARS``
    are for figures and never touch these numbers -- a run whose charge blew up
    should move its own p95, not be quietly trimmed back into the pack.
    """
    g = df.groupby(keys, observed=True)
    out = [g.size().rename('n_tracks').to_frame()]
    for name, (col, *_rest) in VARS.items():
        out.append(_q_table(g, col, name))
    for f in FLAGS:
        out.append(g[f].mean().rename(f'frac_{f}').to_frame())
    for name, (col, test) in PATHOLOGY.items():
        s = df[col]
        out.append(df.assign(_t=test(s).astype(float))
                     .groupby(keys, observed=True)['_t'].mean()
                     .rename(name).to_frame())
    res = pd.concat(out, axis=1).reset_index()
    # A group of 40 tracks has a p95 but not a meaningful one. Carried, not
    # dropped: the figures thin it, the tables keep it, and a reader can see
    # that a wild-looking tag is wild because it holds 12 tracks.
    return res


def histograms(df: pd.DataFrame, var: str, by: str = 'run',
               bins: int = 60) -> pd.DataFrame:
    """Normalised histogram of one VAR per group, long form.

    Shared bin edges across every group -- that is the point of computing them
    here rather than per group, since the comparison is between the curves.
    Log-scaled variables get log-spaced edges.
    """
    col, _label, logx, clip = VARS[var]
    v = pd.to_numeric(df[col], errors='coerce')
    lo, hi = clip if clip else (np.nanpercentile(v, 0.5),
                                np.nanpercentile(v, 99.5))
    if logx:
        lo = max(lo, 1e-12)
        edges = np.geomspace(lo, hi, bins + 1)
    else:
        edges = np.linspace(lo, hi, bins + 1)
    centre = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for key, sub in df.groupby(by, observed=True):
        x = pd.to_numeric(sub[col], errors='coerce').to_numpy()
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        n_under = int((x < edges[0]).sum())
        n_over = int((x > edges[-1]).sum())
        h, _ = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)
        rows.append(pd.DataFrame({
            by: key, 'centre': centre, 'count': h,
            'density': h / max(h.sum(), 1),
            'n': x.size, 'frac_under': n_under / x.size,
            'frac_over': n_over / x.size}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def outliers(per_run: pd.DataFrame, per_arm: pd.DataFrame,
             z_cut: float = 3.5, effect_cut: float = 0.15) -> pd.DataFrame:
    """Which (run, arm) sits far from its own arm's centre, and on what.

    A robust z on the MEDIAN of each variable -- ``(x - median) / (1.4826 *
    MAD)`` within each arm, so an arm that is globally bad (D) does not flag
    all 36 of its runs, and a run that is bad only on D does not drag A with
    it.  MAD, not sigma, because with 36 runs two bad ones would set a sigma
    wide enough to cover themselves.

    **A z on its own is not enough, and the first version of this was useless
    because of it.**  Several statistics are near-identical across runs -- the
    ``t0`` medians agree to a fraction of a nanosecond on most arms -- so their
    across-run MAD collapses toward zero and every run that differs at all
    comes back at ``z = 3e4``.  Two guards:

      the MAD is FLOORED at 2 % of the arm's own track-level IQR (0.002 for a
      fraction), so ``z`` can never be manufactured by a degenerate spread;

      an EFFECT SIZE is required alongside it -- ``|value - arm median|``
      expressed in units of that same track-level IQR.  A run must be both
      unusual *and* moved by a visible fraction of the distribution's real
      width to be flagged.

    Both are reported, so a reader can see a run that is statistically extreme
    but physically tiny, which is a different thing from a run that is broken.

    Returned long: one row per flagged (arm, run, variable), sorted by effect.
    """
    stat_cols = [f'{n}_p50' for n in VARS]
    # `tan_sane` is not a tracking result: it is false for every track of a run
    # whose arm never certified a `k`, so scanning it just re-lists which runs
    # got calibrated. It stays in the tables and out of the outlier hunt.
    frac_cols = [f'frac_{f}' for f in FLAGS if f != 'tan_sane'] + list(PATHOLOGY)
    cols = [c for c in stat_cols + frac_cols if c in per_run.columns]
    ref = per_arm.set_index('arm')
    # Small-integer counts: their median moves in steps of 1 and their IQR is
    # often 1, so a one-count step scores effect = 1.0 and half the campaign
    # "flags". A one-candidate shift is real but carries almost no information,
    # so require more than one unit before it counts as an outlier.
    integral = {f'{n}_p50' for n, (c, *_r) in VARS.items()
                if n.startswith(('n_cand', 'n_strips', 'n_dropped'))}

    def _width(arm: str, col: str) -> float:
        """The arm's own track-level width for this statistic: the IQR of the
        underlying distribution, not the run-to-run scatter of its median."""
        if col.endswith('_p50'):
            base = col[:-4]
            lo, hi = f'{base}_p25', f'{base}_p75'
            if lo in ref.columns and hi in ref.columns:
                w = float(ref.loc[arm, hi]) - float(ref.loc[arm, lo])
                if np.isfinite(w) and w > 0:
                    return w
            return float('nan')
        return 1.0                       # a fraction is already on [0, 1]

    rows = []
    for arm, sub in per_run.groupby('arm', observed=True):
        for c in cols:
            x = pd.to_numeric(sub[c], errors='coerce')
            med = x.median()
            if not np.isfinite(med):
                continue
            # A statistic that takes only two or three values across 36 runs is
            # quantized at the run level -- `drift_len_p50` lands on the depth
            # grid, `n_cand_*_p50` on an integer -- and every run on the minority
            # value then "flags" against a MAD of zero. That is the grid, not a
            # chamber, so such a variable is not scanned at all.
            if x.dropna().nunique() < 4:
                continue
            w = _width(arm, c)
            floor = (0.02 * w) if np.isfinite(w) else 0.0
            floor = max(floor, 0.002) if c in frac_cols else floor
            scale = max(1.4826 * (x - med).abs().median(), floor)
            if not np.isfinite(scale) or scale <= 0:
                continue
            z = (x - med) / scale
            eff = (x - med).abs() / w if np.isfinite(w) and w > 0 else np.nan
            for i in np.flatnonzero((z.abs() >= z_cut).to_numpy()):
                e = float(eff.iloc[i]) if eff is not np.nan else float('nan')
                if np.isfinite(e) and e < effect_cut:
                    continue
                if c in integral and abs(float(x.iloc[i] - med)) <= 1.0:
                    continue
                rows.append(dict(arm=arm, run=sub['run'].iloc[i], variable=c,
                                 value=float(x.iloc[i]), arm_median=float(med),
                                 shift=float(x.iloc[i] - med),
                                 effect=e, z=float(z.iloc[i]),
                                 n_tracks=int(sub['n_tracks'].iloc[i])))
    cols_out = ['arm', 'run', 'variable', 'value', 'arm_median', 'shift',
                'effect', 'z', 'n_tracks']
    if not rows:
        return pd.DataFrame(columns=cols_out)
    out = pd.DataFrame(rows)
    out['_r'] = out['effect'].fillna(out['z'].abs() / 100.0)
    return (out.sort_values('_r', ascending=False)
               .drop(columns='_r')[cols_out].reset_index(drop=True))


#: The 27 July access. Everything before it is the other detector condition
#: (chamber A's connector 8 was dead), so a correlation computed across it is
#: measuring the step, not a drift.
ACCESS = pd.Timestamp('2026-07-27 12:00')


def drift(per_tag: pd.DataFrame, min_tracks: int = 50) -> pd.DataFrame:
    """Is each variable trending with time, per arm? Spearman rho on the tag
    medians against the tag timestamp.

    Spearman rather than a fit: a step (the 27 July access, a gas bottle) and a
    ramp both matter and neither is linear.  ``n_tags`` is carried because rho
    over 12 tags means nothing and over 900 means a lot.

    **Two rhos, and the difference between them is the result.**  ``rho`` runs
    over every tag; ``rho_post`` over the tags after the 27 July access only.
    The first version of this reported one number and it was misleading: the
    access is a genuine step in several variables -- C's cluster size falls by
    a third across it, A's t0 error by more than half -- so a campaign-wide
    Spearman scores that step as a strong monotone trend.  A variable with a
    large ``rho`` and a small ``rho_post`` stepped once and then held; one with
    both is actually drifting.
    """
    from scipy.stats import spearmanr
    cols = [f'{n}_p50' for n in VARS] + [f'frac_{f}' for f in FLAGS]
    cols = [c for c in cols if c in per_tag.columns]
    d = per_tag[per_tag['n_tracks'] >= min_tracks].copy()
    d['t'] = pd.to_datetime(d['t'])

    def _rho(t, x):
        m = np.isfinite(x) & np.isfinite(t)
        if m.sum() < 20 or np.unique(x[m]).size < 3:
            return np.nan, np.nan, 0
        r, p = spearmanr(t[m], x[m])
        return float(r), float(p), int(m.sum())

    rows = []
    for arm, sub in d.groupby('arm', observed=True):
        t = sub['t'].astype('int64').to_numpy(dtype=float)
        post = (sub['t'] >= ACCESS).to_numpy()
        for c in cols:
            x = pd.to_numeric(sub[c], errors='coerce').to_numpy()
            rho, p, n = _rho(t, x)
            if not n:
                continue
            rho_p, p_p, n_p = _rho(t[post], x[post])
            pre_x = x[~post][np.isfinite(x[~post])]
            post_x = x[post][np.isfinite(x[post])]
            rows.append(dict(
                arm=arm, variable=c, rho=rho, p=p, n_tags=n,
                rho_post=rho_p, p_post=p_p, n_tags_post=n_p,
                pre_access=float(np.median(pre_x)) if pre_x.size else np.nan,
                post_access=float(np.median(post_x)) if post_x.size else np.nan,
                first_decile=float(np.median(post_x[:max(n_p // 10, 1)]))
                if post_x.size else np.nan,
                last_decile=float(np.median(post_x[-max(n_p // 10, 1):]))
                if post_x.size else np.nan))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Ordered by the post-access trend: that is the one that is not already
    # explained by a known hardware intervention.
    out['_r'] = out['rho_post'].abs().fillna(0)
    return out.sort_values('_r', ascending=False).drop(columns='_r')


def gap_check(df: pd.DataFrame) -> pd.DataFrame:
    """Does the reconstructed drift span fit inside the chamber?

    A geometric bound on the angle scale that owes nothing to the pointing
    estimators, and therefore checks them.  ``drift_len_mm = t_end * v`` with
    ``v = 42.6 / k``, and the depth grid runs to ``18 * 60 = 1080 ns``, so the
    deepest span the reconstruction can produce is ``1080 * 42.6 / k`` microns.
    That has to fit in the drift gap, which gives

        k  >=  1080 * 0.0426 / gap_mm

    with no reference to the target, the scintillators or the pointing sample.
    An arm that reconstructs a large fraction of its tracks deeper than its own
    gap is telling you that its ``k`` is too small, or that its depth-grid
    origin sits outside the gas -- both calibration faults, and neither
    visible in a chi-squared.
    """
    from ntof_tracking.wft_beam import BEAM_DETS
    rows = []
    for arm, sub in df.groupby('arm', observed=True):
        if arm not in BEAM_DETS:
            continue
        gap = float(BEAM_DETS[arm]['gap_mm'])
        L = pd.to_numeric(sub['drift_len_mm'], errors='coerce')
        k = pd.to_numeric(sub['k_arm'], errors='coerce').median() \
            if 'k_arm' in sub.columns else np.nan
        edge = float(sub['depth_grid_edge_ns'].median()) \
            if 'depth_grid_edge_ns' in sub.columns else np.nan
        v = 42.6 / k if np.isfinite(k) and k else np.nan
        rows.append(dict(
            arm=arm, gap_mm=gap, k_applied=float(k) if np.isfinite(k) else np.nan,
            v_um_ns=float(v) if np.isfinite(v) else np.nan,
            grid_edge_ns=edge,
            max_span_mm=float(edge * v / 1000.0)
            if np.isfinite(edge) and np.isfinite(v) else np.nan,
            span_p50=float(L.median()),
            span_p50_unrailed=float(L[~sub['drift_railed'].to_numpy()].median()),
            frac_over_gap=float((L > gap).mean()),
            frac_railed=float(sub['drift_railed'].mean()),
            k_min_for_gap=float(edge * 0.0426 / gap)
            if np.isfinite(edge) else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def build(src: str | None, out_dir, hist_vars: list[str], z_cut: float,
          effect_cut: float = 0.15) -> dict:
    out_dir = paths.out('tracking_qa') if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f'[qa] reading {src or default_src()}')
    # Read UNGATED. `frac_gated` is the one rate that cannot be computed on the
    # gated sample -- it would be 1.000 everywhere, which is what the first
    # version of this reported -- so the gate rate is taken here, on the full
    # table, and merged back onto each summary afterwards.
    df_all = load(src, gated_only=False)
    df = df_all[df_all['gated'].to_numpy()].copy()
    print(f'[qa] {len(df):,} gated of {len(df_all):,} tracks, '
          f'{df["run"].nunique()} runs, '
          f'{df.groupby(["run", "subrun"], observed=True).ngroups} sub-runs, '
          f'{df["tag"].nunique()} tags')

    def _gate(keys, tab):
        g = (df_all.groupby(keys, observed=True)['gated'].mean()
             .rename('frac_gated'))
        return tab.drop(columns=['frac_gated'], errors='ignore') \
                  .merge(g, left_on=keys, right_index=True, how='left')

    per_arm = _gate(['arm'], summarise(df, ['arm']))
    per_run = _gate(['arm', 'run'], summarise(df, ['arm', 'run']))
    per_tag = _gate(['arm', 'run', 'subrun', 'tag'],
                    summarise(df, ['arm', 'run', 'subrun', 'tag']))
    per_tag['t'] = tag_time(per_tag['tag'])
    per_tag = per_tag.sort_values(['arm', 't', 'tag'])

    # run start time, so the per-run table can also be read as a series.
    # NOT the sort key: run_145's rows in the August blind pass carry the
    # literal tag `prelim` (they were built from an already-merged table), so
    # its timestamp is unknown and it would sort to the end of a time axis.
    # Run NUMBER is monotonic in time across this campaign, so it orders the
    # table and `t_start` stays as information.
    tmin = per_tag.groupby('run', observed=True)['t'].min().rename('t_start')
    per_run = per_run.merge(tmin, left_on='run', right_index=True, how='left')
    per_run['run_no'] = per_run['run'].astype(str).str.split('_').str[-1].astype(int)
    per_run = per_run.sort_values(['arm', 'run_no'])
    n_untagged = int((per_tag['t'].isna()).sum())
    if n_untagged:
        print(f'  [qa] {n_untagged} (run, tag) row(s) have no parseable '
              f'timestamp and are excluded from the drift test')

    o = outliers(per_run, per_arm, z_cut=z_cut, effect_cut=effect_cut)
    dr = drift(per_tag)
    gap = gap_check(df)

    for name, tab in (('per_arm', per_arm), ('per_run', per_run),
                      ('per_tag', per_tag), ('outliers', o), ('drift', dr),
                      ('gap_check', gap)):
        p = os.path.join(out_dir, f'{name}.csv')
        tab.to_csv(p, index=False)
        print(f'  -> {p}  ({len(tab)} rows)')

    for v in hist_vars:
        h = histograms(df, v, by='run')
        h.insert(0, 'arm', 'ALL')
        parts = [h]
        for arm, sub in df.groupby('arm', observed=True):
            ha = histograms(sub, v, by='run')
            if not ha.empty:
                ha.insert(0, 'arm', arm)
                parts.append(ha)
        p = os.path.join(out_dir, f'hist_{v}.csv')
        pd.concat(parts, ignore_index=True).to_csv(p, index=False)
        print(f'  -> {p}')

    meta = dict(
        schema=SCHEMA, generated=dt.datetime.now().isoformat(timespec='seconds'),
        src=str(src or default_src()), n_tracks=int(len(df)),
        n_tracks_ungated=int(len(df_all)),
        n_runs=int(df['run'].nunique()),
        # distinct (run, sub-run) PAIRS -- `subrun` alone is `stat090_0000`
        # style and repeats in every run, which read back as 29 for 293.
        n_subruns=int(df.groupby(['run', 'subrun'], observed=True).ngroups),
        n_tags=int(df['tag'].nunique()),
        arms={a: int(n) for a, n in df['arm'].value_counts().items()},
        quantiles=list(QUANTILES), z_cut=z_cut, effect_cut=effect_cut,
        hist_vars=list(hist_vars),
        n_outliers=int(len(o)),
        note=('gated tracks only; quantiles on raw columns, VARS clips are for '
              'figures only'))
    with open(os.path.join(out_dir, 'tracking_qa.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'  -> {out_dir}/tracking_qa.meta.json')
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=None,
                    help=f'track table; default {default_src()}')
    ap.add_argument('--out', default=None, help='default <out>/tracking_qa')
    ap.add_argument('--z-cut', type=float, default=3.5)
    ap.add_argument('--effect-cut', type=float, default=0.15,
                    help='minimum |shift| in units of the arm track-level IQR')
    ap.add_argument('--hist-vars', default='chi2dof_x,chi2dof_y,n_strips_x,'
                                           'n_strips_y,tan_err_x,q_total,'
                                           't0_x,drift_len,n_dropped_x',
                    help='comma-separated VARS keys to histogram per run')
    a = ap.parse_args()
    bad = [v for v in a.hist_vars.split(',') if v and v not in VARS]
    if bad:
        sys.exit(f'FATAL: unknown --hist-vars {bad}; known: {sorted(VARS)}')
    m = build(a.src, a.out, [v for v in a.hist_vars.split(',') if v],
              a.z_cut, a.effect_cut)
    print(f'\n{m["n_tracks"]:,} tracks profiled, {m["n_outliers"]} '
          f'(run, arm, variable) outlier(s) at |z| >= {a.z_cut}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
