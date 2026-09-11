#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
campaign_efficiency.py -- the scintillator-tagged efficiency, run by run.

`efficiency.py` measures P(gated track | wall AND plastic in the same arm) for
ONE run.  Every acceptance built so far has therefore been run_145's, borrowed
campaign-wide, and `campaign_angle.py` stamps that borrowing as its leading
systematic (`acceptance_source = run_145 (BORROWED)`).  This module measures the
same thing for all 36 runs of the condor full pass, so the acceptance can be
per run and the borrowing can be checked rather than assumed.

WHY THE SINGLE-TRACK EVENTS CARRY THE MAP.  The map is the efficiency binned in
the chamber's own in-plane coordinate `u`, and `u` comes from the fitted track.
On an event the chamber turned into two or three tracks there is no single `u`
to bin -- `x_p0` is one track's and the tag does not say which.  So the map's
denominator is the tagged, seeded events the chamber turned into AT MOST ONE
track, and its numerator is the ones it turned into EXACTLY one.  That keeps
position and efficiency consistent with each other.  Multi-track events are
3.0-3.4 % of the tagged sample, and the fraction is reported per run and per arm
rather than assumed small: it is the cost of the choice.

  * `eff_single_given_seed` -- the single-track map above.  THIS is what the
    acceptance toy's shape uses.
  * `eff_track_given_seed` -- `efficiency.py`'s definition, P(any gated track |
    tagged AND seeded) over all seeded events.  Kept under its original name so
    the old acceptance path reads this product unchanged, and so the two can be
    differenced.

THREE THINGS THAT CHANGE FROM THE SINGLE-RUN MODULE, each because the campaign
forces it:

  1. **The tag is read from the exported parquet, not the ROOT slim.**
     `scintillators.read_slim` opens `<runs>/<run>/<sub>/ntof_hits/*.root`, which
     on this machine exists for run_145 only.  `slim_export.read_export` reads
     the 293 exported sub-runs.  Validated to give the IDENTICAL tag set on
     run_145 arm A (43 105 events either way), so this is a source change and
     not a definition change.
  2. **`p0` is measured per run.** The accidental response probability that
     makes the efficiency meaningful is a rate, and rates moved across the
     campaign.  Carrying run_145's `p0` would be the same borrowing this module
     exists to remove.
  3. **Nothing is written where the single-run module writes.** Products land in
     `<out>/efficiency_campaign/`; `<out>/efficiency/efficiency_*_run_145.csv`
     is the published single-run measurement on `<out>/fullpass` (the ALLOWLIST
     pass, despite the name) and stays as it is for the comparison.

WHAT THIS IS STILL NOT.  Every caveat in `efficiency.py`'s docstring survives
here unchanged -- the denominator includes particles that missed the active
area, so these are lower bounds; B's number is a HIT efficiency and must never
be used as a track efficiency; and the tag says a particle reached the wall and
the plastic, not that it crossed the gas.  Measuring it 36 times does not make
any of that go away, it only stops one run's value standing in for all of them.

    python -m sept26_prelim_analysis.campaign_efficiency --jobs 8
    python -m sept26_prelim_analysis.campaign_efficiency --runs run_86,run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.campaign_imaging import (  # noqa: E402
    PRE_ACCESS_RUNS, condition, in_block, run_number, subruns_of)
from sept26_prelim_analysis.efficiency import (  # noqa: E402
    DT_WINDOW, HITS_ONLY, U_EDGES, WAL_CODE, corrected)

SCHEMA = 'sept26_prelim/campaign_efficiency/1'
ARMS = ('A', 'B', 'C', 'D')
#: Minimum tagged-and-seeded events in a u bin before its efficiency is
#: reported.  Below this the binomial error swamps the variation the map exists
#: to carry, and a noisy shape multiplied into an acceptance is worse than no
#: shape at all.
MIN_BIN = 40


# --------------------------------------------------------------------------- #
# the tag, from the exported slim
# --------------------------------------------------------------------------- #
def tagged_events(run: str, subruns, arm: str, slim: pd.DataFrame) -> tuple:
    """(tag set, {(subrun, event) -> the single wall group, or -1}).

    Same definition as `efficiency.tagged_events`, evaluated on the exported
    parquet instead of the ROOT slim.  The second return carries which wall
    group fired, because that is the abscissa of the incidence measurement
    below: the group says roughly where on the wall the particle arrived and
    therefore at what angle it crossed the chamber.  Events with more than one
    group lit get -1 -- their incidence is ambiguous and they are dropped there
    and kept everywhere else.
    """
    it = slim[(slim.is_control == 0) & (slim.dt_ns >= DT_WINDOW[0])
              & (slim.dt_ns <= DT_WINDOW[1])]
    wal = it[it.det == WAL_CODE[arm]]
    pss = it[it.det == WAL_CODE[arm] + 4]
    kw = set(map(tuple, wal[['subrun', 'eventId']].drop_duplicates().values))
    kp = set(map(tuple, pss[['subrun', 'eventId']].drop_duplicates().values))
    tag = kw & kp
    g = wal.assign(grp=(wal.detn - 1) // 2) \
           .groupby(['subrun', 'eventId']).grp.agg(['min', 'max'])
    grp = {k: (int(lo) if lo == hi else -1)
           for k, lo, hi in zip(g.index, g['min'], g['max']) if k in tag}
    return tag, grp


def chamber_response(run: str, subruns, arm: str, reco: Path) -> pd.DataFrame:
    """What the chamber did per event, from the full pass's merged tables.

    A sub-run whose merged table is missing is raised on rather than skipped:
    a silently absent arm would enter the efficiency as a chamber that never
    responded, which is the one failure mode that looks like a real result.
    """
    out = []
    for sub in subruns:
        p = reco / run / sub / f'mx17_{arm}' / 'events_prelim.parquet'
        if not p.exists():
            raise FileNotFoundError(f'no merged full pass for {arm}/{sub}: {p}')
        out.append(pd.read_parquet(
            p, columns=['event_id', 'x_ok', 'y_ok', 'n_tracks', 'x_p0'])
            .assign(subrun=sub))
    d = pd.concat(out, ignore_index=True)
    # A row exists only for a SEEDED event, so presence is the hit-level
    # response and needs no column of its own.
    return d


def n_triggers(run: str, subruns) -> int:
    """Total triggers in the sub-runs, from the stage-1 census."""
    n = 0
    for sub in subruns:
        c = pd.read_csv(paths.require(
            paths.out('stage1') / f'census_{run}_{sub}.csv',
            f'stage-1 census for {run}/{sub}'))
        n += int(c.loc[c.cls == '(total)', 'n'].iloc[0])
    return n


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def measure_run(run: str, subruns, reco: Path, slim_dir: Path | None = None
                ) -> tuple:
    """(per-arm frame, u-binned map frame, incidence frame) for one run."""
    from sept26_prelim_analysis import slim_export as SE
    from sept26_prelim_analysis import chamber_b as CB
    from ntof_tracking import run145_target_imaging as TI

    slim = SE.read_export(run, subruns, slim_dir)
    n_trig = n_triggers(run, subruns)
    gu = CB.wall_group_u(run)

    rows, maps, inc = [], [], []
    for arm in ARMS:
        tag, grp = tagged_events(run, subruns, arm, slim)
        resp = chamber_response(run, subruns, arm, reco)
        key = list(map(tuple, resp[['subrun', 'event_id']].values))
        is_tag = np.fromiter((k in tag for k in key), bool, len(key))

        seeded = is_tag                      # a row present IS a seed
        trk = resp.n_tracks.to_numpy() > 0
        n_tag = len(tag)
        n_untag = n_trig - n_tag

        # p0: how often the chamber responds when this arm tagged NOTHING.
        p0_hit = float((~is_tag).sum()) / max(n_untag, 1)
        p0_trk = float((trk & ~is_tag).sum()) / max(n_untag, 1)

        # Tagged events with no row were not seeded at all, so the tagged
        # numerators are counted over the rows that ARE present and the
        # denominator is the full tag count.
        p_hit = seeded.sum() / max(n_tag, 1)
        p_trk = float((trk & is_tag).sum()) / max(n_tag, 1)
        multi = resp.n_tracks.to_numpy() > 1
        rows.append(dict(
            run=run, arm=arm, n_subruns=len(subruns), n_triggers=n_trig,
            n_tagged=n_tag, n_untagged=int(n_untag),
            n_tag_seeded=int(seeded.sum()),
            n_tag_tracked=int((trk & is_tag).sum()),
            eff_hit_raw=float(p_hit), eff_track_raw=float(p_trk),
            p0_hit=p0_hit, p0_track=p0_trk,
            eff_hit_corr=corrected(p_hit, p0_hit),
            eff_track_corr=corrected(p_trk, p0_trk),
            err_hit=float(np.sqrt(max(p_hit * (1 - p_hit), 0) / max(n_tag, 1))),
            err_track=float(np.sqrt(max(p_trk * (1 - p_trk), 0)
                                    / max(n_tag, 1))),
            frac_multitrack=float((multi & is_tag).sum()
                                  / max((trk & is_tag).sum(), 1)),
            condition=condition(run), k_block=in_block(run)))

        # ---- the u-binned map -------------------------------------------- #
        xp = resp.x_p0.to_numpy()
        has = np.isfinite(xp) & is_tag
        u = np.full(len(xp), np.nan)
        u[has] = TI.local_x(xp[has]) - TI.PINWHEEL[arm]
        one = resp.n_tracks.to_numpy() == 1
        for lo, hi in zip(U_EDGES[:-1], U_EDGES[1:]):
            band = has & (u >= lo) & (u < hi)
            # single-track basis: at most one track in, exactly one track out
            sden = band & ~multi
            if band.sum() < MIN_BIN:
                continue
            e_all = float(trk[band].mean())
            e_one = (float(one[sden].mean()) if sden.sum() >= MIN_BIN
                     else float('nan'))
            maps.append(dict(
                run=run, arm=arm, u_lo=float(lo), u_hi=float(hi),
                u_mid=float(0.5 * (lo + hi)), n=int(band.sum()),
                n_single_den=int(sden.sum()),
                eff_track_given_seed=e_all,
                eff_single_given_seed=e_one,
                err=float(np.sqrt(max(e_all * (1 - e_all), 0) / band.sum())),
                err_single=(float(np.sqrt(max(e_one * (1 - e_one), 0)
                                          / sden.sum()))
                            if sden.sum() >= MIN_BIN else float('nan')),
                condition=condition(run)))

        # ---- efficiency versus INCIDENCE ---------------------------------- #
        # The tag carries the angle: a particle that reached wall group k came
        # in at tan ~ (u_k - foot) / (d_perp + wall_depth), and one of the four
        # groups lands inside the head-on band while the others do not.  This
        # is `normal_incidence.efficiency_vs_incidence`, per run and off the
        # exported slim.  The positional confound is real and is why the
        # head-on group's two NEIGHBOURS are what it gets compared against
        # downstream, not the arm's mean.
        gk = np.fromiter((grp.get(k, -2) for k in key), int, len(key))
        foot = TI.PINWHEEL[arm]
        for g in sorted(gu.get(arm, {})):
            band = gk == g
            if band.sum() < 200:
                continue
            tan = (gu[arm][g] - foot) / (CB.D_PERP_MM + CB.WALL_DEPTH_MM)
            # The denominator is the tag count for this group, which is the
            # rows present PLUS the tagged events the chamber never seeded --
            # and an unseeded event has no row, so it has to be added back.
            n_g = sum(1 for v in grp.values() if v == g)
            inc.append(dict(
                run=run, arm=arm, grp=int(g), u_wall=float(gu[arm][g]),
                tan_expected=float(tan), head_on=bool(abs(tan) < 0.08),
                n_tagged=int(n_g), n_seeded=int(band.sum()),
                p_seeded=float(band.sum() / max(n_g, 1)),
                p_tracked=float((band & trk).sum() / max(n_g, 1)),
                condition=condition(run), k_block=in_block(run)))
    I = pd.DataFrame(inc)
    if not I.empty:
        # each arm against its own best group: the comparison is within a
        # chamber, so the absolute efficiency differences cannot confuse it
        I['p_tracked_rel'] = I.groupby('arm').p_tracked.transform(
            lambda x: x / x.max())
    return pd.DataFrame(rows), pd.DataFrame(maps), I


def headline(E: pd.DataFrame) -> pd.DataFrame:
    """The number each chamber should be quoted at, per run.

    B on HITS and everything else on TRACKS, for the reason `efficiency.py`
    gives: B has no uniform drift field, so a track in B is not a track.  The
    basis travels in the table so a consumer cannot pick up the wrong column.
    """
    out = []
    for r in E.itertuples():
        hits = r.arm in HITS_ONLY
        out.append(dict(
            run=r.run, arm=r.arm, basis='hits' if hits else 'tracks',
            efficiency=r.eff_hit_corr if hits else r.eff_track_corr,
            err=r.err_hit if hits else r.err_track,
            p0=r.p0_hit if hits else r.p0_track,
            n_tagged=int(r.n_tagged), condition=r.condition,
            k_block=r.k_block))
    return pd.DataFrame(out)


def stability(H: pd.DataFrame) -> pd.DataFrame:
    """Per arm, how much the headline efficiency moves across the campaign.

    The question the whole module is for: is one run's efficiency a fair stand-in
    for the campaign's?  `p10_p90_frac` is the answer -- the p10-to-p90 spread as
    a fraction of the median, the same statistic STATUS.md quotes for `k`, so the
    two can be read against each other.
    """
    rows = []
    for arm, g in H[~H.run.isin(PRE_ACCESS_RUNS)].groupby('arm'):
        e = g.efficiency.dropna().to_numpy()
        if len(e) < 3:
            continue
        p10, p90 = np.percentile(e, [10, 90])
        med = float(np.median(e))
        blk = g[g.k_block].efficiency.dropna().to_numpy()
        out = g[~g.k_block].efficiency.dropna().to_numpy()
        rows.append(dict(
            arm=arm, n_runs=len(e), min=float(e.min()), median=med,
            max=float(e.max()), sd=float(e.std(ddof=1)),
            p10_p90_frac=float((p90 - p10) / med) if med else np.nan,
            in_block=float(blk.mean()) if len(blk) else np.nan,
            outside_block=float(out.mean()) if len(out) else np.nan,
            block_shift_frac=(float(blk.mean() / out.mean() - 1)
                              if len(blk) and len(out) and out.mean()
                              else np.nan)))
    return pd.DataFrame(rows)


def map_shape_spread(M: pd.DataFrame) -> pd.DataFrame:
    """How much the MAP's shape moves run to run, per arm and u bin.

    The acceptance uses the map as a shape times the headline, so a stable
    headline with a moving shape would still make a borrowed acceptance wrong.
    Normalising each run's map to its own mean before comparing is what
    separates the two.
    """
    d = M[np.isfinite(M.eff_single_given_seed)].copy()
    d['shape'] = d.groupby(['run', 'arm']).eff_single_given_seed \
        .transform(lambda s: s / s.mean())
    rows = []
    for (arm, u), g in d.groupby(['arm', 'u_mid']):
        rows.append(dict(arm=arm, u_mid=u, n_runs=int(g.run.nunique()),
                         shape_mean=float(g['shape'].mean()),
                         shape_sd=float(g['shape'].std(ddof=1))
                         if len(g) > 1 else np.nan))
    return pd.DataFrame(rows).sort_values(['arm', 'u_mid'], ignore_index=True)


# --------------------------------------------------------------------------- #
# the campaign
# --------------------------------------------------------------------------- #
def head_on_dip(I: pd.DataFrame) -> pd.DataFrame:
    """The head-on group against its two positional NEIGHBOURS, per run.

    This is what controls the confound `normal_incidence` names: the four wall
    groups sample four incidences but also four places on the chamber, so a
    bare ranking cannot separate angle from surface.  The head-on group is an
    interior one with a neighbour on each side -- a surface effect interpolates
    between them, an angle effect sits below both.
    """
    rows = []
    for (run, arm), g in I.groupby(['run', 'arm']):
        h = g[g.head_on]
        if h.empty:
            continue
        k = int(h.grp.iloc[0])
        nb = g[g.grp.isin((k - 1, k + 1))]
        if len(nb) < 2 or nb.p_tracked.mean() <= 0:
            continue
        rows.append(dict(
            run=run, arm=arm, grp=k, n_head_on=int(h.n_tagged.iloc[0]),
            seed_ratio=float(h.p_seeded.iloc[0] / nb.p_seeded.mean()),
            track_head_on=float(h.p_tracked.iloc[0]),
            track_neighbours=float(nb.p_tracked.mean()),
            track_ratio=float(h.p_tracked.iloc[0] / nb.p_tracked.mean()),
            below_both=bool(h.p_tracked.iloc[0] < nb.p_tracked.min()),
            condition=h.condition.iloc[0], k_block=bool(h.k_block.iloc[0])))
    return pd.DataFrame(rows)


def _one(run: str, reco: str, slim_dir: str | None) -> tuple:
    """Worker: (run, per-arm, map, incidence, error text)."""
    try:
        reco = Path(reco)
        subs, dropped = subruns_of(reco, run)
        if not subs:
            return run, None, None, None, 'no usable sub-runs under the reco tree'
        for s, why in dropped.items():
            print(f'  {run}/{s}: dropped -- {why}', flush=True)
        E, M, I = measure_run(run, subs, reco,
                              Path(slim_dir) if slim_dir else None)
        return run, E, M, I, ''
    except Exception:
        return run, None, None, None, traceback.format_exc(limit=3)


def campaign(reco: Path, runs, jobs: int, slim_dir: Path | None = None
             ) -> tuple:
    E, M, I, failed = [], [], [], {}
    sd = str(slim_dir) if slim_dir else None
    if jobs <= 1:
        results = [_one(r, str(reco), sd) for r in runs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut = {ex.submit(_one, r, str(reco), sd): r for r in runs}
            for f in as_completed(fut):
                results.append(f.result())
    for run, e, m, i, err in sorted(results, key=lambda r: run_number(r[0])):
        if err:
            failed[run] = err.strip().splitlines()[-1]
            print(f'  {run}: FAILED -- {failed[run]}', flush=True)
            continue
        E.append(e)
        M.append(m)
        if i is not None and not i.empty:
            I.append(i)
        print(f'  {run}: {len(e)} arms, {len(m)} map bins, '
              f'{0 if i is None else len(i)} incidence points', flush=True)
    return (pd.concat(E, ignore_index=True) if E else pd.DataFrame(),
            pd.concat(M, ignore_index=True) if M else pd.DataFrame(),
            pd.concat(I, ignore_index=True) if I else pd.DataFrame(),
            failed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--reco', default=str(paths.out('reco_fullpass')),
                    help='the CONDOR FULL PASS reco tree; <out>/fullpass is '
                         'the allowlist pass despite its name')
    ap.add_argument('--runs', default='', help='comma-separated; default all')
    ap.add_argument('--slim', default=None, help='exported slim dir')
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()

    reco = Path(paths.require(a.reco, 'the full-pass reco tree'))
    runs = ([r for r in a.runs.split(',') if r] or
            sorted((p.name for p in reco.iterdir()
                    if p.is_dir() and p.name.startswith('run_')),
                   key=run_number))
    print(f'{len(runs)} run(s) from {reco}\n')

    E, M, I, failed = campaign(reco, runs, a.jobs,
                               Path(a.slim) if a.slim else None)
    if E.empty:
        print('\nnothing measured')
        return 1
    H = headline(E)
    S = stability(H)
    P = map_shape_spread(M)
    Dip = head_on_dip(I) if not I.empty else pd.DataFrame()

    od = paths.out('efficiency_campaign')
    E.to_csv(od / 'per_run_arm.csv', index=False)
    M.to_csv(od / 'map_per_run.csv', index=False)
    H.to_csv(od / 'headline_per_run.csv', index=False)
    S.to_csv(od / 'stability.csv', index=False)
    P.to_csv(od / 'map_shape_spread.csv', index=False)
    I.to_csv(od / 'incidence_per_run.csv', index=False)
    Dip.to_csv(od / 'head_on_dip_per_run.csv', index=False)
    # Per-run files in the shape `acceptance.Chambers` already reads, so the
    # acceptance toy needs no new plumbing to go per run.
    (od / 'per_run').mkdir(exist_ok=True)
    for run, g in H.groupby('run'):
        g.drop(columns=['run']).to_csv(
            od / 'per_run' / f'efficiency_headline_{run}.csv', index=False)
    for run, g in M.groupby('run'):
        g.drop(columns=['run']).to_csv(
            od / 'per_run' / f'efficiency_map_{run}.csv', index=False)
    for run, g in I.groupby('run'):
        g.drop(columns=['run']).to_csv(
            od / 'per_run' / f'efficiency_incidence_{run}.csv', index=False)

    json.dump(dict(
        schema=SCHEMA, reco=str(reco), n_runs=int(E.run.nunique()),
        tag='in-time wall AND plastic coincidence in the same arm',
        tag_source='slim_export.read_export (parquet), validated identical to '
                   'the ROOT slim on run_145 arm A',
        dt_window=list(DT_WINDOW), u_edges=[float(x) for x in U_EDGES],
        min_bin=MIN_BIN, map_basis='single track (n_tracks == 1) among tagged, '
                                   'seeded, at-most-one-track events',
        runs_failed=failed,
        caveat='lower bounds: the denominator includes particles that missed '
               'the active area and accidental wall+plastic pairs. B is a HIT '
               'efficiency and is not a track efficiency.'),
        open(od / 'campaign_efficiency.meta.json', 'w'), indent=1)

    print(f'\nheadline efficiency per arm, over {E.run.nunique()} runs '
          f'(post-access only in the spread)\n')
    print(f'{"arm":>3} {"runs":>5} {"min":>7} {"median":>8} {"max":>7} '
          f'{"p10-p90":>8} {"block":>8}')
    for r in S.itertuples():
        print(f'{r.arm:>3} {r.n_runs:>5} {100 * r.min:>6.1f}% '
              f'{100 * r.median:>7.1f}% {100 * r.max:>6.1f}% '
              f'{100 * r.p10_p90_frac:>7.1f}% '
              f'{100 * r.block_shift_frac:>+7.1f}%')
    if not S.empty:
        print('\nrun_145 against the campaign median -- the borrowing this '
              'module exists to check\n')
        med = S.set_index('arm')['median']
        r145 = H[H.run == 'run_145'].set_index('arm')['efficiency']
        for arm in ARMS:
            if arm in med.index and arm in r145.index:
                print(f'  {arm}: run_145 {100 * r145[arm]:5.1f}%   campaign '
                      f'median {100 * med[arm]:5.1f}%   '
                      f'{100 * (r145[arm] / med[arm] - 1):+6.1f}%')
    if not P.empty:
        print(f'\nmap shape sd, per arm (1 = the run\'s own mean)\n')
        piv = P.pivot(index='arm', columns='u_mid', values='shape_sd')
        print('   ' + piv.round(3).to_string().replace('\n', '\n   '))
    if not Dip.empty:
        print('\nthe HEAD-ON DIP -- tracking rate of the head-on wall group '
              'over its two neighbours\n')
        print(f'{"arm":>3} {"runs":>5} {"median":>8} {"min":>7} {"max":>7} '
              f'{"below both":>11}')
        for arm, g in Dip.groupby('arm'):
            print(f'{arm:>3} {g.run.nunique():>5} '
                  f'{g.track_ratio.median():>8.3f} {g.track_ratio.min():>7.3f} '
                  f'{g.track_ratio.max():>7.3f} '
                  f'{g.below_both.mean():>10.0%}')
    if failed:
        print(f'\n{len(failed)} run(s) failed: {", ".join(failed)}')
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
