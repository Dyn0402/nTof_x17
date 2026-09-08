#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normal_incidence.py -- what a track arriving head-on costs the opening angle.

THE PREMISE HAD TO BE CHECKED FIRST, AND IT IS FALSE.  The standing worry was
that ``wft.reco.TAN_MIN_SLOPE = 0.08`` *blanks* a band of the chamber where
target-pointing tracks arrive perpendicular to the plane.  It does not.  Read
in the source: the constant appears at ``reco.py:242`` only to SET the flag
``slope_reliable``, and the single place that gates on it is
``compat._strip_fit(require_slope=...)``, whose default is False and which no
caller in this repository passes True -- every caller is in the June bench
packages, none in the n_TOF chain.  Confirmed in the data: of run_145's 116 839
gated tracks, **23.4 % carry x_slope_reliable = False and 30.3 % y, and all of
them are present and gated**.  Nothing is blanked.

SO WHAT IS THE REAL COST?  Not acceptance -- resolution.  A track at tan = 0 is
one whose charge column arrives at every strip simultaneously, so its timing
carries no slope information and its ANGLE is meaningless even though its
POSITION is fine.  That matters for exactly one observable: the opening angle,
which is built from two directions.  And the effect is not uniform in the
opening angle, because a pair from a point source has correlated incidence: the
angles at which a chamber sees a head-on leg are the angles at which that
chamber's pairs are most collimated.

WHAT THIS MEASURES.  Through the same toy `acceptance.py` uses -- same vertex
distribution, same geometry, same trigger, same dead channels -- but recording
each accepted leg's TRUE in-plane angles, so the fraction landing in the
unreliable band is a property of the geometry and not of the reconstruction:

  * ``p_unreliable(theta | topology)``  -- pairs with at least one plane of one
    leg inside |tan| < TAN_MIN_SLOPE.  This is the population whose opening
    angle is degraded but NOT removed today.
  * ``acceptance_if_cut(theta | topology)`` -- what the acceptance would become
    if a reliable slope were required on both planes of both legs.  October's
    number, if the cut is ever applied.

The two are different questions and the module reports both, because "we lose
30 % of pairs" and "30 % of pairs have a bad angle" call for opposite
responses.

    python -m sept26_prelim_analysis.normal_incidence --run run_145
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
from sept26_prelim_analysis import acceptance as AC  # noqa: E402

SCHEMA = 'sept26_prelim/normal_incidence/1'
THETA_BINS = np.arange(0.0, 181.0, 10.0)


def tan_in_plane(D: np.ndarray, arm: str) -> tuple:
    """(|tan_u|, |tan_v|) of each direction in one arm's local frame.

    ``tan`` in the reconstruction is du/dw -- how far the track slides across
    the strips per unit of drift depth -- so it is the in-plane component over
    the normal component, computed here from the true direction.
    """
    from ntof_tracking.reco import geometry as G
    w = D @ G.W_HAT[arm]
    u = D @ G.U_HAT[arm]
    v = D @ G.V_HAT
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(u / w), np.abs(v / w)


def throw(ch, n: int = 2_000_000, offset=(0.0, 0.0, 0.0), seed: int = 23,
          tan_min: float = None) -> pd.DataFrame:
    """One row per accepted pair-leg-combination, with its slope reliability."""
    from wft import reco as WR
    tan_min = WR.TAN_MIN_SLOPE if tan_min is None else tan_min
    rng = np.random.default_rng(seed)
    P = AC._vertices(rng, n, offset)
    d1 = AC._isotropic(rng, n)
    th = rng.uniform(0.0, np.pi, n)
    d2 = AC._rotate_by(rng, d1, th)

    hit, rel = {}, {}
    for leg, D in ((1, d1), (2, d2)):
        for a in AC.ARMS:
            u, v, plane, plastic = ch.cross(P, D, a)
            keep = plane & (rng.uniform(0, 1, n) < ch.efficiency(a, u))
            hit[(leg, a)] = dict(plane=plane, plastic=plastic, reco=keep)
            tu, tv = tan_in_plane(D, a)
            # a leg is reliable only if BOTH planes carry slope information:
            # the 3D direction needs both, and the weaker one sets the angle
            rel[(leg, a)] = (tu >= tan_min) & (tv >= tan_min)

    trig = np.zeros(n, bool)
    for leg in (1, 2):
        for a in AC.ARMS:
            trig |= hit[(leg, a)]['plane'] & hit[(leg, a)]['plastic']

    rows = []
    for a1 in AC.ANGLE_ARMS:
        for a2 in AC.ANGLE_ARMS:
            if a2 < a1:
                continue
            for l1, l2 in ((1, 2), (2, 1)):
                if a1 == a2 and (l1, l2) == (2, 1):
                    continue
                m = hit[(l1, a1)]['reco'] & hit[(l2, a2)]['reco'] & trig
                if not m.any():
                    continue
                both = rel[(l1, a1)][m] & rel[(l2, a2)][m]
                rows.append(pd.DataFrame(dict(
                    theta=np.degrees(th[m]), arm1=a1, arm2=a2,
                    topology=AC.topology(a1, a2), both_reliable=both)))
    # per-LEG reliability, per arm, on accepted legs only -- this is the
    # quantity the data can check directly, because `slope_reliable` is
    # recorded per track and per plane.
    legs = []
    for a in AC.ANGLE_ARMS:
        m = np.zeros(n, bool)
        r = np.zeros(n, bool)
        for leg in (1, 2):
            sel = hit[(leg, a)]['reco'] & trig
            m |= sel
            r |= sel & rel[(leg, a)]
        # count each accepted leg once: recompute without the OR collapse
        acc_n = sum(int((hit[(leg, a)]['reco'] & trig).sum()) for leg in (1, 2))
        rel_n = sum(int((hit[(leg, a)]['reco'] & trig & rel[(leg, a)]).sum())
                    for leg in (1, 2))
        legs.append(dict(arm=a, n_legs=acc_n, n_reliable=rel_n,
                         frac_unreliable=1.0 - rel_n / max(acc_n, 1)))
    thrown = pd.DataFrame(dict(theta=np.degrees(th)))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            thrown, pd.DataFrame(legs))


def curves(acc: pd.DataFrame, thrown: pd.DataFrame,
           bins=THETA_BINS) -> pd.DataFrame:
    """Per topology and theta bin: acceptance now, acceptance if cut, and the
    fraction whose opening angle is degraded but kept."""
    n0, _ = np.histogram(thrown.theta, bins=bins)
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    groups = [('all', acc)] + [(t, g) for t, g in acc.groupby('topology')]
    for name, g in groups:
        k, _ = np.histogram(g.theta, bins=bins)
        kr, _ = np.histogram(g[g.both_reliable].theta, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            a_now = np.where(n0 > 0, k / n0, np.nan)
            a_cut = np.where(n0 > 0, kr / n0, np.nan)
            frac = np.where(k > 0, 1.0 - kr / np.clip(k, 1, None), np.nan)
        rows.append(pd.DataFrame(dict(
            group=name, theta=mid, n_acc=k, n_reliable=kr, n_thrown=n0,
            acc_now=a_now, acc_if_cut=a_cut, frac_degraded=frac,
            survival=np.where(k > 0, kr / np.clip(k, 1, None), np.nan))))
    return pd.concat(rows, ignore_index=True)


def measured_rates(run: str, subruns, tan_min: float = None) -> pd.DataFrame:
    """The same fraction, in the DATA -- a cross-check on the toy.

    Uses `slope_reliable` as the reconstruction actually recorded it, so if the
    toy's geometry is right these should agree without anything being tuned.
    """
    from wft import reco as WR
    tan_min = WR.TAN_MIN_SLOPE if tan_min is None else tan_min
    from sept26_prelim_analysis import k_robustness as KR
    from ntof_tracking import run145_target_imaging as TI
    src = paths.out('stage3_fullpass')
    out = []
    for sub in subruns:
        p = paths.require(src / f'tracks_{run}_{sub}.parquet',
                          f'stage-3 tracks for {sub}')
        out.append(pd.read_parquet(p, columns=[
            'arm', 'gated', 'angle_calibrated', 'dca_axis_mm',
            'x_slope_reliable', 'y_slope_reliable', 'x_p0', 'y_p0',
            'x_tan_theta', 'y_tan_theta']))
    t = pd.concat(out, ignore_index=True)
    t = t[t.gated]
    merged = str(paths.out('fullpass') / run)
    rows = []
    for arm, g in t.groupby('arm'):
        pt = g[g.dca_axis_mm < 30]
        # ...and again with the hot cells masked, because a noise cluster has
        # no real slope and would be flagged unreliable for the wrong reason.
        f_hot = np.nan
        if len(pt):
            M = KR.Masks(run, subruns, arm, merged)
            xl = TI.local_x(pt.x_p0.to_numpy())
            yl = TI.IN_PLANE_SIGN * (pt.y_p0.to_numpy() - TI.STRIP_MAP_HALF)
            keep = M.keep(xl, yl, 'no_hot')
            if keep.sum() > 100:
                q = pt[keep]
                f_hot = float((~(q.x_slope_reliable
                                 & q.y_slope_reliable)).mean())
        rows.append(dict(
            arm=arm, n_gated=int(len(g)),
            frac_x_unreliable=float((~g.x_slope_reliable).mean()),
            frac_y_unreliable=float((~g.y_slope_reliable).mean()),
            frac_either=float((~(g.x_slope_reliable
                                 & g.y_slope_reliable)).mean()),
            n_pointing=int(len(pt)),
            frac_either_pointing=float((~(pt.x_slope_reliable
                                          & pt.y_slope_reliable)).mean())
            if len(pt) else np.nan,
            frac_either_pointing_nohot=f_hot))
    return pd.DataFrame(rows)


def efficiency_vs_incidence(run: str, subruns) -> pd.DataFrame:
    """Does the chamber respond less often to a track arriving head-on?

    This is the systematic the toy/data disagreement points at, measured
    directly and WITHOUT the Micromegas -- which is the only way it can be
    measured, since the thing under test is whether the Micromegas responds.

    The trick is that the tag already contains the angle.  A wall+plastic
    coincidence in one arm says a particle from the target crossed that arm,
    and the wall GROUP that fired says roughly where: a particle reaching the
    wall at u_w came in at

        tan ~ (u_w - foot) / (d_perp + wall_depth)

    The four groups sit at u ~ -159, -59, +41, +141 mm, so they sample
    tan ~ -0.5, -0.2, +0.1, +0.4 -- and **one of them lands inside the
    head-on band while the others do not**.  Comparing the chamber's response
    rate across groups is therefore an efficiency-versus-incidence measurement
    with an entirely external abscissa.

    The confound is real and is reported next to the result: the four groups
    also sit at four different places on the chamber, so a response difference
    could be the surface rather than the angle.  The dead/hot maps say how much
    of each group's footprint is compromised, and chamber A -- which has no
    dead channels at all -- is the clean test.
    """
    from sept26_prelim_analysis import efficiency as EF
    from sept26_prelim_analysis import chamber_b as CB
    from ntof_tracking import run145_target_imaging as TI

    gu = CB.wall_group_u(run)
    fullpass = str(paths.out('fullpass') / run)
    rows = []
    for arm in AC.ARMS:
        tag = EF.tagged_events(run, subruns, arm)
        if tag.empty:
            continue
        resp = EF.chamber_response(run, subruns, arm, fullpass)
        j = tag.merge(resp, on=['subrun', 'event_id'], how='left')
        j['seeded'] = j.seeded.fillna(False).astype(bool)
        j['tracked'] = j.tracked.fillna(False).astype(bool)
        # single-group tags only: with two groups lit the incidence is ambiguous
        j = j[j.wall_groups.map(len) == 1].copy()
        if j.empty:
            continue
        j['grp'] = j.wall_groups.map(lambda s: next(iter(s)))
        foot = TI.PINWHEEL[arm]
        for g, h in j.groupby('grp'):
            if g not in gu[arm] or len(h) < 200:
                continue
            tan = (gu[arm][g] - foot) / (CB.D_PERP_MM + CB.WALL_DEPTH_MM)
            rows.append(dict(
                arm=arm, grp=int(g), u_wall=gu[arm][g], tan_expected=tan,
                head_on=bool(abs(tan) < 0.08), n_tagged=int(len(h)),
                p_seeded=float(h.seeded.mean()),
                p_tracked=float(h.tracked.mean())))
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    # normalise each arm to its own best group, so the comparison is within a
    # chamber and the absolute efficiency differences do not confuse it
    R['p_tracked_rel'] = R.groupby('arm').p_tracked.transform(
        lambda x: x / x.max())
    return R


def head_on_dip(EI: pd.DataFrame) -> pd.DataFrame:
    """The head-on group against its two positional NEIGHBOURS.

    This is what controls the confound.  The four wall groups sample four
    incidences, but they also sample four places on the chamber, so a bare
    ranking cannot separate angle from surface.  The head-on group is the
    interior one, with a neighbour on each side: if the dip were the surface it
    would interpolate between them, and if it is the angle it will sit below
    both.  Reported for seeding and for tracking separately, because the two
    answer different questions -- did the chamber see the particle, and did the
    fit return a usable track.
    """
    rows = []
    for arm, g in EI.groupby('arm'):
        h = g[g.head_on]
        if h.empty:
            continue
        k = int(h.grp.iloc[0])
        nb = g[g.grp.isin((k - 1, k + 1))]
        if len(nb) < 2:
            continue
        rows.append(dict(
            arm=arm, grp=k, n_head_on=int(h.n_tagged.iloc[0]),
            seed_head_on=float(h.p_seeded.iloc[0]),
            seed_neighbours=float(nb.p_seeded.mean()),
            seed_ratio=float(h.p_seeded.iloc[0] / nb.p_seeded.mean()),
            track_head_on=float(h.p_tracked.iloc[0]),
            track_neighbours=float(nb.p_tracked.mean()),
            track_ratio=float(h.p_tracked.iloc[0] / nb.p_tracked.mean()),
            below_both=bool(h.p_tracked.iloc[0] < nb.p_tracked.min())))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--n', type=int, default=2_000_000)
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]
    merged = str(paths.out('fullpass') / a.run)

    im = paths.out('imaging')
    meta = json.load(open(paths.require(im / f'imaging_{a.run}.meta.json',
                                        'the source position')))
    v = {d['axis']: d['source_mm'] for d in meta['verdict']}
    offset = (v.get('X', 0.0), 0.0, v.get('Z', 0.0))

    ed = paths.out('efficiency')
    hl = pd.read_csv(paths.require(ed / f'efficiency_headline_{a.run}.csv',
                                   'the efficiency headline'))
    emap = pd.read_csv(paths.require(ed / f'efficiency_map_{a.run}.csv',
                                     'the efficiency map'))
    eff = {r.arm: (r.efficiency if r.basis == 'tracks' else 0.0)
           for r in hl.itertuples()}
    ch = AC.Chambers(a.run, subs, merged, eff_map=emap, eff_headline=eff)

    acc, thrown, legs = throw(ch, a.n, offset=offset)
    C = curves(acc, thrown)
    M = measured_rates(a.run, subs)
    EI = efficiency_vs_incidence(a.run, subs)
    HD = head_on_dip(EI)
    # The cross-check: the toy predicts a per-leg unreliable fraction from
    # geometry alone, and the data records one.  Nothing is tuned between them.
    X = legs.merge(M[['arm', 'frac_either_pointing',
                      'frac_either_pointing_nohot', 'n_pointing']],
                   on='arm', how='left').rename(
        columns={'frac_unreliable': 'toy', 'frac_either_pointing': 'data',
                 'frac_either_pointing_nohot': 'data_nohot'})
    X['ratio'] = X.data / X.toy
    X['ratio_nohot'] = X.data_nohot / X.toy

    from wft import reco as WR
    od = paths.out('angle')
    C.to_csv(od / f'normal_incidence_{a.run}.csv', index=False)
    M.to_csv(od / f'slope_reliable_measured_{a.run}.csv', index=False)
    EI.to_csv(od / f'efficiency_vs_incidence_{a.run}.csv', index=False)
    HD.to_csv(od / f'head_on_dip_{a.run}.csv', index=False)
    X.to_csv(od / f'slope_reliable_crosscheck_{a.run}.csv', index=False)
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs, n_thrown=a.n,
                   tan_min_slope=WR.TAN_MIN_SLOPE,
                   gated_today=True,
                   note='TAN_MIN_SLOPE sets a flag and gates nothing in this '
                        'chain; the cut column is a hypothetical'),
              open(od / f'normal_incidence_{a.run}.meta.json', 'w'), indent=1)

    print(f'TAN_MIN_SLOPE = {WR.TAN_MIN_SLOPE} -- a FLAG, not a cut '
          '(verified in wft/reco.py and wft/compat.py)\n')
    print('IN THE DATA: gated tracks whose slope is flagged unreliable')
    print(M.to_string(index=False))
    print('\nCROSS-CHECK: per-leg unreliable fraction, toy vs data '
          '(pointing sample), nothing tuned')
    print(X[['arm', 'n_legs', 'toy', 'n_pointing', 'data', 'ratio',
             'data_nohot', 'ratio_nohot']].round(3).to_string(index=False))
    print('\nIN THE TOY: what a require-slope cut would cost, per topology')
    for t in ('intra', 'perpendicular', 'opposing', 'all'):
        g = C[C.group == t]
        if g.empty or not np.isfinite(g.n_acc.sum()) or g.n_acc.sum() == 0:
            continue
        surv = g.n_reliable.sum() / max(g.n_acc.sum(), 1)
        print(f'  {t:<14} overall survival {100 * surv:5.1f} %'
              f'   ({int(g.n_acc.sum()):,} -> {int(g.n_reliable.sum()):,})')
    print('\nSURVIVAL vs OPENING ANGLE  (fraction of accepted pairs keeping a '
          'reliable angle)')
    piv = C.pivot(index='theta', columns='group', values='survival')
    cols = [c for c in ('intra', 'perpendicular', 'opposing', 'all')
            if c in piv.columns]
    print((100 * piv[cols]).round(1).dropna(how='all').to_string())
    print('\nEFFICIENCY vs INCIDENCE -- abscissa from the fired wall group, '
          'no Micromegas in it')
    if len(EI):
        print(EI[['arm', 'grp', 'tan_expected', 'head_on', 'n_tagged',
                  'p_seeded', 'p_tracked', 'p_tracked_rel']]
              .round(3).to_string(index=False))
    if len(HD):
        print('\nTHE HEAD-ON GROUP AGAINST ITS TWO POSITIONAL NEIGHBOURS')
        print(HD.round(3).to_string(index=False))
        print(f'  below BOTH neighbours in {int(HD.below_both.sum())} of '
              f'{len(HD)} chambers; median tracking ratio '
              f'{HD.track_ratio.median():.2f}, seeding ratio '
              f'{HD.seed_ratio.median():.2f}')
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
