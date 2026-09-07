#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairs.py -- the two-chamber (X17-topology) rate, measured against a control.

An X17 at 16.8 MeV has a minimum opening angle of 109 deg, so its e+e- pair
lands in TWO chambers, and the chambers sit at 90 deg azimuth: A and C are
opposite each other (measured +94.0 and -85.8 deg), as are B and D.  So the
signal topology is "a track in A and a track in C in the same trigger".

**Counting those directly does not work, and the reason matters.**  The
production trigger is a wall AND plastic coincidence in ONE arm.  That
partitions the events by arm: an event with an A track is overwhelmingly an
A-triggered event, and is therefore *less* likely than average to also carry a
C track.  Measured on run_145: A-pointing and C-pointing event sets overlap
3.5x LESS than independent expectation.  Any "excess" quoted against a
product-of-marginals null is measuring the trigger, not the physics.

The measurement here controls for it.  Fix the track chamber, and vary the
TRIGGER chamber:

    rate of a target-pointing track in C, given the event triggered on A
        versus, given the event triggered on B or on D

B and D are the controls: C is equally "not the trigger arm" in all three, so
acceptance, occupancy and the ambient single-track rate divide out, and a
back-to-back pair signal has to show up as an A-trigger excess.  The controls
need no angle calibration -- the trigger arm comes from the n_TOF slim, not
from the reconstruction -- which is what lets B and D serve while their own
angles are unusable.

A genuine back-to-back signal must be **symmetric**: A-in-C-triggered and
C-in-A-triggered have to move together.  One of the two rising alone is a
fluctuation or an acceptance difference, not a pair.  The report prints both.

    python -m sept26_prelim_analysis.pairs --run run_145
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
from sept26_prelim_analysis.build_tracks import (  # noqa: E402
    IN_PLANE_SIGN, IN_PLANE_SIGN_Y, STRIP_MAP_HALF)

SCHEMA = 'sept26_prelim/pairs/1'
ARMS = ('A', 'B', 'C', 'D')
#: measured from run_config: A +94.0, C -85.8 (179.8 apart); D +3.8, B -176.2
OPPOSITE = {'A': 'C', 'C': 'A', 'B': 'D', 'D': 'B'}
DT_WINDOW = (-100.0, 60.0)
DCA_CUTS = (20.0, 30.0, 50.0)


def trigger_arms(run: str, subruns) -> pd.DataFrame:
    """(subrun, event_id, trig_arm) for events with exactly ONE arm holding an
    in-time wall AND plastic hit -- the production trigger, as recorded.

    Restricting to single-arm triggers keeps the partition clean: an event with
    two arms triggering cannot serve as a control for either.
    """
    import uproot
    arm_of = {i: ARMS[i % 4] for i in range(12)}
    rows = []
    for sub in subruns:
        d = os.path.join(str(paths.root('runs')), run, sub, 'ntof_hits')
        f = sorted(x for x in os.listdir(d) if x.endswith('.root'))
        if not f:
            raise FileNotFoundError(f'no slim n_TOF file under {d}')
        a = uproot.open(os.path.join(d, f[0]))['hits'].arrays(
            ['eventId', 'det', 'dt_ns', 'is_control'], library='np')
        m = ((a['is_control'] == 0) & (a['dt_ns'] >= DT_WINDOW[0])
             & (a['dt_ns'] <= DT_WINDOW[1]))
        t = pd.DataFrame({'event_id': a['eventId'][m], 'det': a['det'][m]})
        t['arm'] = t.det.mod(4).map(arm_of)
        wal = set(map(tuple, t[t.det < 4][['event_id', 'arm']]
                      .drop_duplicates().values))
        pss = set(map(tuple, t[(t.det >= 4) & (t.det < 8)][['event_id', 'arm']]
                      .drop_duplicates().values))
        both = pd.DataFrame(list(wal & pss), columns=['event_id', 'trig_arm'])
        both['subrun'] = sub
        rows.append(both)
    tr = pd.concat(rows, ignore_index=True)
    n = tr.groupby(['subrun', 'event_id']).trig_arm.size()
    solo = n[n == 1].index
    return (tr.set_index(['subrun', 'event_id']).loc[solo].reset_index())


def pointing_tracks(run: str, subruns, k_arm: dict, merged_dir: str) -> pd.DataFrame:
    """Gated tracks with their closest approach to the beam axis, per arm.

    Only arms carrying a certified angle scale appear: a track's *direction* is
    what makes it "target-pointing", and an arm with no k has no direction.
    """
    from ntof_tracking import run145_target_imaging as TI
    from ntof_tracking.reco import geometry as G
    cfg = json.loads((paths.root('runs') / run / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)
    out = []
    for sub in subruns:
        for arm, k in k_arm.items():
            p = os.path.join(merged_dir, sub, f'mx17_{arm}',
                             'events_prelim.parquet')
            if not os.path.exists(p):
                continue
            df = pd.read_parquet(p)
            sel = (df['x_ok'].to_numpy() & df['y_ok'].to_numpy()
                   & (df['n_tracks'].to_numpy() > 0))
            xl = IN_PLANE_SIGN * (df['x_p0'].to_numpy() - STRIP_MAP_HALF)
            yl = IN_PLANE_SIGN_Y * (df['y_p0'].to_numpy() - STRIP_MAP_HALF)
            tx = df['x_tan_theta'].to_numpy() * k
            ty = df['y_tan_theta'].to_numpy() * k
            tr = trs[f'mx17_{arm}']
            P0 = tr.local_to_global(xl, yl, np.zeros_like(xl))
            P1 = tr.local_to_global(xl - tx * 30.0, yl - ty * 30.0,
                                    np.full_like(xl, 30.0))
            D = P1 - P0
            D = D / np.linalg.norm(D, axis=-1, keepdims=True)
            r, y_at, _ = TI.axis_approach(P0, D)
            out.append(pd.DataFrame(dict(
                subrun=sub, event_id=df['event_id'].to_numpy(), arm=arm,
                gated=sel, dca=r, target_y=y_at))[sel])
    if not out:
        raise RuntimeError('no arm has a certified angle scale; nothing to pair')
    return pd.concat(out, ignore_index=True)


def measure(TRK: pd.DataFrame, TRG: pd.DataFrame, cut: float) -> pd.DataFrame:
    """One row per (track arm, trigger arm): the rate and its Poisson error."""
    P = TRK[TRK.dca < cut][['subrun', 'event_id', 'arm']].drop_duplicates()
    rows = []
    for track_arm in sorted(P.arm.unique()):
        pa = P[P.arm == track_arm]
        j = TRG.merge(pa.assign(hit=1), on=['subrun', 'event_id'], how='left')
        for ta in ARMS:
            s = j[j.trig_arm == ta]
            n, tot = int(s.hit.notna().sum()), int(len(s))
            rows.append(dict(track_arm=track_arm, trig_arm=ta, n=n, n_trig=tot,
                             rate=n / tot if tot else np.nan,
                             err=np.sqrt(max(n, 1)) / tot if tot else np.nan,
                             is_self=(ta == track_arm),
                             is_opposite=(ta == OPPOSITE.get(track_arm))))
    return pd.DataFrame(rows).assign(dca_cut=cut)


def excess(M: pd.DataFrame) -> pd.DataFrame:
    """The controlled comparison: opposite-arm trigger against the two
    perpendicular-arm triggers pooled.

    The self-trigger row is excluded from both sides -- it is the trigger arm's
    own track and carries no pair information.
    """
    rows = []
    for track_arm, g in M.groupby('track_arm'):
        sig = g[g.is_opposite]
        ctl = g[~g.is_opposite & ~g.is_self]
        if sig.empty or ctl.empty:
            continue
        n_s, t_s = int(sig.n.iloc[0]), int(sig.n_trig.iloc[0])
        n_c, t_c = int(ctl.n.sum()), int(ctl.n_trig.sum())
        r_c = n_c / t_c
        exp = r_c * t_s
        # Poisson on the signal count, plus the control's own error scaled in
        sig_err = np.sqrt(exp + exp ** 2 * (1.0 / max(n_c, 1)))
        rows.append(dict(
            track_arm=track_arm, trig_arm=OPPOSITE[track_arm],
            n_obs=n_s, n_exp=exp, rate_sig=n_s / t_s, rate_ctl=r_c,
            excess=n_s - exp, sigma=(n_s - exp) / sig_err if sig_err else np.nan,
            n_trig_sig=t_s, n_trig_ctl=t_c, dca_cut=float(g.dca_cut.iloc[0])))
    return pd.DataFrame(rows)


def control_spread(M: pd.DataFrame) -> pd.DataFrame:
    """How much the answer moves if you pick a different control chamber.

    The pooled control assumes the two perpendicular chambers are equivalent
    backgrounds for the track chamber.  They need not be -- occupancy and
    efficiency differ -- so the honest systematic is the spread of the answer
    over the choice.  A signal survives it; a fluctuation does not.
    """
    rows = []
    for (track_arm, cut), g in M.groupby(['track_arm', 'dca_cut']):
        sig = g[g.is_opposite]
        if sig.empty:
            continue
        n_s, t_s = int(sig.n.iloc[0]), int(sig.n_trig.iloc[0])
        for _, c in g[~g.is_opposite & ~g.is_self].iterrows():
            r_c = c.n / c.n_trig
            exp = r_c * t_s
            err = np.sqrt(exp + exp ** 2 / max(c.n, 1))
            rows.append(dict(track_arm=track_arm, dca_cut=cut,
                             control=c.trig_arm, n_obs=n_s, n_exp=exp,
                             excess=n_s - exp,
                             sigma=(n_s - exp) / err if err else np.nan))
    return pd.DataFrame(rows)


def limits(E: pd.DataFrame) -> pd.DataFrame:
    """Turn each null into a 95 % CL upper limit on the second-track rate.

    A null is only useful with a sensitivity attached: "no excess" and "no
    excess above 5 %" are very different statements.  The limit is quoted as a
    fraction of the opposite-arm triggers, which is the quantity a pair
    hypothesis predicts -- how often a trigger in one chamber is accompanied by
    a target-pointing track in the chamber facing it, beyond the ambient rate.

    One-sided Gaussian at 1.645 sigma on the excess, floored at zero: a
    negative central value gives a limit set by the error alone, which is the
    honest reading when the point estimate is below the control.
    """
    rows = []
    for _, r in E.iterrows():
        err = abs(r['excess'] / r['sigma']) if r['sigma'] else np.nan
        ul = max(float(r['excess']), 0.0) + 1.645 * err
        rows.append(dict(
            dca_cut=r['dca_cut'], track_arm=r['track_arm'],
            trig_arm=r['trig_arm'], excess=r['excess'], err=err,
            ul_events=ul, ul_frac=ul / r['n_trig_sig'],
            n_trig=r['n_trig_sig']))
    L = pd.DataFrame(rows)
    # The two directions of one chamber pair are the same physics measured
    # twice, so combine them inverse-variance -- and a real signal cannot
    # average away, which is the point.
    comb = []
    for cut, g in L.groupby('dca_cut'):
        for pair in {frozenset((a, OPPOSITE[a])) for a in g.track_arm}:
            gg = g[g.track_arm.isin(pair) & g.trig_arm.isin(pair)]
            if len(gg) < 2:
                continue
            w = 1.0 / gg.err.to_numpy() ** 2
            x = gg.excess.to_numpy()
            mu = float((w * x).sum() / w.sum())
            se = float(np.sqrt(1.0 / w.sum()))
            nt = float(gg.n_trig.mean())
            comb.append(dict(dca_cut=cut, pair='-'.join(sorted(pair)),
                             excess=mu, err=se, sigma=mu / se,
                             ul_events=max(mu, 0.0) + 1.645 * se,
                             ul_frac=(max(mu, 0.0) + 1.645 * se) / nt,
                             n_trig=nt))
    return L, pd.DataFrame(comb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns', default='stat090_0000,stat090_0001')
    ap.add_argument('--merged', default=str(paths.out('fullpass') / 'run_145'))
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    cal = json.load(open(paths.require(
        paths.out('kcal') / f'k_arm_{a.run}.json', 'angle calibration')))
    k = {arm: float(v) for arm, v in (cal.get('apply') or {}).items()}
    print(f'chambers with a certified angle scale: {", ".join(sorted(k)) or "none"}')
    for arm in ARMS:
        if arm not in k:
            print(f'  {arm}: {cal["arms"][arm]["verdict"]} -- no direction, '
                  f'usable only as a TRIGGER control')

    TRG = trigger_arms(a.run, subs)
    TRK = pointing_tracks(a.run, subs, k, a.merged)
    print(f'\nsingle-arm triggers: {len(TRG):,}')
    print(TRG.trig_arm.value_counts().sort_index().to_string())
    print(f'gated tracks in calibrated chambers: {len(TRK):,}')

    M = pd.concat([measure(TRK, TRG, c) for c in DCA_CUTS], ignore_index=True)
    E = pd.concat([excess(M[M.dca_cut == c]) for c in DCA_CUTS],
                  ignore_index=True)

    od = paths.out('pairs')
    M.to_csv(od / f'pair_rates_{a.run}.csv', index=False)
    E.to_csv(od / f'pair_excess_{a.run}.csv', index=False)
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs,
                   k_arm=k, dca_cuts=list(DCA_CUTS),
                   control='perpendicular-arm triggers, pooled',
                   note='a real back-to-back signal must be symmetric in the '
                        'two directions; one alone is not evidence'),
              open(od / f'pairs_{a.run}.meta.json', 'w'), indent=1)

    print('\nRate of a target-pointing track, by trigger arm  [%]')
    for c in DCA_CUTS:
        m = M[M.dca_cut == c]
        print(f'\n  dca < {c:.0f} mm')
        piv = m.pivot(index='track_arm', columns='trig_arm', values='rate') * 100
        print('   ' + piv.round(2).to_string().replace('\n', '\n   '))
    print('\nControlled excess (opposite-arm trigger vs perpendicular pooled)')
    print(E[['dca_cut', 'track_arm', 'trig_arm', 'n_obs', 'n_exp', 'excess',
             'sigma']].round(2).to_string(index=False))
    S = control_spread(M)
    S.to_csv(od / f'pair_control_spread_{a.run}.csv', index=False)
    print('\nSystematic: the same excess against each control chamber alone')
    piv = S.pivot_table(index=['track_arm', 'dca_cut'], columns='control',
                        values='sigma')
    print('   ' + piv.round(2).to_string().replace('\n', '\n   '))
    rng = (piv.max(axis=1) - piv.min(axis=1))
    print(f'   control choice moves the significance by up to '
          f'{rng.max():.1f} sigma')

    L, C = limits(E)
    L.to_csv(od / f'pair_limits_{a.run}.csv', index=False)
    C.to_csv(od / f'pair_combined_{a.run}.csv', index=False)
    print('\n95 % CL upper limit on the second-track rate, per direction')
    print(L[['dca_cut', 'track_arm', 'trig_arm', 'excess', 'err',
             'ul_events']].round(1).assign(
        ul_pct=(100 * L.ul_frac).round(3)).to_string(index=False))
    if len(C):
        print('\nBoth directions of a pair combined (a real signal cannot '
              'average away)')
        print(C.assign(ul_pct=(100 * C.ul_frac).round(3))[
            ['dca_cut', 'pair', 'excess', 'err', 'sigma', 'ul_events',
             'ul_pct']].round(2).to_string(index=False))
    print(f'\nwrote {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
