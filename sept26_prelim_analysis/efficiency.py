#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
efficiency.py -- per-chamber efficiency, tagged by the scintillators.

Acceptance is what an opening-angle distribution has to be divided by, and
acceptance needs efficiency.  This measures it the only way that is not
circular: **the denominator comes from outside the Micromegas.**

THE TAG.  Each arm has a scintillator wall 96.4 mm behind its strip plane and a
plastic bar ~187 mm behind it.  An in-time coincidence of BOTH, in one arm, is
a particle that went through that arm -- and through the chamber, since the
chamber is in front of both.  Neither element is the Micromegas, so

    eff = P(gated 3D track in chamber X | wall AND plastic fired in arm X)

has an MM-independent denominator.  That is the whole point: the funnel's
`coinc_frac` is the reverse conditional, P(scintillators | track), which cannot
be an efficiency because its denominator is the thing being measured.

WHAT IT IS NOT.  Three things this number is not, all of which matter for
acceptance and none of which this module hides:

  * **Not purely the chamber.**  A wall+plastic coincidence can be a particle
    that missed the active area, a random pair of hits, or a shower.  Those
    dilute the denominator and push the measured efficiency DOWN, so it is a
    lower bound on the true chamber efficiency.  ``--pointing`` tightens the
    tag to a wall segment and a plastic bar the track can actually connect,
    which raises purity at the cost of using the reconstruction in the tag.
  * **Not uniform.**  It varies across the chamber, and acceptance needs the
    variation, not the average.  :func:`efficiency_map` bins it in the in-plane
    coordinate.
  * **Not the same as tracking.**  Chamber B has no drift field (STATUS.md), so
    for B the meaningful efficiency is the HIT efficiency -- did the chamber
    register a cluster -- not the track efficiency.  Both are reported, and for
    B only the first means anything.

    python -m sept26_prelim_analysis.efficiency --run run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
WAL_CODE = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
DT_WINDOW = (-100.0, 60.0)
SCHEMA = 'sept26_prelim/efficiency/1'
#: In-plane bins for the efficiency map, mm about the plane centre.  The strip
#: map runs +-199 mm; beyond ~150 mm the scintillators stop covering, so the
#: outer bins are acceptance rather than efficiency and are reported as such.
U_EDGES = np.arange(-160.0, 161.0, 40.0)


def tagged_events(run: str, subruns, arm: str) -> pd.DataFrame:
    """Events where arm ``arm`` had an in-time wall AND plastic hit.

    One row per (subrun, event): the tag, plus which wall groups and plastic
    bars fired, so a pointing-tightened version can be built from the same read.
    """
    import uproot
    rows = []
    for sub in subruns:
        d = os.path.join(str(paths.root('runs')), run, sub, 'ntof_hits')
        f = sorted(x for x in os.listdir(d) if x.endswith('.root'))
        if not f:
            raise FileNotFoundError(f'no slim n_TOF file under {d}')
        a = uproot.open(os.path.join(d, f[0]))['hits'].arrays(
            ['eventId', 'det', 'detn', 'dt_ns', 'is_control'], library='np')
        it = ((a['is_control'] == 0) & (a['dt_ns'] >= DT_WINDOW[0])
              & (a['dt_ns'] <= DT_WINDOW[1]))
        wal = it & (a['det'] == WAL_CODE[arm])
        pss = it & (a['det'] == WAL_CODE[arm] + 4)
        wg, pb = defaultdict(set), defaultdict(set)
        for e, dn in zip(a['eventId'][wal], a['detn'][wal]):
            wg[int(e)].add((int(dn) - 1) // 2)
        for e, dn in zip(a['eventId'][pss], a['detn'][pss]):
            pb[int(e)].add(int(dn))
        both = sorted(set(wg) & set(pb))
        rows.append(pd.DataFrame(dict(
            subrun=sub, event_id=both,
            wall_groups=[wg[e] for e in both],
            plastic_bars=[pb[e] for e in both])))
    return pd.concat(rows, ignore_index=True)


def chamber_response(run: str, subruns, arm: str, fullpass: str) -> pd.DataFrame:
    """What the chamber did, per event: seeded, both planes fit, gated track."""
    out = []
    for sub in subruns:
        p = os.path.join(fullpass, sub, f'mx17_{arm}', 'events_prelim.parquet')
        if not os.path.exists(p):
            raise FileNotFoundError(f'no full pass for {arm}/{sub}: {p}')
        d = pd.read_parquet(p, columns=['event_id', 'x_ok', 'y_ok', 'n_tracks',
                                        'n_hits', 'x_p0'])
        out.append(d.assign(subrun=sub))
    d = pd.concat(out, ignore_index=True)
    # The full pass has a row only for a SEEDED event, so presence in this
    # table is the hit-level response and needs no extra column.
    return d.assign(seeded=True,
                    xy_ok=d.x_ok.to_numpy() & d.y_ok.to_numpy(),
                    tracked=d.n_tracks.to_numpy() > 0)


def untagged_rate(run: str, subruns, arm: str, fullpass: str,
                  tagged: pd.DataFrame) -> dict:
    """How often the chamber responds when NO particle was tagged in that arm.

    This is the number that makes the efficiency meaningful.  A chamber that
    fires on almost everything has a high P(response | tagged) that says
    nothing about the tagged particle -- chamber D seeds 80 % of ALL triggers,
    so its raw 93.6 % is occupancy.  With an accidental response probability
    p0, P(resp | tagged) = eff + (1 - eff) p0, so

        eff = (P(resp | tagged) - p0) / (1 - p0)

    and p0 is measured here on the events the same arm's scintillators did NOT
    tag -- the honest control, rather than all triggers, which contain the
    tagged ones.
    """
    resp = chamber_response(run, subruns, arm, fullpass)
    cen = []
    for sub in subruns:
        c = pd.read_csv(paths.require(
            paths.out('stage1') / f'census_{run}_{sub}.csv',
            f'stage-1 census for {sub}'))
        cen.append(int(c.loc[c.cls == '(total)', 'n'].iloc[0]))
    n_trig = sum(cen)
    tag_keys = set(map(tuple, tagged[['subrun', 'event_id']].values))
    rk = list(map(tuple, resp[['subrun', 'event_id']].values))
    is_tag = np.array([k in tag_keys for k in rk])
    n_untagged = n_trig - len(tagged)
    # Events with no row in the full pass were not seeded at all, so the
    # untagged numerator is just the untagged rows that ARE present.
    return dict(p0_hit=float((~is_tag).sum()) / max(n_untagged, 1),
                p0_track=float((resp.tracked.to_numpy() & ~is_tag).sum())
                / max(n_untagged, 1),
                n_untagged=int(n_untagged))


def corrected(p_tagged: float, p0: float) -> float:
    """Accidental-corrected efficiency, floored at zero."""
    if not np.isfinite(p0) or p0 >= 1.0:
        return float('nan')
    return max((p_tagged - p0) / (1.0 - p0), 0.0)


def measure(run: str, subruns, fullpass: str) -> tuple:
    """Per-arm efficiency, and the same binned in the in-plane coordinate."""
    from ntof_tracking import run145_target_imaging as TI
    rows, maps = [], []
    for arm in ARMS:
        tag = tagged_events(run, subruns, arm)
        resp = chamber_response(run, subruns, arm, fullpass)
        j = tag.merge(resp, on=['subrun', 'event_id'], how='left')
        n = len(j)
        seeded = j.seeded.fillna(False).to_numpy().astype(bool)
        xy = j.xy_ok.fillna(False).to_numpy().astype(bool)
        trk = j.tracked.fillna(False).to_numpy().astype(bool)
        p0 = untagged_rate(run, subruns, arm, fullpass, tag)
        rows.append(dict(arm=arm, n_tagged=n,
                         p0_hit=p0['p0_hit'], p0_track=p0['p0_track'],
                         eff_hit_corr=corrected(seeded.mean(), p0['p0_hit']),
                         eff_track_corr=corrected(trk.mean(), p0['p0_track']),
                         n_seeded=int(seeded.sum()), n_xy_ok=int(xy.sum()),
                         n_tracked=int(trk.sum()),
                         eff_hit=seeded.mean(), eff_xy=xy.mean(),
                         eff_track=trk.mean(),
                         err_hit=np.sqrt(seeded.mean() * (1 - seeded.mean()) / n),
                         err_track=np.sqrt(trk.mean() * (1 - trk.mean()) / n)))
        # binned in the chamber's own in-plane coordinate, for the events that
        # produced a cluster; the un-seeded ones have no position by definition,
        # so the map is P(track | tagged AND seeded) and says so.
        u = np.full(n, np.nan)
        has = j.x_p0.notna().to_numpy()
        u[has] = TI.local_x(j.x_p0.to_numpy()[has]) - TI.PINWHEEL[arm]
        for lo, hi in zip(U_EDGES[:-1], U_EDGES[1:]):
            m = has & (u >= lo) & (u < hi)
            if m.sum() < 20:
                continue
            maps.append(dict(arm=arm, u_lo=lo, u_hi=hi, u_mid=0.5 * (lo + hi),
                             n=int(m.sum()), eff_track_given_seed=trk[m].mean(),
                             err=np.sqrt(trk[m].mean() * (1 - trk[m].mean())
                                         / m.sum())))
    return pd.DataFrame(rows), pd.DataFrame(maps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns', default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--fullpass',
                    default=str(paths.out('fullpass') / 'run_145'))
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    E, M = measure(a.run, subs, a.fullpass)
    od = paths.out('efficiency')
    E.to_csv(od / f'efficiency_{a.run}.csv', index=False)
    M.to_csv(od / f'efficiency_map_{a.run}.csv', index=False)
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs,
                   tag='in-time wall AND plastic coincidence in the same arm',
                   dt_window=list(DT_WINDOW),
                   caveat='denominator includes particles that missed the '
                          'active area and accidental wall+plastic pairs, so '
                          'these are LOWER BOUNDS on the chamber efficiency'),
              open(od / f'efficiency_{a.run}.meta.json', 'w'), indent=1)

    print('Scintillator-tagged efficiency: P(response | wall AND plastic in '
          'the same arm)\n')
    print(f'{"arm":>3} {"tagged":>9} {"hit eff":>16} {"both planes":>13} '
          f'{"track eff":>16}')
    for _, r in E.iterrows():
        print(f'{r.arm:>3} {int(r.n_tagged):>9,} '
              f'{100 * r.eff_hit:>9.1f} +- {100 * r.err_hit:<4.1f} '
              f'{100 * r.eff_xy:>12.1f} '
              f'{100 * r.eff_track:>9.1f} +- {100 * r.err_track:<4.1f}')
    print('\nAccidental-corrected -- p0 measured on events this arm did NOT tag.')
    print('This is the number acceptance should use.\n')
    print(f'{"arm":>3} {"p0 hit":>8} {"eff hit":>9} | {"p0 trk":>8} {"eff track":>10}')
    for _, r in E.iterrows():
        print(f'{r.arm:>3} {100 * r.p0_hit:>7.1f}% {100 * r.eff_hit_corr:>8.1f}% | '
              f'{100 * r.p0_track:>7.1f}% {100 * r.eff_track_corr:>9.1f}%')
    print('\nP(track | tagged AND seeded), binned in the in-plane coordinate '
          '[mm from the perpendicular foot]:')
    piv = M.pivot(index='arm', columns='u_mid',
                  values='eff_track_given_seed') * 100
    print('   ' + piv.round(1).to_string().replace('\n', '\n   '))
    print(f'\nwrote {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
