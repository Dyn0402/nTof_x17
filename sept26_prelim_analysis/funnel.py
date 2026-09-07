#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
funnel.py -- the reconstruction funnel, per arm, from trigger to confirmed track.

One question: of everything the DAQ wrote, how much survives each step, and what
does n_TOF say about what survived?

The funnel is deliberately built on the FULL waveform pass -- every trigger of
every tag, no stage-2 allowlist, no prescale -- so that no number here inherits
a hits-based selection.  ``combined_hits`` enters at exactly one point, the
seeder (`wft/seed.py`), and only as a set of CHANNELS: which strips carry
charge.  No hit time crosses into the geometry.  That boundary is the whole
point of RECONSTRUCTION_BASIS.md, and the funnel reports the seeder's own
acceptance so the cost of it is visible rather than assumed.

Stages, per arm:

  triggers        every DREAM trigger in the sub-run (stage-1 census)
  seeded          the seeder found >=3 clustered strips on at least one plane
  plane fits      candidate clusters offered to the forward model (up to 3/plane)
  x_ok / y_ok     a plane fit converged and passed the plausibility window
  pairings        (x, y) candidate combinations the pairing considered
  gated tracks    pairings that passed the 3D gate -- the track sample

Then, for each event with >=1 gated track, what the SAME ARM's n_TOF layers say:

  WAL             scintillator wall  (det 0-3),  96.4 mm past the strip plane
  PSS             plastic bars       (det 4-7), ~187 mm past the strip plane
  LIQ             liquid cells       (det 8-11)

reported as an exclusive partition (wall only / plastic only / both / neither)
because "both" is the interesting one: two independent layers at different
depths agreeing is far stronger evidence of a real particle than either alone.

The strongest column is POINTING coincidence: not merely "the arm's wall fired"
but "the track extrapolates to the wall segment and the plastic bar that
actually fired".  That one cannot be faked by an unrelated hit in the same arm,
and it is computed with ntof_tracking.run145_target_imaging's own geometry so
the report and the calibration cannot drift apart.

Every rate is quoted against a stated denominator; a fraction whose denominator
is a selected sample says so in its own column name.

    python -m sept26_prelim_analysis.funnel --run run_145 \
        --subruns stat090_0000,stat090_0001
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
SCHEMA = 'sept26_prelim/funnel/1'

#: The full waveform pass this reads: the NESTED, merged tree, one pair of
#: tables per (sub-run, arm).  ``merge_fullpass.py`` puts the flat August CERN
#: output into this shape, and ``wft_beam`` writes it directly for sub-runs
#: reconstructed here, so one reader covers both.
#:
#: NOT the stage-2 filtered output: the funnel is the one product that must be
#: free of the hits-based selection.
FULLPASS = os.environ.get(
    'X17_FULLPASS', '/media/dylan/data/x17/sept26_prelim/fullpass/run_145')

#: slim n_TOF detector families, by ``det`` code (config.SCINT_TREES order).
FAMILY = {**{i: 'WAL' for i in range(0, 4)},
          **{i: 'PSS' for i in range(4, 8)},
          **{i: 'LIQ' for i in range(8, 12)}}
ARM_OF_DET = {i: ARMS[i % 4] for i in range(12)}

#: in-time window on the slim ``dt_ns`` (the trigger peak sits at ~[-30, -2]).
#: Same window run145_target_imaging uses, so the two agree by construction.
DT_WINDOW = (-100.0, 60.0)


def read_fullpass(run: str, subruns) -> tuple:
    """(events, candidates) for every arm of ``subruns``, from the merged tree.

    ``events`` is one row per (arm, sub-run, event) -- the wide per-event
    table.  ``candidates`` is one row per plane candidate offered to the fit.
    Raises if a sub-run is absent rather than reporting it as a sub-run with
    zero tracks, which is the failure this whole package is written against.
    """
    ev, cand = [], []
    for sub in subruns:
        for arm in ARMS:
            d = os.path.join(FULLPASS, sub, f'mx17_{arm}')
            pe = os.path.join(d, 'events_prelim.parquet')
            pc = os.path.join(d, 'events_prelim.candidates.parquet')
            if not os.path.exists(pe):
                raise FileNotFoundError(
                    f'no full pass for {arm}/{sub} under {FULLPASS}\n'
                    f'  reconstruct it, or run merge_fullpass.py if it exists '
                    f'in the flat per-tag layout. Do NOT drop it from '
                    f'--subruns silently: a missing sub-run and an empty one '
                    f'are different things.')
            e = pd.read_parquet(pe, columns=['event_id', 'n_hits', 'spark',
                                             'x_ok', 'y_ok', 'n_tracks'])
            ev.append(e.assign(arm=arm, subrun=sub))
            if os.path.exists(pc):
                c = pd.read_parquet(pc, columns=['event_id', 'plane', 'rank',
                                                 'track_id', 'track_gated',
                                                 'quality_ok', 'plausible'])
                cand.append(c.assign(arm=arm, subrun=sub))
    if not cand:
        raise FileNotFoundError(
            f'no candidate tables under {FULLPASS}; the pairing count needs them')
    return (pd.concat(ev, ignore_index=True),
            pd.concat(cand, ignore_index=True))


def census(run: str, subruns) -> pd.DataFrame:
    """Stage-1 trigger counts per sub-run -- the funnel's denominator.

    The seeded-event table cannot supply it: it has no row for a trigger the
    seeder rejected, which is exactly the number the first stage measures.
    """
    rows = []
    for sub in subruns:
        p = paths.require(paths.out('stage1') / f'census_{run}_{sub}.csv',
                          f'stage-1 census for {sub}')
        c = pd.read_csv(p)
        tot = c.loc[c.cls == '(total)', 'n']
        if tot.empty:
            raise ValueError(f'{p} has no (total) row')
        rows.append(dict(subrun=sub, n_triggers=int(tot.iloc[0])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the n_TOF
def slim_path(run: str, sub: str) -> str:
    d = os.path.join(str(paths.root('runs')), run, sub, 'ntof_hits')
    hits = sorted(glob.glob(os.path.join(d, 'ntof_hits_*.root')))
    if not hits:
        raise FileNotFoundError(f'no slim n_TOF file under {d}')
    if len(hits) > 1:
        raise ValueError(f'{len(hits)} slim files under {d}; expected 1')
    return hits[0]


def layer_flags(run: str, subruns) -> pd.DataFrame:
    """(subrun, event_id, arm) -> WAL / PSS / LIQ fired in time, same arm.

    ``is_control`` hits are the 100 ms-shifted accidental sample and are
    excluded: they are the measurement of the accidental rate, not signal.
    """
    import uproot
    out = []
    for sub in subruns:
        t = uproot.open(slim_path(run, sub))['hits']
        a = t.arrays(['eventId', 'det', 'dt_ns', 'is_control'], library='np')
        m = ((a['is_control'] == 0)
             & (a['dt_ns'] >= DT_WINDOW[0]) & (a['dt_ns'] <= DT_WINDOW[1]))
        d = pd.DataFrame({'event_id': a['eventId'][m], 'det': a['det'][m]})
        d['arm'] = d.det.map(ARM_OF_DET)
        d['fam'] = d.det.map(FAMILY)
        d['subrun'] = sub
        out.append(d)
    s = pd.concat(out, ignore_index=True)
    fl = (s.assign(one=1)
            .pivot_table(index=['subrun', 'event_id', 'arm'], columns='fam',
                         values='one', aggfunc='max')
            .reindex(columns=['WAL', 'PSS', 'LIQ'])
            .fillna(0).astype(bool).reset_index())
    return fl


def pointing_flags(run: str, sub: str, arm: str, merged_dir: str) -> pd.DataFrame:
    """(event_id, pointing) -- the track extrapolates to the wall segment AND
    the plastic bar that actually fired.

    Delegates to run145_target_imaging so the funnel and the angle calibration
    share one geometry.  Returns an empty frame (not a crash) when the merged
    per-arm table is absent, since pointing is an extra column, not the funnel.

    ``info['n_predictable']`` is the honest denominator: a track extrapolating
    off the edge of the wall is an UNTESTABLE coincidence, not a failed one.
    """
    empty = pd.DataFrame(columns=['subrun', 'event_id', 'arm', 'pointing'])
    from ntof_tracking import run145_target_imaging as TI
    p = os.path.join(merged_dir, sub, f'mx17_{arm}', 'events_prelim.parquet')
    if not os.path.exists(p):
        return empty, dict(n_predictable=0, n_coincident=0)
    df = pd.read_parquet(p)
    sel = (df['x_ok'].to_numpy() & df['y_ok'].to_numpy()
           & (df['n_tracks'].to_numpy() > 0))
    mask, info = TI.pointing_coincidence(slim_path(run, sub), arm, df, sel,
                                         foot_x=TI.PINWHEEL[arm])
    return pd.DataFrame({'subrun': sub, 'event_id': df['event_id'].to_numpy(),
                         'arm': arm, 'pointing': np.asarray(mask, bool)}), info


# ------------------------------------------------------------------ the funnel
def build(run: str, subruns, merged_dir: str = None) -> dict:
    """Every table the report needs, as one dict of DataFrames."""
    cen = census(run, subruns)
    n_trig = int(cen.n_triggers.sum())
    E, C = read_fullpass(run, subruns)
    FL = layer_flags(run, subruns)

    E = E.merge(FL, on=['subrun', 'event_id', 'arm'], how='left')
    for c in ('WAL', 'PSS', 'LIQ'):
        E[c] = E[c].fillna(False).astype(bool)

    pf, npred = [], {a: 0 for a in ARMS}
    if merged_dir:
        for sub in subruns:
            for arm in ARMS:
                d, info = pointing_flags(run, sub, arm, merged_dir)
                pf.append(d)
                npred[arm] += int(info.get('n_predictable', 0))
    if pf:
        P = pd.concat(pf, ignore_index=True)
        E = E.merge(P, on=['subrun', 'event_id', 'arm'], how='left')
    if 'pointing' not in E:
        E['pointing'] = np.nan
    E['pointing'] = E['pointing'].fillna(False).astype(bool)

    rows = []
    for arm in ARMS:
        e = E[E.arm == arm]
        c = C[C.arm == arm]
        trk = e[e.n_tracks > 0]
        w, p = trk.WAL.to_numpy(), trk.PSS.to_numpy()
        # a "pairing" is an (x, y) candidate combination the pairing stage
        # considered; a gated track is one that survived.  n_tracks is the
        # authority on the latter -- the candidate table's track_id marks
        # every pairing, gated or not.
        # event_id is global within a sub-run (verified: tags carve disjoint
        # ranges), so the tag is not part of the key.
        pair = c[c.track_id >= 0].groupby(
            ['subrun', 'event_id', 'track_id']).ngroups
        rows.append(dict(
            arm=arm,
            n_triggers=n_trig,
            n_seeded=len(e),
            n_plane_cand=len(c),
            n_plane_cand_x=int((c.plane == 'x').sum()),
            n_plane_cand_y=int((c.plane == 'y').sum()),
            n_x_ok=int(e.x_ok.sum()),
            n_y_ok=int(e.y_ok.sum()),
            n_xy_ok=int((e.x_ok & e.y_ok).sum()),
            n_pairings=pair,
            n_track_events=len(trk),
            n_tracks=int(e.n_tracks.sum()),
            n_ev_2track=int((e.n_tracks >= 2).sum()),
            wal_only=int((w & ~p).sum()),
            pss_only=int((p & ~w).sum()),
            wal_and_pss=int((w & p).sum()),
            neither=int((~w & ~p).sum()),
            liq=int(trk.LIQ.sum()),
            pointing=int(trk.pointing.sum()),
            n_predictable=int(npred.get(arm, 0)),
            # the control: seeded, but the reconstruction found no track.
            # Same trigger population, same n_TOF -- the difference between
            # this and the row above is what the tracking is worth.
            ctrl_n=int((e.n_tracks == 0).sum()),
            ctrl_wal_and_pss=int((e.WAL & e.PSS & (e.n_tracks == 0)).sum()),
        ))
    F = pd.DataFrame(rows)
    F['seed_eff'] = F.n_seeded / F.n_triggers
    F['track_eff'] = F.n_track_events / F.n_seeded
    F['coinc_frac'] = F.wal_and_pss / F.n_track_events
    F['ctrl_coinc_frac'] = F.ctrl_wal_and_pss / F.ctrl_n
    F['lift'] = F.coinc_frac / F.ctrl_coinc_frac
    # against the predictable denominator, not every tracked event: a track
    # aimed off the wall cannot confirm or refute anything.
    F['pointing_frac'] = F.pointing / F.n_predictable.replace(0, np.nan)

    return dict(funnel=F, census=cen, events=E,
                meta=dict(schema=SCHEMA, run=run, subruns=list(subruns),
                          n_triggers=n_trig, fullpass=FULLPASS,
                          dt_window=list(DT_WINDOW),
                          allowlist='none -- full pass, no prescale',
                          hits_role='seeder channel selection only '
                                    '(wft/seed.py); no hit time used'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns', default='stat090_0000,stat090_0001')
    ap.add_argument('--merged', default=str(paths.out('fullpass') / 'run_145'),
                    help='per-arm merged events_prelim.parquet tree, for the '
                         'pointing column. Omit to skip pointing.')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    subruns = [s for s in a.subruns.split(',') if s]
    r = build(a.run, subruns, merged_dir=a.merged or None)
    od = paths.out('funnel') if a.out is None else a.out
    os.makedirs(od, exist_ok=True)

    F = r['funnel']
    F.to_csv(os.path.join(od, f'funnel_{a.run}.csv'), index=False)
    r['events'].to_parquet(os.path.join(od, f'events_{a.run}.parquet'),
                           index=False)
    json.dump(r['meta'], open(os.path.join(od, f'funnel_{a.run}.meta.json'), 'w'),
              indent=1)

    show = ['arm', 'n_triggers', 'n_seeded', 'n_plane_cand', 'n_xy_ok',
            'n_pairings', 'n_tracks', 'wal_and_pss', 'pointing']
    print(F[show].to_string(index=False))
    print()
    print(F[['arm', 'seed_eff', 'track_eff', 'coinc_frac', 'ctrl_coinc_frac',
             'lift', 'pointing_frac']].round(3).to_string(index=False))
    print(f'\nwrote {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
