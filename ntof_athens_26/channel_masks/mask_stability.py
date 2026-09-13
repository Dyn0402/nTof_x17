#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_stability.py -- is a channel classified hot in one run hot in the others?

THE QUESTION, from ``HANDOFF_CHANNEL_MASKS.md`` Sec. 5.  That handoff found that
no reconstruction in the campaign masks a single dead or hot channel, that
chamber D is half noise by hit count and C is clean, and that all of it was
measured on **run_145 alone**.  Its steps 1 and 2 are: run the classifier over
the whole campaign, and **check stability first** -- "if the classification moves
run to run it is measuring occupancy, not hardware".

This answers step 2 and as much of step 1 as the local tree allows.

WHY IT IS FAST, AND WHAT IT LEAVES OUT.  ``noisy_channels.py`` takes ~15 min for
three sub-runs, almost all of it in ``noise.flag_noise``, which the *shape*
(``noisy``) pass needs.  The ``dead``/``hot`` classification does not: by that
module's own docstring ``occupancy()`` "always used every raw hit".  So this
skips the noise flagging and the clustering entirely and reads only the four
columns a bincount needs -- ~6 s per sub-run instead of ~5 min.  It therefore
reproduces ``dead``, ``hot``, ``n_hot_bands`` and ``hits_in_hot`` **exactly**
(checked against ``noisy_channels_summary_run_145.csv``, all eight planes) and
does **not** produce ``noisy``.  That is the intended trade: stability is a
question about many runs, and ``noisy`` is the class the handoff itself calls
"weaker evidence".

WHAT IT FOUND.

  1. ``THE CLASSIFIER MEASURES HARDWARE.``  It independently recovers the
     A-x connector-8 fault that ``CLAUDE.md`` records from the DAQ side:
     43 of channels 448-511 classified dead in run_79, **0 of 64 in every
     post-access run**.  Nothing in this module knows that connector exists.
  2. ``D IS HALF NOISE IN EVERY RUN, not just run_145.``  ``hits_in_hot`` on
     D-x is 45-56 % and on D-y 33-53 % across five runs spanning six weeks, and
     the hot channel SET has Jaccard 0.52-0.89 against run_145.  The handoff's
     headline generalises.
  3. ``C IS CLEAN ONLY AFTER THE 27 JULY ACCESS.``  **This is new and it
     contradicts a load-bearing sentence of the handoff.**  In run_79 -- the
     first long production run -- C-x carries 15.8 % of its hits in hot channels
     and C-y 15.0 %, and B-x carries 32.7 %.  Post-access all three are ~0.
     Measured in every one of run_79's 11 sub-runs separately, so it is not an
     artefact of pooling.  "Chamber C is clean" is true of run_86 onward and
     false of run_79.

WHAT IS BLOCKED.  Step 1 proper -- the whole campaign -- needs ``combined_hits``
that is not staged locally.  Of the 289 campaign sub-runs with a
``combined_hits_root`` the local tree has **21, in 5 runs**; the other ``combined_hits_root`` directories exist and are empty.
``missing_runs`` reports exactly which, so the staging job is one rsync away
rather than a rediscovery.  Everything above is therefore 5 runs, not 36.

    python ntof_athens_26/channel_masks/mask_stability.py
    X17_ROOT=D:/x17 X17_BEAM_JULY=D:/x17/beam_july python .../mask_stability.py
    ... --per-subrun run_79      # the pooling check, one sub-run at a time
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                          # noqa: E402
from sept26_prelim_analysis import noisy_channels as nc           # noqa: E402

SCHEMA = 'ntof_athens_26/channel_masks/1'

ARMS = ('A', 'B', 'C', 'D')
PLANES = ('x', 'y')
N_STRIPS = 512

#: The columns a raw-occupancy bincount needs.  ``eventId`` and ``time`` are
#: here only for ``drop_duplicates``, which production also does; amplitude and
#: significance are not read at all, which is most of the speed.
HIT_COLS = ['eventId', 'feu', 'channel', 'time']

#: ``CLAUDE.md``: chamber A's x-view connector 8 was electrically disconnected
#: from 22 July to the 27 July access -- every sub-run of run_79 -- and live
#: again from run_83.  Used as a blind validation of the classifier: it is not
#: told this, and it has to find it.
CONNECTOR_8 = (448, 512)
PRE_ACCESS_RUN = 'run_79'


# --------------------------------------------------------------------------- #
# Occupancy
# --------------------------------------------------------------------------- #
def staged_subruns(run: str) -> list:
    """Sub-runs of ``run`` whose ``combined_hits_root`` actually holds files.

    The directory existing is not the same as the data being there: on this
    machine 268 of 289 campaign sub-runs have an EMPTY ``combined_hits_root``.
    Globbing for the directory and trusting it is how a campaign pass silently
    returns 5 runs' worth of answer to a 36-run question.
    """
    base = paths.root('runs') / run
    if not base.is_dir():
        return []
    return sorted(q.name for q in base.iterdir()
                  if q.is_dir() and any((q / 'combined_hits_root').glob('*_datrun_*.root')))


def campaign_runs() -> list:
    """Every ``run_N`` under the runs root, in run-number order."""
    base = paths.root('runs')
    return sorted([p.name for p in base.iterdir()
                   if p.name.startswith('run_') and p.name[4:].isdigit()],
                  key=lambda s: int(s[4:]))


def occupancy_by_run(run: str, subruns=None, verbose: bool = True) -> pd.DataFrame:
    """Raw per-channel hit counts for one run, pooled over its staged sub-runs.

    Long-form ``(arm, plane, channel, occ)`` plus the sub-run and event counts
    the pooling rests on, because a per-run occupancy without them cannot be
    compared to another run's.
    """
    import uproot
    from ntof_tracking.reco import io as rio

    subs = list(subruns) if subruns is not None else staged_subruns(run)
    if not subs:
        return pd.DataFrame()
    lut = rio.build_channel_lut(rio.load_run_config(run))
    acc, n_ev, n_hits, t0 = {}, 0, 0, time.time()
    for sub in subs:
        files = [str(p) for p in
                 sorted((paths.root('runs') / run / sub / 'combined_hits_root')
                        .glob('*.root')) if '_datrun_' in p.name]
        if not files:
            continue
        df = uproot.concatenate([f'{f}:hits' for f in files],
                                expressions=HIT_COLS, library='pd')
        df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
        df = rio.drop_unphysical(df, tag=f'{run}/{sub}', verbose=False)
        n_ev += int(df.eventId.nunique())
        df = df.merge(lut, on=['feu', 'channel'], how='inner')
        n_hits += len(df)
        for (det, pl), g in df.groupby(['det', 'plane'], sort=False):
            key = (str(det).replace('mx17_', ''), str(pl))
            acc[key] = acc.get(key, 0) + np.bincount(
                g.channel.to_numpy(int), minlength=N_STRIPS)[:N_STRIPS]
    if verbose:
        print(f'  {run}: {len(subs):3d} subruns  {n_hits:11,d} hits  '
              f'{n_ev:9,d} events  {time.time() - t0:6.1f}s', flush=True)
    if not acc:
        return pd.DataFrame()
    return pd.concat([
        pd.DataFrame({'run': run, 'arm': arm, 'plane': pl,
                      'channel': np.arange(N_STRIPS), 'occ': b.astype(float),
                      'n_subruns': len(subs), 'n_events': n_ev})
        for (arm, pl), b in acc.items()], ignore_index=True)


def classify(occ_long: pd.DataFrame) -> tuple:
    """``(summary, per-channel)`` from ``noisy_channels.classify_occupancy``.

    The classifier itself is imported, not reimplemented -- the point of this
    module is running it on more runs, and a second copy of the thresholds would
    be a second thing to keep in step.
    """
    srows, crows = [], []
    for (run, arm, pl), g in occ_long.groupby(['run', 'arm', 'plane']):
        g = g.sort_values('channel')
        occ = g.occ.to_numpy(float)
        cls, med = nc.classify_occupancy(occ)
        hot, dead = cls == 'hot', cls == 'dead'
        srows.append(dict(
            run=run, arm=arm, plane=pl,
            n_subruns=int(g.n_subruns.iloc[0]), n_events=int(g.n_events.iloc[0]),
            frac_dead=float(dead.mean()), frac_hot=float(hot.mean()),
            n_hot_bands=int(np.sum(np.diff(np.r_[0, hot.astype(int), 0]) == 1)),
            hits_in_hot=float(occ[hot].sum() / occ.sum()) if occ.sum() else 0.0))
        crows.append(pd.DataFrame({'run': run, 'arm': arm, 'plane': pl,
                                   'channel': np.arange(N_STRIPS), 'occ': occ,
                                   'local_median_occ': med, 'cls': cls}))
    return pd.DataFrame(srows), pd.concat(crows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Stability -- the handoff's step 2
# --------------------------------------------------------------------------- #
def stability(per_ch: pd.DataFrame, ref_run: str = 'run_145',
              kind: str = 'hot', exclude=(PRE_ACCESS_RUN,)) -> pd.DataFrame:
    """Jaccard overlap of one run's ``kind`` set with ``ref_run``'s, per plane.

    ``exclude`` drops the pre-access run by default: the 27 July access changed
    the hardware, so comparing across it measures the access, not the stability
    of the classification.  ``run_79_vs_post`` in the report is that comparison
    made deliberately instead of by accident.

    A Jaccard near 1 means the same physical channels; near 0 with both sets
    non-empty means the classifier is tracking occupancy rather than hardware,
    which is the failure mode the handoff asked to rule out.  **Both sets empty
    is not an answer** and comes back NaN with ``n_ref``/``n_other`` = 0, not 0.0.
    """
    rows = []
    d = per_ch[~per_ch.run.isin(exclude)]
    for (arm, pl), g in d.groupby(['arm', 'plane']):
        ref = set(g.loc[(g.run == ref_run) & (g.cls == kind), 'channel'])
        for run, gr in g.groupby('run'):
            if run == ref_run:
                continue
            other = set(gr.loc[gr.cls == kind, 'channel'])
            union = ref | other
            rows.append(dict(arm=arm, plane=pl, run=run, ref_run=ref_run, kind=kind,
                             n_ref=len(ref), n_other=len(other),
                             n_shared=len(ref & other),
                             jaccard=len(ref & other) / len(union) if union else np.nan))
    return pd.DataFrame(rows)


def access_step(summary: pd.DataFrame) -> pd.DataFrame:
    """Pre-access (run_79) against the post-access mean, per plane.

    Written because the pre/post split turned out to matter and the handoff does
    not mention it.  ``hits_in_hot`` is the column to read: it is what actually
    reaches a fit, and a plane can have few hot channels carrying most of the
    plane's hits.
    """
    pre = summary[summary.run == PRE_ACCESS_RUN].set_index(['arm', 'plane'])
    post = (summary[summary.run != PRE_ACCESS_RUN]
            .groupby(['arm', 'plane'])[['frac_hot', 'hits_in_hot', 'frac_dead']]
            .mean())
    j = pre[['frac_hot', 'hits_in_hot', 'frac_dead']].join(
        post, rsuffix='_post', how='outer')
    return j.reset_index().rename(columns={
        'frac_hot': 'frac_hot_pre', 'hits_in_hot': 'hits_in_hot_pre',
        'frac_dead': 'frac_dead_pre'})


def connector8_validation(per_ch: pd.DataFrame) -> pd.DataFrame:
    """The blind test: does the classifier find A-x connector 8 dead in run_79?

    ``CLAUDE.md`` records the fault from the DAQ side (channels 448-511 of FEU 3
    electrically disconnected until the 27 July access).  Nothing in this module
    is told that.  If the classification is measuring hardware it must find it
    in run_79 and must NOT find it afterwards -- and that is a stronger check on
    the thresholds than any amount of self-consistency across runs.
    """
    lo, hi = CONNECTOR_8
    rows = []
    for run, g in per_ch[(per_ch.arm == 'A') & (per_ch.plane == 'x')].groupby('run'):
        g = g.sort_values('channel')
        occ, cls = g.occ.to_numpy(float), g.cls.to_numpy()
        blk, rest = occ[lo:hi], np.r_[occ[:lo], occ[hi:]]
        rows.append(dict(run=run, mean_occ_connector8=float(blk.mean()),
                         mean_occ_rest=float(rest.mean()),
                         ratio=float(blk.mean() / rest.mean()) if rest.mean() else np.nan,
                         n_zero_occupancy=int((blk == 0).sum()),
                         n_classified_dead=int((cls[lo:hi] == 'dead').sum()),
                         n_channels=hi - lo))
    return pd.DataFrame(rows)


def missing_runs() -> pd.DataFrame:
    """What step 1 is blocked on: per run, sub-runs present vs sub-runs staged.

    A directory listing, not a measurement -- but it is the difference between
    "the campaign pass says D is stable" and "the campaign pass ran on 5 of 36
    runs", and it belongs in the product rather than in someone's memory.
    """
    rows = []
    for run in campaign_runs():
        base = paths.root('runs') / run
        subs = [q for q in base.iterdir() if q.is_dir()]
        withdir = [q for q in subs if (q / 'combined_hits_root').is_dir()]
        staged = staged_subruns(run)
        if not withdir:
            continue
        rows.append(dict(run=run, n_subruns=len(withdir),
                         n_staged=len(staged),
                         staged=len(staged) > 0,
                         missing=len(withdir) - len(staged)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def per_subrun_check(run: str) -> pd.DataFrame:
    """One sub-run at a time -- is a per-run result an artefact of pooling?

    Pooling 11 sub-runs shrinks the Poisson error on every channel, so a
    threshold test can cross on statistics alone.  Run_79's B and C hot excess
    is the claim this was written for; it survives, appearing in every sub-run
    separately.
    """
    rows = []
    for sub in staged_subruns(run):
        occ = occupancy_by_run(run, [sub], verbose=False)
        if occ.empty:
            continue
        summ, _ = classify(occ)
        rows.append(summ.assign(subrun=sub))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='*', default=None,
                    help='runs to classify (default: every staged run)')
    ap.add_argument('--ref', default='run_145', help='reference run for stability')
    ap.add_argument('--per-subrun', default=None, metavar='RUN',
                    help='also classify RUN one sub-run at a time (pooling check)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else paths.out('channel_masks')
    out.mkdir(parents=True, exist_ok=True)

    miss = missing_runs()
    runs = a.runs if a.runs else list(miss.loc[miss.staged, 'run'])
    print(f'[masks] {len(runs)} of {len(miss)} campaign runs are staged locally: '
          f'{", ".join(runs)}')
    print(f'[masks] {int(miss.n_staged.sum())} of {int(miss.n_subruns.sum())} '
          f'sub-runs have combined_hits on this machine')

    occ = pd.concat([occupancy_by_run(r) for r in runs], ignore_index=True)
    summary, per_ch = classify(occ)
    tables = {
        'summary': summary,
        'stability_hot': stability(per_ch, a.ref, 'hot'),
        'stability_dead': stability(per_ch, a.ref, 'dead'),
        'access_step': access_step(summary),
        'connector8_validation': connector8_validation(per_ch),
        'missing_runs': miss,
    }
    if a.per_subrun:
        tables['per_subrun'] = per_subrun_check(a.per_subrun)

    per_ch.to_csv(out / 'channels_by_run.csv', index=False)
    for name, t in tables.items():
        t.to_csv(out / f'{name}.csv', index=False)
        print(f'\n=== {name}')
        print(t.to_string(index=False))

    (out / 'channel_masks.meta.json').write_text(json.dumps(dict(
        schema=SCHEMA, runs=runs, ref_run=a.ref,
        n_runs_staged=int(miss.staged.sum()), n_runs_total=int(len(miss)),
        n_subruns_staged=int(miss.n_staged.sum()),
        n_subruns_total=int(miss.n_subruns.sum()),
        hot_factor=nc.HOT_FACTOR, dead_thresh=nc.DEAD_THRESH,
        window_strips=nc.WINDOW_STRIPS,
        note='dead/hot only; the shape (noisy) pass is not run here',
    ), indent=1))
    print(f'\n[masks] wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
