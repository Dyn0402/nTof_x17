#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
allowlist.py -- turn a stage-1 class table into the stage-2 event-id allowlist.

Stage 2 runs the full waveform-first reconstruction (`ntof_tracking/wft_beam.py`)
on condor, one job per **(arm, file tag)**.  Reconstructing every trigger in
every arm is ~50x more compute than the physics needs, so each job is handed a
list of event ids and fits only those.  This module builds that list.

The unit of selection is the **(event, arm) pair**, not the event.  An `INTER`
event needs its two lit arms fitted; the other two chambers have nothing in
them and fitting them buys nothing.  That scoping is most of the saving -- see
`plan()`.

    per class            arms reconstructed
    -------------------  ---------------------------------------------------
    INTER                the 2 lit arms
    INTRA                the 1 lit arm
    IMPLIED              the lit arm PLUS every arm with an n_TOF coincidence
                         and no track -- forced, because "was there a hint of
                         a track the hit-level finder missed?" is the whole
                         question IMPLIED exists to ask (PLAN.md stage 2)
    control (prescaled)  all four arms.  A control that only fitted the arms
                         stage 1 already liked could not measure what stage 1
                         missed, which is its only job.

The control is drawn from `SINGLE`, `BUSY` and `NONE` at **per-class** rates,
not one global rate.  Those classes differ by two orders of magnitude in
population and by much more in how likely they are to be hiding a pair, so a
single rate would either bankrupt us on `NONE` or leave `BUSY` with single
digits.  Defaults in :data:`CONTROL_PRESCALE`; tune them, do not remove them.

**The draw is a hash, not an RNG.**  ``keep = blake2b(salt|run|subrun|event) <
rate``.  It is reproducible on any machine with no stored seed state, it is
stable if one tag is rebuilt on its own, and adding a run to the sample does
not repartition the runs already drawn.  The salt is recorded in the output so
a future rebuild can prove it drew the same events.

Outputs, per (run, sub-run), into ``<out>/stage2/``:

  ``allowlist_<run>_<subrun>.json``     what the condor job reads: arm -> tag
                                        -> [event ids], plus the policy header
  ``allowlist_<run>_<subrun>.parquet``  one row per (arm, tag, event, class,
                                        reason) -- the join key stage 3 uses to
                                        get a reconstructed track back to *why*
                                        its event was selected

Usage:
    python -m sept26_prelim_analysis.allowlist --run run_145 --subrun stat090_0000
    python -m sept26_prelim_analysis.allowlist --plan-only      # cost, no write
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from . import paths
except ImportError:                                    # run as a script
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sept26_prelim_analysis import paths

ARMS = ('A', 'B', 'C', 'D')

#: Classes reconstructed in full.  Every event of these is selected; what
#: varies is which arms (see module docstring).
SIGNAL_CLASSES = ('INTER', 'INTRA', 'IMPLIED')

#: Fraction of each background class drawn into the control sample, all four
#: arms each.  These are the numbers to tune when the budget moves:
#:
#:  SINGLE 5 %   one arm lit is where a missed second track most plausibly
#:               hides, so this is the class the efficiency measurement leans
#:               on hardest.  5 % of run_145/0000 is ~480 events.
#:  BUSY  100 %  all of it -- see below.
#:  NONE    1 %  79 % of all triggers.  Its job is to bound the "both arms
#:               missed" rate, which needs a number, not precision.
#:
#: **Why BUSY is taken whole.**  It looks like the expensive class and is not.
#: Measured on run_145/stat090_0000 against the full August pass: a `BUSY`
#: event's arms are flooded, not crowded -- 97 % of them carry more than 120
#: clean strips and the median arm that fails to seed has **493**, roughly half
#: the chamber's channels.  Those planes are over `wft_beam.BUSY_PLANE_HITS`
#: (150 hits) and the seeder drops them, correctly: they are discharges and
#: flashes, not multi-track events.  Seeding is 100 % (16/16) below 200 strips
#: and 8.7 % above 400.
#:
#: So only ~17 % of `BUSY` arm-events cost anything to fit, and a 10 % prescale
#: on top of that left **44** of them -- not a measurement of anything.  Taking
#: the class whole costs ~12 % more budget (it is 1.2 % of triggers) and buys
#: both a real number on the reconstructable part and a measured count of the
#: flooded part.  No prescale recovers the flooded part; that information is
#: not in the data.
CONTROL_PRESCALE = {'SINGLE': 0.05, 'BUSY': 1.00, 'NONE': 0.01}

#: Changing this redraws the whole control sample.  Bump it only deliberately,
#: and say so in STATUS.md -- a silently redrawn control is a control whose
#: efficiency cannot be compared to the one before it.
PRESCALE_SALT = 'sept26-stage2-v1'

SCHEMA = 'sept26_prelim/allowlist/1'


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #
def _draw(run: str, subrun: str, event_id: int, salt: str = PRESCALE_SALT) -> float:
    """A stable uniform in [0, 1) for one trigger.

    Deterministic across machines and python versions (blake2b is specified,
    ``hash()`` is not), and independent per event, so a rate change moves only
    the events near the old threshold rather than reshuffling everything.
    """
    h = hashlib.blake2b(f'{salt}|{run}|{subrun}|{event_id}'.encode(),
                        digest_size=8).digest()
    return int.from_bytes(h, 'big') / 2 ** 64


def _draw_series(df: pd.DataFrame, salt: str = PRESCALE_SALT) -> pd.Series:
    return pd.Series(
        [_draw(r, s, e, salt) for r, s, e in
         zip(df['run'], df['subrun'], df['eventId'])],
        index=df.index, dtype=float)


# --------------------------------------------------------------------------- #
# Arm scoping
# --------------------------------------------------------------------------- #
def arms_for_row(row) -> list[tuple[str, str]]:
    """Which arms to reconstruct for one trigger, and why.

    Returns ``[(arm, reason), ...]``.  ``reason`` is carried all the way to the
    parquet so a track's provenance says whether its arm was fitted because
    stage 1 saw a track there (``lit``), because n_TOF said something crossed
    it and stage 1 saw nothing (``forced_silent``), or because the event was
    drawn into the control (``control``).
    """
    cls = row['cls']
    lit = [a for a in str(row.get('arms_lit') or '') if a in ARMS]

    if cls in ('INTER', 'INTRA'):
        return [(a, 'lit') for a in lit]

    if cls == 'IMPLIED':
        out = [(a, 'lit') for a in lit]
        # The point of IMPLIED: n_TOF's scintillators say a particle crossed a
        # chamber that the hit-level finder found no track in.  Fit it anyway.
        for a in ARMS:
            if a not in lit and int(row.get(f'coinc_{a}', 0) or 0) > 0:
                out.append((a, 'forced_silent'))
        return out

    return [(a, 'control') for a in ARMS]           # SINGLE / BUSY / NONE


def select(cand: pd.DataFrame,
           control_prescale: dict[str, float] | None = None,
           salt: str = PRESCALE_SALT) -> pd.DataFrame:
    """Stage-1 candidates -> one row per (arm, tag, event) to reconstruct.

    Columns: ``run, subrun, tag, eventId, arm, cls, reason, draw``.
    """
    control_prescale = CONTROL_PRESCALE if control_prescale is None else control_prescale
    missing = {'cls', 'arms_lit', 'eventId', 'tag', 'run', 'subrun'} - set(cand.columns)
    if missing:
        raise KeyError(f'stage-1 table is missing {sorted(missing)} -- this is '
                       f'not a candidates_*.parquet from candidate_filter.py')

    draw = _draw_series(cand, salt)
    keep_signal = cand['cls'].isin(SIGNAL_CLASSES)
    rate = cand['cls'].map(control_prescale).fillna(0.0)
    keep_control = ~keep_signal & (draw < rate)
    sel = cand[keep_signal | keep_control].copy()
    sel['draw'] = draw[keep_signal | keep_control]

    rows = []
    for row in sel.itertuples(index=False):
        d = row._asdict()
        for arm, reason in arms_for_row(d):
            rows.append((d['run'], d['subrun'], d['tag'], int(d['eventId']),
                         arm, d['cls'], reason, round(float(d['draw']), 6)))
    out = pd.DataFrame(rows, columns=['run', 'subrun', 'tag', 'eventId', 'arm',
                                      'cls', 'reason', 'draw'])
    if len(out) and out.duplicated(['run', 'subrun', 'eventId', 'arm']).any():
        raise AssertionError('an (event, arm) was selected twice -- '
                             'arms_for_row must return each arm at most once')
    return out.sort_values(['arm', 'tag', 'eventId']).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def plan(cand: pd.DataFrame, sel: pd.DataFrame,
         core_h_per_1e3_arm_events: float | None = None) -> pd.DataFrame:
    """What the selection costs, per class, in (arm, event) fits.

    The denominator is ``n_triggers x 4``, because that is what "reconstruct
    everything" would actually mean.  Quoting the saving against the trigger
    count instead would flatter it by the arm scoping twice over.
    """
    n_trig = len(cand)
    rows = []
    for cls in ['INTER', 'INTRA', 'IMPLIED', 'SINGLE', 'BUSY', 'NONE']:
        n_cls = int((cand['cls'] == cls).sum())
        s = sel[sel['cls'] == cls]
        n_ev = int(s['eventId'].nunique()) if len(s) else 0
        n_arm = len(s)
        rows.append(dict(cls=cls, n_triggers=n_cls,
                         frac_of_sample=round(n_cls / max(n_trig, 1), 5),
                         n_events_selected=n_ev,
                         frac_of_class=round(n_ev / max(n_cls, 1), 4),
                         n_arm_events=n_arm,
                         arms_per_event=round(n_arm / max(n_ev, 1), 2)))
    tot = dict(cls='TOTAL', n_triggers=n_trig, frac_of_sample=1.0,
               n_events_selected=int(sel['eventId'].nunique()) if len(sel) else 0,
               frac_of_class=round((sel['eventId'].nunique() if len(sel) else 0)
                                   / max(n_trig, 1), 4),
               n_arm_events=len(sel),
               arms_per_event=round(len(sel) / max(sel['eventId'].nunique(), 1), 2)
               if len(sel) else 0.0)
    rows.append(tot)
    df = pd.DataFrame(rows)
    # The number that sets the condor budget.
    df['frac_of_full_reco'] = (df['n_arm_events'] / max(n_trig * len(ARMS), 1)).round(5)
    if core_h_per_1e3_arm_events:
        df['core_hours'] = (df['n_arm_events'] / 1e3
                            * core_h_per_1e3_arm_events).round(2)
    return df


def seed_efficiency(sel: pd.DataFrame, reco_dir, tags=None) -> pd.DataFrame:
    """How many allowlisted (arm, event)s the beam seeder actually seeds.

    ``reco_dir`` holds ``mx17_<arm>/events_<tag>.parquet`` from a **full,
    unfiltered** pass (`wft_beam` with no allowlist), whose event ids are by
    construction exactly the events that seeded.  Comparing the allowlist
    against it costs nothing and answers the question the allowlist cannot
    answer about itself: *of the events stage 1 chose, how many can stage 2
    even see?*

    That number is not a curiosity.  An allowlisted event with no seed produces
    no row, and a missing row is indistinguishable downstream from an event
    that was never selected -- so an unmeasured seeding loss silently becomes a
    reconstruction inefficiency attributed to physics.

    Returns one row per (cls, reason) with ``n``, ``n_seeded``, ``frac``.
    """
    reco_dir = Path(reco_dir)
    seeded: dict[tuple[str, str], set[int]] = {}
    for arm in ARMS:
        for p in sorted((reco_dir / f'mx17_{arm}').glob('events_*.parquet')):
            if p.name.endswith('.candidates.parquet'):
                continue
            tag = p.name[len('events_'):-len('.parquet')]
            if tags and tag not in tags:
                continue
            seeded[(arm, tag)] = set(
                int(e) for e in pd.read_parquet(p, columns=['event_id'])['event_id'])
    if not seeded:
        raise FileNotFoundError(f'no events_*.parquet under {reco_dir}/mx17_*/')

    have = sel[[(a, t) in seeded for a, t in zip(sel['arm'], sel['tag'])]].copy()
    if not len(have):
        # The usual cause: a full pass exists, but for a DIFFERENT sub-run.
        # Returning zeros here would read as "nothing seeds", which is the
        # opposite of "nothing was compared".
        raise ValueError(
            f'no (arm, tag) of this allowlist has a table under {reco_dir} -- '
            f'allowlist tags {sorted(set(sel["tag"]))[:3]}..., reco tags '
            f'{sorted({t for _, t in seeded})[:3]}...  Nothing was compared.')
    have['seeded'] = [e in seeded[(a, t)]
                      for a, t, e in zip(have['arm'], have['tag'], have['eventId'])]

    def _agg(g, **extra):
        return dict(n=len(g), n_seeded=int(g['seeded'].sum()),
                    frac=round(float(g['seeded'].mean()), 4), **extra)

    rows = [_agg(g, cls=c, reason=r) for (c, r), g in
            have.groupby(['cls', 'reason'], sort=True)]
    rows += [_agg(g, cls='(all)', reason=r) for r, g in
             have.groupby('reason', sort=True)]
    rows += [_agg(g, cls='(all)', reason=f'arm {a}') for a, g in
             have.groupby('arm', sort=True)]
    rows.append(_agg(have, cls='(all)', reason='(all)'))
    return pd.DataFrame(rows)[['cls', 'reason', 'n', 'n_seeded', 'frac']]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def to_json(sel: pd.DataFrame, cand: pd.DataFrame, run: str, subrun: str,
            sources: list[Path], salt: str = PRESCALE_SALT,
            control_prescale: dict[str, float] | None = None) -> dict:
    """The document a condor job reads: ``arm -> tag -> [event ids]``.

    Keyed by arm first because a job is one (arm, tag) and should be able to
    find its own list without walking anyone else's.
    """
    control_prescale = CONTROL_PRESCALE if control_prescale is None else control_prescale
    events: dict[str, dict[str, list[int]]] = {}
    for (arm, tag), g in sel.groupby(['arm', 'tag'], sort=True):
        events.setdefault(arm, {})[tag] = sorted(int(e) for e in g['eventId'])
    return dict(
        schema=SCHEMA,
        run=run, subrun=subrun,
        built=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        source=[dict(path=str(p), sha256=_sha256(p)) for p in sources],
        policy=dict(signal_classes=list(SIGNAL_CLASSES),
                    implied_forces_silent_coincident_arms=True,
                    control_arms='all',
                    control_prescale=dict(control_prescale),
                    prescale_salt=salt,
                    draw='blake2b(salt|run|subrun|eventId)[:8] / 2**64'),
        counts=dict(n_triggers=int(len(cand)),
                    n_events_selected=int(sel['eventId'].nunique()) if len(sel) else 0,
                    n_arm_events=int(len(sel)),
                    n_arm_events_full_reco=int(len(cand) * len(ARMS)),
                    by_class={c: int((sel['cls'] == c).sum())
                              for c in sorted(sel['cls'].unique())},
                    by_reason={r: int((sel['reason'] == r).sum())
                               for r in sorted(sel['reason'].unique())},
                    by_arm={a: int((sel['arm'] == a).sum()) for a in ARMS}),
        events=events)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path) -> dict[str, dict[str, set[int]]]:
    """Read an allowlist back as ``{arm: {tag: {event ids}}}``.

    Kept here rather than in ``wft_beam`` so the format has exactly one reader
    and one writer.  ``wft_beam`` has its own tiny loader for the same file --
    it must not import this package, which is analysis-side and not shipped to
    the workers.
    """
    doc = json.loads(Path(path).read_text())
    if doc.get('schema') != SCHEMA:
        raise ValueError(f'{path}: schema {doc.get("schema")!r}, expected {SCHEMA!r}')
    return {arm: {tag: set(ids) for tag, ids in tags.items()}
            for arm, tags in doc['events'].items()}


# --------------------------------------------------------------------------- #
def build(run: str, subrun: str, stage1_dir: Path | None = None,
          out_dir: Path | None = None,
          control_prescale: dict[str, float] | None = None,
          salt: str = PRESCALE_SALT, write: bool = True):
    """Build (and by default write) the allowlist for one sub-run."""
    stage1_dir = Path(stage1_dir) if stage1_dir else paths.out('stage1')
    src = paths.require(stage1_dir / f'candidates_{run}_{subrun}.parquet',
                        f'stage-1 candidates for {run}/{subrun} '
                        f'(run candidate_filter.py --all-tags first)')
    cand = pd.read_parquet(src)
    sel = select(cand, control_prescale=control_prescale, salt=salt)
    cost = plan(cand, sel)
    doc = to_json(sel, cand, run, subrun, [src], salt=salt,
                  control_prescale=control_prescale)
    if write:
        out_dir = Path(out_dir) if out_dir else paths.out('stage2')
        out_dir.mkdir(parents=True, exist_ok=True)
        jp = out_dir / f'allowlist_{run}_{subrun}.json'
        jp.write_text(json.dumps(doc, indent=1))
        sel.to_parquet(out_dir / f'allowlist_{run}_{subrun}.parquet', index=False)
        cost.to_csv(out_dir / f'allowlist_cost_{run}_{subrun}.csv', index=False)
        print(f'  -> {jp}  ({len(sel):,} (arm, event) fits)')
    return sel, cost, doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000',
                    help='one sub-run, or "all" for every candidates_* of the run')
    ap.add_argument('--stage1', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--single', type=float, default=CONTROL_PRESCALE['SINGLE'])
    ap.add_argument('--busy', type=float, default=CONTROL_PRESCALE['BUSY'])
    ap.add_argument('--none', type=float, default=CONTROL_PRESCALE['NONE'])
    ap.add_argument('--salt', default=PRESCALE_SALT)
    ap.add_argument('--plan-only', action='store_true',
                    help='print the cost table, write nothing')
    ap.add_argument('--seed-eff', type=Path, default=None,
                    help='directory of a FULL unfiltered reco '
                         '(mx17_<arm>/events_<tag>.parquet) to measure how '
                         'many allowlisted events the seeder can see')
    a = ap.parse_args()

    pre = {'SINGLE': a.single, 'BUSY': a.busy, 'NONE': a.none}
    stage1 = Path(a.stage1) if a.stage1 else paths.out('stage1')
    if a.subrun == 'all':
        subs = sorted(p.name[len(f'candidates_{a.run}_'):-len('.parquet')]
                      for p in stage1.glob(f'candidates_{a.run}_*.parquet')
                      if p.name.count('_') == 4)     # no per-tag files
    else:
        subs = [a.subrun]

    for s in subs:
        print(f'\n{a.run}/{s}')
        sel, cost, doc = build(a.run, s, stage1_dir=stage1, out_dir=a.out,
                               control_prescale=pre, salt=a.salt,
                               write=not a.plan_only)
        print(cost.to_string(index=False))
        print('  by reason:', doc['counts']['by_reason'])
        print('  by arm:   ', doc['counts']['by_arm'])
        if a.seed_eff:
            try:
                eff = seed_efficiency(sel, a.seed_eff)
            except ValueError as exc:      # no full pass for THIS sub-run
                print(f'\n  seed efficiency: skipped -- {exc}')
                continue
            print('\nseed efficiency -- allowlisted (arm, event)s the beam '
                  f'seeder produces a cluster for\n(vs the full pass in '
                  f'{a.seed_eff})\n')
            print(eff.to_string(index=False))
            if not a.plan_only:
                od = Path(a.out) if a.out else paths.out('stage2')
                eff.to_csv(od / f'seed_efficiency_{a.run}_{s}.csv', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
