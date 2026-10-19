#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_campaign.py -- per-tag stage-2 output -> one merged table per sub-run.

`wft_beam` writes one `events_<tag>.parquet` (plus `.candidates.parquet`) per
file tag.  `k_arm.coincident_tracks` wants `events_prelim.parquet`, one merged
table per (sub-run, arm).  This makes the second from the first, across the
whole campaign.

WHY NOT `merge_fullpass.py`.  That module solves the same problem for the
August CERN pass, but it maps tag -> sub-run through a hardcoded three-entry
table (`TAG_SUBRUN`) that only covers run_145 and would silently mis-file
every other run.  Here the campaign fetch has already nested the products as
`<fullpass>/<run>/<subrun>/mx17_<arm>/`, so the sub-run is the directory and
there is no map to go stale.

`build_tracks` is NOT a consumer of this: it reads the per-tag
`.candidates.parquet` files directly (`load_reco`), because the events table
is a reduction that cannot express a second track.  So the merge exists for
`k_arm` and the other events-level consumers, and the per-tag files stay.

A `tag` column is added so nothing about the file split is lost.

    python -m sept26_prelim_analysis.merge_campaign            # everything present
    python -m sept26_prelim_analysis.merge_campaign --run run_79
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')


def merge_subrun(d: Path, force: bool = False) -> int:
    """Merge one `<run>/<subrun>` directory in place. Returns arms merged."""
    n = 0
    for arm in ARMS:
        ad = d / f'mx17_{arm}'
        if not ad.is_dir():
            continue
        # ONLY the events table is merged, never the candidates.
        # `build_tracks.load_reco` globs `events_*.candidates.parquet`, which
        # would match a merged `events_prelim.candidates.parquet` AS WELL AS
        # the per-tag files it is built from -- every track counted twice, with
        # nothing in the output to show it. `k_arm` reads `events_prelim.parquet`
        # and never the candidates, so the merge is not needed for it either.
        for suffix, out in (('.parquet', 'events_prelim.parquet'),):
            dest = ad / out
            if dest.exists() and not force:
                continue
            # '.parquet' also matches the candidates files, so exclude them
            # explicitly -- merging candidates INTO the events table would
            # produce a table with two different row meanings and no way to
            # tell them apart afterwards.
            src = [p for p in sorted(ad.glob(f'events_*{suffix}'))
                   if p.name != out
                   and not (suffix == '.parquet'
                            and p.name.endswith('.candidates.parquet'))]
            if not src:
                continue
            frames = []
            for p in src:
                tag = p.name[len('events_'):-len(suffix)]
                f = pd.read_parquet(p)
                f['tag'] = tag
                frames.append(f)
            pd.concat(frames, ignore_index=True).to_parquet(dest, index=False)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--fullpass', type=Path, default=None,
                    help='default <out>/fullpass')
    ap.add_argument('--run', default=None, help='one run; default all present')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    base = a.fullpass or paths.out('fullpass')

    runs = [base / a.run] if a.run else sorted(
        p for p in base.iterdir() if p.is_dir() and p.name.startswith('run_'))
    tot_sub = tot_arm = 0
    for r in runs:
        if not r.is_dir():
            print(f'  {r.name}: absent, skipped')
            continue
        for sub in sorted(p for p in r.iterdir() if p.is_dir()):
            k = merge_subrun(sub, force=a.force)
            if k:
                tot_sub += 1
                tot_arm += k
    print(f'merged {tot_arm} (sub-run, arm) table(s) across {tot_sub} sub-run(s)'
          f' under {base}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
