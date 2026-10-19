#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_fullpass.py -- put every full-pass sub-run into one layout.

The full waveform pass exists in two shapes, for historical reasons:

  flat, per tag    the August CERN pass: <out>/mx17_<arm>/events_<tag>.parquet
                   plus events_<tag>.candidates.parquet, all sub-runs mixed
                   together in one directory and told apart only by the tag
  nested, merged   what wft_beam writes when pointed at one sub-run:
                   <root>/<subrun>/mx17_<arm>/events_prelim.parquet (+ its
                   .candidates.parquet)

Every consumer downstream wants the second, so this makes the first look like
it: one merged pair per (sub-run, arm), with a ``tag`` column preserved so
nothing about the file split is lost.

Idempotent, and it will not silently mix layouts: a sub-run already present in
the nested tree (because wft_beam wrote it directly) is left alone.

    python -m sept26_prelim_analysis.merge_fullpass
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
FLAT = os.environ.get(
    'X17_FULLPASS_FLAT') or str(paths.spell(
        'analysis', 'wft_beam145', 'extracted', 'out'))
TAG_SUBRUN = {'260805_14H06': 'stat090_0000',
              '260805_15H07': 'stat090_0001',
              '260805_16H07': 'stat090_0002'}


def merge(run: str, dest: str, force: bool = False) -> int:
    n = 0
    for arm in ARMS:
        per = {}
        for f in sorted(glob.glob(os.path.join(FLAT, f'mx17_{arm}',
                                               'events_*.parquet'))):
            base = os.path.basename(f)
            tag = base.split('events_')[1] \
                .replace('.candidates.parquet', '').replace('.parquet', '')
            sub = TAG_SUBRUN.get(tag[:12])
            if sub is None:
                raise KeyError(f'tag {tag} maps to no sub-run; extend TAG_SUBRUN')
            kind = 'cand' if base.endswith('.candidates.parquet') else 'events'
            per.setdefault((sub, kind), []).append((tag, f))

        for (sub, kind), items in sorted(per.items()):
            od = os.path.join(dest, sub, f'mx17_{arm}')
            name = ('events_prelim.candidates.parquet' if kind == 'cand'
                    else 'events_prelim.parquet')
            out = os.path.join(od, name)
            if os.path.exists(out) and not force:
                print(f'  {arm}/{sub}/{kind}: exists, left alone')
                continue
            os.makedirs(od, exist_ok=True)
            d = pd.concat([pd.read_parquet(f).assign(tag=t) for t, f in items],
                          ignore_index=True)
            d.to_parquet(out, index=False)
            print(f'  {arm}/{sub}/{kind}: {len(items)} tags -> {len(d):,} rows')
            n += 1
            if kind == 'events':
                m = json.load(open(items[0][1].replace('.parquet',
                                                       '.meta.json')))
                m['tags_done'] = [t for t, _ in items]
                m['n_events'] = int(len(d))
                m['merged_from'] = FLAT
                json.dump(m, open(os.path.join(od, 'events_prelim.meta.json'),
                                  'w'), indent=1)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--dest', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    dest = a.dest or str(paths.out('fullpass') / a.run)
    print(f'{FLAT}\n  -> {dest}')
    n = merge(a.run, dest, force=a.force)
    print(f'\nwrote {n} merged table(s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
