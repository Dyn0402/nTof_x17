#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
availability.py -- what n_TOF data exists for every DREAM beam sub-run, 2026-08-10.

The second pass over the processed data, after the discovery that 55 runs were
reconstructed but never merged (`README.md`). It classifies every second of DREAM
beam time by the BEST n_TOF state that covers it:

  AVAILABLE         a merged file in done/, OR a complete partial set in
                    completed/<run>/. Both are v12 and both are readable today;
                    `slim_pipeline.config.ntof_files()` finds either.
  NEEDS_PROCESSING  covered only by a run whose raw is staged but which has no
                    processed output at all (224688-224718).
  NO_NTOF           no n_TOF run was live at that time.

The point of the AVAILABLE class is that merged-vs-unmerged is NOT a data
distinction. It was treated as one only because `done/` was the only place
anyone looked.

Inputs, all regenerated 2026-08-10 (see README.md section 6):
  inputs/inventory_2026-08-10.csv                  per-run processing state
  inputs/ntof_index_times_partials_2026-08-10.txt  spans from the partials
  inputs/ntof_raw_times_2026-08-10.txt             spans from raw mtimes
  inputs/dream_eos_subruns_2026-08-10.txt          DREAM sub-runs on EOS
  ../slim_study/coverage_inputs/ntof_index_times.txt   spans for merged runs

Times: the `index` tree's Date/Time are LOCAL (UTC+2), so index-derived spans
carry INDEX_LOCAL_SHIFT_S; raw mtimes are true UTC and carry -RAW_WRITE_LAG_S.
Both conventions are coverage_map's, reused deliberately.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'slim_study'))

import coverage_map as cm            # noqa: E402

INP = HERE / 'inputs'
OLD = HERE.parent / 'slim_study' / 'coverage_inputs'

AVAILABLE_STATES = {'MERGED', 'PARTIALS_ONLY', 'MERGE_EMPTY'}


def load_state():
    with (INP / 'inventory_2026-08-10.csv').open() as fh:
        return {int(r['run']): r['state'] for r in csv.DictReader(fh)}


def _spans(path, shift_s, skip_short=True):
    return {r: (a, b) for r, a, b in cm._spans(path, skip_short=skip_short,
                                               shift_s=shift_s)}


def load_spans():
    """run -> (t0, t1). Partial-derived index times WIN for the runs that have
    them, because they are read from the data that actually exists."""
    spans = _spans(OLD / 'ntof_index_times.txt', cm.INDEX_LOCAL_SHIFT_S)
    spans.update(_spans(INP / 'ntof_index_times_partials_2026-08-10.txt',
                        cm.INDEX_LOCAL_SHIFT_S))
    for r, s in _spans(INP / 'ntof_raw_times_2026-08-10.txt',
                       -cm.RAW_WRITE_LAG_S, skip_short=False).items():
        spans.setdefault(r, s)        # only where nothing better exists
    return spans


def load_dream():
    subs, per_file = cm.load_dream(INP / 'dream_eos_subruns_2026-08-10.txt',
                                   OLD / 'dream_daq_subruns.txt')
    return subs, per_file


def classify(t0, t1, spans, state):
    """(seconds available, seconds needing processing, seconds with no n_TOF)."""
    avail = need = 0.0
    for r, (a, b) in spans.items():
        ov = min(b, t1) - max(a, t0)
        if ov <= 0:
            continue
        st = state.get(r)
        if st in AVAILABLE_STATES:
            avail += ov
        elif st == 'RAW_ONLY':
            need += ov
    total = t1 - t0
    avail = min(avail, total)
    need = min(need, max(total - avail, 0.0))
    return avail, need, max(total - avail - need, 0.0)


def main() -> int:
    state = load_state()
    spans = load_spans()
    subs, per_file = load_dream()
    print(f'{len(state)} n_TOF runs inventoried, {len(spans)} with a time span')
    print(f'{len(subs)} DREAM beam sub-runs, {per_file:.0f} s per decoded file\n')

    by_run = {}
    for (run, sub), (t0, t1, n) in subs.items():
        if t1 is None:
            t1 = t0 + n * per_file
        by_run.setdefault(run, []).append(classify(t0, t1, spans, state))

    print(f'{"DREAM run":>10} {"subs":>5} {"hours":>7} {"AVAIL":>7} '
          f'{"NEEDPROC":>9} {"NO nTOF":>8}')
    print('-' * 52)
    tA = tN = tX = 0.0
    for run in sorted(by_run, key=lambda x: int(x.split('_')[1])):
        a = sum(x[0] for x in by_run[run])
        nd = sum(x[1] for x in by_run[run])
        x = sum(x[2] for x in by_run[run])
        tot = a + nd + x
        tA, tN, tX = tA + a, tN + nd, tX + x
        if tot <= 0:
            continue
        print(f'{run:>10} {len(by_run[run]):5d} {tot/3600:7.1f} '
              f'{100*a/tot:6.0f}% {100*nd/tot:8.0f}% {100*x/tot:7.0f}%')
    tot = tA + tN + tX
    print('-' * 52)
    print(f'{"TOTAL":>10} {len(subs):5d} {tot/3600:7.1f} '
          f'{100*tA/tot:6.0f}% {100*tN/tot:8.0f}% {100*tX/tot:7.0f}%')
    print(f'\n  AVAILABLE        {tA/3600:6.1f} h  (merged or unmerged partials)')
    print(f'  NEEDS PROCESSING {tN/3600:6.1f} h  (raw staged, nothing processed)')
    print(f'  NO n_TOF         {tX/3600:6.1f} h  (no n_TOF run live)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
