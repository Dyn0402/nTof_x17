#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_empty_filter.py -- the empty-pulse filter, on the join it actually sees.

    python test_empty_filter.py           # 7 cases, < 1 s, no data needed

`slim.bunch_table` decides which bunches reach the fit, the file and every
analysis downstream, from one column of the beam record. The rules it has to
get right are small and each one has a way of being silently wrong:

  * an empty pulse (< C.EMPTY_PULSE_E10) is dropped, WITH all its triggers;
  * its bunch still appears in the table, with has_beam = 0 and its ORIGINAL
    trigger count, because the table is the record of what was dropped;
  * a NaN intensity is NOT an empty pulse -- a burst the join could not place
    is an unknown, and an unknown is not a reason to throw data away;
  * the threshold classifies, it does not tune: a parasitic pulse at ~413e10
    is beam, exactly like a dedicated one at ~851e10.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ntof_processing.slim_pipeline import config as C          # noqa: E402
from ntof_processing.slim_pipeline import slim as S            # noqa: E402


COLS = ('eventId', 'BunchNumber', 'is_flash', 't_since_flash_ns',
        'bunch_intensity_e10')


def joined(spec):
    """A joined-events frame: spec is {bunch: (n_triggers, intensity_e10)}.

    Always carries the columns, even with no rows -- `dream_event_to_bunch`
    does, and a fixture that does not would make the zero-join case fail here
    for a reason the real code never has.
    """
    rows = []
    for b, (n, inten) in spec.items():
        for i in range(n):
            rows.append(dict(eventId=len(rows), BunchNumber=b,
                             is_flash=(i == 0), t_since_flash_ns=1e6 * (i + 1),
                             bunch_intensity_e10=inten))
    return pd.DataFrame(rows, columns=list(COLS))


def check(name, cond, detail=''):
    print(f'  {"ok  " if cond else "FAIL"}  {name}' + (f'   {detail}' if detail
                                                       else ''))
    return [] if cond else [name]


def main() -> int:
    quiet = lambda *a, **k: None                                # noqa: E731
    bad = []

    ev = joined({10: (90, 851.0),      # dedicated
                 11: (74, 413.0),      # parasitic
                 12: (2, 0.3),         # empty pulse
                 13: (1, 0.0),         # empty pulse
                 14: (80, np.nan)})    # unplaceable burst: unknown, not empty
    tbl, keep = S.bunch_table(ev, log=quiet)

    bad += check('every bunch is in the table, empty ones included',
                 list(tbl['bunch']) == [10, 11, 12, 13, 14])
    bad += check('has_beam marks exactly the two empty pulses',
                 list(tbl['has_beam']) == [True, True, False, False, True])
    bad += check('NaN intensity counts as beam',
                 bool(tbl['has_beam'][4]))
    bad += check('the table keeps the ORIGINAL trigger counts',
                 list(tbl['n_triggers']) == [90, 74, 2, 1, 80])
    bad += check('the empty pulses\' triggers are dropped',
                 int((~keep).sum()) == 3 and keep.sum() == 244,
                 f'{int(keep.sum())} kept of {keep.size}')
    bad += check('no surviving event belongs to an empty bunch',
                 not np.isin(ev['BunchNumber'].to_numpy()[keep],
                             tbl['bunch'][~tbl['has_beam']]).any())
    bad += check('parasitic pulses are beam, not empties',
                 bool(tbl['has_beam'][1])
                 and 413.0 < C.PARASITIC_E10
                 and 413.0 >= C.EMPTY_PULSE_E10)

    # A segment with beam everywhere must be untouched -- the campaign's
    # reference pair (224572) has zero empty pulses and has to slim bit for bit
    # as it did before the filter existed.
    ev2 = joined({20: (90, 851.0), 21: (74, 413.0)})
    tbl2, keep2 = S.bunch_table(ev2, log=quiet)
    bad += check('a segment with full beam loses nothing',
                 keep2.all() and tbl2['has_beam'].all())

    # An EMPTY join is not a no-beam segment. Both end with an empty frame, and
    # `run_segment` has to tell them apart or it blames the accelerator for a
    # proposal that simply did not overlap -- which is what it did on
    # run_116/stat090_0017 x 224636 until 2026-08-10.
    ev3 = joined({})
    tbl3, keep3 = S.bunch_table(ev3, log=quiet)
    bad += check('an empty join produces an empty table, not a crash',
                 tbl3['bunch'].size == 0 and keep3.size == 0)
    ev4 = joined({30: (2, 0.1), 31: (1, 0.0)})
    tbl4, keep4 = S.bunch_table(ev4, log=quiet)
    bad += check('an all-empty-pulse segment is distinguishable from it',
                 tbl4['bunch'].size == 2 and not tbl4['has_beam'].any()
                 and not keep4.any())

    print()
    if bad:
        print(f'{len(bad)} PROBLEM(S): ' + ', '.join(bad))
        return 1
    print('all cases behaved as specified')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
