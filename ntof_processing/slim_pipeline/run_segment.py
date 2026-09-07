#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_segment.py -- slim one (DREAM sub-run x n_TOF run) segment.

USAGE
    # the reference pair, from the local v12 copy, into a scratch directory
    python run_segment.py run_79 stat090_0000 224572 \
        --ntof-source /media/dylan/data/x17/ntof_reproc/v12_liqpileup \
        --out /tmp/slim_test

    # production: reprocessed n_TOF on EOS, output beside the DREAM sub-run
    python run_segment.py run_79 stat090_0000 224572
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ntof_processing.slim_pipeline import config as C     # noqa: E402
from ntof_processing.slim_pipeline.slim import (           # noqa: E402
    Segment, apply_fixes, load_burst_fixes, run_segment)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dream_run')
    ap.add_argument('dream_subrun')
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('--ntof-source', default=None,
                    help=f'directory of n_TOF partials (default {C.NTOF_DONE})')
    ap.add_argument('--out', default=None,
                    help=f'output base (default {C.EOS_JULY})')
    ap.add_argument('--slim-ns', type=float, default=C.SLIM_NS)
    ap.add_argument('--nb', type=int, default=None,
                    help='use only the first N bunches -- for smoke tests')
    ap.add_argument('--burst-fixes', default=None,
                    help=f'burst_bruteforce.py overrides (default '
                         f'{C.BURST_FIXES.name} beside the package)')
    ap.add_argument('--no-burst-fixes', action='store_true')
    ap.add_argument('--min-events', type=int, default=C.MIN_EVENTS)
    args = ap.parse_args()

    seg = Segment(args.dream_run, args.dream_subrun, args.ntof_run,
                  ntof_source=Path(args.ntof_source) if args.ntof_source else None)
    min_events = args.min_events
    if not args.no_burst_fixes:
        used = apply_fixes(seg, load_burst_fixes(args.burst_fixes))
        if used and used.get('lock', {}).get('min_events') is not None:
            min_events = int(used['lock']['min_events'])
    if args.nb:
        import numpy as np
        from ntof_dream_merge.bunch_join import dream_event_to_bunch
        ev = dream_event_to_bunch(seg.dream_run, seg.dream_subrun, seg.ntof_run)
        b = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())
        seg.bunches = b[:args.nb]
        print(f'smoke test: first {args.nb} bunches ({b[0]}-{seg.bunches[-1]})')

    _, meta = run_segment(seg, out_base=Path(args.out) if args.out else None,
                          slim_ns=args.slim_ns, min_events=min_events)
    q = meta['qa']
    print(f'efficiency {q["efficiency"]:.4%}  accidental {q["accidental"]:.4%}  '
          f'{q["n_hits"]:,} hits  {q["seconds"]:.0f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
