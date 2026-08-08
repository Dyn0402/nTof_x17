#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slim_run.py -- slim every DREAM sub-run that overlaps ONE n_TOF run.

This is the condor job unit. One n_TOF run is 30 GB and gets copied to node
scratch once; the bunch index is built once; then each overlapping DREAM sub-run
is a ~6 minute segment on top of that. Making the job a single *segment* instead
would pay the copy and the index two or three times over.

Segments come from `segments.py`, which proposes them on wall clock. The
proposal is checked here: `bunch_join` matches each DREAM burst to a real n_TOF
bunch through the beam record, and a segment that joins fewer than
`--min-events` events is a wall-clock miss and is skipped with a loud line
rather than writing a near-empty file.

USAGE
    python slim_run.py 224572 --out ./out
    python slim_run.py 224572 --out ./out --ntof-source /path/to/partials
    python slim_run.py 224572 --out ./out --only stat090_0000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ntof_processing.slim_pipeline import config as C           # noqa: E402
from ntof_processing.slim_pipeline import segments as SEG       # noqa: E402
from ntof_processing.slim_pipeline.slim import Segment, run_segment  # noqa: E402

MIN_EVENTS = 500          # below this the wall-clock proposal did not pan out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('--out', required=True, help='output base directory')
    ap.add_argument('--ntof-source', default=None)
    ap.add_argument('--segments-dir', default=str(SEG.DATA))
    ap.add_argument('--only', default=None,
                    help='restrict to one DREAM sub-run name')
    ap.add_argument('--slim-ns', type=float, default=C.SLIM_NS)
    ap.add_argument('--min-events', type=int, default=MIN_EVENTS)
    ap.add_argument('--allow-unreprocessed', action='store_true',
                    help='also slim if the n_TOF run is not on the v12 list')
    args = ap.parse_args()

    props = SEG.for_ntof_run(args.ntof_run, Path(args.segments_dir),
                             ready_only=not args.allow_unreprocessed)
    if args.only:
        props = [p for p in props if p.dream_subrun == args.only]
    if not props:
        print(f'no segments for n_TOF {args.ntof_run} '
              f'(reprocessed-only={not args.allow_unreprocessed})')
        return 0

    print(f'n_TOF {args.ntof_run}: {len(props)} segment(s) proposed')
    for p in props:
        print(f'  {p.dream_run}/{p.dream_subrun}  overlap '
              f'{p.overlap_s/60:.1f} min ({p.fraction:.0%})')
    print()

    out_base = Path(args.out)
    results, t_all = [], time.time()
    for p in props:
        seg = Segment(p.dream_run, p.dream_subrun, args.ntof_run,
                      ntof_source=Path(args.ntof_source) if args.ntof_source
                      else None)
        rec = dict(dream_run=p.dream_run, dream_subrun=p.dream_subrun,
                   ntof_run=args.ntof_run, proposed_overlap_s=p.overlap_s)
        try:
            path, meta = run_segment(seg, out_base=out_base,
                                     slim_ns=args.slim_ns)
            q = meta['qa']
            if q['n_physics'] < args.min_events:
                rec.update(status='SKIPPED_LOW_JOIN', n_events=q['n_physics'])
                print(f'!! {seg}: only {q["n_physics"]} events joined -- the '
                      f'wall-clock proposal missed. File written anyway; treat '
                      f'as suspect.\n')
            else:
                rec.update(status='OK', path=str(path), **{
                    k: q[k] for k in ('efficiency', 'accidental', 'n_events',
                                      'n_physics', 'n_hits', 'seconds')})
        except Exception as e:                                   # noqa: BLE001
            rec.update(status='FAILED', error=f'{type(e).__name__}: {e}')
            print(f'!! {seg} FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
        results.append(rec)

    out_base.mkdir(parents=True, exist_ok=True)
    summary = out_base / f'slim_summary_{args.ntof_run}.json'
    summary.write_text(json.dumps(dict(
        ntof_run=args.ntof_run, seconds=round(time.time() - t_all, 1),
        segments=results), indent=2))

    ok = [r for r in results if r['status'] == 'OK']
    print(f'\n{"=" * 64}')
    for r in results:
        eff = f'{r["efficiency"]:.2%}' if 'efficiency' in r else '-'
        print(f'  {r["status"]:<18} {r["dream_run"]}/{r["dream_subrun"]:<16} '
              f'eff {eff}')
    print(f'{len(ok)}/{len(results)} segments OK in '
          f'{(time.time() - t_all)/60:.1f} min -> {summary}')
    return 0 if len(ok) == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
