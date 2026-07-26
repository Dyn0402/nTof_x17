#!/usr/bin/env python3
"""
Process all complete run_53/run_55 sub-runs into cached event/segment tables.
Safe to re-run as run_55 grows (skips cached sub-runs). Parallel over sub-runs.

Run: .venv/bin/python ntof_july_analysis/hv_track_scan/process.py [--jobs N]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import scan_lib  # noqa: E402


def worker(run, subrun):
    t0 = time.time()
    try:
        ev, segs = scan_lib.build_subrun_tables(run, subrun)
        if ev is None:
            return run, subrun, 'EMPTY', 0, 0, time.time() - t0
        n_seg = 0 if segs is None or segs.empty else len(segs)
        return run, subrun, 'ok', len(ev), n_seg, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return run, subrun, f'ERROR {e!r}', 0, 0, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--runs', default='run_53,run_55')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    todo = []
    for run in args.runs.split(','):
        for d in scan_lib.list_subruns(run):
            ev_p, _ = scan_lib._cache_paths(run, d['name'])
            if args.force or not os.path.exists(ev_p):
                todo.append((run, d['name']))
    print(f'{len(todo)} sub-runs to process, {args.jobs} jobs', flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(worker, r, s) for r, s in todo]
        for i, f in enumerate(as_completed(futs)):
            run, subrun, status, n_ev, n_seg, dt = f.result()
            print(f'[{i + 1}/{len(todo)}] {run}/{subrun}: {status} '
                  f'{n_ev} events, {n_seg} track segs ({dt:.0f}s)', flush=True)
    print(f'done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
