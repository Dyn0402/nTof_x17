#!/usr/bin/env python3
"""
Process the lead-shielding-comparison stat090 sub-runs (run_130/132/139) into
cached event/segment/drift tables. Safe to re-run (skips cached sub-runs).

Same worker-pool discipline as run67_scan/process.py, learned on this 15 GB
box: fresh worker per sub-run (max_tasks_per_child=1), pool auto-restart on
OOM, ~4 GB per worker -> --jobs 2 by default.

Run: .venv/bin/python ntof_july_analysis/leadshield_compare/process.py [--jobs N]
     [--force] [--only SUBSTR] [--runs run_132,run_139]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import lib as L  # noqa: E402


def worker(run, subrun, force=False):
    t0 = time.time()
    try:
        ev, segs, spec = L.build_subrun_tables(run, subrun, force=force)
        if ev is None:
            return run, subrun, 'EMPTY', 0, 0, time.time() - t0
        n_seg = 0 if segs is None or segs.empty else len(segs)
        return run, subrun, 'ok', len(ev), n_seg, time.time() - t0
    except Exception as e:  # noqa: BLE001
        import traceback
        return run, subrun, f'ERROR {e!r}\n{traceback.format_exc()}', 0, 0, \
            time.time() - t0


def hits_mtime(run, subrun):
    import glob
    fs = glob.glob(os.path.join(L.io.BASE_PATH, run, subrun,
                                'combined_hits_root', '*_datrun_*.root'))
    return max((os.path.getmtime(f) for f in fs), default=0.0)


def pending(runs, force, only, rebuilt=()):
    todo = []
    for run in runs:
        for d in L.list_subruns(run):
            if only and only not in d['name']:
                continue
            ev_p, _, spec_p = L._cache_paths(run, d['name'])
            have = os.path.exists(ev_p) and os.path.exists(spec_p)
            key = (run, d['name'])
            if force:
                done = key in rebuilt or (
                    have and os.path.getmtime(ev_p) > hits_mtime(run, d['name']))
            else:
                done = have
            if not done:
                todo.append(key)
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--runs', default=','.join(L.RUNS))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--max-restarts', type=int, default=20)
    args = ap.parse_args()
    runs = [r.strip() for r in args.runs.split(',') if r.strip()]

    t0 = time.time()
    done = 0
    rebuilt = set()
    for attempt in range(args.max_restarts + 1):
        todo = pending(runs, args.force, args.only, rebuilt)
        if not todo:
            break
        print(f'--- pass {attempt}: {len(todo)} sub-runs to process, '
              f'{args.jobs} jobs ---', flush=True)
        try:
            with ProcessPoolExecutor(max_workers=args.jobs,
                                     max_tasks_per_child=1) as ex:
                futs = [ex.submit(worker, r, s, args.force) for r, s in todo]
                for f in as_completed(futs):
                    run, subrun, status, n_ev, n_seg, dt = f.result()
                    done += 1
                    if status in ('ok', 'EMPTY'):
                        rebuilt.add((run, subrun))
                    print(f'[{done}] {run}/{subrun}: {status[:200]} '
                          f'{n_ev} events, {n_seg} track segs ({dt:.0f}s)',
                          flush=True)
        except BrokenProcessPool:
            print('!! pool broke (likely OOM) — restarting with remaining '
                  'sub-runs', flush=True)
            continue
    left = pending(runs, args.force, args.only, rebuilt)
    print(f'done in {(time.time() - t0) / 60:.1f} min; '
          f'{len(left)} sub-run(s) still missing', flush=True)
    for r, s in left:
        print('  MISSING', r, s)


if __name__ == '__main__':
    main()
