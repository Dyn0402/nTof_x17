#!/usr/bin/env python3
"""
Process all complete run_67 sub-runs into cached event/segment/drift tables.
Safe to re-run as the scan grows (skips cached sub-runs).

MEMORY (learned the hard way on this 15 GB box, 2026-07-23): a RAW n32 sub-run's
hit table is 0.5-0.7 GB on disk and several GB once in pandas, and the lowest
plastic threshold (m090) is the heaviest block because it has the highest
trigger rate. Pools of 4 AND of 2 workers were both killed by the OOM reaper
mid-run. Two mitigations, both on by default:

  * ``max_tasks_per_child=1`` — every sub-run gets a FRESH worker process, so
    its memory is returned to the OS on completion instead of accumulating
    across tasks in a long-lived worker (pandas/uproot do not reliably release
    arenas back to the OS).
  * pool auto-restart — a BrokenProcessPool (i.e. the OOM reaper took a worker)
    no longer aborts the run: the pool is rebuilt and the still-missing sub-runs
    are retried, up to --max-restarts times. Completed sub-runs are already
    cached, so finished work is never lost.

Pick --jobs from FREE memory, not core count: budget ~4 GB per worker.

Run: .venv/bin/python ntof_july_analysis/run67_scan/process.py [--jobs N] [--force]
     [--only SUBSTR]   restrict to sub-runs whose name contains SUBSTR
                       (e.g. --only m090 to finish one threshold block first)
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

import scan_lib  # noqa: E402


def worker(run, subrun, force=False):
    t0 = time.time()
    try:
        ev, segs, spec = scan_lib.build_subrun_tables(run, subrun, force=force)
        if ev is None:
            return run, subrun, 'EMPTY', 0, 0, time.time() - t0
        n_seg = 0 if segs is None or segs.empty else len(segs)
        return run, subrun, 'ok', len(ev), n_seg, time.time() - t0
    except Exception as e:  # noqa: BLE001
        import traceback
        return run, subrun, f'ERROR {e!r}\n{traceback.format_exc()}', 0, 0, \
            time.time() - t0


def hits_mtime(run, subrun):
    """Newest combined_hits mtime for a sub-run (0.0 if none found)."""
    import glob
    fs = glob.glob(os.path.join(scan_lib.io.BASE_PATH, run, subrun,
                                'combined_hits_root', '*_datrun_*.root'))
    return max((os.path.getmtime(f) for f in fs), default=0.0)


def pending(run, force, only, rebuilt=()):
    """Sub-runs still to do.

    Normal mode: a sub-run is done when its cache files exist.

    ``--force`` mode: mere existence is NOT a done-marker — the whole point is
    that the on-disk cache is stale (e.g. after the runs are re-decoded). A
    sub-run counts as done if EITHER this invocation rebuilt it (`rebuilt`, for
    the OOM-restart case within one run) OR its cache is already newer than the
    combined_hits it would be built from. The mtime test is what makes --force
    resumable ACROSS invocations: without it, killing and relaunching a 7-hour
    re-reco would start again from zero.
    """
    todo = []
    for d in scan_lib.list_subruns(run):
        if only and only not in d['name']:
            continue
        ev_p, _, spec_p = scan_lib._cache_paths(run, d['name'])
        have = os.path.exists(ev_p) and os.path.exists(spec_p)
        if force:
            done = d['name'] in rebuilt or (
                have and os.path.getmtime(ev_p) > hits_mtime(run, d['name']))
        else:
            done = have
        if not done:
            todo.append((run, d['name']))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--run', default=scan_lib.RUN)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', default=None,
                    help='only sub-runs whose name contains this substring')
    ap.add_argument('--max-restarts', type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    done = 0
    rebuilt = set()          # sub-runs THIS invocation actually rewrote
    for attempt in range(args.max_restarts + 1):
        # Recomputed each pass: anything finished by a previous (possibly
        # killed) pass drops out automatically, so a restart never redoes
        # finished work. Under --force "finished" means "in `rebuilt`", not
        # "file exists" — the on-disk files start out stale.
        todo = pending(args.run, args.force, args.only, rebuilt)
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
                        rebuilt.add(subrun)
                    print(f'[{done}] {run}/{subrun}: {status[:80]} '
                          f'{n_ev} events, {n_seg} track segs ({dt:.0f}s)',
                          flush=True)
        except BrokenProcessPool:
            # OOM reaper took a worker. Completed sub-runs are recorded in
            # `rebuilt`; rebuild the pool and carry on with what is still left.
            print('!! pool broke (likely OOM) — restarting with remaining '
                  'sub-runs', flush=True)
            continue
    left = pending(args.run, args.force, args.only, rebuilt)
    print(f'done in {(time.time() - t0) / 60:.1f} min; '
          f'{len(left)} sub-run(s) still missing', flush=True)
    for _, s in left:
        print('  MISSING', s)


if __name__ == '__main__':
    main()
