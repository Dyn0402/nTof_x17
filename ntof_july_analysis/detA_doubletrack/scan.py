#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan.py — run the Det-A double-track finder over whole runs and cache results.

For each sub-run of a run:
  * load Det-A combined hits, noise-flag per event;
  * reconstruct the singles/doubles burst structure (trigger_timestamp) to skip
    the γ-flash leader and saturated flash-pile-up events (dt≈0);
  * cheap pre-filter (enough clean hits in BOTH planes, not a busy discharge);
  * run dtrack_lib.analyze_event on survivors.

Two cached tables per sub-run (under <ANALYSIS>/July_HV_Scan/detA_doubletrack/cache/<run>/):
  <subrun>_ev.parquet    — one row per reco'd event: n_xline, n_yline, n_pair,
                           is_double, dt_ms, charges, topology tag.
  <subrun>_cand.pkl      — full detail (line params + clean-hit dumps) for every
                           is_double candidate, so displays regenerate offline.

Usage:
  .venv/bin/python .../scan.py process run_58 [run_61 run_62 run_63] [--jobs N] [--force]
  .venv/bin/python .../scan.py process-subrun run_58 sngPS_dr300_r555_041
"""
import argparse
import glob
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dtrack_lib as D  # noqa: E402
from ntof_tracking.reco import io, noise, geometry as geo  # noqa: E402

DETA_FEUS = (3, 4)          # Det A: feu 3 = x plane, feu 4 = y plane
LOAD_STEP = 500_000         # uproot batch size (rows) -> bounds peak RAM


def load_detA_hits(run, subrun):
    """Low-memory Det-A-only loader: stream the combined-hits tree in batches
    and keep only Det-A FEUs (3,4), so we never materialise the B/C/D rows.
    Peak RAM ~ one batch + the Det-A subset instead of all four detectors."""
    hits_dir = os.path.join(io.BASE_PATH, run, subrun, 'combined_hits_root')
    srcs = io._real_files(hits_dir, '.root')
    cols = io.HIT_COLUMNS + ['trigger_timestamp_ns']
    parts = []
    for s in srcs:
        try:
            with uproot.open(s) as f:
                if 'hits' not in f:
                    continue
        except Exception:
            continue
        for batch in uproot.iterate(f'{s}:hits', cols, step_size=LOAD_STEP,
                                    library='pd'):
            parts.append(batch[batch['feu'].isin(DETA_FEUS)])
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
    lut_a = io.build_channel_lut(io.load_run_config(run))
    lut_a = lut_a[lut_a['det'] == D.DET_A]
    df = df.merge(lut_a, on=['feu', 'channel'], how='inner')
    return df.sort_values(['eventId', 'plane', 'time']).reset_index(drop=True)

OUT_BASE = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'detA_doubletrack')
CACHE_DIR = os.path.join(OUT_BASE, 'cache')

SUBRUN_RE = re.compile(
    r'(?P<pre>sngPS|sng|dblPS|dbl)_dr(?P<drift>\d+)_r(?P<resist>\d+)_(?P<seq>\d+)$')

# burst / flash model (mirrors run58_scan.scan_lib)
BURST_GAP_S = 0.10
FLASH_AMP = 1000
FLASH_NBIG = 150

# cheap pre-filter before the (relatively expensive) RANSAC line search
MIN_CLEAN_PER_PLANE = 10     # need >=2 lines of >=5 hits in each plane
# A genuine multi-track event is inherently busier than a single track, so we
# do NOT hard-cut at the search.py 120-strip "busy" line (that kills real
# doubles like ev977). We only skip TRUE full-plane discharges (RANSAC would be
# slow and its diagonals accidental) and TAG the moderately-busy survivors.
BUSY_SKIP_STRIPS = 300       # >this many clean strips = discharge -> skip
BUSY_TAG_STRIPS = 120        # >this = flag 'busy' for separate ranking


def parse_subrun(name):
    m = SUBRUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return dict(name=name, pre=d['pre'], drift=int(d['drift']),
                resist=int(d['resist']), seq=int(d['seq']))


def list_subruns(run):
    out = []
    run_dir = os.path.join(io.BASE_PATH, run)
    for name in sorted(os.listdir(run_dir)):
        d = parse_subrun(name)
        if not d:
            continue
        if not glob.glob(os.path.join(run_dir, name, 'combined_hits_root',
                                      '*_datrun_*.root')):
            continue
        d['run'] = run
        out.append(d)
    return out


def _cache_paths(run, subrun):
    d = os.path.join(CACHE_DIR, run)
    os.makedirs(d, exist_ok=True)
    return (os.path.join(d, f'{subrun}_ev.parquet'),
            os.path.join(d, f'{subrun}_cand.pkl'))


def _burst_model(hits):
    """Per-event burst id, leader flag, flash flag, dt-since-leader [ms]."""
    h = hits.sort_values('eventId', kind='stable')
    ev_ids = h['eventId'].to_numpy()
    uev, first = np.unique(ev_ids, return_index=True)
    tns = h['trigger_timestamp_ns'].to_numpy(np.float64)[first]
    big = (h['amplitude'].to_numpy() >= FLASH_AMP).astype(np.int64)
    n_big = np.add.reduceat(big, first)
    order = np.argsort(tns, kind='stable')
    uev, tns, n_big = uev[order], tns[order], n_big[order]
    t_s = (tns - tns[0]) / 1e9
    new_b = np.append(True, np.diff(t_s) > BURST_GAP_S)
    bid = np.cumsum(new_b) - 1
    lead_idx = np.flatnonzero(new_b)
    dt_ms = (t_s - t_s[lead_idx][bid]) * 1e3
    return pd.DataFrame({'eventId': uev, 'is_leader': new_b,
                         'n_big': n_big, 'dt_ms': dt_ms})


def process_subrun(run, subrun, lut=None, cfg=None, force=False):
    ev_path, cand_path = _cache_paths(run, subrun)
    if not force and os.path.exists(ev_path) and os.path.exists(cand_path):
        return pd.read_parquet(ev_path)
    hits = load_detA_hits(run, subrun)      # low-memory Det-A-only stream
    if hits is None or hits.empty:
        return None

    bm = _burst_model(hits).set_index('eventId')
    drift_hv = io.parse_drift_hv(subrun) or 800.0
    drift = geo.DriftModel.from_drift_hv(drift_hv)

    rows, cands = [], []
    for evid, g in hits.groupby('eventId', sort=False):
        binfo = bm.loc[evid]
        if bool(binfo['is_leader']) or binfo['n_big'] > FLASH_NBIG:
            continue                                     # flash / leader
        g = noise.flag_noise(g)
        cl = g[g['clean']]
        nx = int(((cl['plane'] == 'x')).sum())
        ny = int(((cl['plane'] == 'y')).sum())
        if nx < MIN_CLEAN_PER_PLANE or ny < MIN_CLEAN_PER_PLANE:
            continue
        nstr = cl['channel'].nunique()
        if nstr > BUSY_SKIP_STRIPS:
            continue                                     # full-plane discharge
        res = D.analyze_event(g, drift)
        if res is None:
            continue
        row = dict(eventId=int(evid), run=run, subrun=subrun,
                   drift=drift_hv, dt_ms=float(binfo['dt_ms']),
                   n_big=int(binfo['n_big']),
                   n_clean_x=nx, n_clean_y=ny, n_clean_strips=int(nstr),
                   busy=bool(nstr > BUSY_TAG_STRIPS),
                   n_xline=res['n_xline'], n_yline=res['n_yline'],
                   n_pair=res['n_pair'], is_double=res['is_double'],
                   q_lines=res['q_lines'], min_r2=res['min_r2'],
                   topo=res['topo'].get('tag', ''))
        rows.append(row)
        if res['is_double']:
            # keep full detail + the clean-hit dump for offline displays
            dump = {}
            for pl in ('x', 'y'):
                gp = g[(g['plane'] == pl) & g['clean']]
                dump[pl] = dict(pos=gp['pos_mm'].to_numpy(),
                                time=gp['time'].to_numpy(),
                                amp=gp['amplitude'].to_numpy())
                gpn = g[(g['plane'] == pl) & ~g['clean']]
                dump[pl + '_noise'] = dict(pos=gpn['pos_mm'].to_numpy(),
                                           time=gpn['time'].to_numpy())
            cands.append(dict(meta=row, res={k: res[k] for k in
                              ('xlines', 'ylines', 'pairs', 'topo',
                               'n_xline', 'n_yline', 'n_pair', 'eventId')},
                              dump=dump))

    ev = pd.DataFrame(rows)
    ev.to_parquet(ev_path)
    with open(cand_path, 'wb') as f:
        pickle.dump(cands, f)
    return ev


def _worker(args):
    run, subrun, force = args
    try:
        ev = process_subrun(run, subrun, force=force)
        n = 0 if ev is None else int(ev['is_double'].sum()) if len(ev) else 0
        return (run, subrun, None if ev is None else len(ev), n)
    except Exception as e:              # keep the scan going on a bad subrun
        import traceback
        return (run, subrun, 'ERR', f'{e}\n{traceback.format_exc()[-500:]}')


def process(runs, jobs=4, force=False):
    tasks = []
    for run in runs:
        for d in list_subruns(run):
            tasks.append((run, d['name'], force))
    print(f'{len(tasks)} sub-runs across {runs} on {jobs} workers')
    tot_ev = tot_dbl = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs)):
            run, subrun, nev, ndbl = fut.result()
            if nev == 'ERR':
                print(f'  [{i+1}/{len(tasks)}] {run}/{subrun} ERROR: {ndbl}')
                continue
            tot_ev += nev or 0
            tot_dbl += ndbl or 0
            print(f'  [{i+1}/{len(tasks)}] {run}/{subrun}: '
                  f'{nev} reco ev, {ndbl} doubles  (running total doubles={tot_dbl})')
    print(f'\nDONE. {tot_ev} reco events, {tot_dbl} Det-A double-track candidates.')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('process')
    p.add_argument('runs', nargs='+')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--force', action='store_true')
    q = sub.add_parser('process-subrun')
    q.add_argument('run')
    q.add_argument('subrun')
    q.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'process':
        process(a.runs, a.jobs, a.force)
    else:
        ev = process_subrun(a.run, a.subrun, force=a.force)
        if ev is None:
            print('no data')
        else:
            print(f'{len(ev)} reco events, {int(ev["is_double"].sum())} doubles')


if __name__ == '__main__':
    main()
