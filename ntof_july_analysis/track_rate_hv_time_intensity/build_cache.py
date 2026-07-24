#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cache.py — tracks vs HV vs time-since-flash, run_67, on the CNS-REPROCESSED hits.

run_67 (2026-07-22/23) is the 2D HV scan: drift {700,600,500,(400)} x resist
{550..520} x plastic threshold {1.41, 1.13, 0.90} MIP, PS+SINGLES, RAW 32 smp,
mesh circuit fully off. That makes it the scan for "tracks vs HV vs time since flash".

The earlier run_67/run_61 track analyses were built on hits with NO common-noise
subtraction, which the run_70 study showed are common-mode dominated (~1500x inflated
track rate, ±90° fragments of full-plane bands). CNS was re-enabled 2026-07-23 and
run_67 is being reprocessed; this script caches tracking on the CLEAN hits only.

Per-detector, per-subrun it writes one row per track and reuses the DECODED trigger
list as the denominator (with CNS on, a trigger whose only signal was common mode now
has zero hits and is absent from combined_hits — counting the denominator there would
undercount by ~40%).

Outputs (cache/):
  tracks.csv  det, subrun, mip, drift, resist, event_id, projection, n_hits,
              angle_deg, time_min, time_span, pos_span, dt_ms
  events.csv  det-independent: subrun, mip, drift, resist, dt_ms   (one row per
              physics trigger; flash events excluded)
  subruns.csv bookkeeping: n_flash, n_phys, per-det track counts

Only subruns whose combined_hits are CNS-era (mtime >= CNS_CUTOFF) are used; re-run as
the reprocess advances and it picks up the new ones (already-cached subruns are skipped).

Run: ~/PycharmProjects/nTof_x17/.venv/bin/python build_cache.py
"""
from __future__ import annotations

import os
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, str(Path.home() / 'beam_july/analysis/flash_timing_threshold'))
import flash_timing_lib as FT  # noqa: E402

NTOF = '/home/mx17/PycharmProjects/nTof_x17'
sys.path.insert(0, NTOF)
_tq = types.ModuleType('tqdm')
_tq.tqdm = lambda x=None, **k: x if x is not None else (lambda y: y)
sys.modules.setdefault('tqdm', _tq)
import beam_track_finding as bt          # noqa: E402
from common.Mx17StripMap import RunConfig  # noqa: E402

RUNS_DIR = Path('/mnt/data/x17/beam_july/runs')
MAP = f'{NTOF}/mx17_m1_map.csv'
OUT = Path(__file__).resolve().parent

# combined_hits written before this are the CNS-OFF (common-mode) files — unusable.
# See docs/METHOD_track_rate_vs_hv_time_intensity.md §1. Any run processed after CNS was
# re-enabled (2026-07-23) passes trivially; set to 0 to disable the check.
CNS_CUTOFF = datetime(2026, 7, 23, 20, 0).timestamp()

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
DET_FEUS = {'mx17_A': [3, 4], 'mx17_B': [5, 6], 'mx17_C': [7, 8], 'mx17_D': [1, 2]}
FLASH_GAP_MS = 200.0
MIN_TRACK_HITS = 4
bt.MIN_TRACK_HITS = MIN_TRACK_HITS

# Sub-run name -> scan axes. Each entry is (regex, parser -> dict of axis values).
# Add a pattern here to point this analysis at another run; anything that does not match
# any pattern is skipped, so a run with no HV axis simply yields nothing.
SUB_PATTERNS = [
    # run_67 / run_64 style: m090On_dr500_r520_062  (mip, drift, resist)
    (re.compile(r'^m(\d{3})(?:On|Off)_dr(\d{3})_r(\d{3})_(\d{3})$'),
     lambda m: dict(mip=int(m.group(1)) / 100.0, drift=int(m.group(2)),
                    resist=int(m.group(3)))),
    # run_71 style: acmeshOff_dr600_ri0_0041  (mesh on/off, drift, resist INDEX)
    (re.compile(r'^\w*mesh(On|Off)_dr(\d{3})_ri(\d+)_(\d+)$'),
     lambda m: dict(mesh=m.group(1), drift=int(m.group(2)), resist_idx=int(m.group(3)))),
    # run_70 style: m141On_mip1p41_006  (plastic threshold only, no HV axis)
    (re.compile(r'^m(\d{3})On_mip(\d)p(\d{2})_(\d{3})$'),
     lambda m: dict(mip=float(f'{m.group(2)}.{m.group(3)}'))),
]
AXES = ['mip', 'drift', 'resist', 'resist_idx', 'mesh']


def parse_subrun(sub):
    """Scan-axis values for a sub-run name, or None if no pattern matches."""
    for rx, fn in SUB_PATTERNS:
        m = rx.match(sub)
        if m:
            return fn(m)
    return None


def cns_subruns(rundir):
    """(subrun, axes-dict) for every sub-run whose combined_hits are CNS-era."""
    out = []
    for sub in sorted(os.listdir(rundir)):
        ax = parse_subrun(sub)
        if ax is None:
            continue
        chd = rundir / sub / 'combined_hits_root'
        files = list(chd.glob('*_feu-combined_hits.root')) if chd.is_dir() else []
        if not files or min(f.stat().st_mtime for f in files) < CNS_CUTOFF:
            continue
        out.append((sub, ax))
    return out


def load_hits(rundir, sub):
    d = rundir / sub / 'combined_hits_root'
    files = [f'{d}/{f}:hits' for f in os.listdir(d) if f.endswith('.root') and '_datrun_' in f]
    return uproot.concatenate(files, ['eventId', 'feu', 'channel', 'amplitude', 'time',
                                      'trigger_timestamp_ns'], library='pd')


def process(run, rundir, detobj, sub, axes):
    df = load_hits(rundir, sub)

    # flash tagging from the combined-hits trigger timestamps (same anchor as
    # flash_timing_lib: first event of each burst, >200 ms gap before it)
    ev = df.groupby('eventId')['trigger_timestamp_ns'].first().sort_values()
    t_ms = ev.values / 1e6
    is_flash = np.concatenate([[True], np.diff(t_ms) > FLASH_GAP_MS])
    t0 = np.maximum.accumulate(np.where(is_flash, t_ms, -1e18))
    ev_dt = pd.Series(t_ms - t0, index=ev.index)
    flash_ids = set(ev.index[is_flash].tolist())

    # denominator: every decoded physics trigger, incl. ones with zero hits after CNS
    ft = FT.load_subrun(run, sub)
    events = pd.DataFrame({'dt_ms': ft['dt_ms']})
    events['subrun'] = sub
    for a in AXES:
        events[a] = axes.get(a, np.nan)

    tracks, counts = [], {}
    for det in DETS:
        dfp = df[df['feu'].isin(DET_FEUS[det]) & (df['amplitude'] >= bt.MIN_HIT_AMP)
                 & (~df['eventId'].isin(flash_ids))].copy()
        if dfp.empty:
            counts[det] = 0
            continue
        dfp = bt.add_xy_pos(dfp, detobj[det])
        xs = np.sort(dfp['x_position_mm'].dropna().unique())
        ys = np.sort(dfp['y_position_mm'].dropna().unique())
        px = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.78
        py = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.78
        trk = bt.collect_all_tracks(dfp, np.array([]), np.array([]), px, py)
        if len(trk) == 0:
            counts[det] = 0
            continue
        trk = trk[['event_id', 'projection', 'n_hits', 'angle_deg',
                   'time_min', 'time_span', 'pos_span']].copy()
        trk['dt_ms'] = trk['event_id'].map(ev_dt)
        trk['det'], trk['subrun'] = det[-1], sub
        for a in AXES:
            trk[a] = axes.get(a, np.nan)
        tracks.append(trk)
        counts[det] = len(trk)

    info = dict(subrun=sub, **{a: axes.get(a, np.nan) for a in AXES},
                n_flash=ft['n_flash'], n_phys=len(events),
                **{f'n_trk_{d[-1]}': counts.get(d, 0) for d in DETS})
    print(f'  {sub}: flashes={ft["n_flash"]} phys={len(events)} '
          + ' '.join(f'{d[-1]}={counts.get(d, 0)}' for d in DETS), flush=True)
    return (pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame()), events, info


def _append(df, path):
    """Append rows to a CSV, aligning to the existing header so an older cache written
    with fewer axis columns stays consistent (missing columns filled with NaN)."""
    if path.exists():
        cols = list(pd.read_csv(path, nrows=0).columns)
        df = df.reindex(columns=cols)
    df.to_csv(path, mode='a', header=not path.exists(), index=False)


def main(run='run_67', cache=None):
    rundir = RUNS_DIR / run
    cache = Path(cache) if cache else (OUT / 'cache')
    cache.mkdir(parents=True, exist_ok=True)

    rc = RunConfig(str(rundir / 'run_config.json'), MAP)
    detobj = {d: rc.get_detector(d) for d in DETS}

    subs = cns_subruns(rundir)
    print(f'{run}: {len(subs)} CNS-reprocessed sub-runs -> {cache}', flush=True)

    done = set()
    tpath, epath, spath = cache / 'tracks.csv', cache / 'events.csv', cache / 'subruns.csv'
    if spath.exists():
        done = set(pd.read_csv(spath)['subrun'])
        print(f'  {len(done)} already cached, skipping', flush=True)

    for sub, axes in subs:
        if sub in done:
            continue
        t, e, i = process(run, rundir, detobj, sub, axes)
        # append incrementally so a kill mid-run keeps what is finished.
        # NEVER run two builders against one cache dir — both append and silently
        # double-count. Check subruns.csv for duplicate sub-run names if unsure.
        if len(t):
            _append(t, tpath)
        _append(e, epath)
        _append(pd.DataFrame([i]), spath)
    print('done ->', cache, flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_67', help='run name under %s' % RUNS_DIR)
    ap.add_argument('--cache', default=None, help='cache dir (default ./cache)')
    a = ap.parse_args()
    main(a.run, a.cache)
