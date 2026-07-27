#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/pulse_match.py — per-event beam-pulse intensity for July runs.

Matches DREAM events to individual PS pulses using the beam_watcher CSV
(`nTof_x17_DAQ/beam_monitor/logs/beam_intensity_<date>.csv`), which logs every
PS basic-period record with a sub-second NXCALS timestamp and the pulse
intensity (1e10 protons; non-nTOF destinations appear as ~0 and are ignored).

Method (descendant of ntof_may_analysis/dream_timber_time_sync_flash.py, but
with a local per-pulse log instead of raw Timber files):
  1. Cluster the subrun's event times (trigger_timestamp_ns) with a 0.5 s gap —
     one cluster per beam pulse (flash blocks: 1 event/cluster; rand/scint
     blocks: one ~30 ms burst/cluster). The cluster START approximates the
     pulse time (flash trigger fires at the pulse).
  2. Anchor the DREAM-relative clock with the datrun filename (…_YYMMDD_HHhMM)
     and fit the residual offset by maximizing the number of clusters matching
     a CSV pulse (intensity > MIN_E10) within TOL_S, scanning ±SEARCH_S.
  3. Every event inherits its cluster's matched pulse intensity (NaN if the
     cluster found no pulse — e.g. beam-off cosmic triggers).

Public API:
    match_subrun(run, subrun) -> dict(offset_s, match_frac, n_clusters,
                                      event_e10={eventId: intensity})
Results are cached in <this dir>/cache_pulse_match/.
"""
from __future__ import annotations
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.beam_july_paths import RUNS_DIR, BEAM_LOG_DIR  # noqa: E402
CACHE = Path(__file__).resolve().parent / 'cache_pulse_match'
CACHE.mkdir(exist_ok=True)

MIN_E10 = 50.0      # a real nTOF pulse
GAP_S = 0.5         # event-time gap that separates pulse clusters
TOL_S = 0.35        # cluster-to-pulse match tolerance
SEARCH_S = 120.0    # offset scan half-range around the filename anchor
STEP_S = 0.05       # offset scan step

_FNAME_T = re.compile(r'_datrun_(\d{6})_(\d{2})H(\d{2})_')


def _combined_files(run, subrun):
    d = RUNS_DIR / run / subrun / 'combined_hits_root'
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir()
                  if f.suffix == '.root' and 'feu-combined' in f.name
                  and '_datrun_' in f.name and '_pedestals_' not in f.name)


def _event_times(run, subrun):
    """(eventIds sorted by time, t_rel_s) from combined hits."""
    eids, tns = [], []
    for f in _combined_files(run, subrun):
        with uproot.open(f) as uf:
            if 'hits' not in uf:
                continue
            a = uf['hits'].arrays(['eventId', 'trigger_timestamp_ns'], library='np')
        eid = a['eventId']
        _, first = np.unique(eid, return_index=True)
        eids.append(eid[first])
        tns.append(a['trigger_timestamp_ns'][first])
    if not eids:
        return None, None, None
    eid = np.concatenate(eids)
    t = np.concatenate(tns).astype(np.float64) / 1e9
    # events can repeat across file parts; keep first
    eid, idx = np.unique(eid, return_index=True)
    t = t[idx]
    o = np.argsort(t)
    anchor = _anchor_epoch(run, subrun)
    return eid[o], t[o] - t.min(), anchor


def _anchor_epoch(run, subrun):
    """Wall-clock epoch of the datrun start from any combined filename."""
    for f in _combined_files(run, subrun):
        m = _FNAME_T.search(f.name)
        if m:
            ymd, hh, mm = m.groups()
            dt = datetime.strptime(f'20{ymd} {hh}:{mm}', '%Y%m%d %H:%M')
            return dt.timestamp()
    return None


def _load_pulses(day_epochs):
    """(t_epoch, e10) of real pulses covering the given epochs (± a day edge)."""
    days = sorted({datetime.fromtimestamp(e).strftime('%Y-%m-%d') for e in day_epochs})
    ts, e10 = [], []
    for day in days:
        p = BEAM_LOG_DIR / f'beam_intensity_{day}.csv'
        if not p.exists():
            continue
        raw = np.genfromtxt(p, delimiter=',', names=True,
                            dtype=None, encoding='utf-8')
        ts.append(np.atleast_1d(raw['unix_ts']).astype(float))
        e10.append(np.atleast_1d(raw['intensity_e10']).astype(float))
    if not ts:
        return np.array([]), np.array([])
    t = np.concatenate(ts)
    i = np.concatenate(e10)
    good = i >= MIN_E10
    o = np.argsort(t[good])
    return t[good][o], i[good][o]


def match_subrun(run: str, subrun: str, rebuild: bool = False) -> dict | None:
    cpath = CACHE / f'{run}_{subrun}.json'
    if cpath.exists() and not rebuild:
        d = json.loads(cpath.read_text())
        d['event_e10'] = {int(k): v for k, v in d['event_e10'].items()}
        return d

    eid, t_rel, anchor = _event_times(run, subrun)
    if eid is None or anchor is None:
        return None
    # clusters
    starts = np.concatenate([[0], np.where(np.diff(t_rel) > GAP_S)[0] + 1])
    c_t = t_rel[starts]
    c_of_ev = np.searchsorted(starts, np.arange(len(t_rel)), side='right') - 1

    # cover the whole sub-run span (a 24 h sub-run crosses a day boundary)
    span_s = float(t_rel.max()) if len(t_rel) else 600.0
    pt, pe = _load_pulses([anchor + s for s in
                           np.arange(0, span_s + 600 + 43200, 43200)])
    if pt.size == 0:
        return None

    # offset scan: DREAM cluster epoch = anchor + t_rel + offset
    offsets = np.arange(-SEARCH_S, SEARCH_S + STEP_S, STEP_S)
    best_n, best_off = -1, 0.0
    for off in offsets:
        cand = anchor + c_t + off
        j = np.searchsorted(pt, cand)
        j0 = np.clip(j - 1, 0, len(pt) - 1)
        j1 = np.clip(j, 0, len(pt) - 1)
        d = np.minimum(np.abs(pt[j0] - cand), np.abs(pt[j1] - cand))
        n = int((d < TOL_S).sum())
        if n > best_n:
            best_n, best_off = n, float(off)
    # refine: median residual at best offset
    cand = anchor + c_t + best_off
    j = np.searchsorted(pt, cand)
    j0 = np.clip(j - 1, 0, len(pt) - 1)
    j1 = np.clip(j, 0, len(pt) - 1)
    pick = np.where(np.abs(pt[j0] - cand) <= np.abs(pt[j1] - cand), j0, j1)
    res = pt[pick] - cand
    m = np.abs(res) < TOL_S
    if m.sum() >= 3:
        best_off += float(np.median(res[m]))
        cand = anchor + c_t + best_off
        j = np.searchsorted(pt, cand)
        j0 = np.clip(j - 1, 0, len(pt) - 1)
        j1 = np.clip(j, 0, len(pt) - 1)
        pick = np.where(np.abs(pt[j0] - cand) <= np.abs(pt[j1] - cand), j0, j1)
        res = pt[pick] - cand
        m = np.abs(res) < TOL_S

    c_e10 = np.full(len(c_t), np.nan)
    c_e10[m] = pe[pick[m]]
    ev_e10 = c_e10[c_of_ev]

    out = dict(run=run, subrun=subrun, offset_s=best_off,
               n_clusters=int(len(c_t)), n_matched=int(m.sum()),
               match_frac=float(m.mean()),
               resid_rms_ms=float(np.std(res[m]) * 1e3) if m.any() else None,
               event_e10={int(e): (None if np.isnan(v) else float(v))
                          for e, v in zip(eid, ev_e10)})
    cpath.write_text(json.dumps(out))
    out['event_e10'] = {int(k): v for k, v in out['event_e10'].items()}
    return out


if __name__ == '__main__':
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else 'run_30'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'flashOn_A700_00'
    r = match_subrun(run, sub, rebuild='--rebuild' in sys.argv)
    if r is None:
        print('no data')
    else:
        e10s = np.array([v for v in r['event_e10'].values() if v is not None])
        print(f"{run}/{sub}: offset {r['offset_s']:+.2f} s, matched "
              f"{r['n_matched']}/{r['n_clusters']} clusters "
              f"({r['match_frac']:.0%}), resid RMS {r['resid_rms_ms']:.0f} ms")
        if e10s.size:
            print(f"  intensity: median {np.median(e10s):.0f}e10, "
                  f"quartiles {np.percentile(e10s, 25):.0f}-{np.percentile(e10s, 75):.0f}, "
                  f"range {e10s.min():.0f}-{e10s.max():.0f}")
