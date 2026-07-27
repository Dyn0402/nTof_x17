#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bunch_join.py -- DREAM event -> n_TOF BunchNumber, the coarse half of the merge.

This is PLAN.md section 3 (a)->(c) implemented as one function. The chain:

    DREAM event --(a)--> burst --(b)--> PS pulse --(c)--> n_TOF BunchNumber

(a) Burst clustering. `trigger_timestamp_ns` (10 ns granularity) split on a 0.5 s
    gap. The FIRST event of a burst is the PS/flash trigger and defines t = 0; the
    N93B gate then admits scintillator singles from ~1 ms (measured flash -> first
    single 1.0045 ms median on run_79/stat090_0000).

(b) Burst -> PS pulse. Already solved by ntof_july_analysis/pulse_match.py, which
    anchors the DREAM clock on the datrun filename and fits the residual offset
    against the DAQ's beam-intensity CSV. run_79/stat090_0000: offset +27.917 s,
    1012/1012 bursts, residual RMS 6 ms.

(c) PS pulse -> BunchNumber. The CSV and PKUP.psTime are the same pulse stream
    with a rigid offset, so once (b) has put the bursts on the CSV clock a single
    constant lands them on psTime. Measured here, not assumed:

        median(burst_epoch - psTime) = 0.8290 s

    which reproduces the NXCALS publication latency PLAN.md measured independently
    from the CSV/PKUP pair. Residual MAD 5.3 ms, max 11.5 ms.

Result on the reference pair (run_79 stat090_0000 <-> 224572): **1012/1012 bursts
matched to a unique bunch, zero duplicates**, and the nearest bunch is >=23x closer
than the runner-up. PLAN.md expected 886/1012 with 126 misses at exactly -1.2 s
(one PS basic period); that deficit does not survive doing the match through
pulse_match's fitted offset, so it was an artifact of the coarse-side scan and NOT
a real n_TOF acquisition gap. Verified independently: PKUP.PulseIntensity and the
pulse_match CSV intensity agree to <= 0.0005e10 on all 1012 bursts, and those come
from two files that share no code path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / 'ntof_july_analysis'))

import pulse_match as pm                                    # noqa: E402
from ntof_dream_merge.ntof_io import pkup_bunches           # noqa: E402

GAP_S = 0.5             # burst split, same convention as pulse_match
MATCH_TOL_S = 0.05      # burst<->bunch accept window; ~10x the 5 ms residual MAD
                        # and ~1/24 of the 1.2 s PS basic period, so it cannot
                        # reach the neighbouring pulse


def dream_events(run: str, subrun: str) -> pd.DataFrame:
    """
    Per-event table for one DREAM subrun, with burst structure and flash timing.

    Columns: eventId, trigger_ns (raw), burst_id, is_flash, t_since_flash_ns.
    Times are kept as int64 nanoseconds throughout -- the intra-burst alignment
    downstream cares about microseconds, and a float64 seconds representation of a
    wall-clock epoch has already lost that.
    """
    eids, tns = [], []
    for f in pm._combined_files(run, subrun):
        with uproot.open(f) as uf:
            if 'hits' not in uf:
                continue
            a = uf['hits'].arrays(['eventId', 'trigger_timestamp_ns'], library='np')
        eid = a['eventId']
        _, first = np.unique(eid, return_index=True)     # one row per event
        eids.append(eid[first])
        tns.append(a['trigger_timestamp_ns'][first])
    if not eids:
        raise FileNotFoundError(f'no combined hits for {run}/{subrun}')

    eid = np.concatenate(eids)
    t = np.concatenate(tns).astype(np.int64)
    eid, idx = np.unique(eid, return_index=True)         # events repeat across parts
    t = t[idx]
    o = np.argsort(t)
    eid, t = eid[o], t[o]

    burst_id = np.cumsum(np.concatenate([[0], np.diff(t) > GAP_S * 1e9]))
    flash_t = pd.Series(t).groupby(burst_id).transform('first').to_numpy()
    is_flash = t == flash_t

    return pd.DataFrame(dict(eventId=eid, trigger_ns=t, burst_id=burst_id,
                             is_flash=is_flash, t_since_flash_ns=t - flash_t))


def burst_epochs(run: str, subrun: str, events: pd.DataFrame | None = None):
    """(burst_id, wall-clock epoch of each burst's flash) using pulse_match's fit."""
    ev = dream_events(run, subrun) if events is None else events
    mr = pm.match_subrun(run, subrun)
    if mr is None:
        raise RuntimeError(f'pulse_match has no result for {run}/{subrun}')
    anchor = pm._anchor_epoch(run, subrun)
    t0 = ev['trigger_ns'].min()
    flash = ev.loc[ev['is_flash'], ['burst_id', 'trigger_ns']]
    epoch = anchor + (flash['trigger_ns'].to_numpy() - t0) / 1e9 + mr['offset_s']
    return flash['burst_id'].to_numpy(), epoch, mr


def dream_event_to_bunch(run: str, subrun: str, ntof_run: int) -> pd.DataFrame:
    """
    The section-3 chain, end to end.

    Returns the per-event table with the n_TOF bunch attached:
      eventId, burst_id, is_flash, t_since_flash_ns,
      BunchNumber, join_resid_s, bunch_intensity_e10, pulse_e10, pstime_recovered
    Events whose burst found no bunch get BunchNumber = -1.
    """
    ev = dream_events(run, subrun)
    bids, epoch, mr = burst_epochs(run, subrun, ev)
    pk = pkup_bunches(ntof_run)
    ps = pk['psTime_s']

    # The offset is DEFINED by the matched pairs, so bootstrap it: coarse scan for
    # the assignment, then take the median of (epoch - psTime) as the constant.
    def assign(delta):
        cand = epoch - delta
        k = np.clip(np.searchsorted(ps, cand), 1, len(ps) - 1)
        return np.where(np.abs(ps[k - 1] - cand) <= np.abs(ps[k] - cand), k - 1, k)

    best_n, best_delta = -1, 0.0
    for delta in np.arange(-3.0, 3.0, 0.001):
        k = assign(delta)
        n = int((np.abs(ps[k] - (epoch - delta)) < MATCH_TOL_S).sum())
        if n > best_n:
            best_n, best_delta = n, float(delta)
    delta = float(np.median(epoch - ps[assign(best_delta)]))
    k = assign(delta)
    resid = (epoch - delta) - ps[k]
    ok = np.abs(resid) < MATCH_TOL_S

    b_bunch = np.where(ok, pk['BunchNumber'][k], -1)
    per_burst = pd.DataFrame(dict(
        burst_id=bids, BunchNumber=b_bunch,
        join_resid_s=np.where(ok, resid, np.nan),
        bunch_intensity_e10=np.where(ok, pk['intensity_e10'][k], np.nan),
        pstime_recovered=np.where(ok, pk['pstime_recovered'][k], False)))

    out = ev.merge(per_burst, on='burst_id', how='left')
    e10 = mr['event_e10']
    out['pulse_e10'] = [e10.get(int(e)) for e in out['eventId']]
    out.attrs.update(delta_s=delta, pulse_match_offset_s=mr['offset_s'],
                     n_bursts=len(bids), n_matched=int(ok.sum()),
                     resid_mad_s=float(np.median(np.abs(resid[ok] - np.median(resid[ok])))),
                     ntof_run=ntof_run, run=run, subrun=subrun)
    return out


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572

    t = dream_event_to_bunch(run, sub, nt)
    a = t.attrs
    print(f'{run}/{sub} <-> n_TOF {nt}')
    print(f'  events {len(t):,}   bursts {a["n_bursts"]}   '
          f'matched {a["n_matched"]} ({100*a["n_matched"]/a["n_bursts"]:.1f} %)')
    print(f'  pulse_match offset {a["pulse_match_offset_s"]:+.3f} s')
    print(f'  burst_epoch - psTime = {a["delta_s"]:.4f} s   '
          f'residual MAD {1e3*a["resid_mad_s"]:.1f} ms')
    b = t.loc[t['BunchNumber'] > 0, 'BunchNumber']
    print(f'  bunches {b.min()}-{b.max()}, {b.nunique()} distinct, '
          f'duplicates {a["n_matched"] - b.nunique()}')
    both = t.dropna(subset=['pulse_e10', 'bunch_intensity_e10'])
    d = (both['bunch_intensity_e10'] - both['pulse_e10']).abs()
    print(f'  intensity cross-check (PKUP vs CSV, independent files): '
          f'median |diff| {d.median():.4f}e10, max {d.max():.4f}e10')
    print(f'  interpolated-psTime events: {int(t["pstime_recovered"].sum()):,}')
