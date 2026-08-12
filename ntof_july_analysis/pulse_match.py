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

# ---- lock selection (2026-08-12). The count-only scan is DEGENERATE under
# the accelerator supercycle: pulse timing AND the parasitic/dedicated
# intensity schedule repeat (39.6 s / 43.2 s hours measured), so locks whole
# cycles apart match ~100 % of clusters each and the old `n > best_n` scan
# silently kept the most negative tying lock — 107 of 291 campaign segments,
# 25.7 % of attempted beam (ntof_processing/join_mislock/). What the schedule
# repeats, the per-pulse intensity FLUCTUATIONS do not: cluster size vs
# matched-pulse e10 correlation separates true from shifted locks (measured
# r 0.925 true vs 0.508 shifted on a dead 607-607 count tie). Selection is
# count first; near-ties are arbitrated by that correlation; anything the
# instruments cannot separate RAISES instead of returning a silent winner.
LOCK_GROUP_S = 5.0       # offsets closer than this are the same lock
MIN_LOCK_N = 10          # fewer matched clusters than this is not a lock
MIN_LOCK_FRAC = 0.2      # ... and neither is matching <20 % of clusters
MARGIN_CLEAR = 10        # count margin at/above which count wins outright
                         # (margin study: failures all <=8, fitted median 23)
R_SIG = 3.0              # Fisher-z sigmas for intensity arbitration


class NoLock(RuntimeError):
    """No offset matches the beam record — do not guess."""


class AmbiguousLock(RuntimeError):
    """Two locks the instruments cannot separate — do not pick silently."""


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


def _pick(pt, cand):
    """Index of the nearest pulse to each candidate epoch."""
    j = np.searchsorted(pt, cand)
    j0 = np.clip(j - 1, 0, len(pt) - 1)
    j1 = np.clip(j, 0, len(pt) - 1)
    return np.where(np.abs(pt[j0] - cand) <= np.abs(pt[j1] - cand), j0, j1)


def _at_offset(c_t, anchor, pt, off):
    """(pick, residuals, matched mask) for one candidate offset."""
    cand = anchor + c_t + off
    pick = _pick(pt, cand)
    res = pt[pick] - cand
    return pick, res, np.abs(res) < TOL_S


def _refine(c_t, anchor, pt, off):
    """Centre an offset on its own median residual, once."""
    _, res, m = _at_offset(c_t, anchor, pt, off)
    if m.sum() >= 3:
        off += float(np.median(res[m]))
    return float(off)


def select_lock(c_t, sizes, anchor, pt, pe):
    """Choose the wall-clock lock, or refuse loudly.

    Returns (offset_s, locks, diag). `locks` is every candidate alignment as
    dict(off_s, n, frac, r): refined offset, matched clusters, matched
    fraction, and the cluster-size <-> pulse-intensity correlation. `diag`
    records margin / chosen_by / r_sig for provenance.

    Raises NoLock when nothing matches (the old code returned the scan edge
    with 0 matches and everything downstream failed mysteriously), and
    AmbiguousLock when the top locks are separated neither by count
    (>= MARGIN_CLEAR) nor by intensity correlation (>= R_SIG Fisher sigmas).
    An ambiguous sub-run needs a bunch-shift scan
    (slim_pipeline/segment_diagnose.py --span 200), not a coin flip.
    """
    offsets = np.arange(-SEARCH_S, SEARCH_S + STEP_S, STEP_S)
    counts = np.empty(len(offsets), int)
    for i, off in enumerate(offsets):
        cand = anchor + c_t + off
        j = np.searchsorted(pt, cand)
        j0 = np.clip(j - 1, 0, len(pt) - 1)
        j1 = np.clip(j, 0, len(pt) - 1)
        d = np.minimum(np.abs(pt[j0] - cand), np.abs(pt[j1] - cand))
        counts[i] = int((d < TOL_S).sum())

    best_n = int(counts.max()) if len(counts) else 0
    if best_n < MIN_LOCK_N or best_n < MIN_LOCK_FRAC * len(c_t):
        raise NoLock(
            f'no offset in ±{SEARCH_S:g} s matches the beam record: best '
            f'{best_n} of {len(c_t)} clusters. The beam CSV may not cover '
            f'this hour, or the anchor is off by more than the search range. '
            f'Refusing to return a lock rather than guessing.')

    # group contiguous above-threshold offsets into candidate locks
    thr = max(MIN_LOCK_N, int(0.5 * best_n))
    hot = counts >= thr
    cand_offs = []
    i = 0
    while i < len(offsets):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < len(offsets) and hot[j]:
            j += 1
        cand_offs.append(float(offsets[i + int(counts[i:j].argmax())]))
        i = j

    refined = []
    for off0 in cand_offs:
        off = _refine(c_t, anchor, pt, off0)
        pick, res, m = _at_offset(c_t, anchor, pt, off)
        nm = int(m.sum())
        r = float('nan')
        if nm > 10:
            with np.errstate(invalid='ignore'):
                r = float(np.corrcoef(sizes[m], pe[pick[m]])[0, 1])
        refined.append(dict(off_s=off, n=nm, frac=float(m.mean()), r=r))
    refined.sort(key=lambda L: -L['n'])
    locks = []                     # dedupe: keep the strongest of each group
    for L in refined:
        if not any(abs(L['off_s'] - K['off_s']) < LOCK_GROUP_S for K in locks):
            locks.append(L)

    n1 = locks[0]['n']
    margin = n1 - (locks[1]['n'] if len(locks) > 1 else 0)
    if margin >= MARGIN_CLEAR or len(locks) == 1:
        chosen = locks[0]
        diag = dict(margin=margin, chosen_by='count', r_sig=None)
        return chosen['off_s'], locks, diag

    # near-tie: arbitrate among all locks within MARGIN_CLEAR of the best,
    # by the intensity-fluctuation correlation (Fisher z on matched counts)
    cont = [L for L in locks if n1 - L['n'] < MARGIN_CLEAR
            and np.isfinite(L['r'])]
    if len(cont) >= 2:
        cont.sort(key=lambda L: -L['r'])
        za = np.arctanh(np.clip(cont[0]['r'], -0.999999, 0.999999))
        zb = np.arctanh(np.clip(cont[1]['r'], -0.999999, 0.999999))
        se = np.sqrt(1.0 / max(cont[0]['n'] - 3, 1)
                     + 1.0 / max(cont[1]['n'] - 3, 1))
        sig = float((za - zb) / se)
        if sig >= R_SIG:
            diag = dict(margin=margin, chosen_by='intensity', r_sig=sig)
            return cont[0]['off_s'], locks, diag
    else:
        sig = None

    tab = '; '.join(f"{L['off_s']:+.2f}s n={L['n']} r={L['r']:.3f}"
                    for L in locks[:6])
    raise AmbiguousLock(
        f'count margin {margin} (< {MARGIN_CLEAR}) and intensity '
        f'correlation cannot separate the top locks '
        f'(r_sig {sig if sig is None else round(sig, 2)}, need {R_SIG:g}): '
        f'{tab}. This sub-run needs a bunch-shift scan '
        f'(segment_diagnose --span 200) before it can be joined — refusing '
        f'to pick one silently. That silent pick cost 25.7 % of the July '
        f'campaign beam (ntof_processing/join_mislock/).')


def match_subrun(run: str, subrun: str, rebuild: bool = False,
                 accept_offset_s: float | None = None) -> dict | None:
    """Wall-clock lock for one sub-run; raises rather than guessing.

    `accept_offset_s` is the scan-verified override: when a bunch-shift scan
    (segment_diagnose --span 200) has independently established the lock, pass
    its offset and the nearest candidate lock (within LOCK_GROUP_S) is
    accepted even if the automatic selection would be ambiguous. The result
    records lock_chosen_by = 'verified' so the provenance survives into the
    slim products. It is an override for VERIFIED locks, not a seed — an
    unverified guess here recreates the bug this function exists to prevent.
    """
    cpath = CACHE / f'{run}_{subrun}.json'
    if cpath.exists() and not rebuild and accept_offset_s is None:
        d = json.loads(cpath.read_text())
        if d.get('cache_version') == 2:
            if d.get('ambiguous'):
                raise AmbiguousLock(d.get('error', f'{run}/{subrun}: cached '
                                    'ambiguous lock — scan before joining'))
            d['event_e10'] = {int(k): v for k, v in d['event_e10'].items()}
            return d
        # pre-2026-08-12 cache: selected by the count-only scan — rebuild

    eid, t_rel, anchor = _event_times(run, subrun)
    if eid is None or anchor is None:
        return None
    # clusters, and their sizes (events per cluster = the DREAM-side
    # intensity proxy that the arbitration correlates against the pulses)
    starts = np.concatenate([[0], np.where(np.diff(t_rel) > GAP_S)[0] + 1])
    c_t = t_rel[starts]
    sizes = np.diff(np.r_[starts, len(t_rel)]).astype(float)
    c_of_ev = np.searchsorted(starts, np.arange(len(t_rel)), side='right') - 1

    # cover the whole sub-run span (a 24 h sub-run crosses a day boundary)
    span_s = float(t_rel.max()) if len(t_rel) else 600.0
    pt, pe = _load_pulses([anchor + s for s in
                           np.arange(0, span_s + 600 + 43200, 43200)])
    if pt.size == 0:
        return None

    auto_ok = False
    try:
        best_off, locks, diag = select_lock(c_t, sizes, anchor, pt, pe)
        auto_ok = True
    except AmbiguousLock as e:
        if accept_offset_s is None:
            cpath.write_text(json.dumps(dict(
                run=run, subrun=subrun, cache_version=2, ambiguous=True,
                error=str(e), anchor_epoch=anchor, t_span_s=span_s)))
            raise
        # scan-verified override: lock onto the verified offset
        best_off = _refine(c_t, anchor, pt, float(accept_offset_s))
        if abs(best_off - accept_offset_s) >= LOCK_GROUP_S:
            raise AmbiguousLock(
                f'no candidate lock within {LOCK_GROUP_S:g} s of the '
                f'verified offset {accept_offset_s:+.2f} s') from e
        locks = []
        diag = dict(margin=None, chosen_by='verified', r_sig=None)
    if auto_ok and accept_offset_s is not None and \
            abs(best_off - accept_offset_s) >= LOCK_GROUP_S:
        # a CONFIDENT automatic selection contradicting the verified offset
        # is not overridable — something is wrong on one side; look by hand
        raise AmbiguousLock(
            f'verified offset {accept_offset_s:+.2f} s disagrees with the '
            f'confident automatic selection {best_off:+.2f} s '
            f'(margin {diag["margin"]}, by {diag["chosen_by"]}) — resolve '
            f'by hand before joining')

    pick, res, m = _at_offset(c_t, anchor, pt, best_off)
    c_e10 = np.full(len(c_t), np.nan)
    c_e10[m] = pe[pick[m]]
    ev_e10 = c_e10[c_of_ev]

    out = dict(run=run, subrun=subrun, cache_version=2,
               offset_s=best_off,
               n_clusters=int(len(c_t)), n_matched=int(m.sum()),
               match_frac=float(m.mean()),
               resid_rms_ms=float(np.std(res[m]) * 1e3) if m.any() else None,
               lock_margin=diag['margin'], lock_chosen_by=diag['chosen_by'],
               lock_r_sig=diag['r_sig'],
               locks=[dict(off_s=L['off_s'], n=L['n'],
                           r=None if not np.isfinite(L['r']) else L['r'])
                      for L in locks[:8]],
               anchor_epoch=anchor, t_span_s=span_s,
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
