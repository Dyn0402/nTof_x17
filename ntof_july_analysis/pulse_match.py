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
    """Two locks the instruments cannot separate — do not pick silently.

    Carries `.locks`, the candidates that tied: a list of
    {off_s, n, r} with r None where the correlation is undefined. A downstream
    arbiter (coincidence_arbiter) needs exactly this list, and until
    2026-08-13 the only way to get it was to re-parse the message text — which
    dropped every candidate whose r was NaN, because `r=nan` does not match a
    numeric pattern. Short segments are precisely the ambiguous ones AND the
    ones where the intensity correlation is undefined, so that silently hid
    the candidates most in need of arbitration.
    """

    def __init__(self, *args, locks=None):
        super().__init__(*args)
        self.locks = list(locks or [])


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


def _lock_records(locks, limit=8):
    """Candidate locks as plain JSON-able dicts; r is None where undefined."""
    return [dict(off_s=float(L['off_s']), n=int(L['n']),
                 r=None if not np.isfinite(L['r']) else float(L['r']))
            for L in locks[:limit]]


def enumerate_candidates(c_t, sizes, anchor, pt, pe, search_s=None):
    """EVERY candidate alignment, ranked by count. NO decision is made here.

    `search_s` widens the offset scan past SEARCH_S for the segments whose
    true lock lies outside +-120 s of the filename anchor. Measured 2026-08-14
    on the three "dark hours" (run_116/stat090_0027, run_118/stat090_0017,
    run_122/stat090_0000): every candidate inside +-120 s measured 0 %
    coincidence, and a +-200-bunch scan found the lock at +60 bunches =
    +172.8 s, exactly four 43.2 s supercycles away, S/N 1377-1619. Only the
    coincidence may choose among the extra candidates -- a wider count scan
    has MORE degenerate ties, not fewer.

    Returns `locks`, a list of dict(off_s, n, frac, r): refined offset, matched
    clusters, matched fraction, and the cluster-size <-> pulse-intensity
    correlation. Ranked best-count-first and deduped to one entry per lock
    group.

    This is the SEARCH, and it is all the count scan is good for. Choosing
    among these is a separate question answered by a far stronger instrument
    (the wall+plastic coincidence, ntof_processing/slim_pipeline/
    coincidence_arbiter.py): at 50 ms tolerance the count ties routinely under
    the PS supercycle, while the coincidence separates right from wrong locks
    by three orders of magnitude.

    Raises NoLock when nothing matches at all -- that is an absence of data,
    not a decision, so it belongs here. `r` is computed and returned as a
    DIAGNOSTIC; it no longer decides anything for the slim products.
    """
    half = float(SEARCH_S if search_s is None else search_s)
    offsets = np.arange(-half, half + STEP_S, STEP_S)
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
            f'no offset in ±{half:g} s matches the beam record: best '
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
    return locks


def select_lock(c_t, sizes, anchor, pt, pe):
    """Pick a lock by count, then by intensity -- the WEAK path, kept for the
    analysis scripts that have no coincidence available (run30_flash_intensity,
    leadshield_compare and friends read per-event intensity and never build a
    slim). The slim itself does not use this any more: it enumerates and lets
    the coincidence choose, so a tie here can no longer reach a product.

    Returns (offset_s, locks, diag); raises AmbiguousLock when neither count
    nor intensity separates the top locks, because a silent pick on this
    instrument is what cost 25.7 % of the July campaign beam.
    """
    locks = enumerate_candidates(c_t, sizes, anchor, pt, pe)
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
    # the intensity discriminant is carried by sparse schedule-break events
    # (~one per 10-40 min of beam), not accumulated per cluster — a short
    # segment typically contains NONE, so ambiguity there is the expected
    # outcome, not bad luck (join_mislock/arbitration_floor.py: r-separation
    # is a step function of window length; 0.001 apart below it)
    short = (f' Segment is only {len(c_t)} clusters: below ~200 clusters '
             f'intensity arbitration has no power in principle, and the '
             f'bunch-shift scan is the STANDARD route, not an exception.'
             if len(c_t) < 200 else '')
    raise AmbiguousLock(
        f'count margin {margin} (< {MARGIN_CLEAR}) and intensity '
        f'correlation cannot separate the top locks '
        f'(r_sig {sig if sig is None else round(sig, 2)}, need {R_SIG:g}): '
        f'{tab}. This sub-run needs a bunch-shift scan '
        f'(segment_diagnose --span 200) before it can be joined — refusing '
        f'to pick one silently. That silent pick cost 25.7 % of the July '
        f'campaign beam (ntof_processing/join_mislock/).{short}',
        locks=_lock_records(locks))


def enumerate_locks(run: str, subrun: str,
                    search_s: float | None = None) -> dict | None:
    """Candidate locks for one sub-run, with NO decision taken.

    The entry point for the coincidence path: the slim asks for every
    candidate, measures the wall+plastic coincidence at each, and applies the
    winner through `match_subrun(..., accept_offset_s=..., accept_source=
    'coincidence')`. Nothing here can raise AmbiguousLock, because nothing
    here chooses -- ambiguity is only a problem for an instrument that has to
    decide, and this one does not.

    Returns None when the sub-run has no events or the beam CSV does not cover
    it; raises NoLock when no offset matches the beam record at all.

        {run, subrun, locks: [{off_s, n, r}, ...], n_clusters, anchor_epoch}
    """
    eid, t_rel, anchor = _event_times(run, subrun)
    if eid is None or anchor is None:
        return None
    starts = np.concatenate([[0], np.where(np.diff(t_rel) > GAP_S)[0] + 1])
    c_t = t_rel[starts]
    sizes = np.diff(np.r_[starts, len(t_rel)]).astype(float)
    span_s = float(t_rel.max()) if len(t_rel) else 600.0
    pt, pe = _load_pulses([anchor + s_ for s_ in
                           np.arange(0, span_s + 600 + 43200, 43200)])
    if pt.size == 0:
        return None
    locks = enumerate_candidates(c_t, sizes, anchor, pt, pe,
                                 search_s=search_s)
    # EVERY candidate, not the top 8. `_lock_records`' default limit exists
    # for the cached diagnostic record; applied here it silently withheld
    # candidates from the coincidence arbiter -- on 2026-08-13 the three
    # "dark hours" were refused with "best 0 % of 8 measured" while their
    # true lock (~+63 s, well inside +-120 s) was the 9th-or-later count tie
    # and was never offered. Found 2026-08-15 when a +-400 s enumeration
    # returned eight candidates all between -300 and -400 s.
    return dict(run=run, subrun=subrun, locks=_lock_records(locks, limit=None),
                n_clusters=int(len(c_t)), anchor_epoch=anchor,
                t_span_s=span_s,
                search_s=float(SEARCH_S if search_s is None else search_s))


def match_subrun(run: str, subrun: str, rebuild: bool = False,
                 accept_offset_s: float | None = None,
                 accept_source: str = 'verified') -> dict | None:
    """Wall-clock lock for one sub-run; raises rather than guessing.

    `accept_offset_s` is the evidence-backed override: when something OTHER
    than the count scan has established the lock, pass its offset and the
    nearest candidate lock (within LOCK_GROUP_S) is accepted even if the
    automatic selection would be ambiguous. It is an override for ESTABLISHED
    locks, not a seed — an unverified guess here recreates the bug this
    function exists to prevent.

    `accept_source` says WHICH evidence, and is what gets recorded as
    lock_chosen_by so the provenance survives into the slim products:

        'verified'    a hand-run bunch-shift scan (segment_diagnose --span 200)
        'coincidence' the wall+plastic coincidence, via coincidence_arbiter

    The two are not interchangeable and the override result is CACHED, so the
    label outlives the run that made it. Collapsing both to 'verified' would
    make an automatic decision indistinguishable from a human one for as long
    as the cache lives.
    """
    cpath = CACHE / f'{run}_{subrun}.json'
    if cpath.exists() and not rebuild and accept_offset_s is None:
        d = json.loads(cpath.read_text())
        if d.get('cache_version') == 2:
            if d.get('ambiguous'):
                # the candidates too, not just the message: an arbiter reading
                # a CACHED refusal must see the same list as one reading a
                # fresh raise, or it silently has nothing to arbitrate
                raise AmbiguousLock(
                    d.get('error', f'{run}/{subrun}: cached ambiguous lock '
                          '— scan before joining'),
                    locks=d.get('locks'))
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
                error=str(e), locks=getattr(e, 'locks', []),
                anchor_epoch=anchor, t_span_s=span_s)))
            raise
        # scan-verified override: lock onto the verified offset
        best_off = _refine(c_t, anchor, pt, float(accept_offset_s))
        if abs(best_off - accept_offset_s) >= LOCK_GROUP_S:
            raise AmbiguousLock(
                f'no candidate lock within {LOCK_GROUP_S:g} s of the '
                f'verified offset {accept_offset_s:+.2f} s') from e
        locks = []
        diag = dict(margin=None, chosen_by=accept_source, r_sig=None)
    override_disagreed = None
    if auto_ok and accept_offset_s is not None and \
            abs(best_off - accept_offset_s) >= LOCK_GROUP_S:
        # THE OVERRIDE WINS, and says so. This used to raise: a confident
        # automatic selection contradicting the offered offset was treated as
        # unresolvable. That made sense while the only overrides were hand
        # scans; it is wrong now that the offer comes from the wall+plastic
        # coincidence, which separates right from wrong locks by three orders
        # of magnitude (96 % vs 0.00 % measured over ~190 candidate
        # evaluations) while this selection is a cluster count at 50 ms
        # tolerance. The weaker instrument does not get to veto the stronger.
        #
        # Measured 2026-08-13: this guard alone blocked 31 of 36 candidate
        # evaluations on the unmatched campaign -- the coincidence was never
        # allowed to be computed on them.
        #
        # The disagreement is still INFORMATION, so it is recorded rather than
        # discarded: a segment where the count scan and the coincidence point
        # to different locks is worth a human eye even though the coincidence
        # is the one to believe.
        override_disagreed = dict(count_lock_s=float(best_off),
                                  count_margin=diag['margin'],
                                  count_chosen_by=diag['chosen_by'])
        nearest = min(locks, key=lambda L: abs(L['off_s'] - accept_offset_s)) \
            if locks else None
        best_off = (float(nearest['off_s']) if nearest is not None
                    and abs(nearest['off_s'] - accept_offset_s) < LOCK_GROUP_S
                    else _refine(c_t, anchor, pt, float(accept_offset_s)))
        diag = dict(margin=None, chosen_by=accept_source, r_sig=None)

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
               # set only when the count scan pointed elsewhere and was
               # overruled; null on the ordinary path
               lock_override_disagreed=override_disagreed,
               locks=_lock_records(locks),
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
