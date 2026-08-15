#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pulse_ledger.py -- every DREAM burst, accounted for.

The QA question Dylan actually asks is not "did the segments that fitted fit
well" but "how many of the pulses we took were NOT confidently matched to an
n_TOF pulse, and why, per sub-run" -- with 99 % matched as the target for runs
after the trigger system locked (~run_79). No existing product can answer it:
bursts whose BunchNumber never resolved are dropped uncounted at the join,
clockfit's efficiency excludes unfitted bunches from numerator AND denominator,
and a sub-run that fails to join never enters any QA denominator at all.

This module owns the DENOMINATOR. It walks every DREAM burst of every sub-run
(from the DREAM files alone -- not from the products, not from the segments
that happened to fit) and assigns each one exactly one terminal state:

    MATCHED           >= ACCEPT_FRAC of its physics triggers have a same-arm
                      wall+plastic partner inside the accept window
    LOW_COINC         joined to a bunch, coincidence below the bar -- the real
                      follow-up list, with the leg breakdown attached
    UNKNOWN_COINC     joined, but the product predates the per-pulse rows in
                      clock_qa.json (re-run clock_qa on it to resolve)
    TOO_FEW_TRIGGERS  too few physics triggers to judge an 80 % bar
    EMPTY_PULSE       the PS pulse carried < EMPTY_PULSE_E10 protons
                      (diverted/dud) -- expected unmatched, not a failure
    NTOF_NO_BUNCH     a real beam pulse, but no n_TOF run covers its epoch:
                      DAQ reset / inter-run gap / acquisition off
    NO_BEAM_PULSE     the burst matches no PS pulse at all (beam-off or cosmic
                      block) -- excluded from the 99 % denominator but shown
    UNJOINED          inside an n_TOF run that joined, but this burst is not in
                      the join's burst->bunch map
    SEGMENT_FAILED    the (sub-run x n_TOF run) segment never joined; carries
                      the refusal reason
    NOT_ATTEMPTED     an n_TOF run covers the epoch but no segment was ever run

Inputs, and the contract with the slim side (2026-08-13, agreed with the
matching session):
  * the DREAM combined files            -> burst census (cached per sub-run)
  * cache_pulse_match/<run>_<subrun>    -> the wall-clock lock, or its refusal
  * the beam-intensity CSVs (ALL rows,  -> pulse existence + emptiness
    unlike pulse_match which cuts at 50e10)
  * slim_study/coverage_inputs/ntof_index_times.txt -> n_TOF run coverage
  * per-segment calibration.json "join" block, EXTENDED with the burst->bunch
    map {burst_id[], bunch[], resid_ms[]} the join already computes
  * per-segment clock_qa.json "pulses" block, EXTENDED with per-bunch arrays
    {bunch[], n_trig[], n_coinc[], wall_only[], pss_only[], neither[],
    wrong_arm[]}
  * the campaign inventory.csv          -> status + reason for failed segments

Everything here degrades gracefully when the extended fields are absent
(products from before the contract): joined bursts fall back to
UNKNOWN_COINC and the counts stay honest -- unknown is reported as unknown,
never as matched.

CLI:
    python3 pulse_ledger.py census <dream_run> [subrun]        build the census
    python3 pulse_ledger.py classify <dream_run> [subrun] --qa-root DIR
    python3 pulse_ledger.py campaign --qa-root DIR [--since-run 79]
Ledgers land in --out (default <qa-root>/pulse_ledger/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / 'ntof_july_analysis'))

from common.beam_july_paths import BEAM_LOG_DIR, RUNS_DIR      # noqa: E402
from ntof_processing.slim_pipeline import config as C          # noqa: E402
import pulse_match as pm                                       # noqa: E402

CACHE = HERE / 'cache_burst_census'
INDEX_TIMES = HERE.parent / 'slim_study' / 'coverage_inputs' / 'ntof_index_times.txt'

STATES = ('MATCHED', 'LOW_COINC', 'UNKNOWN_COINC', 'TOO_FEW_TRIGGERS',
          'EMPTY_PULSE', 'NTOF_NO_BUNCH', 'NO_BEAM_PULSE',
          'NOT_COINC_TRIGGERED', 'UNJOINED', 'SEGMENT_FAILED',
          'NOT_ATTEMPTED')
# States that a perfect chain could still not match -- they come out of the
# 99 % denominator. UNKNOWN/UNJOINED/FAILED stay IN it: they are our problem.
NOT_OURS = frozenset({'EMPTY_PULSE', 'NO_BEAM_PULSE', 'NOT_COINC_TRIGGERED'})
# n_TOF not recording is a THIRD kind: real beam pulses DREAM triggered on,
# for which no n_TOF data exists (run transitions, DAQ resets, one 25-min
# gap). Fully understood, irrecoverable, and worth its own headline ("how much
# did n_TOF miss") -- so it is reported separately and, since 2026-08-15 at
# Dylan's request, kept OUT of the matching denominator like the empty pulses,
# rather than mixed in with LOW_COINC / SEGMENT_FAILED which are ours to fix.
NTOF_OFF = frozenset({'NTOF_NO_BUNCH'})
NOT_IN_DENOM = NOT_OURS | NTOF_OFF

# Trigger-mode lists, DERIVED FROM OBSERVATION (the 08-13 clock_qa sweep
# aggregate, 392 slims), not from the trigger definition: the N1081B config is
# not shipped with the products, so mode is keyed on the sub-run name prefix.
# The sweep's per-mode coincidence separates them cleanly -- stat090 95.9 %,
# acmeshOff 95.7 %, mOn 96.0 % (mesh/HV scans: detector settings vary, the
# trigger is still the wall+plastic coincidence) versus scint 64 %, sngPSmesh
# 38 %, scintd 31 %, frand 26 % (not coincidence-triggered, cannot match by
# construction). A first stat090-only rule would have discarded ~10k good
# acmeshOff/mOn pulses and hidden their 14 real misses; classifying a MODE by
# its 20-segment aggregate is not the per-segment circularity trap (one
# mis-locked segment cannot drag a mode from 96 % to 26 %). Revisit against
# the DAQ-side N1081B configs when they are to hand.
COINC_TRIGGER_PREFIXES = ('stat090', 'acmeshOff', 'mOn')
NON_COINC_PREFIXES = ('scint', 'scintd', 'sngPSmesh', 'frand')

ACCEPT_FRAC = C.PULSE_MIN_FRAC   # single source (config.py, contract item
                                 # delivered 2026-08-13); populations sit at
                                 # 96 % and 0.00 %, the value between them
                                 # cannot matter
MIN_JUDGE_TRIG = 10     # below this an 80 % bar is a coin toss, not a verdict
CSV_TOL_S = pm.TOL_S    # burst-to-CSV-pulse tolerance, same as the lock scan
RUN_EDGE_PAD_S = 2.0    # index times are first/last BUNCH; a pulse this close
                        # to the edge still belongs to the run


# --------------------------------------------------------------- burst census

def burst_census(run: str, subrun: str, rebuild: bool = False) -> dict | None:
    """Every burst of the sub-run, from the DREAM files alone. Cached.

    Reuses pulse_match's event reader and the SAME 0.5 s gap convention as
    bunch_join.dream_events, so burst_id here is burst_id everywhere.
    """
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f'{run}_{subrun}.json'
    if cpath.exists() and not rebuild:
        return json.loads(cpath.read_text())
    eid, t_rel, anchor = pm._event_times(run, subrun)
    if eid is None or anchor is None:
        return None
    starts = np.concatenate([[0], np.where(np.diff(t_rel) > pm.GAP_S)[0] + 1])
    sizes = np.diff(np.r_[starts, len(t_rel)])
    out = dict(run=run, subrun=subrun, anchor_epoch=anchor,
               n_bursts=int(len(starts)),
               burst_id=list(range(len(starts))),
               t_rel_s=[round(float(t), 4) for t in t_rel[starts]],
               # first trigger of a burst is the flash; physics = size - 1
               n_trig=[int(s) for s in sizes])
    cpath.write_text(json.dumps(out))
    return out


def write_census_from_events(ev, run: str, subrun: str, out_dir: Path):
    """The census, computed in-job from bunch_join.dream_events' frame.

    For the slim job (contract 2026-08-13): `ev` is the frame join_events
    already built -- eventId / trigger_ns / burst_id / is_flash -- so this is
    ~1 s of numpy on data already in memory, no second pass over the DREAM
    files. Writes burst_census.json next to burst_map.json; identical content
    to burst_census() because dream_events uses the same reader, dedup and
    0.5 s gap. The census is a property of the SUB-RUN, so any segment's copy
    serves; the ledger reads whichever it finds first.

        from ntof_processing.slim_pipeline import pulse_ledger
        pulse_ledger.write_census_from_events(ev, seg.dream_run,
                                              seg.dream_subrun, out_dir)

    Call it with the UNFILTERED frame (before the BunchNumber > 0 cut) or the
    denominator is wrong by exactly the bursts it exists to count.
    """
    t = np.asarray(ev['trigger_ns'], np.int64)
    bid = np.asarray(ev['burst_id'], int)
    starts = np.r_[0, np.flatnonzero(np.diff(bid)) + 1]
    sizes = np.diff(np.r_[starts, len(bid)])
    t_rel = (t[starts] - t.min()) / 1e9
    anchor = pm._anchor_epoch(run, subrun)
    out = dict(run=run, subrun=subrun, anchor_epoch=anchor,
               n_bursts=int(len(starts)),
               burst_id=[int(b) for b in bid[starts]],
               t_rel_s=[round(float(x), 4) for x in t_rel],
               n_trig=[int(s) for s in sizes])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'burst_census.json').write_text(json.dumps(out))
    return out


def harvest_censuses(qa_root: Path, log=print):
    """Copy in-job burst_census.json files into the local census cache."""
    CACHE.mkdir(exist_ok=True)
    n = 0
    for p in qa_root.rglob('burst_census.json'):
        c = json.loads(p.read_text())
        dst = CACHE / f'{c["run"]}_{c["subrun"]}.json'
        if not dst.exists():
            dst.write_text(json.dumps(c))
            n += 1
    log(f'harvested {n} new census(es) from {qa_root}')
    return n


def subruns_of(run: str):
    """Sub-run names for one DREAM run, from the local tree."""
    d = RUNS_DIR / run
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / 'combined_hits_root').is_dir())


# ------------------------------------------------------------------- beam side

def load_all_pulses(day_epochs):
    """(t_epoch, e10) of EVERY logged PS pulse -- no intensity cut.

    pulse_match cuts at 50e10 because only real pulses can anchor a lock; the
    ledger needs the empty ones too, because "empty" is a terminal state.
    """
    from datetime import datetime
    days = sorted({datetime.fromtimestamp(e).strftime('%Y-%m-%d')
                   for e in day_epochs})
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
    o = np.argsort(t)
    return t[o], i[o]


# ntof_index_times.txt is on LOCAL Geneva time, 2 h ahead of the psTime/CSV
# base everything else here lives on. The constant has ONE home,
# coverage_map.INDEX_LOCAL_SHIFT_S, with both independent measurements (raw
# mtimes over 109 runs; PKUP psTime on 224572) documented beside it --
# redeclaring it here is how the ledger and the segment proposals would
# silently drift apart. A PKUP cache, where present, overrides the index
# span entirely.
from ntof_processing.slim_study.coverage_map import (        # noqa: E402
    INDEX_LOCAL_SHIFT_S as INDEX_TIME_OFFSET_S)


def load_index_times(path: Path = INDEX_TIMES):
    """{ntof_run: (first_bunch_epoch, last_bunch_epoch, n_bunches)},
    shifted onto the psTime base (see INDEX_TIME_OFFSET_S)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        r, t0, t1, n = line.split()
        out[int(r)] = (float(t0) + INDEX_TIME_OFFSET_S,
                       float(t1) + INDEX_TIME_OFFSET_S, int(n))
    return out


PKUP_CACHE = HERE / 'cache_pkup_times'
DELTA_PKUP_S = 0.829    # burst_epoch - psTime. MEASURED, not assumed: median
                        # 0.8290 s / MAD 0.1 ms / range 0.8221-0.8396 over all
                        # 241 fitted segments of the 8-13 recovery campaign.
PKUP_TOL_S = 0.35       # pulse-existence tolerance about epoch - DELTA


def build_pkup_cache(ntof_runs, log=print):
    """Per-run {bunch, ps, e10} npz via the PRODUCTION reader.

    This is the per-pulse existence table the ledger uses to tell an in-run
    DAQ gap from a join miss. It calls ntof_io.pkup_bunches -- the production
    PKUP reader, including its psTime grid repair -- so there is no parallel
    interpretation of the file; run it wherever the n_TOF files are reachable
    (lxplus, or a box with the bunchidx caches).
    """
    from ntof_dream_merge import ntof_io
    PKUP_CACHE.mkdir(exist_ok=True)
    for r in ntof_runs:
        p = PKUP_CACHE / f'{r}.npz'
        if p.exists():
            continue
        try:
            pk = ntof_io.pkup_bunches(int(r))
        except Exception as e:                        # noqa: BLE001
            log(f'  {r}: {type(e).__name__}: {e}')
            continue
        np.savez_compressed(p, bunch=pk['BunchNumber'],
                            ps=pk['psTime_s'], e10=pk['intensity_e10'])
        log(f'  {r}: {len(pk["BunchNumber"])} bunches cached')


def pkup_lookup(ntof_run):
    """bunch-at-epoch lookup from the cache, or None when not built yet."""
    p = PKUP_CACHE / f'{ntof_run}.npz'
    if not p.exists():
        return None
    z = np.load(p)
    ps, bn = z['ps'], z['bunch']
    o = np.argsort(ps)
    ps, bn = ps[o], bn[o]

    def look(epoch, delta=DELTA_PKUP_S):
        cand = epoch - delta
        j = np.clip(np.searchsorted(ps, cand), 1, len(ps) - 1)
        k = j - 1 if abs(ps[j - 1] - cand) <= abs(ps[j] - cand) else j
        return int(bn[k]) if abs(ps[k] - cand) < PKUP_TOL_S else None
    return look


def run_covering(index, epoch):
    """The n_TOF run whose bunch span covers this epoch, or None."""
    for r, (t0, t1, _) in index.items():
        if t0 - RUN_EDGE_PAD_S <= epoch <= t1 + RUN_EDGE_PAD_S:
            return r
    return None


def nearest_gap(index, epoch):
    """'between 224601 and 224602, 3.2 min into a 7.5 min gap' -- the reason
    string for NTOF_NO_BUNCH, so the follow-up list names the gap, not just
    the fact."""
    ends = sorted((t1, r) for r, (t0, t1, _) in index.items() if t1 < epoch)
    starts = sorted((t0, r) for r, (t0, t1, _) in index.items() if t0 > epoch)
    if not ends or not starts:
        side = 'before first' if not ends else 'after last'
        return f'{side} indexed n_TOF run'
    (t1, ra), (t0, rb) = ends[-1], starts[0]
    return (f'between {ra} and {rb}, {(epoch - t1) / 60:.1f} min into a '
            f'{(t0 - t1) / 60:.1f} min gap')


# ------------------------------------------------------------ product sidecars

def collect_segments(qa_root: Path, run: str, subrun: str):
    """Per (sub-run x n_TOF run) sidecars found under qa_root.

    -> {ntof_run: dict(join=..., pulses=..., verdict=...)}; missing pieces are
    None. Directory layout is the campaign one: .../runs/<run>/<subrun>/ntof_hits/.
    """
    segs = {}
    for d in qa_root.rglob(f'{run}/{subrun}/ntof_hits'):
        cal, cq = d / 'calibration.json', d / 'clock_qa.json'
        join = pulses = verdict = ntof_run = burst_map = None
        if cal.exists():
            j = json.loads(cal.read_text())
            join = j.get('join')
        if cq.exists():
            q = json.loads(cq.read_text())
            pulses = q.get('pulses')
            verdict = q.get('verdict')
            ntof_run = (q.get('segment') or {}).get('ntof_run')
        # burst_map.json is its own sidecar (contract 2026-08-13): ALL bursts
        # of the sub-run, bunch = -1 where the join found none, resid in ms
        bmp = d / 'burst_map.json'
        if bmp.exists():
            burst_map = json.loads(bmp.read_text())
        elif join is not None:
            burst_map = join.get('burst_map')   # early drafts put it here
        if ntof_run is None and join is not None:
            ntof_run = join.get('ntof_run')
        if ntof_run is None:
            continue
        rec = dict(join=join, pulses=pulses, verdict=verdict,
                   burst_map=burst_map)
        # a qa_root can hold several vintages of the same segment (a stale
        # pre-sweep mirror next to the swept tree) -- keep the one carrying
        # the most contract fields, not whichever the walk found last
        def _score(r):
            return (2 * bool(r['pulses'] and 'bunch' in r['pulses'])
                    + bool(r['burst_map']))
        old = segs.get(int(ntof_run))
        if old is None or _score(rec) > _score(old):
            segs[int(ntof_run)] = rec
    return segs


def load_inventory(qa_root: Path):
    """{(run, subrun, ntof_run): (status, reason)} for failed segments.

    Two sources, summaries winning over the CSV: the campaign inventory CSV
    (status only), and slim_summary_*.json whose records now carry
    rec['arbiter'] on a refusal (contract 2026-08-13) -- its `reason` keeps
    "no candidate lock reached 80%" distinguishable from "the coincidence
    never got to decide this segment", which is exactly the distinction the
    follow-up list must preserve.
    """
    out = {}
    for p in list(qa_root.glob('inventory*.csv')) + \
            list(qa_root.glob('slim_inventory*.csv')):
        import csv
        with open(p) as f:
            for row in csv.DictReader(f):
                key = (row.get('dream_run'), row.get('sub_run'),
                       int(row['ntof_run']))
                out[key] = (row.get('status', '?'),
                            row.get('reason') or row.get('error') or '')
    for p in qa_root.rglob('slim_summary_*.json'):
        try:
            s = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        recs = s if isinstance(s, list) else s.get('segments', [])
        for rec in recs:
            key = (rec.get('dream_run'), rec.get('sub_run') or
                   rec.get('dream_subrun'), int(rec.get('ntof_run', 0)))
            arb = rec.get('arbiter') or {}
            reason = arb.get('reason') or rec.get('error') or ''
            out[key] = (rec.get('status', '?'), reason)
    return out


# -------------------------------------------------------------- classification

def _coinc_lookup(pulses):
    """bunch -> (frac, n_trig, legs dict) from the extended clock_qa arrays,
    or None when the product predates them."""
    if not pulses or 'bunch' not in pulses:
        return None
    b = np.asarray(pulses['bunch'], int)
    nt = np.asarray(pulses['n_trig'], float)
    nc = np.asarray(pulses['n_coinc'], float)
    legs = {k: np.asarray(pulses.get(k, np.zeros(len(b))), int)
            for k in ('wall_only', 'pss_only', 'neither', 'wrong_arm')}
    idx = {int(x): i for i, x in enumerate(b)}

    def look(bunch):
        i = idx.get(int(bunch))
        if i is None:
            return None
        n = nt[i]
        return (float(nc[i] / n) if n else 0.0, int(n),
                {k: int(v[i]) for k, v in legs.items()})
    return look


def classify_subrun(run: str, subrun: str, qa_root: Path,
                    index=None, inventory=None) -> dict | None:
    """One terminal state per burst. The heart of the ledger."""
    cen = burst_census(run, subrun)
    if cen is None:
        return None
    index = load_index_times() if index is None else index
    inventory = load_inventory(qa_root) if inventory is None else inventory
    anchor = cen['anchor_epoch']
    t_rel = np.asarray(cen['t_rel_s'], float)
    n_trig = np.asarray(cen['n_trig'], int)
    nb = len(t_rel)

    state = np.full(nb, -1, int)
    ntof_run = np.zeros(nb, int)
    bunch = np.full(nb, -1, int)
    frac = np.full(nb, np.nan)
    reasons: list[str] = []
    reason_idx = np.full(nb, -1, int)

    def set_reason(mask, text):
        reasons.append(text)
        reason_idx[mask] = len(reasons) - 1

    # Cosmic-bounce blocks are not beam-triggered BY CONSTRUCTION -- they
    # cannot match, the coverage map proposes them anyway, and letting their
    # NoLock land in SEGMENT_FAILED puts them in the 99 % denominator and
    # understates the headline (found on the 08-13 re-slim: 6 of the first 10
    # "failures" were cosbounce sub-runs with 1-6 clusters).
    if 'cosbounce' in subrun:
        state[:] = STATES.index('NO_BEAM_PULSE')
        set_reason(np.ones(nb, bool),
                   'cosmic-bounce block -- not beam-triggered by construction')
        return _emit(run, subrun, cen, state, ntof_run, bunch, frac,
                     reasons, reason_idx, lock=None)
    if subrun.startswith(NON_COINC_PREFIXES):
        # on beam, but not triggered on the wall+plastic coincidence, so the
        # coincidence is absent by construction. A mode in NEITHER list stays
        # in the denominator on purpose -- unclassified modes should show up
        # as misses and force a classification, not vanish quietly.
        state[:] = STATES.index('NOT_COINC_TRIGGERED')
        set_reason(np.ones(nb, bool),
                   f'{subrun.split("_")[0]} trigger mode -- not triggered on '
                   f'the wall+plastic coincidence, cannot match by '
                   f'construction (mode list from the 08-13 sweep aggregate)')
        return _emit(run, subrun, cen, state, ntof_run, bunch, frac,
                     reasons, reason_idx, lock=None)

    segs = collect_segments(qa_root, run, subrun)

    # ---- the wall-clock lock: prefer the one RECORDED in a product sidecar
    # (calibration.json join block) -- it is the lock the products were built
    # with, and it makes classification runnable on a host with neither the
    # DREAM files nor the pulse_match cache. Fall back to pulse_match.
    offset = chosen_by = None
    lock_err = None
    for nr, seg in sorted(segs.items()):
        j = seg.get('join') or {}
        if j.get('pulse_match_offset_s') is not None:
            offset = float(j['pulse_match_offset_s'])
            chosen_by = j.get('pulse_match_chosen_by')
            break
    if offset is None:
        try:
            mr = pm.match_subrun(run, subrun)
            if mr is None:
                # No DREAM data and no cache ON THIS HOST. If the campaign
                # record says every attempted segment of this sub-run FAILED,
                # that verdict stands without a lock -- these pulses belong in
                # the denominator as failures, and hiding them behind
                # lock_pending inflated the headline (48 sub-runs, found on
                # the first 98.7 % pass). Only a sub-run with NO recorded
                # attempt is a genuine availability gap.
                tried = {k: v for k, v in inventory.items()
                         if k[0] == run and k[1] == subrun}
                if tried and all(v[0] != 'OK' for v in tried.values()):
                    why = '; '.join(f'x {k[2]}: {v[0]}'
                                    + (f' -- {v[1][:200]}' if v[1] else '')
                                    for k, v in sorted(tried.items()))
                    state[:] = STATES.index('SEGMENT_FAILED')
                    set_reason(np.ones(nb, bool), why)
                    return _emit(run, subrun, cen, state, ntof_run, bunch,
                                 frac, reasons, reason_idx, lock=None)
                return _emit(run, subrun, cen, state, ntof_run, bunch, frac,
                             reasons, reason_idx, lock=None,
                             lock_pending='lock not computable on this host '
                                          '(no DREAM data, no cached lock, '
                                          'no product sidecar)')
            offset, chosen_by = mr['offset_s'], mr.get('lock_chosen_by')
        except pm.AmbiguousLock as e:
            lock_err = f'ambiguous lock: {e}'
        except pm.NoLock as e:
            lock_err = f'no lock: {e}'

    if offset is None:
        state[:] = STATES.index('SEGMENT_FAILED')
        set_reason(np.ones(nb, bool), lock_err)
        return _emit(run, subrun, cen, state, ntof_run, bunch, frac,
                     reasons, reason_idx, lock=None)

    epoch = anchor + t_rel + offset

    # ---- CSV pulse under each burst (all intensities)
    span = [epoch.min(), epoch.max()]
    pt, pe = load_all_pulses(np.arange(span[0], span[1] + 43200, 43200))
    # A missing CSV day is a hole on THIS HOST, not evidence of no beam --
    # without this guard every burst of an uncovered sub-run classified
    # NO_BEAM_PULSE (157k bursts on the first full run, laptop mirror ends
    # 07-28; the full set lives on EOS july_beam/slow_control).
    if pt.size == 0 or pt.min() > span[0] or pt.max() < span[1]:
        return _emit(run, subrun, cen, state, ntof_run, bunch, frac,
                     reasons, reason_idx, lock=dict(offset_s=offset,
                                                    chosen_by=chosen_by),
                     lock_pending='beam CSV does not cover this sub-run on '
                                  'this host -- sync beam_intensity or run '
                                  'where it is complete')
    if pt.size:
        j = np.clip(np.searchsorted(pt, epoch), 1, len(pt) - 1)
        near = np.where(np.abs(pt[j - 1] - epoch) <= np.abs(pt[j] - epoch),
                        j - 1, j)
        has_pulse = np.abs(pt[near] - epoch) < CSV_TOL_S
        e10 = np.where(has_pulse, pe[near], np.nan)
    else:
        has_pulse = np.zeros(nb, bool)
        e10 = np.full(nb, np.nan)

    # ABSENCE of a CSV row is held back until AFTER the join: the beam-watcher
    # log has gaps (dropped NXCALS records), and a burst the join matched to a
    # real n_TOF bunch is real beam no matter what the CSV failed to log.
    # Applying NO_BEAM_PULSE first cost exactly 30 matched pulses across 15
    # sub-runs (found 2026-08-13 when the ledger and the per-pulse arrays
    # disagreed by precisely that count: full 72-90-trigger bursts, epochs on
    # the psTime grid to <= 6 ms, PKUP bunch present -- CSV row absent).
    # EMPTY_PULSE stays pre-join: it is a POSITIVE statement (row present,
    # intensity measured low), and deferring it let the PKUP fallback map
    # 3,335 empty-pulse dark-count bursts into TOO_FEW_TRIGGERS inside the
    # denominator. Absence defers to evidence; a measurement does not.
    no_pulse = ~has_pulse
    empty = has_pulse & (e10 < C.EMPTY_PULSE_E10)
    state[empty] = STATES.index('EMPTY_PULSE')
    if empty.any():
        set_reason(empty, f'PS pulse under {C.EMPTY_PULSE_E10:g}e10 protons')

    # ---- the join, segment by segment
    def score(i, b, nr, look):
        """One burst against its bunch's per-pulse row."""
        ntof_run[i], bunch[i] = nr, b
        if n_trig[i] - 1 < MIN_JUDGE_TRIG:
            state[i] = STATES.index('TOO_FEW_TRIGGERS')
            return
        hit = look(b) if look else None
        if hit is None:
            state[i] = STATES.index('UNKNOWN_COINC')
            return
        frac[i], _, legs = hit
        if frac[i] >= ACCEPT_FRAC:
            state[i] = STATES.index('MATCHED')
        else:
            state[i] = STATES.index('LOW_COINC')
            worst = max(legs, key=legs.get)
            reasons.append(f'coincidence {frac[i]:.0%}; dominant miss: '
                           f'{worst} ({legs[worst]} of {n_trig[i] - 1})')
            reason_idx[i] = len(reasons) - 1

    for nr, seg in segs.items():
        look = _coinc_lookup(seg.get('pulses'))
        bm = seg.get('burst_map') or {}
        if bm:
            bid = np.asarray(bm.get('burst_id', []), int)
            bbn = np.asarray(bm.get('bunch', []), int)
            joined = bbn >= 0        # the map carries unmatched bursts as -1
            for i, b in zip(bid[joined], bbn[joined]):
                if i < nb and state[i] < 0:
                    score(i, b, nr, look)
            continue
        # PRE-CONTRACT product: no burst_map, but the per-pulse arrays exist
        # (the 08-13 clock_qa sweep rebuilt them from the stored trees). The
        # burst->bunch step is REPRODUCED, not re-decided: nearest PKUP psTime
        # at the product's own recorded delta -- the same arithmetic and the
        # same delta the shipped join used.
        jd = (seg.get('join') or {}).get('delta_s')
        pk = pkup_lookup(nr) if (look and jd is not None) else None
        if pk is None:
            continue
        for i in np.flatnonzero(state < 0):
            b = pk(epoch[i], jd)
            if b is not None:
                score(i, b, nr, look)
    if any(s >= 0 and STATES[s] == 'UNKNOWN_COINC' for s in state):
        m = state == STATES.index('UNKNOWN_COINC')
        set_reason(m, 'joined, but no per-pulse row resolves this burst -- '
                      'stale clock_qa or bunch outside the arrays')

    # deferred absence verdict: only bursts the join could NOT vouch for
    m = no_pulse & (state < 0)
    state[m] = STATES.index('NO_BEAM_PULSE')
    if m.any():
        set_reason(m, 'no PS pulse within tolerance of the burst epoch')

    # ---- the rest: real pulses that never joined
    pkup = {}
    for i in np.flatnonzero(state < 0):
        r = run_covering(index, epoch[i])
        if r is None:
            state[i] = STATES.index('NTOF_NO_BUNCH')
            reasons.append(nearest_gap(index, epoch[i]))
            reason_idx[i] = len(reasons) - 1
            continue
        ntof_run[i] = r
        # per-pulse existence from PKUP where the cache is built; a run-span
        # answer is only span-level and the reason string says so
        if r not in pkup:
            pkup[r] = pkup_lookup(r)
        b_at = pkup[r](epoch[i]) if pkup[r] else None
        if pkup[r] and b_at is None:
            state[i] = STATES.index('NTOF_NO_BUNCH')
            reasons.append(f'inside n_TOF {r} span but no PKUP bunch at this '
                           f'pulse (in-run DAQ gap)')
            reason_idx[i] = len(reasons) - 1
            continue
        if b_at is not None:
            bunch[i] = b_at
        if r in segs:
            if segs[r].get('burst_map'):
                # the map exists and says this burst found no bunch -- a real
                # miss inside a run that was otherwise joined
                state[i] = STATES.index('UNJOINED')
                pk_note = ('' if b_at is None else
                           f' (PKUP has bunch {b_at} at this pulse)')
                reasons.append(f'inside n_TOF {r} which joined, but the join '
                               f'matched no bunch to this burst{pk_note}')
            else:
                # the product simply predates the burst_map sidecar: we know
                # the segment joined, we cannot say whether THIS burst did.
                # Unknown is reported as unknown, never as a miss.
                state[i] = STATES.index('UNKNOWN_COINC')
                reasons.append(f'n_TOF {r} joined but its product predates '
                               f'burst_map.json -- re-slim or re-run clock_qa')
        else:
            st, why = inventory.get((run, subrun, r), (None, ''))
            if st in (None, ''):
                state[i] = STATES.index('NOT_ATTEMPTED')
                reasons.append(f'n_TOF {r} covers this pulse; no segment run')
            else:
                state[i] = STATES.index('SEGMENT_FAILED')
                reasons.append(f'segment x {r}: {st}'
                               + (f' -- {why}' if why else ''))
        reason_idx[i] = len(reasons) - 1

    return _emit(run, subrun, cen, state, ntof_run, bunch, frac,
                 reasons, reason_idx,
                 lock=dict(offset_s=offset, chosen_by=chosen_by))


def _emit(run, subrun, cen, state, ntof_run, bunch, frac, reasons, reason_idx,
          lock, lock_pending=None):
    counts = {s: int((state == k).sum()) for k, s in enumerate(STATES)}
    denom = sum(v for s, v in counts.items() if s not in NOT_IN_DENOM)
    return dict(
        run=run, subrun=subrun, n_bursts=int(len(state)), lock=lock,
        lock_pending=lock_pending,
        states=counts,
        denominator=denom,
        matched_frac=(counts['MATCHED'] / denom) if denom else None,
        anchor_epoch=cen['anchor_epoch'],
        bursts=dict(
            burst_id=cen['burst_id'],
            t_rel_s=cen['t_rel_s'],
            n_trig=cen['n_trig'],
            state=[STATES[s] for s in state],
            ntof_run=[int(x) for x in ntof_run],
            bunch=[int(x) for x in bunch],
            frac=[None if not np.isfinite(f) else round(float(f), 4)
                  for f in frac],
            reason=[None if i < 0 else reasons[i] for i in reason_idx]))


# ------------------------------------------------------------------- campaign

def known_subruns(qa_root: Path, inventory=None):
    """Every (run, subrun) the campaign knows about, from the product tree
    and the inventory -- the completeness reference for the census."""
    subs = set()
    for d in qa_root.rglob('ntof_hits'):
        # layout .../runs/<run>/<subrun>/ntof_hits
        subs.add((d.parent.parent.name, d.parent.name))
    for (run, subrun, _r) in (inventory or {}):
        subs.add((run, subrun))
    return {(r, s) for r, s in subs if r and s and r.startswith('run_')}


def campaign(qa_root: Path, out: Path, since_run: int | None = None,
             log=print):
    """Every sub-run with a census, classified; plus the headline.

    The census write in the slim job is deliberately NON-FATAL (a good
    product should not die over the accounting), so the guarantee that no
    sub-run silently drops out of the denominator lives HERE, in the
    consumer: any sub-run the campaign knows about that has no census is
    reported as missing_census, loudly, instead of being quietly absent
    from the totals.
    """
    out.mkdir(parents=True, exist_ok=True)
    index = load_index_times()
    inventory = load_inventory(qa_root)
    total = {s: 0 for s in STATES}
    rows = []
    missing = []
    for run, subrun in sorted(known_subruns(qa_root, inventory)):
        try:
            if since_run is not None and int(run.split('_')[1]) < since_run:
                continue
        except (ValueError, IndexError):
            continue
        if not (CACHE / f'{run}_{subrun}.json').exists():
            missing.append(f'{run}/{subrun}')
    if missing:
        log(f'!! {len(missing)} sub-run(s) with segments but NO CENSUS -- '
            f'their pulses are not in any total below. Run the standalone '
            f'census (or harvest) for: ' + ', '.join(missing[:10])
            + (' ...' if len(missing) > 10 else ''))
    for cpath in sorted(CACHE.glob('*.json')):
        # census cache names are '<run>_<subrun>' with run like 'run_79'
        parts = cpath.stem.split('_')
        run, subrun = '_'.join(parts[:2]), '_'.join(parts[2:])
        if since_run is not None and int(parts[1]) < since_run:
            continue
        led = classify_subrun(run, subrun, qa_root, index, inventory)
        if led is None:
            continue
        if led.get('lock_pending'):
            missing.append(f'{run}/{subrun} [lock pending: '
                           f'{led["n_bursts"]} bursts]')
            continue
        (out / f'{run}_{subrun}.json').write_text(json.dumps(led))
        for s, v in led['states'].items():
            total[s] += v
        rows.append(dict(run=run, subrun=subrun, n=led['n_bursts'],
                         denominator=led['denominator'],
                         matched_frac=led['matched_frac'],
                         states=led['states']))
        log(f'  {run}/{subrun}: {led["n_bursts"]} bursts, '
            f'{led["states"]["MATCHED"]} matched'
            + (f' ({led["matched_frac"]:.1%} of ours)'
               if led['matched_frac'] is not None else ''))
    denom = sum(v for s, v in total.items() if s not in NOT_IN_DENOM)
    beam = sum(v for s, v in total.items() if s not in NOT_OURS)
    ntof_off = sum(total.get(s, 0) for s in NTOF_OFF)
    summary = dict(since_run=since_run, states=total, denominator=denom,
                   matched_frac=(total['MATCHED'] / denom) if denom else None,
                   # beam pulses DREAM saw = denominator + n_TOF-off; the
                   # n_TOF loss is quoted against THIS
                   beam_pulses=beam, ntof_off=ntof_off,
                   ntof_off_frac=(ntof_off / beam) if beam else None,
                   n_subruns=len(rows), subruns=rows,
                   missing_census=missing)
    (out / 'campaign_ledger.json').write_text(json.dumps(summary))
    log(f'\n{denom:,} pulses in the denominator'
        + (f', {summary["matched_frac"]:.2%} matched'
           if summary['matched_frac'] is not None else '')
        + f'; n_TOF not recording for {ntof_off:,} of {beam:,} beam pulses'
        + (f' ({summary["ntof_off_frac"]:.2%})'
           if summary['ntof_off_frac'] is not None else '')
        + f'; states: ' + ', '.join(f'{s} {v:,}' for s, v in total.items()
                                    if v))
    return summary


# ------------------------------------------------------------------ dashboard

def build_dashboard_section(ledger_dir: Path, since_run: int | None = 79,
                            drill_cap: int = 50) -> str:
    """The 'every pulse accounted for' section, as HTML for the clock
    dashboard to embed (contract 2026-08-13: delivered as a function so the
    dashboard file stays single-owner).

    Uses the dashboard's own CSS classes (tbl / sub / good / warn / bad).
    Reads campaign_ledger.json plus the per-sub-run ledgers for the
    drill-down. Returns '' when there is no ledger yet -- the section simply
    does not exist until the data does.
    """
    import html as H
    camp = ledger_dir / 'campaign_ledger.json'
    if not camp.exists():
        return ''
    c = json.loads(camp.read_text())
    tot, denom = c['states'], c['denominator']
    matched = tot.get('MATCHED', 0)
    frac = c.get('matched_frac')
    known = denom - tot.get('UNKNOWN_COINC', 0) - tot.get('NOT_ATTEMPTED', 0)
    out = ['<h2>Every pulse accounted for</h2>']
    tgt = ('' if since_run is None else
           f' Target: 99&nbsp;% for runs &ge; run_{since_run}.')
    out.append(
        f'<div class="sub"><b>{matched:,} of {denom:,}</b> pulses matched'
        + (f' ({frac:.2%})' if frac is not None else '')
        + f' &mdash; denominator is every DREAM burst except empty/beam-off '
        f'pulses and the {c.get("ntof_off", 0):,} beam pulses n_TOF was not '
        f'recording'
        + (f' ({c["ntof_off_frac"]:.2%} of beam pulses, irrecoverable)'
           if c.get('ntof_off_frac') is not None else '')
        + f', over {c["n_subruns"]} sub-runs.{tgt} '
        f'{known:,} pulses have a definitive state; the rest await '
        f'per-pulse rows or a segment run.</div>')
    miss = c.get('missing_census') or []
    if miss:
        out.append(
            f'<div class="sub bad"><b>{len(miss)} sub-run(s) have segments '
            f'but no census</b> &mdash; their pulses are in NO total on this '
            f'page: ' + ', '.join(H.escape(m) for m in miss[:12])
            + (' &hellip;' if len(miss) > 12 else '') + '</div>')

    # state totals, worst-first, colour-coded by whose problem it is
    cls = dict(MATCHED='good', LOW_COINC='bad', UNJOINED='bad',
               SEGMENT_FAILED='bad', UNKNOWN_COINC='warn',
               NOT_ATTEMPTED='warn', TOO_FEW_TRIGGERS='warn',
               NTOF_NO_BUNCH='', EMPTY_PULSE='', NO_BEAM_PULSE='',
               NOT_COINC_TRIGGERED='')
    out.append('<table class="tbl"><tr>'
               + ''.join(f'<th>{s}</th>' for s in STATES if tot.get(s))
               + '</tr><tr>'
               + ''.join(f'<td class="{cls[s]}">{tot[s]:,}</td>'
                         for s in STATES if tot.get(s))
               + '</tr></table>')

    # per-sub-run table, most unmatched first
    def bad_n(r):
        return sum(r['states'].get(s, 0) for s in
                   ('LOW_COINC', 'UNJOINED', 'SEGMENT_FAILED'))
    rows = sorted(c['subruns'], key=lambda r: -bad_n(r))
    out.append('<table class="tbl"><tr><th>sub-run</th><th>bursts</th>'
               '<th>matched</th><th>low coinc</th><th>unjoined</th>'
               '<th>seg failed</th><th>unknown</th><th>n_TOF off</th>'
               '<th>not ours</th></tr>')
    for r in rows:
        s = r['states']
        notours = sum(s.get(k, 0) for k in NOT_OURS)
        ntoff = sum(s.get(k, 0) for k in NTOF_OFF)
        unk = s.get('UNKNOWN_COINC', 0) + s.get('NOT_ATTEMPTED', 0) \
            + s.get('TOO_FEW_TRIGGERS', 0)
        mf = r.get('matched_frac')
        out.append(
            f'<tr><td>{H.escape(r["run"])}/{H.escape(r["subrun"])}</td>'
            f'<td>{r["n"]:,}</td>'
            f'<td>{s.get("MATCHED", 0):,}'
            + (f' ({mf:.1%})' if mf is not None else '') + '</td>'
            + ''.join(f'<td class="{cls[k] if s.get(k) else ""}">'
                      f'{s.get(k, 0):,}</td>'
                      for k in ('LOW_COINC', 'UNJOINED', 'SEGMENT_FAILED'))
            + f'<td class="{"warn" if unk else ""}">{unk:,}</td>'
            f'<td>{ntoff:,}</td>'
            f'<td>{notours:,}</td></tr>')
    out.append('</table>')

    # drill-down: the actual follow-up pulses, reasons attached
    for r in rows:
        if not bad_n(r):
            continue
        led_p = ledger_dir / f'{r["run"]}_{r["subrun"]}.json'
        if not led_p.exists():
            continue
        led = json.loads(led_p.read_text())
        b = led['bursts']
        bad = [i for i, s in enumerate(b['state'])
               if s in ('LOW_COINC', 'UNJOINED', 'SEGMENT_FAILED')]
        out.append(f'<details><summary>{H.escape(r["run"])}/'
                   f'{H.escape(r["subrun"])} &mdash; {len(bad)} follow-up '
                   f'pulse(s)</summary><table class="tbl">'
                   '<tr><th>burst</th><th>t (s)</th><th>state</th>'
                   '<th>n_TOF</th><th>bunch</th><th>frac</th>'
                   '<th>reason</th></tr>')
        for i in bad[:drill_cap]:
            fr = b['frac'][i]
            out.append(
                f'<tr><td>{b["burst_id"][i]}</td>'
                f'<td>{b["t_rel_s"][i]:.0f}</td>'
                f'<td class="{cls[b["state"][i]]}">{b["state"][i]}</td>'
                f'<td>{b["ntof_run"][i] or ""}</td>'
                f'<td>{b["bunch"][i] if b["bunch"][i] >= 0 else ""}</td>'
                f'<td>{"" if fr is None else format(fr, ".0%")}</td>'
                f'<td>{H.escape(b["reason"][i] or "")}</td></tr>')
        if len(bad) > drill_cap:
            out.append(f'<tr><td colspan="7">&hellip; and '
                       f'{len(bad) - drill_cap} more</td></tr>')
        out.append('</table></details>')
    return '\n'.join(out)


# ------------------------------------------------------------------------ CLI

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('cmd', choices=['census', 'classify', 'campaign',
                                    'harvest'])
    ap.add_argument('dream_run', nargs='?')
    ap.add_argument('subrun', nargs='?')
    ap.add_argument('--qa-root', type=Path,
                    help='campaign tree holding runs/<run>/<subrun>/ntof_hits')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--since-run', type=int, default=None)
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()

    if a.cmd == 'census':
        subs = [a.subrun] if a.subrun else subruns_of(a.dream_run)
        for s in subs:
            c = burst_census(a.dream_run, s, rebuild=a.rebuild)
            print(f'{a.dream_run}/{s}: '
                  + ('no data' if c is None else f'{c["n_bursts"]} bursts'))
        return 0
    if not a.qa_root:
        ap.error('--qa-root is required for classify/campaign/harvest')
    if a.cmd == 'harvest':
        harvest_censuses(a.qa_root)
        return 0
    if a.cmd == 'classify':
        led = classify_subrun(a.dream_run, a.subrun, a.qa_root)
        if led is None:
            print('no census/data')
            return 1
        print(json.dumps({k: v for k, v in led.items() if k != 'bursts'},
                         indent=1))
        st = led['bursts']['state']
        rs = led['bursts']['reason']
        for s in STATES:
            idx = [i for i, x in enumerate(st) if x == s]
            if idx and s != 'MATCHED':
                print(f'\n{s} ({len(idx)}):')
                for i in idx[:10]:
                    print(f'  burst {i:4d}  t+{led["bursts"]["t_rel_s"][i]:9.1f}s'
                          f'  {rs[i] or ""}')
                if len(idx) > 10:
                    print(f'  ... and {len(idx) - 10} more')
        return 0
    out = a.out or (a.qa_root / 'pulse_ledger')
    campaign(a.qa_root, out, since_run=a.since_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
