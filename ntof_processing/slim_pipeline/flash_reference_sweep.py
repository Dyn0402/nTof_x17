#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flash_reference_sweep.py -- is every DREAM burst's t = 0 really its gamma flash?

THE QUESTION. `bunch_join` defines a burst's time base by its FIRST trigger:
that trigger is called the flash and every other trigger is timed from it. The
2026-08-16 brute-force pass found three bursts where that is false -- the flash
trigger was not recorded, the first scintillator single ~1 ms later was tagged
in its place, and the whole burst therefore sits ~1 ms off the n_TOF clock (one
more, run_102/stat090_0002 burst 0, was recorded 4.4 ms into the gate). All
four failed their per-bunch clock fit and were caught as unmatched pulses. The
open question is whether any burst with a mis-tagged flash was matched ANYWAY,
absorbing the offset silently -- which would be a wrong time base inside a
product that looks healthy.

THE TWO SIGNATURES, both measured on run_79/stat090_0000 (1,012 bursts) and
both computed here from the DREAM files alone -- no n_TOF, no lock, no fit:

  gap1   flash -> first physics trigger. The N93B gate admits singles only
         from ~1 ms after the flash, so this is a HARD EDGE: median 1.0045 ms,
         full range 993 us - 1.19 ms, not one burst below 900 us. The next gap
         (single to single) has median 15 us. A burst whose tagged flash is
         really a single therefore shows gap1 in the tens of microseconds --
         two orders of magnitude below the edge.

  nhits  hits in the tagged flash event. The gamma flash saturates everything:
         3,853-4,032 hits (5th-95th pct) against 16-831 for a physics trigger.
         Disjoint, with a factor ~5 between the extremes.

They are independent -- one is timing, one is pulse height -- so a burst
flagged by both is not a statistical statement. Both are self-calibrated
against the SUB-RUN's own medians rather than against run_79's numbers: the
gate width is a DAQ setting and the flash hit count depends on which chambers
were live (chamber A's connector 8 was dead through run_79), and neither is
allowed to become a campaign-wide constant here.

WHAT ELSE IT COLLECTS. Where a slim product exists for the sub-run, the
`bunches` tree's per-bunch fit record (fitted, da_ns, dk, n_core) is read
alongside, so the report can answer the actual question: of the bursts this
flags, which ones nonetheless got a per-bunch correction and count as matched?
That join is done in `flash_reference_report.py`, which also holds the
verdict; this module only measures.

    python3 flash_reference_sweep.py run_124 --out sweep
    python3 flash_reference_sweep.py run_124 --subrun stat090_0006   # one only

Writes <out>/<dream_run>.json. One condor job per DREAM run: see
lxplus/flash.sub. Reads DREAM off EOS FUSE (two branches of the hits tree) and
the products off EOS; nothing is staged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / 'ntof_july_analysis'))

from ntof_processing.slim_pipeline import config as C            # noqa: E402

GAP_S = 0.5              # burst split, same convention as bunch_join
# A burst is flagged when its gap1 falls below this fraction of the sub-run's
# own median gap1. On run_79 the observed spread is 0.99-1.19 of the median and
# the single-to-single gap is 0.015 of it, so anything in between separates;
# 0.5 is the middle of an empty decade.
GAP_FRAC = 0.5
# ... or when its tagged flash carries below this fraction of the sub-run's
# median flash hit count. Observed spread 0.98-1.02 of the median; a physics
# trigger is 0.004-0.21 of it.
NHITS_FRAC = 0.35
MIN_BURST_TRIG = 5       # below this there is no gap1 worth measuring


def subruns_of(run: str) -> list:
    """Sub-run names of a DREAM run, from the burst-census cache."""
    out = []
    for p in sorted((HERE / 'cache_burst_census').glob(f'{run}_*.json')):
        out.append(p.stem[len(run) + 1:])
    return out


def sweep_subrun(run: str, subrun: str, log=print) -> dict | None:
    """Per-burst gap1 and flash hit count, from the DREAM files alone."""
    import uproot
    import pulse_match as pm

    files = pm._combined_files(run, subrun)
    if not files:
        log(f'  {run}/{subrun}: no combined hits')
        return None
    t0 = time.time()
    # per-event trigger time and HIT COUNT. Events repeat across parts, so the
    # counts are summed per eventId and the time taken from the first sight of
    # it -- the same dedupe `bunch_join.dream_events` does, plus the count it
    # throws away.
    ids, cnts, tns = [], [], []
    for f in files:
        with uproot.open(f) as uf:
            if 'hits' not in uf:
                continue
            a = uf['hits'].arrays(['eventId', 'trigger_timestamp_ns'],
                                  library='np')
        if a['eventId'].size == 0:
            continue
        u, first, c = np.unique(a['eventId'], return_index=True,
                                return_counts=True)
        ids.append(u)
        cnts.append(c)
        tns.append(a['trigger_timestamp_ns'][first].astype(np.int64))
    if not ids:
        log(f'  {run}/{subrun}: no hits tree')
        return None
    eid = np.concatenate(ids)
    cnt = np.concatenate(cnts)
    tns = np.concatenate(tns)
    order = np.argsort(eid, kind='stable')
    eid, cnt, tns = eid[order], cnt[order], tns[order]
    # sum counts over the parts an event was split across; keep its first time
    uniq, start = np.unique(eid, return_index=True)
    nh = np.add.reduceat(cnt.astype(np.int64), start)
    tt = tns[start]
    o = np.argsort(tt)
    uniq, nh, tt = uniq[o], nh[o], tt[o]

    burst = np.cumsum(np.r_[0, (np.diff(tt) > GAP_S * 1e9).astype(np.int64)])
    t_first = tt[np.r_[0, np.flatnonzero(np.diff(burst)) + 1]]
    anchor0 = tt[0]

    rec = dict(burst_id=[], n_trig=[], t_rel_s=[], gap1_ns=[], gap2_ns=[],
               flash_nhits=[], phys_nhits_med=[])
    for b in range(burst[-1] + 1):
        m = burst == b
        t, h = tt[m], nh[m]
        rec['burst_id'].append(int(b))
        rec['n_trig'].append(int(t.size))
        rec['t_rel_s'].append(float((t[0] - anchor0) / 1e9))
        rec['flash_nhits'].append(int(h[0]))
        rec['gap1_ns'].append(float(t[1] - t[0]) if t.size > 1 else float('nan'))
        rec['gap2_ns'].append(float(t[2] - t[1]) if t.size > 2 else float('nan'))
        rec['phys_nhits_med'].append(float(np.median(h[1:])) if t.size > 1
                                     else float('nan'))
    n = np.asarray(rec['n_trig'])
    big = n >= MIN_BURST_TRIG
    g1 = np.asarray(rec['gap1_ns'])
    fh = np.asarray(rec['flash_nhits'], float)
    ref = dict(
        n_bursts=int(n.size), n_big=int(big.sum()),
        gap1_med_ns=float(np.median(g1[big])) if big.any() else None,
        gap1_mad_ns=(float(1.4826 * np.median(np.abs(g1[big] - np.median(g1[big]))))
                     if big.any() else None),
        flash_nhits_med=float(np.median(fh[big])) if big.any() else None,
        phys_nhits_med=(float(np.nanmedian(np.asarray(rec['phys_nhits_med'])[big]))
                        if big.any() else None))
    log(f'  {run}/{subrun}: {n.size} bursts, gap1 median '
        f'{(ref["gap1_med_ns"] or float("nan"))/1e3:.1f} us, flash hits median '
        f'{ref["flash_nhits_med"]}, physics {ref["phys_nhits_med"]} '
        f'[{time.time()-t0:.0f} s]')
    return dict(bursts=rec, ref=ref, anchor_first_ns=int(anchor0),
                n_files=len(files))


def read_products(run: str, subrun: str, base: Path, log=print) -> dict:
    """{ntof_run: per-bunch fit record} from any published slim product."""
    import uproot
    d = base / 'runs' / run / subrun / 'ntof_hits'
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob('ntof_hits_*.root')):
        nt = p.stem.split('_')[-1]
        try:
            with uproot.open(p) as uf:
                if 'bunches' not in uf:
                    continue
                a = uf['bunches'].arrays(
                    ['bunch', 'n_triggers', 'has_beam', 'fitted', 'da_ns',
                     'dk', 'n_core'], library='np')
        except Exception as e:                                   # noqa: BLE001
            log(f'  !! {p.name}: {type(e).__name__}: {e}')
            continue
        out[nt] = {k: v.tolist() for k, v in a.items()}
        # the burst -> bunch map, so a flagged BURST can be found in the
        # per-bunch table without re-deriving the join
        bm = d / 'burst_map.json'
        if bm.exists():
            try:
                out[nt]['burst_map'] = json.loads(bm.read_text())
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dream_run')
    ap.add_argument('--subrun', default=None,
                    help='one sub-run instead of every sub-run of the run')
    ap.add_argument('--out', default='sweep')
    ap.add_argument('--products', default=str(C.EOS_JULY),
                    help='base holding runs/<run>/<subrun>/ntof_hits/ '
                         '(default %(default)s); "" to skip the product read')
    a = ap.parse_args()

    run = a.dream_run
    subs = [a.subrun] if a.subrun else subruns_of(run)
    if not subs:
        print(f'{run}: no sub-runs in the census cache')
        return 1
    print(f'== {run}: {len(subs)} sub-run(s)')
    t0 = time.time()
    out = dict(run=run, subruns={}, products={},
               created=time.strftime('%Y-%m-%dT%H:%M:%S'))
    for s in subs:
        try:
            r = sweep_subrun(run, s)
        except Exception as e:                                   # noqa: BLE001
            print(f'  !! {run}/{s}: {type(e).__name__}: {e}')
            out['subruns'][s] = dict(error=f'{type(e).__name__}: {e}')
            continue
        if r is None:
            continue
        out['subruns'][s] = r
        if a.products:
            try:
                p = read_products(run, s, Path(a.products))
                if p:
                    out['products'][s] = p
            except Exception as e:                               # noqa: BLE001
                print(f'  !! products {run}/{s}: {type(e).__name__}: {e}')
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{run}.json').write_text(json.dumps(out))
    print(f'-> {od / (run + ".json")}  [{time.time()-t0:.0f} s total]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
