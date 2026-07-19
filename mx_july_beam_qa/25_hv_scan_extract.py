#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_hv_scan_extract.py — per-event cluster extraction for the run_55 cyclical
resist-HV scan (r560→r520, drift 800 B/C/D + 600 A, scint-doubles trigger,
30 ms beam gate).

v2: keeps the TOP-5 clusters per plane (by summed amplitude), not just the
largest.  Reason: late in the beam gate the ³He-capture products (n+³He→p+t,
764 keV, heavily ionizing) pile up in the 1.92 µs MM window and produce very
wide high-amplitude clusters that would otherwise mask the MIP track cluster
of the triggering particle.  Cluster classification (MIP-like vs capture-like)
happens downstream in 25b.

Per subrun:
  - full trigger list + timestamps from decoded_root *_01.root (10 ns ticks);
  - burst structure (gap > 100 ms ⇒ new beam pulse); t0 = first trigger of the
    burst; the first event's MM-garbage signature (n_raw_hits ≳ 1500) marks a
    genuine gamma flash — bursts without it get flash_ok = False;
  - combined_hits mapped to detector/plane/strip position;
  - per event × detector × plane: top-5 gap-clusters above threshold.

Writes one npz per subrun to cache/25_run55/<subrun>.npz covering ALL triggers
(also the ~27 % with zero stored hits — saturation dead-time candidates that
must stay in the efficiency denominator).

Run:  venv/bin/python mx_july_beam_qa/25_hv_scan_extract.py
"""

import glob
import os
import re
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.Mx17StripMap import RunConfig  # noqa: E402

RUN_DIR = os.path.expanduser('~/x17/beam_july/runs/run_55')
MAP_CSV = os.path.join(os.path.dirname(__file__), '..', 'mx17_m1_map.csv')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache', '25_run55')

THR_HIT = 400.0          # ADC, per-strip hit threshold (July QA convention)
GAP_MM = 12.0            # 1-D gap clustering split (bench GAP_THRESHOLD_MM)
NCLUS = 5                # top clusters kept per plane
FLASH_RAW_HITS = 1500    # first-in-burst n_raw_hits above this ⇒ genuine flash
BURST_GAP_NS = 100e6     # >100 ms trigger gap ⇒ new beam pulse

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
CF = ['n', 'sum_amp', 'max_amp', 'centroid', 'extent', 'smp_lo', 'smp_hi',
      'nsat']


def subrun_meta(name):
    m = re.match(r'scintd_r(\d+)_dr\d+dA\d+_c(\d+)_(\d+)', name)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_triggers(subdir):
    f = sorted(glob.glob(os.path.join(subdir, 'decoded_root', '*_01.root')))[0]
    a = uproot.open(f)['nt'].arrays(['eventId', 'timestamp'], library='np')
    eid = a['eventId'].astype(np.int64)
    ts = a['timestamp'].astype(np.int64) * 10  # 10 ns ticks → ns
    o = np.argsort(ts, kind='stable')
    return eid[o], ts[o]


def burst_tag(ts):
    bid = np.concatenate([[0], np.cumsum(np.diff(ts) > BURST_GAP_NS)])
    t0 = np.zeros(bid[-1] + 1, dtype=np.int64)
    for b in range(bid[-1] + 1):
        t0[b] = ts[bid == b][0]
    return bid, (ts - t0[bid]) / 1e6


def clusters_plane(pos, amp, smp, sat):
    """Top-NCLUS gap-clusters (by summed amp) of thresholded hits, one plane.

    Returns list of tuples matching CF. Duplicate strip positions (two pulses
    in one window) collapse to the larger pulse.
    """
    if len(pos) == 0:
        return []
    o = np.argsort(pos, kind='stable')
    pos, amp, smp, sat = pos[o], amp[o], smp[o], sat[o]
    keep = np.ones(len(pos), bool)
    for i in range(1, len(pos)):
        if pos[i] == pos[i - 1]:
            keep[i if amp[i] < amp[i - 1] else i - 1] = False
    pos, amp, smp, sat = pos[keep], amp[keep], smp[keep], sat[keep]
    cuts = np.where(np.diff(pos) > GAP_MM)[0] + 1
    segs = np.split(np.arange(len(pos)), cuts)
    segs.sort(key=lambda s: -amp[s].sum())
    out = []
    for seg in segs[:NCLUS]:
        p, a, sm, st = pos[seg], amp[seg], smp[seg], sat[seg]
        out.append((len(seg), float(a.sum()), float(a.max()),
                    float(np.average(p, weights=a)),
                    float(p.max() - p.min()),
                    int(sm.min()), int(sm.max()), int(st.sum())))
    return out


def process_subrun(subdir, pos_lut):
    name = os.path.basename(subdir.rstrip('/'))
    eid_all, ts_all = load_triggers(subdir)
    bid, tin = burst_tag(ts_all)
    n_ev = len(eid_all)

    fh = [f for f in glob.glob(os.path.join(
        subdir, 'combined_hits_root', '*combined_hits.root'))
        if '_pedestals_' not in f]
    h = uproot.open(fh[0])['hits'].arrays(
        ['eventId', 'feu', 'channel', 'amplitude', 'max_sample', 'saturated'],
        library='np')
    heid = h['eventId'].astype(np.int64)

    eidx = {e: i for i, e in enumerate(eid_all)}

    # total stored hits (pre-threshold) per event → flash signature
    n_raw = np.zeros(n_ev, np.int32)
    ue, cnt = np.unique(heid, return_counts=True)
    for e, c in zip(ue, cnt):
        if e in eidx:
            n_raw[eidx[e]] = c

    # flash-anchored burst quality: first event of burst must look like flash
    first = np.concatenate([[True], np.diff(bid) > 0])
    flash_ok_burst = np.zeros(bid[-1] + 1, bool)
    for i in np.where(first)[0]:
        flash_ok_burst[bid[i]] = n_raw[i] >= FLASH_RAW_HITS

    sel = h['amplitude'] > THR_HIT
    heid_s = heid[sel]
    lut = pos_lut[h['feu'][sel], h['channel'][sel]]
    hdet, hplane, hpos = lut[:, 0], lut[:, 1], lut[:, 2]
    hamp = h['amplitude'][sel].astype(float)
    hsmp = h['max_sample'][sel].astype(np.int32)
    hsat = h['saturated'][sel].astype(bool)

    o = np.argsort(heid_s, kind='stable')
    heid_s, hdet, hplane, hpos, hamp, hsmp, hsat = (
        x[o] for x in (heid_s, hdet, hplane, hpos, hamp, hsmp, hsat))
    ue2 = np.unique(heid_s)
    lo = np.searchsorted(heid_s, ue2)
    hi = np.append(lo[1:], len(heid_s))

    out = {}
    for di in range(4):
        for pname in 'xy':
            for f in CF:
                out[f'd{di}_{pname}_{f}'] = np.zeros((n_ev, NCLUS), np.float32)
        out[f'd{di}_nthr'] = np.zeros(n_ev, np.int32)

    for e, l, r in zip(ue2, lo, hi):
        if e not in eidx:
            continue
        i = eidx[e]
        dets_e = hdet[l:r]
        for di in range(4):
            md = dets_e == di
            out[f'd{di}_nthr'][i] = md.sum()
            for pl, pname in [(0, 'x'), (1, 'y')]:
                m = md & (hplane[l:r] == pl)
                cl = clusters_plane(hpos[l:r][m], hamp[l:r][m],
                                    hsmp[l:r][m], hsat[l:r][m])
                for j, vals in enumerate(cl):
                    for f, v in zip(CF, vals):
                        out[f'd{di}_{pname}_{f}'][i, j] = v

    resist, cycle, step = subrun_meta(name)
    np.savez_compressed(
        os.path.join(CACHE_DIR, name + '.npz'),
        eventId=eid_all, ts_ns=ts_all, burst=bid, t_ms=tin,
        n_raw_hits=n_raw, is_first=first,
        flash_ok=flash_ok_burst[bid],
        resist_v=resist, cycle=cycle, step=step,
        thr_hit=THR_HIT, gap_mm=GAP_MM, nclus=NCLUS,
        **out)
    print(f'{name}: {n_ev} trig, {bid[-1]+1} bursts '
          f'({flash_ok_burst.sum()} flash-anchored), '
          f'{(n_raw > 0).sum()} with hits')


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rc = RunConfig(os.path.join(RUN_DIR, 'run_config.json'), MAP_CSV)
    # pos_lut[feu][ch] = (det_idx, plane 0=x/1=y, pos_mm)
    pos_lut = np.full((9, 512, 3), np.nan)
    for di, det in enumerate(DETS):
        d = rc.get_detector(det)
        for feu in sorted(set(v[0] for v in d.dream_feus.values())):
            for ch in range(512):
                x, y = d.map_hit(feu, ch)
                if x is not None:
                    pos_lut[feu, ch] = (di, 0, x)
                elif y is not None:
                    pos_lut[feu, ch] = (di, 1, y)

    subs = sorted(glob.glob(os.path.join(RUN_DIR, 'scintd_*')))
    todo = [s for s in subs
            if not os.path.exists(os.path.join(
                CACHE_DIR, os.path.basename(s) + '.npz'))]
    print(f'{len(subs)} subruns, {len(todo)} to process')
    for s in todo:
        process_subrun(s, pos_lut)
    print('donzo')


if __name__ == '__main__':
    main()
