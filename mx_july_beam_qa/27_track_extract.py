#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_track_extract.py — per-STRIP micro-TPC candidate extraction for run_55.

Motivation (from the 25/26 HV-scan work + the 27 exploration):
  "any x/y cluster = track" is dominated by ³He-capture blobs, gamma-flash
  garbage and coherent ringing.  The 27 exploration showed that the REAL
  tracks are a rare, clean population: compact clusters (few strips, small
  position extent) whose per-strip drift-time (max_sample) forms a MONOTONIC
  trail spanning the FULL drift gap (~11-12 samples of 60 ns), OR — for
  head-on/normal-incidence tracks — a single strip with a LONG (full-gap)
  saturated waveform.  Selecting them needs per-STRIP information, which the
  25 cache (cluster summaries only) threw away.

This script re-extracts, per subrun, the per-strip content of every candidate
cluster (n_strip in [2,40], position extent <= 45 mm — i.e. track-like, not the
100+ mm flood blobs) for all four detectors, both planes, keeping enough to
(a) build the realism gate (duration, trail monotonicity, head-on pulse width,
X/Y time & length match) and (b) reconstruct micro-TPC angles for the
source-hypothesis alignment.  feu/channel are kept per strip so 27w can pull
the decoded waveform of any candidate strip for true T_sat.

Data-source & conventions mirror 25_hv_scan_extract.py (flash/burst tagging,
top-of-burst t0, THR_HIT=400 ADC, GAP=12 mm).  Output: one npz per subrun in
cache/27_run55/.  Resumable (skips existing).

Run:  venv/bin/python mx_july_beam_qa/27_track_extract.py [--limit N] [subrun...]
"""
import glob
import os
import re
import sys
import argparse

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.Mx17StripMap import RunConfig  # noqa: E402

RUN_DIR = os.path.expanduser('~/x17/beam_july/runs/run_55')
MAP_CSV = os.path.join(os.path.dirname(__file__), '..', 'mx17_m1_map.csv')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache', '27_run55')

THR_HIT = 400.0          # ADC per-strip membership threshold (25/July convention)
GAP_MM = 12.0            # 1-D position-gap clustering split (bench)
BURST_GAP_NS = 100e6     # >100 ms trigger gap => new beam pulse
FLASH_RAW_HITS = 1500    # first-in-burst n_raw_hits above this => genuine flash
CAND_NMAX = 40           # store per-strip only for clusters up to this many strips
CAND_EXTENT_MAX = 45.0   # ...and up to this position extent [mm]
CAND_NMIN = 2
TOPK_PLANE = 16          # cap candidate clusters kept per plane (by charge)

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']


def subrun_meta(name):
    m = re.match(r'scintd_r(\d+)_dr\d+dA\d+_c(\d+)_(\d+)', name)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_triggers(subdir):
    f = sorted(glob.glob(os.path.join(subdir, 'decoded_root', '*_01.root')))[0]
    a = uproot.open(f)['nt'].arrays(['eventId', 'timestamp'], library='np')
    eid = a['eventId'].astype(np.int64)
    ts = a['timestamp'].astype(np.int64) * 10  # 10 ns ticks -> ns
    o = np.argsort(ts, kind='stable')
    return eid[o], ts[o]


def burst_tag(ts):
    bid = np.concatenate([[0], np.cumsum(np.diff(ts) > BURST_GAP_NS)])
    t0 = np.zeros(bid[-1] + 1, dtype=np.int64)
    for b in range(bid[-1] + 1):
        t0[b] = ts[bid == b][0]
    return bid, (ts - t0[bid]) / 1e6


def build_lut(rc):
    """(feu, ch) -> (det_idx, plane 0=x/1=y, pos_mm); nan where unmapped."""
    lut = np.full((9, 512, 3), np.nan)
    for di, det in enumerate(DETS):
        d = rc.get_detector(det)
        for feu in sorted(set(v[0] for v in d.dream_feus.values())):
            for ch in range(512):
                x, y = d.map_hit(feu, ch)
                if x is not None:
                    lut[feu, ch] = (di, 0, x)
                elif y is not None:
                    lut[feu, ch] = (di, 1, y)
    return lut


def cluster_indices(pos, gap_mm=GAP_MM):
    """Return list of index-arrays (into the given plane hit list), one per
    gap-cluster, split on position gaps > gap_mm.  Duplicate strip positions
    (two pulses same strip) are kept — both drift times are informative."""
    o = np.argsort(pos, kind='stable')
    cuts = np.where(np.diff(pos[o]) > gap_mm)[0] + 1
    return [o[s] for s in np.split(np.arange(len(pos)), cuts)]


def process_subrun(subdir, lut):
    name = os.path.basename(subdir.rstrip('/'))
    eid_all, ts_all = load_triggers(subdir)
    bid, tin = burst_tag(ts_all)
    n_ev = len(eid_all)
    eidx = {int(e): i for i, e in enumerate(eid_all)}

    fh = [f for f in glob.glob(os.path.join(
        subdir, 'combined_hits_root', '*combined_hits.root'))
        if '_pedestals_' not in f][0]
    h = uproot.open(fh)['hits'].arrays(
        ['eventId', 'feu', 'channel', 'amplitude', 'max_sample',
         'time_of_max', 'integral', 'time_over_threshold',
         'left_sample', 'right_sample', 'saturated'], library='np')
    heid = h['eventId'].astype(np.int64)

    # flash signature: total STORED hits per event (pre-threshold)
    n_raw = np.zeros(n_ev, np.int32)
    ue, cnt = np.unique(heid, return_counts=True)
    for e, c in zip(ue, cnt):
        i = eidx.get(int(e))
        if i is not None:
            n_raw[i] = c
    first = np.concatenate([[True], np.diff(bid) > 0])
    flash_ok_burst = np.zeros(bid[-1] + 1, bool)
    for i in np.where(first)[0]:
        flash_ok_burst[bid[i]] = n_raw[i] >= FLASH_RAW_HITS

    # map + threshold
    l = lut[h['feu'], h['channel']]
    good = np.isfinite(l[:, 0])
    sel = (h['amplitude'] > THR_HIT) & good
    s_eid = heid[sel]
    s_det = l[sel, 0].astype(np.int8)
    s_pln = l[sel, 1].astype(np.int8)
    s_pos = l[sel, 2].astype(np.float32)
    s_amp = h['amplitude'][sel].astype(np.float32)
    s_smp = h['max_sample'][sel].astype(np.float32)
    s_tmx = h['time_of_max'][sel].astype(np.float32)
    s_int = h['integral'][sel].astype(np.float32)
    s_tot = h['time_over_threshold'][sel].astype(np.float32)
    s_lsm = h['left_sample'][sel].astype(np.float32)
    s_rsm = h['right_sample'][sel].astype(np.float32)
    s_sat = h['saturated'][sel].astype(bool)
    s_feu = h['feu'][sel].astype(np.int16)
    s_ch = h['channel'][sel].astype(np.int16)

    order = np.argsort(s_eid, kind='stable')
    (s_eid, s_det, s_pln, s_pos, s_amp, s_smp, s_tmx, s_int, s_tot,
     s_lsm, s_rsm, s_sat, s_feu, s_ch) = (
        a[order] for a in (s_eid, s_det, s_pln, s_pos, s_amp, s_smp, s_tmx,
                           s_int, s_tot, s_lsm, s_rsm, s_sat, s_feu, s_ch))
    uev, ufirst = np.unique(s_eid, return_index=True)
    ubound = np.append(ufirst, len(s_eid))

    # per (det,plane) event context
    nhit = np.zeros((n_ev, 4, 2), np.int32)      # thr hits per det,plane
    nblob = np.zeros((n_ev, 4, 2), np.int32)     # wide (non-candidate) clusters
    maxext = np.zeros((n_ev, 4, 2), np.float32)  # max cluster extent (context)

    # candidate cluster table + strip store
    C = {k: [] for k in ('eid', 'det', 'pln', 'n', 'qsum', 'extent', 'cen',
                         'smplo', 'smphi', 'dur', 'mono', 'nsat', 'wfmax',
                         'slope', 'off', 'len')}
    SS = {k: [] for k in ('pos', 'amp', 'smp', 'tmx', 'integ', 'tot',
                         'lsm', 'rsm', 'sat', 'feu', 'ch')}
    soff = 0

    for k, e in enumerate(uev):
        ie = eidx.get(int(e))
        if ie is None:
            continue
        a, b = ubound[k], ubound[k + 1]
        det_e = s_det[a:b]
        pln_e = s_pln[a:b]
        for di in range(4):
            for pli in range(2):
                m = (det_e == di) & (pln_e == pli)
                if not m.any():
                    continue
                idxp = np.nonzero(m)[0] + a          # indices into global strip arrays
                nhit[ie, di, pli] = len(idxp)
                pos = s_pos[idxp]
                clusters = cluster_indices(pos)
                # sort clusters by charge desc, keep candidates
                clu_q = [(s_amp[idxp[c]].sum(), c) for c in clusters]
                clu_q.sort(key=lambda t: -t[0])
                kept = 0
                mx = 0.0
                for q, c in clu_q:
                    gi = idxp[c]                       # global strip indices
                    n = len(gi)
                    P = s_pos[gi]
                    ext = float(P.max() - P.min())
                    mx = max(mx, ext)
                    is_cand = (CAND_NMIN <= n <= CAND_NMAX
                               and ext <= CAND_EXTENT_MAX and kept < TOPK_PLANE)
                    if not is_cand:
                        nblob[ie, di, pli] += 1
                        continue
                    A = s_amp[gi]
                    S = s_smp[gi]
                    cen = float(np.average(P, weights=A))
                    smplo = float(S.min())
                    smphi = float(S.max())
                    dur = smphi - smplo
                    nsat = int(s_sat[gi].sum())
                    wfmax = float(np.max(s_rsm[gi] - s_lsm[gi]))
                    if n >= 3 and np.std(S) > 0 and np.std(P) > 0:
                        mono = float(np.corrcoef(P, S)[0, 1])
                    else:
                        mono = np.nan
                    # anchored slope (earliest sample) ns/mm
                    i0 = int(np.argmin(S))
                    dp = P - P[i0]
                    dtn = (S - S[i0]) * 60.0
                    den = float(np.sum(A * dp * dp))
                    slope = float(np.sum(A * dp * dtn) / den) if den > 0 else np.nan
                    for key, arr in (('pos', P), ('amp', A), ('smp', S),
                                     ('tmx', s_tmx[gi]), ('integ', s_int[gi]),
                                     ('tot', s_tot[gi]), ('lsm', s_lsm[gi]),
                                     ('rsm', s_rsm[gi]), ('sat', s_sat[gi]),
                                     ('feu', s_feu[gi]), ('ch', s_ch[gi])):
                        SS[key].append(arr)
                    C['eid'].append(int(e)); C['det'].append(di); C['pln'].append(pli)
                    C['n'].append(n); C['qsum'].append(float(A.sum()))
                    C['extent'].append(ext); C['cen'].append(cen)
                    C['smplo'].append(smplo); C['smphi'].append(smphi)
                    C['dur'].append(dur); C['mono'].append(mono)
                    C['nsat'].append(nsat); C['wfmax'].append(wfmax)
                    C['slope'].append(slope); C['off'].append(soff); C['len'].append(n)
                    soff += n
                    kept += 1
                maxext[ie, di, pli] = mx

    def cat(d, dtype):
        return (np.concatenate(d).astype(dtype) if d else np.array([], dtype))

    resist, cycle, step = subrun_meta(name)
    out = dict(
        eventId=eid_all, ts_ns=ts_all, burst=bid, t_ms=tin.astype(np.float32),
        n_raw_hits=n_raw, is_first=first, flash_ok=flash_ok_burst[bid],
        nhit=nhit, nblob=nblob, maxext=maxext,
        resist_v=resist, cycle=cycle, step=step,
        thr_hit=THR_HIT, gap_mm=GAP_MM,
        # candidate cluster table
        c_eid=np.array(C['eid'], np.int64), c_det=np.array(C['det'], np.int8),
        c_pln=np.array(C['pln'], np.int8), c_n=np.array(C['n'], np.int16),
        c_qsum=np.array(C['qsum'], np.float32), c_extent=np.array(C['extent'], np.float32),
        c_cen=np.array(C['cen'], np.float32), c_smplo=np.array(C['smplo'], np.float32),
        c_smphi=np.array(C['smphi'], np.float32), c_dur=np.array(C['dur'], np.float32),
        c_mono=np.array(C['mono'], np.float32), c_nsat=np.array(C['nsat'], np.int16),
        c_wfmax=np.array(C['wfmax'], np.float32), c_slope=np.array(C['slope'], np.float32),
        c_off=np.array(C['off'], np.int64), c_len=np.array(C['len'], np.int32),
        # strip store
        s_pos=cat(SS['pos'], np.float32), s_amp=cat(SS['amp'], np.float32),
        s_smp=cat(SS['smp'], np.float32), s_tmx=cat(SS['tmx'], np.float32),
        s_integ=cat(SS['integ'], np.float32), s_tot=cat(SS['tot'], np.float32),
        s_lsm=cat(SS['lsm'], np.float32), s_rsm=cat(SS['rsm'], np.float32),
        s_sat=cat(SS['sat'], np.bool_), s_feu=cat(SS['feu'], np.int16),
        s_ch=cat(SS['ch'], np.int16),
    )
    np.savez_compressed(os.path.join(CACHE_DIR, name + '.npz'), **out)
    ncand = len(C['eid'])
    print(f'{name}: {n_ev} trig, {bid[-1]+1} bursts, '
          f'{ncand} candidate clusters, {soff} strips stored')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subruns', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    rc = RunConfig(os.path.join(RUN_DIR, 'run_config.json'), MAP_CSV)
    lut = build_lut(rc)

    subs = sorted(glob.glob(os.path.join(RUN_DIR, 'scintd_*')))
    if args.subruns:
        subs = [s for s in subs if os.path.basename(s) in args.subruns]
    todo = [s for s in subs
            if not os.path.exists(os.path.join(
                CACHE_DIR, os.path.basename(s) + '.npz'))]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(subs)} subruns, {len(todo)} to process')
    for s in todo:
        process_subrun(s, lut)
    print('donzo')


if __name__ == '__main__':
    main()
