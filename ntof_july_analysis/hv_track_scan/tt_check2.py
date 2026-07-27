#!/usr/bin/env python3
"""
Discriminate the two DAQ-accept models for run_53 doubles bursts:

  A) buffer takes the FIRST 16 triggers (all flash-afterglow, <1 ms); the
     8-12 / 17-23 ms stamps are readout-batch artifacts.
  B) 10 slots fill in the afterglow; freed slots accept genuinely-late
     triggers whose stamps (8-12 / 17-23 ms) are ~physical.

Tests, per event class (early-stamped <1 ms vs late-stamped >5 ms):
  1. residual of each late accept stamp to the nearest TT edge / TT doubles
     candidate in its burst (B => sub-ms, A => no correlation);
  2. MM content: hits/event, hits >= 1000 ADC (afterglow junk is big and
     busy; genuinely-late events are quiet).

Run: .venv/bin/python ntof_july_analysis/hv_track_scan/tt_check2.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import uproot

RUNS_ROOT = '/mnt/data/x17/beam_july/runs/run_53'
TT_DIR = os.path.expanduser('~/beam_july/test/tt_stream_qualify')
SUBRUNS = ['scintd_r560_dr800dA600_c01_009', 'scintd_r520_dr800dA600_c01_017']

TICK = 1e-9
CLUSTER_W = 200e-9
BURST_GAP_S = 0.10


def load_events_full(subdir):
    """Absolute event times + per-event MM summary from combined hits."""
    start = None
    with open(os.path.join(subdir, 'raw_daq_data', 'run_time.txt')) as f:
        for line in f:
            if 'Run Start Time' in line:
                start = float(line.split(':')[-1])
    files = sorted(glob.glob(os.path.join(subdir, 'combined_hits_root', '*_datrun_*.root')))
    evs, tns, nbig, nhits = [], [], [], []
    for fp in files:
        t = uproot.open(fp)['hits']
        a = t.arrays(['eventId', 'trigger_timestamp_ns', 'amplitude'], library='np')
        ev = a['eventId']
        order = np.argsort(ev, kind='stable')
        ev, ts, amp = ev[order], a['trigger_timestamp_ns'][order], a['amplitude'][order]
        uev, idx = np.unique(ev, return_index=True)
        evs.append(uev); tns.append(ts[idx].astype(np.float64))
        nbig.append(np.add.reduceat((amp >= 1000).astype(np.int64), idx))
        nhits.append(np.diff(np.append(idx, len(ev))))
    ev = np.concatenate(evs)
    tns, nbig, nhits = (np.concatenate(v) for v in (tns, nbig, nhits))
    t_abs = start + tns / 1e9
    order = np.argsort(t_abs, kind='stable')
    return ev[order], t_abs[order], nbig[order], nhits[order]


def load_edges(t_lo, t_hi):
    frames = []
    for f in sorted(glob.glob(os.path.join(TT_DIR, '*', 'edges.csv'))):
        try:
            it = pd.read_csv(f, chunksize=2_000_000)
        except Exception:
            continue
        for df in it:
            m = (df.host_unix >= t_lo) & (df.host_unix <= t_hi)
            if m.any():
                frames.append(df[m])
    if not frames:
        return np.array([]), np.array([])
    d = pd.concat(frames, ignore_index=True)
    anchor = np.median(d.host_unix.values - d.t_board_ns.values * TICK)
    t_abs = d.t_board_ns.values * TICK + anchor
    order = np.argsort(t_abs)
    return t_abs[order], d.channel.values[order]


def doubles_candidates(t_abs, chans):
    out = []
    i, n = 0, len(t_abs)
    while i < n:
        j = i + 1
        seen = {chans[i]}
        while j < n and t_abs[j] - t_abs[i] <= CLUSTER_W:
            seen.add(chans[j])
            j += 1
        if len(seen) >= 2:
            out.append(t_abs[i])
            i = j
        else:
            i += 1
    return np.asarray(out)


def nearest(sorted_ref, x):
    k = np.searchsorted(sorted_ref, x)
    lo = np.clip(k - 1, 0, len(sorted_ref) - 1)
    hi = np.clip(k, 0, len(sorted_ref) - 1)
    dlo = x - sorted_ref[lo]
    dhi = sorted_ref[hi] - x
    return np.where(np.abs(dlo) <= np.abs(dhi), -dlo, dhi)


def main():
    for subrun in SUBRUNS:
        subdir = os.path.join(RUNS_ROOT, subrun)
        print(f'\n=== {subrun} ===')
        ev, t_ev, nbig, nhits = load_events_full(subdir)
        t_edge, chans = load_edges(t_ev.min() - 35, t_ev.max() + 35)
        if len(t_edge) < 100:
            print('  no TT coverage'); continue
        cands = doubles_candidates(t_edge, chans)

        new_b = np.append(True, np.diff(t_ev) > BURST_GAP_S)
        bid = np.cumsum(new_b) - 1
        lead_t = t_ev[new_b]

        # align TT to DREAM: coarse cross-correlation of 10 ms binned rates,
        # then median refinement on leaders
        lo = min(t_ev.min(), cands.min()) - 30.0
        hi = max(t_ev.max(), cands.max()) + 30.0
        nb = int((hi - lo) / 0.010) + 1
        he, _ = np.histogram(t_ev, bins=nb, range=(lo, lo + nb * 0.010))
        hc, _ = np.histogram(cands, bins=nb, range=(lo, lo + nb * 0.010))
        ml = int(30.0 / 0.010)
        corr = np.correlate(he - he.mean(), hc - hc.mean(), mode='full')
        mid = len(corr) // 2
        off = (np.argmax(corr[mid - ml: mid + ml + 1]) - ml) * 0.010
        d0 = nearest(cands + off, lead_t)
        m0 = np.abs(d0) < 0.02
        off += np.median(d0[m0]) if m0.any() else 0.0
        cands_al = cands + off
        edges_al = t_edge + off
        d_lead = nearest(cands_al, lead_t)
        print(f'  TT->DREAM offset {off:+.4f} s; leader residual p50 '
              f'{np.median(np.abs(d_lead)) * 1e6:.0f} us '
              f'(90% {np.percentile(np.abs(d_lead), 90) * 1e6:.0f} us)')
        # per-burst local t0: TT flash = densest edge cluster near the leader
        # (use nearest candidate within 10 ms of leader; leaders without one
        # are dropped from the residual test)
        bl_ok = np.abs(d_lead) < 0.010
        lead_tt = lead_t - d_lead        # TT flash time per burst (aligned frame)
        print(f'  bursts with TT flash lock (<10 ms): {bl_ok.sum()}/{len(lead_t)}')

        dt_ms = (t_ev - lead_t[bid]) * 1e3
        early = ~new_b & (dt_ms < 1.0)
        late1 = ~new_b & (dt_ms > 5) & (dt_ms < 14)
        late2 = ~new_b & (dt_ms > 14)
        for lab, m in [('leader', new_b), ('early(<1ms)', early),
                       ('late(8-12)', late1), ('late(17-23)', late2)]:
            if not m.any():
                continue
            de = nearest(edges_al, t_ev[m]) * 1e3
            dc = nearest(cands_al, t_ev[m]) * 1e3
            print(f'  {lab:12s} n={m.sum():5d}  '
                  f'dt nearest edge (signed, stamp-edge) p10/50/90 = '
                  f'{np.percentile(-de, [10, 50, 90]).round(3)} ms | '
                  f'|dbl-cand| p50={np.median(np.abs(dc)):7.3f}')
        print(f'  MM content (median [p90]):')
        for lab, m in [('leader', new_b), ('early(<1ms)', early),
                       ('late(8-12)', late1), ('late(17-23)', late2)]:
            if not m.any():
                continue
            print(f'  {lab:12s} n_hits {np.median(nhits[m]):7.0f} '
                  f'[{np.percentile(nhits[m], 90):7.0f}]   '
                  f'n_big {np.median(nbig[m]):5.0f} [{np.percentile(nbig[m], 90):5.0f}]')

        # TT edge & doubles-candidate rate vs dt since the DREAM leader stamp
        # (leader stamp is physical; profile shows where TT coverage lives)
        rel_e, rel_c = [], []
        for t0 in lead_t:
            rel_e.append(edges_al[(edges_al > t0 - 0.005) & (edges_al < t0 + 0.035)] - t0)
            rel_c.append(cands_al[(cands_al > t0 - 0.005) & (cands_al < t0 + 0.035)] - t0)
        e = np.array([-5, -1, 0, .1, .5, 1, 2, 3, 5, 8, 12, 17, 23, 32])
        for lab, rel in [('edges', rel_e), ('dbl-cands', rel_c)]:
            rel = [r for r in rel if len(r)]
            if not rel:
                print(f'  TT {lab}: none in any burst window')
                continue
            h, _ = np.histogram(np.concatenate(rel) * 1e3, e)
            rate = h / np.diff(e) / len(lead_t)
            print(f'  TT {lab}/ms/burst:',
                  ' '.join(f'{lo:g}..{hi:g}:{r:.2f}'
                           for lo, hi, r in zip(e[:-1], e[1:], rate)))


if __name__ == '__main__':
    main()
