#!/usr/bin/env python3
"""
Pilot validation for the run_53/55 doubles-trigger HV-scan track analysis.

Checks, on a couple of sub-runs, that
  1. gamma-flash burst leaders are present and taggable in the doubles stream
     (burst gap > 0.1 s; leader n_big > FLASH_NBIG hits >= FLASH_AMP ADC),
  2. the dt-since-flash distribution of the non-flash doubles events has usable
     coverage over the 30 ms window (no fatal readout-gap holes),
  3. the reco track finder runs at an acceptable per-event cost.

Run: .venv/bin/python ntof_july_analysis/hv_track_scan/pilot.py
"""
import os
import sys
import time

import numpy as np
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from ntof_tracking.reco import io, noise, segments as segmod, pairing  # noqa: E402
from ntof_tracking.reco import geometry as geo  # noqa: E402

RUN = 'run_53'
SUBRUNS = ['scintd_r560_dr800dA600_c00_000', 'scintd_r520_dr800dA600_c00_008']

BURST_GAP_S = 0.10   # flash_recovery convention
FLASH_AMP = 1000     # scint_doubles convention
FLASH_NBIG = 150
WINDOW_MS = 30.0


def event_table(run, subrun):
    """Per-event: tns (trigger time ns), n_big, n_hits — from combined hits."""
    sub_dir = os.path.join(io.BASE_PATH, run, subrun, 'combined_hits_root')
    files = sorted(f for f in os.listdir(sub_dir)
                   if f.endswith('.root') and '_datrun_' in f)
    evs, tns, nbig, nhits = [], [], [], []
    for f in files:
        t = uproot.open(os.path.join(sub_dir, f))['hits']
        a = t.arrays(['eventId', 'trigger_timestamp_ns', 'amplitude'],
                     library='np')
        ev = a['eventId']
        order = np.argsort(ev, kind='stable')
        ev, ts, amp = ev[order], a['trigger_timestamp_ns'][order], a['amplitude'][order]
        uev, idx = np.unique(ev, return_index=True)
        evs.append(uev)
        tns.append(ts[idx].astype(np.float64))
        nbig.append(np.add.reduceat((amp >= FLASH_AMP).astype(np.int64), idx))
        nhits.append(np.diff(np.append(idx, len(ev))))
    ev = np.concatenate(evs)
    tns, nbig, nhits = (np.concatenate(v) for v in (tns, nbig, nhits))
    # events can span file boundaries? (combined file is one per subrun, but be safe)
    order = np.argsort(tns, kind='stable')
    return ev[order], tns[order], nbig[order], nhits[order]


def burst_dt(tns, nbig):
    t_s = (tns - tns[0]) / 1e9
    new_burst = np.append(True, np.diff(t_s) > BURST_GAP_S)
    bid = np.cumsum(new_burst) - 1
    leader_idx = np.flatnonzero(new_burst)
    is_flash_leader = np.zeros(len(tns), bool)
    is_flash_leader[leader_idx] = nbig[leader_idx] > FLASH_NBIG
    flash_ok = is_flash_leader[leader_idx][bid]          # per-event: burst confirmed
    dt_ms = (t_s - t_s[leader_idx][bid]) * 1e3
    return bid, is_flash_leader, flash_ok, dt_ms


def main():
    cfg = io.load_run_config(RUN)
    lut = io.build_channel_lut(cfg)
    trs = geo.detector_transforms(cfg)

    for subrun in SUBRUNS:
        print(f'\n=== {RUN}/{subrun} ===')
        ev, tns, nbig, nhits = event_table(RUN, subrun)
        bid, is_fl, flash_ok, dt_ms = burst_dt(tns, nbig)
        n_b = bid[-1] + 1
        lead = np.flatnonzero(np.append(True, np.diff(bid) > 0))
        n_fl = int(is_fl.sum())
        print(f'{len(ev)} events, {n_b} bursts, {n_fl} flash-confirmed leaders '
              f'({100 * n_fl / n_b:.0f}% of bursts)')
        print(f'leader n_big percentiles: {np.percentile(nbig[lead], [5, 50, 95]).round(0)}')
        nonlead_nbig = nbig[np.setdiff1d(np.arange(len(ev)), lead)]
        print(f'non-leader n_big > {FLASH_NBIG}: {int((nonlead_nbig > FLASH_NBIG).sum())} '
              f'of {len(nonlead_nbig)} (mid-burst flash-like)')
        # events per burst
        cnt = np.bincount(bid)
        print(f'events/burst: median {np.median(cnt):.0f}, p95 {np.percentile(cnt, 95):.0f}, '
              f'max {cnt.max()}')
        # burst period sanity (PS cycle)
        if n_b > 2:
            per = np.diff(tns[lead]) / 1e9
            print(f'burst spacing s: p10/50/90 = {np.percentile(per, [10, 50, 90]).round(2)}')
        # dt coverage of probes (non-leader events in flash-ok bursts)
        probe = flash_ok & ~is_fl
        d = dt_ms[probe]
        d = d[(d > 0) & (d < WINDOW_MS + 5)]
        print(f'probes (non-leader, flash-ok burst, dt<{WINDOW_MS + 5:.0f}ms): {len(d)}')
        edges = np.array([0, .1, .5, 1, 2, 3, 5, 8, 12, 17, 23, 30])
        h, _ = np.histogram(d, edges)
        for lo, hi, n in zip(edges[:-1], edges[1:], h):
            bar = '#' * int(60 * n / max(1, h.max()))
            print(f'  {lo:5.1f}-{hi:5.1f} ms: {n:5d} {bar}')
        print(f'  earliest probe dt: {d.min():.4f} ms' if len(d) else '  no probes!')
        # late tail (dt > window => burst gap logic or beam structure oddity)
        late = dt_ms[probe & (dt_ms >= WINDOW_MS + 5)]
        if len(late):
            print(f'  [note] {len(late)} probes with dt>{WINDOW_MS + 5:.0f}ms '
                  f'(max {late.max():.0f} ms)')

        # --- reco timing on a sample of probes ---
        hits = io.load_subrun_hits(RUN, subrun, lut)
        probe_evs = ev[probe][:80]
        t0 = time.time()
        n_trk = n_pair = 0
        drift = geo.DriftModel.from_drift_hv(io.parse_drift_hv(subrun) or 800.0)
        for e in probe_evs:
            g = hits[hits['eventId'] == e]
            if g.empty:
                continue
            g = noise.flag_noise(g)
            segs = segmod.segments_for_event(g)
            prs = pairing.pair_xy_3d(segs, drift)
            n_trk += sum(1 for s in segs if s['cls'] == 'track')
            n_pair += len(prs)
        dt_run = time.time() - t0
        print(f'reco on {len(probe_evs)} probe events: {dt_run:.1f} s '
              f'({1e3 * dt_run / max(1, len(probe_evs)):.0f} ms/evt) — '
              f'{n_trk} track segments, {n_pair} 3D pairs')


if __name__ == '__main__':
    main()
