#!/usr/bin/env python3
"""
Decisive TT-vs-DREAM check for the run_53 doubles HV scan:

Are the mid-burst DREAM accepts a COMPLETE record of the scint-doubles
triggers (with pipeline-batched timestamps), or does the DAQ only accept
triggers inside the 0-0.1 / 8-12 / 17-23 ms service windows (real loss)?

Per burst: count TT doubles candidates in the 30 ms window after the flash
vs DREAM accepted events, and compare the dt distributions.

Run: .venv/bin/python ntof_july_analysis/hv_track_scan/tt_check.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

RUNS_ROOT = '/mnt/data/x17/beam_july/runs/run_53'
TT_DIR = os.path.expanduser('~/beam_july/test/tt_stream_qualify')
SUBRUNS = ['scintd_r560_dr800dA600_c01_009', 'scintd_r520_dr800dA600_c01_017',
           'scintd_r540_dr800dA600_c03_031']

TICK = 1e-9
CLUSTER_W = 200e-9      # >=2 distinct channels within 200 ns = doubles candidate
BURST_GAP_S = 0.10
WINDOW_S = 0.032


def load_events(subdir):
    files = sorted(glob.glob(os.path.join(subdir, 'decoded_root', '*_01.root')))
    start = None
    with open(os.path.join(subdir, 'raw_daq_data', 'run_time.txt')) as f:
        for line in f:
            if 'Run Start Time' in line:
                start = float(line.split(':')[-1])
    ts_all = []
    for fp in files:
        with uproot.open(fp) as uf:
            arr = uf['nt'].arrays(['eventId', 'timestamp'], library='np')
        _, idx = np.unique(arr['eventId'], return_index=True)
        ts_all.append(arr['timestamp'][idx].astype(np.float64))
    ts = np.sort(np.concatenate(ts_all))
    return start + ts / 1e8


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


def coarse_offset(t_events, t_cands, max_lag=30.0, bin_s=0.010):
    lo = min(t_events.min(), t_cands.min()) - max_lag
    hi = max(t_events.max(), t_cands.max()) + max_lag
    nb = int((hi - lo) / bin_s) + 1
    he, _ = np.histogram(t_events, bins=nb, range=(lo, lo + nb * bin_s))
    hc, _ = np.histogram(t_cands, bins=nb, range=(lo, lo + nb * bin_s))
    ml = int(max_lag / bin_s)
    corr = np.correlate(he - he.mean(), hc - hc.mean(), mode='full')
    mid = len(corr) // 2
    return (np.argmax(corr[mid - ml: mid + ml + 1]) - ml) * bin_s


def main():
    for subrun in SUBRUNS:
        subdir = os.path.join(RUNS_ROOT, subrun)
        print(f'\n=== {subrun} ===')
        t_ev = load_events(subdir)
        t_abs, chans = load_edges(t_ev.min() - 35, t_ev.max() + 35)
        if len(t_abs) < 100:
            print('  no TT coverage, skipping')
            continue
        cands = doubles_candidates(t_abs, chans)
        print(f'  {len(t_ev)} DREAM events, {len(t_abs)} edges, '
              f'{len(cands)} doubles candidates')
        lag = coarse_offset(t_ev, cands)
        cands = cands + lag
        print(f'  coarse offset {lag:+.3f} s')

        # DREAM bursts
        new_b = np.append(True, np.diff(t_ev) > BURST_GAP_S)
        bid = np.cumsum(new_b) - 1
        lead_t = t_ev[new_b]

        # refine: median (leader - nearest candidate) to lock flash alignment
        d_lead = []
        for lt in lead_t:
            k = np.searchsorted(cands, lt)
            near = [cands[m] for m in (k - 1, k) if 0 <= m < len(cands)]
            if near:
                d_lead.append(lt - min(near, key=lambda c: abs(c - lt)))
        med = np.median(d_lead)
        cands = cands + med
        print(f'  leader-vs-candidate median offset {med * 1e3:+.2f} ms '
              f'(rms {np.std(d_lead) * 1e3:.2f} ms), applied')

        n_acc, n_cand = [], []
        cand_dts, acc_dts = [], []
        for b in range(len(lead_t)):
            lt = lead_t[b]
            # flash candidate = nearest candidate to leader within 20 ms
            k = np.searchsorted(cands, lt)
            near = [cands[m] for m in (k - 1, k) if 0 <= m < len(cands)]
            if not near:
                continue
            fc = min(near, key=lambda c: abs(c - lt))
            if abs(fc - lt) > 0.020:
                continue
            in_win = cands[(cands > fc) & (cands <= fc + WINDOW_S)]
            acc = t_ev[(bid == b) & ~new_b[bid == b]] if False else t_ev[bid == b][1:]
            n_cand.append(len(in_win))
            n_acc.append(len(acc))
            cand_dts.append(in_win - fc)
            acc_dts.append(acc - lt)
        n_acc, n_cand = np.array(n_acc), np.array(n_cand)
        print(f'  {len(n_acc)} bursts matched to a flash candidate')
        print(f'  accepts/burst:    p10/50/90 = '
              f'{np.percentile(n_acc, [10, 50, 90]).round(1)}')
        print(f'  candidates/burst: p10/50/90 = '
              f'{np.percentile(n_cand, [10, 50, 90]).round(1)}')
        eq = (n_acc == n_cand).mean()
        print(f'  bursts with equal counts: {100 * eq:.0f}%  '
              f'(cand>acc: {100 * (n_cand > n_acc).mean():.0f}%, '
              f'cand<acc: {100 * (n_cand < n_acc).mean():.0f}%)')
        cd = np.concatenate(cand_dts) * 1e3
        ad = np.concatenate(acc_dts) * 1e3
        edges = np.array([0, .1, .5, 1, 2, 3, 5, 8, 12, 17, 23, 32])
        hc_, _ = np.histogram(cd, edges)
        ha_, _ = np.histogram(ad, edges)
        print(f'  dt histogram [ms]   TT-candidates | DREAM-accepts')
        for lo, hi, c, a in zip(edges[:-1], edges[1:], hc_, ha_):
            print(f'   {lo:5.1f}-{hi:5.1f}: {c:6d} | {a:6d}')


if __name__ == '__main__':
    main()
