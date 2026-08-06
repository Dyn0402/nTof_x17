#!/usr/bin/env python3
"""
run58_columns.py — per-subrun drift-column statistics for the run_58 2-D scan.

**Standalone by design**: the LCG view on the batch nodes has numpy/pandas/
uproot and nothing else, so this file carries its own copy of the clustering
and imports nothing from the repo. The strip map arrives as `run58_stripmap.npz`
(see `make_run58_stripmap.py`).

What it measures, per (event, detector, plane), on the largest spatial cluster:

  onset = earliest peak sample in the cluster   (prompt, near the mesh)
  edge  = latest   peak sample                  (deep, near the cathode)
  span  = edge - onset                          -> the full-gap drift time
  ladder= |rank corr(strip position, peak sample)|  ~1 for a real micro-TPC
          column, ~0 for a block of channels ringing together
  pos   = amplitude-weighted mean strip position
  ceil  = did the deepest strip peak in the LAST sample bin (truncated)?

The point of the exercise: `span` versus drift voltage, per detector. run_58
sweeps drift 700 -> 200 V with a 64-sample (3.84 us) window that contains the
whole column at every point, so it is the one dataset that can say whether a
chamber's drift field responds to its supply. See
`mx_july_beam_qa/HANDOFF_2026-07-30_readout_window_and_detB.md` §4.

Notes that matter for correctness:
  * run_58 predates the 2026-07-24 analyzer -> there is NO `significance`
    branch. The relative floor is skipped and an ABSOLUTE amplitude cut is
    used instead, so every subrun is selected identically.
  * A channel can carry several hits per event (pileup / post-saturation
    ringing); keep the largest pulse per channel or a late secondary pulse is
    read as the column's deep edge.

    python3 run58_columns.py <hits_dir> <subrun_name> [--out out] [--amp-min 300]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import uproot

MIN_STRIPS = 5
GAP_MM = 12.0            # production spatial clustering gap
BUSY_STRIPS = 200        # beam busy/flash veto, per plane
COLS = ['eventId', 'feu', 'channel', 'amplitude', 'max_sample']


def rank_corr(a, b):
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def largest_cluster(pos, channels, amps, gap_mm=GAP_MM, min_strips=MIN_STRIPS):
    good = np.isfinite(pos)
    pos, channels, amps = pos[good], channels[good], amps[good]
    if len(pos) < min_strips:
        return None
    o = np.argsort(pos)
    pos, channels, amps = pos[o], channels[o], amps[o]
    lab = np.concatenate([[0], np.cumsum(np.diff(pos) > gap_mm)])
    counts = np.bincount(lab)
    c = int(np.argmax(counts))
    m = lab == c
    if m.sum() < min_strips:
        return None
    return channels[m].astype(np.int64)


def subrun_meta(name):
    """sngPS_dr700_r560_004 -> (700, 560, 4)"""
    dr = re.search(r'_dr(\d+)', name)
    rs = re.search(r'_r(\d+)_', name)
    sq = re.search(r'_(\d+)$', name)
    return (int(dr.group(1)) if dr else -1,
            int(rs.group(1)) if rs else -1,
            int(sq.group(1)) if sq else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('hits_dir')
    ap.add_argument('subrun')
    ap.add_argument('--stripmap', default='run58_stripmap.npz')
    ap.add_argument('--out', default='out')
    ap.add_argument('--amp-min', type=float, default=300.0)
    ap.add_argument('--busy-strips', type=int, default=BUSY_STRIPS)
    ap.add_argument('--n-sample', type=int, default=64)
    args = ap.parse_args()

    sm = np.load(args.stripmap)
    planes = []          # (det, plane, feu, pos_array)
    for det in 'ABCD':
        for plane in 'xy':
            planes.append((det, plane, int(sm[f'{det}{plane}_feu'][0]),
                           sm[f'{det}{plane}']))

    files = sorted(glob.glob(os.path.join(args.hits_dir, '*combined_hits.root')))
    if not files:
        print(f'FATAL: no hits files in {args.hits_dir}', file=sys.stderr)
        return 2
    print(f'{args.subrun}: {len(files)} hits file(s)', flush=True)

    frames = []
    for f in files:
        t = uproot.open(f)['hits']
        have = [c for c in COLS if c in t.keys()]
        missing = set(COLS) - set(have)
        if missing:
            print(f'FATAL: {os.path.basename(f)} missing {missing}',
                  file=sys.stderr)
            return 2
        frames.append(t.arrays(have, library='pd'))
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    df = df[df['amplitude'] > args.amp_min]
    # largest pulse per (event, channel)
    df = df.sort_values('amplitude').drop_duplicates(
        ['eventId', 'feu', 'channel'], keep='last')
    print(f'  {n_raw:,} hits -> {len(df):,} after amp>{args.amp_min:.0f} + dedup',
          flush=True)

    drift, resist, seq = subrun_meta(args.subrun)
    rows = []
    for det, plane, feu, pos_map in planes:
        d = df[df['feu'] == feu]
        if len(d) == 0:
            continue
        for eid, g in d.groupby('eventId', sort=False):
            if len(g) > args.busy_strips:
                rows.append(dict(det=det, plane=plane, eventId=int(eid),
                                 busy=True))
                continue
            ch = g['channel'].to_numpy().astype(int)
            sel = largest_cluster(pos_map[ch], ch, g['amplitude'].to_numpy())
            if sel is None:
                continue
            gg = g.set_index('channel').loc[sel]
            m = gg['max_sample'].to_numpy()
            p = pos_map[sel]
            a = gg['amplitude'].to_numpy()
            ok = np.isfinite(m) & np.isfinite(p)
            if ok.sum() < MIN_STRIPS:
                continue
            m, p, a = m[ok], p[ok], a[ok]
            rows.append(dict(
                det=det, plane=plane, eventId=int(eid), busy=False,
                n=int(len(m)), onset=float(m.min()), edge=float(m.max()),
                span=float(m.max() - m.min()),
                pos=float((p * a).sum() / a.sum()) if a.sum() > 0 else float(p.mean()),
                amp=float(a.max()), ladder=abs(rank_corr(p, m)),
                ceil=bool(m.max() >= args.n_sample - 1.5)))

    out = pd.DataFrame(rows)
    out['drift'] = drift
    out['resist'] = resist
    out['seq'] = seq
    out['subrun'] = args.subrun
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, f'columns_{args.subrun}.parquet')
    out.to_parquet(dest, index=False)
    live = out[out['busy'] == False] if len(out) else out          # noqa: E712
    print(f'  wrote {dest}: {len(out):,} rows, {len(live):,} clusters, '
          f'drift={drift} resist={resist}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
