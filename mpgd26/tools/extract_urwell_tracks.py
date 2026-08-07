#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_urwell_tracks.py -- real SPS beam tracks from the two EIC uRWELLs.

RUNS ON LXPLUS (the merged hit file is 11 GB and lives there):

    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    python3 extract_urwell_tracks.py --mapping mapping_urwell.csv \
            --subrun 23 --n-events 40000 --out urwell_tracks.csv

Both uRWELLs sit on FEU 1: channels 0-127 = front x, 128-255 = front y,
256-383 = back x, 384-511 = back y.  ``mapping_urwell.csv`` already carries the
resolved wiring (view_mode, axis_flipped) and the final ``position_mm`` per
channel, so nothing here has to re-derive the connector order -- which is the
part with four candidate answers and a mirror ambiguity, and the part that
would quietly poison the result if guessed.

A track is two points: the front cluster at z = 0 and the back cluster at
z = 1370 mm.  Events are kept only when all four views have exactly one
cluster.

**The validation is the point.**  Before writing anything, the script fits
``back = slope * front + offset`` per axis and compares to the published
alignment (mapping_alignment.json, urwell_front_to_back).  If the slopes,
offsets and core widths do not reproduce, the extraction is wrong somewhere and
the script says so rather than handing back plausible-looking nonsense.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

FEU = 1
Z_FRONT, Z_BACK = 0.0, 1370.0

# published in sps_beam_test_26/analysis/urw_mapping/mapping_alignment.json
PUBLISHED = {
    'x': dict(slope=0.9996010584080552, offset=-0.9584662880558853,
              sigma=0.76837300114191),
    'y': dict(slope=1.0417679522587129, offset=-4.107225316797881,
              sigma=0.7209845879062221),
}


def load_mapping(path):
    """channel -> (detector, view, position_mm), as plain numpy lookups."""
    import csv
    ch_det, ch_view, ch_pos = {}, {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['feu']) != FEU:
                continue
            c = int(row['channel'])
            ch_det[c] = row['detector']
            ch_view[c] = row['view']
            ch_pos[c] = float(row['position_mm'])
    n = max(ch_pos) + 1
    pos = np.full(n, np.nan)
    key = np.full(n, -1, dtype=np.int8)      # 0 fx, 1 fy, 2 bx, 3 by
    order = {('EIC_uRWELL_front', 'x'): 0, ('EIC_uRWELL_front', 'y'): 1,
             ('EIC_uRWELL_back', 'x'): 2, ('EIC_uRWELL_back', 'y'): 3}
    for c in ch_pos:
        pos[c] = ch_pos[c]
        key[c] = order[(ch_det[c], ch_view[c])]
    return pos, key


def subrun_entry_offset(index_path, subrun_id):
    """First tree entry of ``subrun_id``, assuming subruns are stored in order.

    Verified against the data before use -- see ``main``.
    """
    with open(index_path) as f:
        idx = json.load(f)
    off = 0
    for s in idx['subruns']:
        if s['subrun_id'] == subrun_id:
            return off, s
        off += s['n_hits']
    raise SystemExit(f'subrun {subrun_id} not in index')


def cluster_1d(chans, poss, amps, max_gap=2):
    """Charge-weighted centroids of contiguous strip groups, sorted by channel."""
    o = np.argsort(chans)
    chans, poss, amps = chans[o], poss[o], amps[o]
    breaks = np.flatnonzero(np.diff(chans) > max_gap) + 1
    out = []
    for grp in np.split(np.arange(len(chans)), breaks):
        if grp.size == 0:
            continue
        w = amps[grp]
        if w.sum() <= 0:
            continue
        out.append((float((poss[grp] * w).sum() / w.sum()), int(grp.size),
                    float(w.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default=os.path.expanduser(
        '~/x17/p2_sps_july/merged/all_subruns_hits.root'))
    ap.add_argument('--index', default=None)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--subrun', type=int, default=23,
                    help='23 = highstat_eff_1/beam_commissioning_00, the run '
                         'the published alignment was fitted on')
    ap.add_argument('--n-hits', type=int, default=6_000_000,
                    help='tree entries to read from the subrun start')
    ap.add_argument('--amp-min', type=float, default=25.0)
    ap.add_argument('--min-strips', type=int, default=2)
    ap.add_argument('--out', default='urwell_tracks.csv')
    ap.add_argument('--n-out', type=int, default=400)
    args = ap.parse_args()
    args.index = args.index or args.file.replace('.root', '_index.json')

    import uproot

    pos_lut, key_lut = load_mapping(args.mapping)
    offset, meta = subrun_entry_offset(args.index, args.subrun)
    print(f'subrun {args.subrun}: {meta["run"]}/{meta["sub_run"]}  '
          f'entry offset {offset:,}  reading {args.n_hits:,}')

    tree = uproot.open(args.file)['hits']
    a = tree.arrays(['eventId', 'channel', 'amplitude', 'feu', 'subrun_id'],
                    entry_start=offset, entry_stop=offset + args.n_hits,
                    library='np')

    got = np.unique(a['subrun_id'])
    if got.size != 1 or got[0] != args.subrun:
        raise SystemExit(f'ORDERING ASSUMPTION BROKEN: window holds subruns '
                         f'{got[:5]}, expected only {args.subrun}. '
                         f'Do not trust the entry-offset shortcut.')
    print(f'  subrun_id check OK ({got[0]})')

    sel = (a['feu'] == FEU) & (a['amplitude'] > args.amp_min) & \
          (a['channel'] < len(pos_lut))
    ev = a['eventId'][sel]
    ch = a['channel'][sel]
    am = a['amplitude'][sel].astype(float)
    ky = key_lut[ch]
    ps = pos_lut[ch]
    ok = ky >= 0
    ev, ch, am, ky, ps = ev[ok], ch[ok], am[ok], ky[ok], ps[ok]
    print(f'  {len(ev):,} mapped uRWELL hits over '
          f'{np.unique(ev).size:,} events')

    order = np.lexsort((ch, ky, ev))
    ev, ch, am, ky, ps = ev[order], ch[order], am[order], ky[order], ps[order]
    bounds = np.flatnonzero(np.diff(ev)) + 1

    rows = []
    for grp in np.split(np.arange(len(ev)), bounds):
        if grp.size < 4:
            continue
        vals = {}
        good = True
        for k in range(4):
            m = grp[ky[grp] == k]
            if m.size == 0:
                good = False
                break
            cl = [c for c in cluster_1d(ch[m], ps[m], am[m])
                  if c[1] >= args.min_strips]
            if len(cl) != 1:                 # exactly one cluster per view
                good = False
                break
            vals[k] = cl[0][0]
        if good:
            rows.append((int(ev[grp[0]]), vals[0], vals[1], vals[2], vals[3]))

    if not rows:
        raise SystemExit('no clean 4-view events found')
    r = np.array([[x[1], x[2], x[3], x[4]] for x in rows])
    print(f'  {len(rows):,} events with exactly one cluster in all four views')

    # ---- the gate: reproduce the published front->back alignment ------------
    print('\n  front -> back fit vs published alignment')
    print('  axis   slope (pub)          offset (pub)        core sigma (pub)')
    verdict = True
    for i, ax in enumerate(('x', 'y')):
        f, b = r[:, i], r[:, 2 + i]
        keep = np.ones(len(f), bool)
        for _ in range(6):                   # robust: iterate on the core
            s, o = np.polyfit(f[keep], b[keep], 1)
            res = b - (s * f + o)
            sd = np.percentile(res[keep], [25, 75])
            iqr = (sd[1] - sd[0]) / 1.349
            keep = np.abs(res - np.median(res[keep])) < 4 * max(iqr, 0.05)
        p = PUBLISHED[ax]
        d_s = abs(s - p['slope'])
        d_o = abs(o - p['offset'])
        d_g = abs(iqr - p['sigma'])
        flag = 'OK' if (d_s < 0.01 and d_o < 3.0 and d_g < 0.5) else 'MISMATCH'
        verdict &= flag == 'OK'
        print(f'  {ax}    {s:8.5f} ({p["slope"]:.5f})   '
              f'{o:7.2f} ({p["offset"]:6.2f})   '
              f'{iqr:5.2f} ({p["sigma"]:.2f})   {flag}')

    if not verdict:
        raise SystemExit(
            '\nVALIDATION FAILED -- the extraction does not reproduce the '
            'published front->back alignment, so the clustering or the '
            'mapping is being applied wrongly. Not writing tracks.')
    print('\n  VALIDATION PASSED')

    # ---- write a sample, preferring events near the beam core ---------------
    rng = np.random.default_rng(20260807)
    pick = rng.choice(len(rows), size=min(args.n_out, len(rows)), replace=False)
    with open(args.out, 'w') as fh:
        fh.write('eventId,front_x_mm,front_y_mm,back_x_mm,back_y_mm,'
                 'z_front_mm,z_back_mm\n')
        for i in pick:
            e, fx, fy, bx, by = rows[i]
            fh.write(f'{e},{fx:.4f},{fy:.4f},{bx:.4f},{by:.4f},'
                     f'{Z_FRONT:.1f},{Z_BACK:.1f}\n')
    print(f'  wrote {args.out}  ({len(pick)} tracks)')


if __name__ == '__main__':
    main()
