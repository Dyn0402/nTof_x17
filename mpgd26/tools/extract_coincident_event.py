#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_coincident_event.py -- real FIVE-detector coincidences from the SPS run.

RUNS ON LXPLUS (the merged hit file is 11 GB and lives there):

    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    python3 extract_coincident_event.py \
        --mapping mapping_urwell.csv --align mapping_alignment.json \
        --run-config run_config.json --map-csv P2_BASKET_mapping.csv \
        --subrun 23 --out coincident_events.json

Companion to ``extract_urwell_tracks.py``, which produces uRWELL tracks alone.
This one additionally requires a **measured P2 cluster in all three stations**
for the same ``eventId``, so what comes out is a genuine muon seen by all five
chambers -- not a uRWELL track with the crossed pads predicted.

Per event it demands:

  * exactly one cluster in each of the four uRWELL views (front x/y, back x/y)
    -- the same requirement ``extract_urwell_tracks.py`` uses;
  * a mapped P2 pad hit above threshold on each of FEU 3 / 4 / 5
    (= P2_IN / P2_MID / P2_OUT, from the merged file's own ``feu_detector``);
  * the leading pad of each station within ``--probe-r`` of where the uRWELL
    track crosses that station, after the published rigid uRWELL -> pad-frame
    transform.  This is the same 15 mm probe the efficiency analysis uses.

**The validation is the point.**  As in the companion script, the uRWELL
front->back alignment is refitted and compared with the published values before
anything is written; and the per-station residual of the accepted events is
reported, because it has to come out at the pad quantisation (~3.4 mm) or the
transform is being applied wrongly.

Memory: the subrun is *streamed* in ``--chunk`` entry steps (default 2 M over
five branches, ~60 MB per step) and only hits belonging to the first
``--n-events`` distinct ``geventId`` are retained -- for the defaults a few
million kept hits, i.e. a few hundred MB peak.  Both knobs scale peak memory
linearly; check before raising either.

Why ``geventId`` and not ``eventId``: within a subrun the hits are ordered
chunk-then-FEU and ``eventId`` restarts in every file chunk, so a single entry
window sees only FEU 1 and ``eventId`` is not a usable event key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

FEU_URW = 1
FEU_STATION = {3: 'P2_IN', 4: 'P2_MID', 5: 'P2_OUT'}
Z_FRONT, Z_BACK = 0.0, 1370.0

# published in mapping_alignment.json -> urwell_front_to_back
PUBLISHED = {
    'x': dict(slope=0.9996010584080552, offset=-0.9584662880558853,
              sigma=0.76837300114191),
    'y': dict(slope=1.0417679522587129, offset=-4.107225316797881,
              sigma=0.7209845879062221),
}

# the c_5_top ribbon of P2_IN is mounted flipped; every other connector half of
# every station is 'reverse'.  Mirrors sps_config.STRATEGY_OVERRIDES.
STRATEGY_OVERRIDES = {'P2_IN': {(5, 'top'): 'linear'}}


# ------------------------------------------------------------ uRWELL side --- #
def load_urw_mapping(path):
    """channel -> (view key, position_mm) for FEU 1, as numpy lookups."""
    import csv
    ch_pos, ch_key = {}, {}
    order = {('EIC_uRWELL_front', 'x'): 0, ('EIC_uRWELL_front', 'y'): 1,
             ('EIC_uRWELL_back', 'x'): 2, ('EIC_uRWELL_back', 'y'): 3}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['feu']) != FEU_URW:
                continue
            c = int(row['channel'])
            ch_pos[c] = float(row['position_mm'])
            ch_key[c] = order[(row['detector'], row['view'])]
    n = max(ch_pos) + 1
    pos = np.full(n, np.nan)
    key = np.full(n, -1, dtype=np.int8)
    for c, p in ch_pos.items():
        pos[c] = p
        key[c] = ch_key[c]
    return pos, key


def cluster_1d(chans, poss, amps, max_gap=2):
    """Charge-weighted centroids of contiguous strip groups."""
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


def subrun_entry_offset(index_path, subrun_id):
    with open(index_path) as f:
        idx = json.load(f)
    off = 0
    for s in idx['subruns']:
        if s['subrun_id'] == subrun_id:
            return off, s
        off += s['n_hits']
    raise SystemExit(f'subrun {subrun_id} not in index')


# --------------------------------------------------------------- P2 side --- #
def station_pad_lut(run_config, map_csv, det_name, feu):
    """(channel -> pad_cx, pad_cy, channel_id) for one station's FEU.

    Uses the group's own p2_mapping so this cannot disagree with the
    efficiency analysis about which channel is which pad.
    """
    import p2_mapping as pmap
    tab = pmap.build_channel_table(
        run_config, map_csv, det_type='P2', det_name=det_name,
        strategy='reverse',
        strategy_overrides=STRATEGY_OVERRIDES.get(det_name, {}))
    tab = tab[(tab['feu'] == feu) & tab['mapped']]
    if tab.empty:
        raise SystemExit(f'{det_name}: no mapped channels on FEU {feu}')
    n = int(tab['channel'].max()) + 1
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    cid = np.full(n, -1, dtype=np.int32)
    cx[tab['channel'].to_numpy()] = tab['pad_cx'].to_numpy()
    cy[tab['channel'].to_numpy()] = tab['pad_cy'].to_numpy()
    cid[tab['channel'].to_numpy()] = tab['channel_id'].to_numpy()
    return cx, cy, cid, len(tab)


def rigid(align, det_name):
    """(2x2 rotation, translation) taking a uRWELL track point to pad frame."""
    st = align['urwell_to_p2_frame']['stations'][det_name]
    r = st['rigid_mean']
    th = np.deg2rad(r['theta_deg'])
    A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return A, np.array([r['dx'], r['dy']]), float(st['z_mm'])


# ------------------------------------------------------------------ main --- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default=os.path.expanduser(
        '~/x17/p2_sps_july/merged/all_subruns_hits.root'))
    ap.add_argument('--index', default=None)
    ap.add_argument('--mapping', required=True, help='mapping_urwell.csv')
    ap.add_argument('--align', required=True, help='mapping_alignment.json')
    ap.add_argument('--run-config', required=True,
                    help="the run's run_config.json (carries dream_feus)")
    ap.add_argument('--map-csv', required=True, help='P2_BASKET_mapping.csv')
    ap.add_argument('--subrun', type=int, default=23,
                    help='23 = highstat_eff_1/beam_commissioning_00')
    ap.add_argument('--n-events', type=int, default=200_000,
                    help='distinct geventIds to keep, taken from the start of '
                         'the subrun; bounds the accumulators')
    ap.add_argument('--chunk', type=int, default=2_000_000,
                    help='tree entries per streaming step')
    ap.add_argument('--amp-min', type=float, default=25.0)
    ap.add_argument('--p2-amp-min', type=float, default=40.0)
    ap.add_argument('--min-strips', type=int, default=2)
    ap.add_argument('--probe-r', type=float, default=15.0,
                    help='mm; the efficiency analysis uses 15')
    ap.add_argument('--single-pad', action='store_true',
                    help='require exactly one pad above threshold per station')
    ap.add_argument('--n-out', type=int, default=25)
    ap.add_argument('--out', default='coincident_events.json')
    args = ap.parse_args()
    args.index = args.index or args.file.replace('.root', '_index.json')

    import uproot

    align = json.load(open(args.align))
    pos_lut, key_lut = load_urw_mapping(args.mapping)
    offset, meta = subrun_entry_offset(args.index, args.subrun)
    print(f'subrun {args.subrun}: {meta["run"]}/{meta["sub_run"]}   '
          f'entry offset {offset:,}   {meta["n_hits"]:,} hits   '
          f'keeping the first {args.n_events:,} events')

    stations = {}
    for feu, name in FEU_STATION.items():
        cx, cy, cid, npads = station_pad_lut(args.run_config, args.map_csv,
                                             name, feu)
        A, t, z = rigid(align, name)
        stations[feu] = dict(name=name, cx=cx, cy=cy, cid=cid, A=A, t=t, z=z)
        print(f'  {name}: FEU {feu}, {npads} mapped channels, z = {z:.0f} mm')

    # ---- stream the subrun ---------------------------------------------- #
    # Within a subrun the hits are ordered chunk-then-FEU, and ``eventId``
    # restarts in every file chunk -- so a single entry window sees only the
    # uRWELLs, and eventId is not a usable key.  ``geventId`` (the merge's own
    # global id) is, and streaming keeps peak memory at one chunk plus the
    # bounded accumulators: for the defaults that is ~60 MB of chunk and
    # ~3 M kept hits, i.e. a few hundred MB in total.
    tree = uproot.open(args.file)['hits']
    branches = ['geventId', 'channel', 'amplitude', 'feu', 'subrun_id']
    keep_ev = None                      # the bounded set of geventIds
    acc = {f: [[], [], []] for f in (FEU_URW, *FEU_STATION)}   # ev, ch, amp
    n_read = 0
    for a in tree.iterate(branches, entry_start=offset,
                          entry_stop=offset + meta['n_hits'],
                          step_size=args.chunk, library='np'):
        got = np.unique(a['subrun_id'])
        if got.size != 1 or got[0] != args.subrun:
            raise SystemExit(f'ORDERING ASSUMPTION BROKEN: chunk holds '
                             f'subruns {got[:5]}, expected only {args.subrun}')
        gev, ch, am, fe = (a['geventId'], a['channel'],
                           a['amplitude'].astype(np.float32), a['feu'])
        n_read += len(gev)
        if keep_ev is None:
            u = np.unique(gev)
            keep_ev = u[:args.n_events]
            print(f'  event window: geventId {keep_ev[0]} .. {keep_ev[-1]} '
                  f'({len(keep_ev):,} events)')
        m = np.isin(gev, keep_ev)
        if not m.any():
            continue
        gev, ch, am, fe = gev[m], ch[m], am[m], fe[m]
        for f in acc:
            s = fe == f
            if s.any():
                acc[f][0].append(gev[s])
                acc[f][1].append(ch[s])
                acc[f][2].append(am[s])
    print(f'  streamed {n_read:,} entries; kept ' + ', '.join(
        f'FEU{f} {sum(len(x) for x in acc[f][0]):,}' for f in sorted(acc)))

    def cat(f):
        if not acc[f][0]:
            return (np.array([], np.int64), np.array([], np.int32),
                    np.array([], np.float32))
        return tuple(np.concatenate(acc[f][i]) for i in range(3))

    ev_u, ch_u, am_u = cat(FEU_URW)
    p2_raw = {f: cat(f) for f in FEU_STATION}
    del acc

    # ---- uRWELL tracks -------------------------------------------------- #
    sel = (am_u > args.amp_min) & (ch_u < len(pos_lut))
    ev, ch, am = ev_u[sel], ch_u[sel], am_u[sel].astype(float)
    ky, ps = key_lut[ch], pos_lut[ch]
    ok = ky >= 0
    ev, ch, am, ky, ps = ev[ok], ch[ok], am[ok], ky[ok], ps[ok]
    print(f'  {len(ev):,} mapped uRWELL hits over '
          f'{np.unique(ev).size:,} events')

    order = np.lexsort((ch, ky, ev))
    ev, ch, am, ky, ps = ev[order], ch[order], am[order], ky[order], ps[order]
    bounds = np.flatnonzero(np.diff(ev)) + 1

    tracks = {}
    for grp in np.split(np.arange(len(ev)), bounds):
        if grp.size < 4:
            continue
        vals, good = {}, True
        for k in range(4):
            m = grp[ky[grp] == k]
            if m.size == 0:
                good = False
                break
            cl = [c for c in cluster_1d(ch[m], ps[m], am[m])
                  if c[1] >= args.min_strips]
            if len(cl) != 1:
                good = False
                break
            vals[k] = cl[0][0]
        if good:
            tracks[int(ev[grp[0]])] = vals
    print(f'  {len(tracks):,} events with one cluster in all four uRWELL views')
    if not tracks:
        raise SystemExit('no clean 4-view events')

    # ---- the gate: reproduce the published front->back alignment --------- #
    r = np.array([[v[0], v[1], v[2], v[3]] for v in tracks.values()])
    print('\n  front -> back fit vs published')
    print('  axis   slope (pub)            offset (pub)          sigma (pub)')
    verdict = True
    for i, ax in enumerate(('x', 'y')):
        f, b = r[:, i], r[:, 2 + i]
        keep = np.ones(len(f), bool)
        for _ in range(6):
            s, o = np.polyfit(f[keep], b[keep], 1)
            res = b - (s * f + o)
            q = np.percentile(res[keep], [25, 75])
            iqr = (q[1] - q[0]) / 1.349
            keep = np.abs(res - np.median(res[keep])) < 4 * max(iqr, 0.05)
        p = PUBLISHED[ax]
        flag = ('OK' if (abs(s - p['slope']) < 0.01 and
                         abs(o - p['offset']) < 3.0 and
                         abs(iqr - p['sigma']) < 0.5) else 'MISMATCH')
        verdict &= flag == 'OK'
        print(f'   {ax}    {s:.5f} ({p["slope"]:.5f})   '
              f'{o:+.2f} ({p["offset"]:+.2f}) mm   '
              f'{iqr:.2f} ({p["sigma"]:.2f}) mm   {flag}')
    if not verdict:
        raise SystemExit('\nALIGNMENT NOT REPRODUCED -- refusing to write.')
    print('  alignment reproduced\n')

    # ---- P2 leading pad per station per event ---------------------------- #
    want = np.fromiter(tracks.keys(), dtype=np.int64)
    want.sort()
    lead = {}          # feu -> {geventId: (cx, cy, channel_id, amp, n_pads)}
    for feu, S in stations.items():
        e0, c0, w0 = p2_raw[feu]
        m = ((w0 > args.p2_amp_min) & (c0 < len(S['cx'])) &
             np.isin(e0, want))
        e, c, w = e0[m], c0[m], w0[m]
        good = np.isfinite(S['cx'][c])
        e, c, w = e[good], c[good], w[good]
        if e.size == 0:
            lead[feu] = {}
            print(f'  {S["name"]}: no mapped pad hits on the tracked events')
            continue
        o = np.lexsort((-w, e))                 # highest amplitude first
        e, c, w = e[o], c[o], w[o]
        first = np.concatenate(([True], np.diff(e) != 0))
        counts = dict(zip(*np.unique(e, return_counts=True)))
        d = {}
        for ei, ci, wi in zip(e[first], c[first], w[first]):
            d[int(ei)] = (float(S['cx'][ci]), float(S['cy'][ci]),
                          int(S['cid'][ci]), float(wi), int(counts[ei]))
        lead[feu] = d
        print(f'  {S["name"]}: leading pad found in {len(d):,} of '
              f'{len(tracks):,} tracked events')

    # ---- require all three stations, in the probe window ----------------- #
    cands = []
    for eid, v in tracks.items():
        if not all(eid in lead[f] for f in stations):
            continue
        f_xy = np.array([v[0], v[1]])
        b_xy = np.array([v[2], v[3]])
        hits, dists, npads = {}, [], []
        ok = True
        for feu, S in stations.items():
            u = f_xy + (b_xy - f_xy) * (S['z'] - Z_FRONT) / (Z_BACK - Z_FRONT)
            pred = S['A'] @ u + S['t']
            cxh, cyh, cid, amp, npad = lead[feu][eid]
            d = float(np.hypot(cxh - pred[0], cyh - pred[1]))
            if d > args.probe_r or (args.single_pad and npad != 1):
                ok = False
                break
            hits[S['name']] = dict(
                pad_cx_mm=cxh, pad_cy_mm=cyh, channel_id=cid,
                amplitude=amp, n_pads=npad, z_mm=S['z'],
                pred_cx_mm=float(pred[0]), pred_cy_mm=float(pred[1]),
                residual_mm=d)
            dists.append(d)
            npads.append(npad)
        if not ok:
            continue
        cands.append(dict(
            geventId=eid,
            urwell=dict(front_x_mm=v[0], front_y_mm=v[1],
                        back_x_mm=v[2], back_y_mm=v[3],
                        z_front_mm=Z_FRONT, z_back_mm=Z_BACK),
            stations=hits,
            max_residual_mm=float(max(dists)),
            mean_residual_mm=float(np.mean(dists)),
            total_pads=int(sum(npads))))

    if not cands:
        raise SystemExit('no five-detector coincidences found -- loosen '
                         '--probe-r or drop --single-pad')

    res = np.array([c['mean_residual_mm'] for c in cands])
    print(f'\n  {len(cands):,} FIVE-detector coincidences '
          f'({len(cands) / len(tracks) * 100:.1f}% of tracked events)')
    print(f'  per-station residual: mean {res.mean():.2f} mm, '
          f'median {np.median(res):.2f} mm  '
          f'(expect ~3.4 mm = 12 mm/sqrt(12) pad quantisation)')

    # cleanest first: tightest residuals, fewest pads
    cands.sort(key=lambda c: (c['max_residual_mm'], c['total_pads']))
    out = dict(
        source=dict(run=meta['run'], sub_run=meta['sub_run'],
                    subrun_id=args.subrun, file=args.file),
        selection=dict(probe_r_mm=args.probe_r, amp_min=args.amp_min,
                       p2_amp_min=args.p2_amp_min,
                       single_pad=bool(args.single_pad),
                       min_strips=args.min_strips),
        validation=dict(urwell_alignment='reproduced',
                        n_tracked=len(tracks), n_coincident=len(cands),
                        residual_mean_mm=float(res.mean()),
                        residual_median_mm=float(np.median(res))),
        events=cands[:args.n_out])
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\n  [wrote {args.out}: {len(out["events"])} events]')
    print('\n  best candidates')
    print('  geventId     max resid   pads   P2_IN / P2_MID / P2_OUT pad ids')
    for c in cands[:8]:
        ids = ' / '.join(str(c['stations'][s]['channel_id'])
                         for s in ('P2_IN', 'P2_MID', 'P2_OUT'))
        print(f'  {c["geventId"]:>9}   {c["max_residual_mm"]:6.2f} mm   '
              f'{c["total_pads"]:>3}    {ids}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
