#!/usr/bin/env python3
"""
16_multitrack_gate.py — gate the multi-track generalisation on bench data.

Two questions, both answered from the bench cache (real waveforms, M3 truth):

1. RECOVERY — synthetic double-track events: merge the candidate windows of
   two well-separated single-track events into one payload and require that
   the reconstruction returns BOTH tracks (n_tracks == 2) with each (x, y)
   pair within tolerance of its parent event's own single-track fit. This
   exercises the whole chain the way an n_TOF double-track event would:
   two candidate clusters per plane, fit_plane_candidates fitting all of
   them, select_tracks pairing them across planes.

2. GHOST RATE — real single-track (cosmic) events: the fraction reporting
   n_tracks >= 2 is the fake-double-track rate the gate buys us. Cosmics are
   single muons, so this should be ~0; every excess event is a split cluster
   or coincident noise that slipped the plausibility gate.

    ../.venv/bin/python mx_june_wft/16_multitrack_gate.py sat_det3 \
        --n-pairs 40 --n-ghost 400 --jobs 8
"""
import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

SEP_MIN_MM = 40.0        # parent tracks must be at least this far apart (x AND y)
TOL_MM = 3.0             # recovered p0 must sit this close to the parent's fit


def _clean_singles(events, rows_by_eid):
    """Events with exactly one candidate window per plane and a good fit."""
    out = []
    for eid, ev in events.items():
        w = ev['wins']
        if len(w.get('x') or []) != 1 or len(w.get('y') or []) != 1:
            continue
        r = rows_by_eid.get(eid)
        if r is None or not (r['x_ok'] and r['y_ok']):
            continue
        if r['n_tracks'] != 1:
            continue
        out.append(eid)
    return out


def _merge_payload(eid, ev_a, ev_b):
    wins = {p: list(ev_a['wins'][p]) + list(ev_b['wins'][p]) for p in ('x', 'y')}
    seeds = {p: (list(ev_a['seeds'].get(p) or []) +
                 list(ev_b['seeds'].get(p) or [])) for p in ('x', 'y')}
    n_hits = ev_a['n_hits'] + ev_b['n_hits']
    return (eid, wins, seeds, n_hits, False, ev_a.get('ftst') or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--n-pairs', type=int, default=40)
    ap.add_argument('--n-ghost', type=int, default=400)
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(args.run_key)
    cache = args.cache or os.path.join(cfg.OUT_BASE, 'wft', 'bench_cache_ftst.pkl')
    with open(cache, 'rb') as f:
        data = pickle.load(f)
    events, meta = data['events'], data['meta']
    bundle = args.bundle or meta['bundle']
    print(f'{len(events):,} cached events, bundle {os.path.basename(bundle)}')

    from wft import reco as wr

    # ---- pass 1: fit real events (ghost rate + parents for the merge) ------
    keys = sorted(events)[:max(args.n_ghost, 4 * args.n_pairs)]
    payloads = [(e, events[e]['wins'], events[e]['seeds'], events[e]['n_hits'],
                 events[e]['spark'], events[e].get('ftst') or {}) for e in keys]
    with ProcessPoolExecutor(max_workers=args.jobs,
                             initializer=wr._worker_init,
                             initargs=(bundle,)) as pool:
        rows = list(pool.map(wr._worker_fit, payloads, chunksize=8))
    rows_by_eid = {r['event_id']: r for r in rows}

    fitted = [r for r in rows if r['x_ok'] and r['y_ok']]
    multi = [r for r in fitted if r['n_tracks'] >= 2]
    zero = [r for r in fitted if r['n_tracks'] == 0]
    print(f'\nGHOST RATE (real single-muon events)')
    print(f'  fitted both planes : {len(fitted):,}')
    print(f'  n_tracks >= 2      : {len(multi):,}  '
          f'({100.0 * len(multi) / max(len(fitted), 1):.2f} %)  <- fake doubles')
    print(f'  n_tracks == 0      : {len(zero):,}  '
          f'({100.0 * len(zero) / max(len(fitted), 1):.2f} %)  '
          f'(winner failed the track gate)')

    # ---- pass 2: synthetic double tracks ------------------------------------
    singles = _clean_singles({e: events[e] for e in keys}, rows_by_eid)
    rng = np.random.RandomState(42)
    rng.shuffle(singles)
    pairs, used = [], set()
    for a in singles:
        if a in used or len(pairs) >= args.n_pairs:
            continue
        ra = rows_by_eid[a]
        for b in singles:
            if b in used or b == a:
                continue
            rb = rows_by_eid[b]
            if (abs(ra['x_p0'] - rb['x_p0']) >= SEP_MIN_MM
                    and abs(ra['y_p0'] - rb['y_p0']) >= SEP_MIN_MM):
                pairs.append((a, b))
                used.update((a, b))
                break
    print(f'\nSYNTHETIC DOUBLES: {len(pairs)} merged events '
          f'(parents >= {SEP_MIN_MM:.0f} mm apart, both planes)')

    merged = [_merge_payload(10_000_000 + k, events[a], events[b])
              for k, (a, b) in enumerate(pairs)]
    with ProcessPoolExecutor(max_workers=args.jobs,
                             initializer=wr._worker_init,
                             initargs=(bundle,)) as pool:
        mrows = list(pool.map(wr._worker_fit, merged, chunksize=4))

    n_two, n_matched = 0, 0
    for (a, b), row in zip(pairs, mrows):
        cands = row.get('_cand', [])
        got = {}
        for tid in (0, 1):
            tx = [c for c in cands if c['plane'] == 'x' and c['track_id'] == tid
                  and c['track_gated']]
            ty = [c for c in cands if c['plane'] == 'y' and c['track_id'] == tid
                  and c['track_gated']]
            if tx and ty:
                got[tid] = (tx[0]['p0'], ty[0]['p0'])
        two = row['n_tracks'] >= 2 and len(got) >= 2
        n_two += two
        if two:
            truth = [(rows_by_eid[e]['x_p0'], rows_by_eid[e]['y_p0'])
                     for e in (a, b)]
            recos = list(got.values())
            # best assignment of the 2 recos to the 2 parents
            d = [[np.hypot(r[0] - t[0], r[1] - t[1]) for t in truth]
                 for r in recos]
            ok = ((d[0][0] < TOL_MM and d[1][1] < TOL_MM)
                  or (d[0][1] < TOL_MM and d[1][0] < TOL_MM))
            n_matched += ok
            if not ok:
                print(f'  MISMATCH parents ({a},{b}): reco {recos} vs '
                      f'parents {truth}')
        else:
            print(f'  MISSED parents ({a},{b}): n_tracks={row["n_tracks"]}')
    n = max(len(pairs), 1)
    print(f'  both tracks found  : {n_two}/{len(pairs)} '
          f'({100.0 * n_two / n:.0f} %)')
    print(f'  both within {TOL_MM:.0f} mm of parents: {n_matched}/{len(pairs)} '
          f'({100.0 * n_matched / n:.0f} %)')


if __name__ == '__main__':
    main()
