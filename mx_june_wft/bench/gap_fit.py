#!/usr/bin/env python3
"""
gap_fit.py — the portable half of the drift-gap study: fit events, dump the
per-event NNLS charge-arrival profiles. No qa_config, no ROOT, no run registry:
everything it needs is a bench cache and a calibration bundle, so it runs on a
condor worker with numpy + scipy alone.

    gap_fit.py --cache bench_cache.pkl --bundle calib_bundle_lp2 \
               --out results/ --label det3_sat [--shard 3/8] [--jobs 1]

Sharding splits the event list N ways (`--shard i/N`, i in 0..N-1); every shard
writes its own parquet and `gap_merge.py` combines them into the endpoint fits,
the topography map and gap_study.json. A shard is a few minutes on one core.

The stacked-profile fitting, endpoint models and maps all live in gap_merge.py,
so this stage is pure, restartable compute.
"""
import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

_CAL = None


def _init(bundle, v_override, k_bins=None):
    global _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(bundle)
    if v_override:
        _CAL.v_drift = float(v_override)
        _CAL.kw = {'x': 1.0, 'y': 1.0}
    wm.use_calibration(_CAL)
    # A slow chamber's 30 mm column can be longer than the default 18 x 60 ns
    # basis (1080 ns): at v = 27 um/ns it needs 1110 ns, so the endpoint falls
    # off the end of the model and the erfc fit rails. Deepening the basis is
    # the fix; gap_merge.py reads the depth from the q columns.
    if k_bins:
        wm.set_depth_bins(int(k_bins))


def _fit_one(payload):
    """Exactly the production fit path (wft.reco._global_start + fit_plane_raw)."""
    from wft import model as wm
    from wft import reco as wr
    eid, wins = payload
    out = {'eid': eid}
    for plane in ('x', 'y'):
        P = wins.get(plane)
        if P is None:
            continue
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        try:
            p0s, _w, t0s = wm.init_guess(P, plane)
            p0s, w0, t0s = wr._global_start(P, plane, p0s, t0s, wm.HYPER)
            r = wm.fit_plane_raw(P, plane, p0s, w0, t0s)
        except Exception:
            continue
        if r is None or not np.isfinite(r['chi2']):
            continue
        out[plane] = dict(q=np.asarray(r['q'], float), w=float(r['w']),
                          t0=float(r['t0']), p0=float(r['p0']),
                          chi2dof=float(r['chi2'] / max(r['dof'], 1)))
    return out


def select_events(events, box, limit=None):
    """Single-cluster, undropped events with a finite reference, plus the
    geometric containment flag (track crosses the whole gap inside the box)."""
    payloads, klass = [], {}
    for eid, ev in sorted(events.items()):
        t = ev['truth']
        if not all(np.isfinite([t['ref_x'], t['ref_y'], t['tan_x'], t['tan_y']])):
            continue
        m = 3.0
        cont = bool(
            box['x'][0] + m + 15.5 * abs(t['tan_x']) <= t['ref_x']
            <= box['x'][1] - m - 15.5 * abs(t['tan_x'])
            and box['y'][0] + m + 15.5 * abs(t['tan_y']) <= t['ref_y']
            <= box['y'][1] - m - 15.5 * abs(t['tan_y']))
        wins = {}
        for p in ('x', 'y'):
            cand = ev['wins'].get(p)
            s = ev['seeds'].get(p)
            if cand and s and s[0]['n_dropped'] == 0 and len(cand) == 1:
                wins[p] = cand[0]
        if not wins:
            continue
        klass[eid] = (cont, t)
        payloads.append((eid, wins))
        if limit and len(payloads) >= limit:
            break
    return payloads, klass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='bench_cache.pkl')
    ap.add_argument('--bundle', required=True, help='calibration bundle dir')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--label', default='fit', help='tag for the output files')
    ap.add_argument('--shard', default='0/1', help='i/N')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--k-bins', type=int, default=None,
                    help='depth bins in the charge basis (default 18 x 60 ns '
                         '= 1080 ns); slow chambers need 22')
    ap.add_argument('--v-override', type=float, default=None,
                    help='refit with this drift speed (kw reset to 1)')
    args = ap.parse_args()

    import pandas as pd
    i_sh, n_sh = (int(x) for x in args.shard.split('/'))
    with open(args.cache, 'rb') as f:
        data = pickle.load(f)
    events, meta = data['events'], data['meta']
    box = meta['box']
    payloads, klass = select_events(events, box, args.limit)
    mine = payloads[i_sh::n_sh]
    print(f'{len(payloads):,} selected, shard {i_sh}/{n_sh} -> {len(mine):,} '
          f'events', flush=True)

    rows = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                                 initargs=(args.bundle, args.v_override,
                                           args.k_bins)) as pool:
            outs = list(pool.map(_fit_one, mine, chunksize=8))
    else:
        _init(args.bundle, args.v_override, args.k_bins)
        outs = [_fit_one(p) for p in mine]

    for o in outs:
        cont, t = klass[o['eid']]
        for plane in ('x', 'y'):
            d = o.get(plane)
            if d is None:
                continue
            rows.append(dict(eid=o['eid'], plane=plane, contained=cont,
                             ref_x=t['ref_x'], ref_y=t['ref_y'],
                             tan=t[f'tan_{plane}'], w=d['w'], t0=d['t0'],
                             chi2dof=d['chi2dof'], qsum=float(d['q'].sum()),
                             **{f'q{k}': float(v) for k, v in enumerate(d['q'])}))
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    stem = os.path.join(args.out, f'profiles_{args.label}_{i_sh:03d}')
    try:
        df.to_parquet(stem + '.parquet', index=False)
        print(f'wrote {stem}.parquet ({len(df):,} plane-rows)')
    except Exception as e:          # no pyarrow on the worker: csv still merges
        df.to_csv(stem + '.csv.gz', index=False)
        print(f'parquet unavailable ({e}); wrote {stem}.csv.gz')


if __name__ == '__main__':
    main()
