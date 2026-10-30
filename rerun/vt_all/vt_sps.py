#!/usr/bin/env python3
"""vt_sps.py --out-dir D [--blocks N] [--per-block 1000] [--t-min KEY=S ...] <sps registry key> [...]

Within-row v(t) of an SPS det4 row (VT_PLAN_2026-10-30) by the column duration (ws_spsv; rerun/sps_followup/vbox.py):
the row's plateau is cut into N equal wall-clock windows, each window's first --per-block events go through the pass's
cache builder (sps_build_cache.build_cache, same baseline as the pass), and ws_spsv's zsstack.py fits the stacked
strip-sum box per window and view.  v = 28.8 mm / T_box where the column ends at an edge (vbox's rule, v_table.py).
The ratio between windows needs no gap and no v: T_box(t) / T_box(row) is 1 / v_rel(t).  Independent of the strip map
(the strip sum).  Writes <out-dir>/vt_sps.json; the per-window caches are deleted.
"""
import argparse, copy, csv, json, os, pickle, subprocess, sys

import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
BUNDLE = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle'
GAP_MM = 28.8
EDGE_S = 20.0          # sps_build_cache's plateau-edge guard, applied to the row's outer edges only
RAW_CHUNK = 1000       # entries per read on RAW rows (sps_build_cache.CHUNK)


def defined(q):
    """vbox's rule (v_table.py): box with gradient and flat box agree to 8 %, gradient not railed, T_box <= 2900 ns."""
    return bool(q and q.get('T_box') and q.get('T_flat') and abs(q['T_box'] / q['T_flat'] - 1) <= 0.08
                and q['grad'] > -0.99 and q['T_box'] <= 2900)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+'); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--blocks', type=int, default=3); ap.add_argument('--per-block', type=int, default=1000)
    ap.add_argument('--t-min', nargs='*', default=[]); ap.add_argument('--bundle', default=BUNDLE)
    a = ap.parse_args()
    import datasets
    import sps_build_cache as sbc
    os.makedirs(a.out_dir, exist_ok=True)
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
    tmin = {k: float(v) for k, v in (s.split('=') for s in a.t_min)}
    res, caches = {}, []
    get0 = datasets.get
    for key in a.keys:
        row = reg[key]
        # RAW entries carry every channel x sample: read 1000 at a time (~1 GB) instead of the builder's 20000 (~17 GB)
        sbc.CHUNK = RAW_CHUNK if int(row['raw']) else 20000
        D = get0(row['dataset'])
        n, lo, hi, dv, rv = next(p for p in D['plateaus'] if p[0] == row['plateau'])
        lo, hi = max(lo + EDGE_S, tmin.get(key, -1e18)), hi - EDGE_S
        edges = np.linspace(lo, hi, a.blocks + 1)
        sub = [(f'{n}#b{i}', float(edges[i]), float(edges[i + 1]), dv, rv) for i in range(a.blocks)]
        D2 = copy.deepcopy(D); D2['plateaus'] = sub
        datasets.get = lambda name, _D=D2: _D
        evs = {s[0]: {} for s in sub}
        for sr in row['subruns'].split():
            if all(len(evs[s[0]]) >= a.per_block for s in sub):
                break
            got = sbc.build_cache(row['run'], sr, row['dataset'], per_plateau=a.per_block, edge_s=0.0,
                                  only=[s[0] for s in sub], verbose=False)
            for b, d in got.items():
                for eid, ev in d.items():
                    if len(evs[b]) < a.per_block:
                        evs[b][f'{sr}:{eid}'] = ev
        datasets.get = get0
        blocks = []
        for i, s in enumerate(sub):
            name = f'{key}__b{i}' + ('_raw' if int(row['raw']) and 'raw' not in key else '')
            p = os.path.join(a.out_dir, f'{name}.pkl'); pickle.dump(evs[s[0]], open(p, 'wb'), protocol=4); caches.append(p)
            tw = [e['t_wall'] for e in evs[s[0]].values()]
            blocks.append(dict(name=name, lo=s[1], hi=s[2], n=len(tw), t=float(np.median(tw)) if tw else None))
            print(f'{key} block {i}: {len(tw)} events {s[1]:.0f}-{s[2]:.0f} s', flush=True)
        res[key] = dict(field_Vcm=float(row['field_Vcm']), gas=row['gas'], mount=row['mount'], raw=int(row['raw']),
                        plateau=[lo, hi], blocks=blocks)
    out = os.path.join(a.out_dir, 'zsstack.json')
    subprocess.run([sys.executable, os.path.join(REPO, 'rerun', 'model_audit', 'ws_spsv', 'zsstack.py'), '--bundle', a.bundle,
                    '--out', out] + caches, check=True)
    Z = json.load(open(out))
    for key, r in res.items():
        for b in r['blocks']:
            for v in 'xy':
                q = Z.get(b['name'], {}).get(v)
                if q:
                    b[v] = {k: q.get(k) for k in ('n', 'T_box', 'T_flat', 'T_edge', 'T_50', 'grad')}
                    b[v]['defined'] = defined(q)
                    b[v]['v_box'] = GAP_MM * 1e3 / q['T_box'] if q.get('T_box') else None
        print(key, [(round(b['t'] or 0), {v: (b.get(v, {}).get('T_box'), b.get(v, {}).get('defined')) for v in 'xy'})
                    for b in r['blocks']], flush=True)
    json.dump(res, open(os.path.join(a.out_dir, 'vt_sps.json'), 'w'), indent=1)
    for p in caches:
        os.remove(p)


if __name__ == '__main__':
    main()
