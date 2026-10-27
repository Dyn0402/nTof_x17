#!/usr/bin/env python3
"""build_masks.py -- dead lists as bundle content under one rule (FE item 12, policy of ws_dead_charge F12).

Rule: a dataset's dead list = the chamber's CHAMBER-SIDE faults, carried by strip position (flagged in >= 2 datasets
read through different electronics: f12_table.json class 'detector'), + that dataset's OWN faults measured on its own
hits (census dead / shorts of the M3 crossing census, flank-dead, silent), minus census-dead channels that the flank test
on the same hits finds firing normally beside bright neighbours (the census over-call at ~20-30 crossings per channel,
ws_dead_charge: 35 of det4 X's 54).  Electronics faults of OTHER datasets are not carried (they follow the FEU channel).

Outputs (this directory, masks/):
  bench_<det>.json      the five golden rows (bench channel numbers), with what changed against the promoted bundle
  ntof_<arm>_<run>.json n_TOF chambers A-D (det3/det2/det6/det7) for run_55 / run_79 / run_145 (n_TOF channel numbers)
Inputs: ws_dead_charge/f12_table.json, census/flank_*.json, rerun/channel_layer/<det>_<key>.json, the promoted bundles'
dead lists (~/x17/cosmic_bench/model_audit/<det>/calib_bundle).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
WS = os.path.join(REPO, 'rerun', 'model_audit', 'ws_dead_charge')
M = os.path.expanduser('~/x17/cosmic_bench/model_audit')
GOLD = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
NTOF = {'A': 'det3', 'B': 'det2', 'C': 'det6', 'D': 'det7'}
C = np.arange(512)
BPOS = np.round(49.92 * (C // 64) + (63 - C % 64) * 0.78, 2)       # every June bench row (strip_position_map, checked)
ACTIVE = {'x': (0.0, 398.6), 'y': (18.0, 380.0)}
HEALTHY_F, HEALTHY_N = 0.6, 20                                     # mask_audit.py's flank-healthy rule


def chamber_side(T, det, v):
    lo, hi = ACTIVE[v]
    return sorted(r['pos'] for r in T[f'{det}_{v}']['positions'] if r['cls'] == 'detector' and lo <= r['pos'] <= hi)


def nearest(pos, p, tol=0.3):
    ok = np.isfinite(pos)
    if not ok.any():
        return None
    i = int(np.nanargmin(np.where(ok, np.abs(pos - p), np.inf)))
    return i if abs(pos[i] - p) <= tol else None


def bench(T):
    out = {}
    for det, key in GOLD.items():
        b = json.load(open(os.path.join(M, det, 'calib_bundle', 'bundle.json')))
        cen = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', f'{det}_{key}.json')))
        fl = json.load(open(os.path.join(WS, 'census', f'flank_{det}_{key}.json')))
        rec = dict(det=det, run_key=key, rule=__doc__.split('\n\n')[1].strip(), planes={})
        for v in 'xy':
            old = sorted(int(c) for c in b['dead'].get(v, []))
            f = fl.get(v) or {}
            healthy = sorted(int(c) for c in cen[v]['dead']
                             if f and f['f'][c] > HEALTHY_F and max(f['n1'][c], f['n2'][c]) >= HEALTHY_N)
            cs = chamber_side(T, det, v)
            cs_ch = sorted({int(nearest(BPOS.astype(float), p)) for p in cs if nearest(BPOS.astype(float), p) is not None})
            own = set(int(c) for c in cen[v]['dead']) | set(int(c) for c in cen[v].get('shorts', []))
            own |= set(int(c) for c in f.get('dead', [])) | set(int(c) for c in f.get('silent', []))
            new = sorted((own - set(healthy)) | set(cs_ch))
            rec['planes'][v] = dict(dead=new, n_old=len(old), n_new=len(new),
                                    added=sorted(set(new) - set(old)), removed=sorted(set(old) - set(new)),
                                    chamber_side=cs_ch, census_flank_healthy=healthy,
                                    added_mm=[float(BPOS[c]) for c in sorted(set(new) - set(old))],
                                    removed_mm=[float(BPOS[c]) for c in sorted(set(old) - set(new))])
            print(f'{det} {v}: {len(old)} -> {len(new)}  +{len(set(new) - set(old))} -{len(set(old) - set(new))} '
                  f'(chamber-side {len(cs_ch)}, flank-healthy census {len(healthy)})')
        out[det] = rec
    return out


def ntof(T):
    out = {}
    for run in ('run_55', 'run_79', 'run_145'):
        F = json.load(open(os.path.join(WS, 'census', f'flank_ntof_{run}.json')))
        for arm, det in NTOF.items():
            rec = dict(arm=arm, det=det, run=run, rule=__doc__.split('\n\n')[1].strip(), planes={})
            for v in 'xy':
                r = F[arm][v]
                pos = np.array(r['pos'], float)
                cs = chamber_side(T, det, v)
                cs_ch = sorted({nearest(pos, p) for p in cs} - {None})
                own = sorted(set(int(c) for c in r['dead']) | set(int(c) for c in r.get('silent', [])))
                new = sorted(set(cs_ch) | set(own))
                rec['planes'][v] = dict(dead=[int(c) for c in new], feu=int(r['feu']), chamber_side=[int(c) for c in cs_ch],
                                        own=own, n_judged=len(r['judged']),
                                        chamber_side_not_cabled=len(cs) - len(cs_ch))
            out[f'{arm}_{run}'] = rec
            print(f'n_TOF {arm} ({det}) {run}: x {len(rec["planes"]["x"]["dead"])} (own {len(rec["planes"]["x"]["own"])}, '
                  f'judged {rec["planes"]["x"]["n_judged"]}), y {len(rec["planes"]["y"]["dead"])} '
                  f'(own {len(rec["planes"]["y"]["own"])}, judged {rec["planes"]["y"]["n_judged"]})')
    return out


def main():
    T = json.load(open(os.path.join(WS, 'f12_table.json')))
    d = os.path.join(HERE, 'masks'); os.makedirs(d, exist_ok=True)
    for det, rec in bench(T).items():
        json.dump(rec, open(os.path.join(d, f'bench_{det}.json'), 'w'), indent=1)
    for k, rec in ntof(T).items():
        json.dump(rec, open(os.path.join(d, f'ntof_{k}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
