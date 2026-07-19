#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27x_features.py — bench hits6 features for the run_55 real-track clusters, from
combined_hits at LOW threshold (100 ADC), so the neighbour-based features
(q_frac, n_raw, a_asym, t_delay) are faithful (the 27 gate cache stored only
>=400 ADC strips, which truncates them).

For every gate-selected INCLINED cluster (trackcache.tag_tracklike, b1/b2), we
re-read combined_hits, take all strips >=100 ADC in the cluster's position
window of that (event, det, plane), and compute microtpc_lib.hit_features
(tot_lead, q_frac, n_raw, a_asym, a_lead, t_delay + signed asymmetries) plus a
low-threshold anchored time-fit angle.  Saturated lead strips are flagged (their
combined_hits amplitude overshoots — see 27z waveform check); a_lead is capped.

Output: calib/27_features.npz — one row per inclined cluster, aligned by a
stable key (sub, eid, det, pln, cen), consumed by 27y_regression_align.py.

Run:  venv/bin/python mx_july_beam_qa/27x_features.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import trackcache as tc
from common.Mx17StripMap import RunConfig
from ntof_tracking import microtpc_lib as mt

HERE = os.path.dirname(__file__)
RUN_DIR = os.path.expanduser('~/x17/beam_july/runs/run_55')
MAP_CSV = os.path.join(HERE, '..', 'mx17_m1_map.csv')
CALIB = os.path.join(HERE, 'calib')
THR_LOW = 100.0          # bench THR_HIT
WIN_MM = 20.0            # position half-window around the gated cluster centroid
DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
FEATS = list(mt.FEATS_HITS6)


def build_lut(rc):
    lut = np.full((9, 512, 3), np.nan)
    for di, det in enumerate(DETS):
        d = rc.get_detector(det)
        for feu in sorted(set(v[0] for v in d.dream_feus.values())):
            for ch in range(512):
                x, y = d.map_hit(feu, ch)
                if x is not None:
                    lut[feu, ch] = (di, 0, x)
                elif y is not None:
                    lut[feu, ch] = (di, 1, y)
    return lut


def main():
    rc = RunConfig(os.path.join(RUN_DIR, 'run_config.json'), MAP_CSV)
    lut = build_lut(rc)

    # gate selection (inclined, b1/b2), from the 27 cache
    cl, ss, ev = tc.load_all()
    cl = tc.add_derived(cl, ss)
    evk = ev.set_index(['sub', 'eid'])
    cl = cl.join(evk[['t_ms']], on=['sub', 'eid'])
    cl['b12'] = (cl['t_ms'] > 7) & (cl['t_ms'] < 30)
    cl = tc.tag_tracklike(cl)
    inc = cl[cl['b12'] & cl['inclined']].copy().reset_index(drop=True)
    inc['cid'] = np.arange(len(inc))
    print(f'inclined clusters to feature-ise: {len(inc)}')

    rows = []
    for sub, g in inc.groupby('sub'):
        subdir = os.path.join(RUN_DIR, sub)
        fh = [f for f in glob.glob(os.path.join(
            subdir, 'combined_hits_root', '*combined_hits.root'))
            if '_pedestals_' not in f][0]
        h = uproot.open(fh)['hits'].arrays(
            ['eventId', 'feu', 'channel', 'amplitude', 'time_of_max',
             'time_over_threshold', 'saturated'], library='np')
        l = lut[h['feu'], h['channel']]
        good = np.isfinite(l[:, 0]) & (h['amplitude'] > THR_LOW)
        eid = h['eventId'][good].astype(np.int64)
        det = l[good, 0].astype(np.int8); pln = l[good, 1].astype(np.int8)
        pos = l[good, 2].astype(np.float32)
        amp = np.clip(h['amplitude'][good], 0, 4200).astype(np.float32)
        tim = h['time_of_max'][good].astype(np.float32)   # drift arrival [ns]
        tot = h['time_over_threshold'][good].astype(np.float32)
        sat = h['saturated'][good].astype(bool)
        # index hits by (eid, det, pln)
        key = (eid.astype(np.int64) << 6) | (det.astype(np.int64) << 1) | pln
        order = np.argsort(key, kind='stable')
        key = key[order]; pos = pos[order]; amp = amp[order]
        tim = tim[order]; tot = tot[order]; sat = sat[order]
        uk, first = np.unique(key, return_index=True)
        bnd = np.append(first, len(key))
        kmap = {int(k): (first[i], bnd[i + 1]) for i, k in enumerate(uk)}

        for _, r in g.iterrows():
            kk = (int(r['eid']) << 6) | (int(r['det']) << 1) | int(r['pln'])
            rng = kmap.get(kk)
            rec = dict(cid=int(r['cid']))
            if rng is None:
                rows.append(rec); continue
            a, b = rng
            sel = np.abs(pos[a:b] - r['cen']) <= WIN_MM
            P = pos[a:b][sel]; A = amp[a:b][sel]; T = tim[a:b][sel]
            Q = tot[a:b][sel]; S = sat[a:b][sel]
            if len(P) < 3:
                rows.append(rec); continue
            fe = mt.hit_features(P, A, T, Q)
            if fe is not None:
                for f in FEATS:
                    rec[f] = fe[f]
                rec['a_asym_sgn'] = fe['a_asym_sgn']
                rec['t_asym_sgn'] = fe['t_asym_sgn']
                rec['pos_lead'] = fe['pos_lead_mm']
                rec['lead_sat'] = bool(S[np.argmax(A)])
            # low-threshold anchored fit angle
            aft = mt.anchored_time_fit(P, T, A)
            if aft is not None:
                rec['slope_lo'] = aft['slope_ns_per_mm']
                rec['n_lo'] = aft['n_strips']
                rec['extent_lo'] = aft['extent_mm']
            rows.append(rec)
    F = pd.DataFrame(rows)
    # merge back cluster metadata
    keep = ['cid', 'sub', 'detn', 'plnn', 'det', 'pln', 'cen', 'resist_v',
            'slope', 'qsum', 'n', 'extent', 'dur', 'mono']
    out = inc[keep].merge(F, on='cid', how='left')
    valid = out[FEATS].notna().all(axis=1)
    print(f'features computed for {int(valid.sum())}/{len(out)} clusters '
          f'({100*valid.mean():.0f}%)')
    print(out.loc[valid, FEATS].describe().round(2).T[['mean', 'std', '50%']])
    os.makedirs(CALIB, exist_ok=True)
    np.savez_compressed(os.path.join(CALIB, '27_features.npz'),
                        **{c: out[c].values for c in out.columns})
    print(f'saved calib/27_features.npz ({len(out)} rows)')


if __name__ == '__main__':
    main()
