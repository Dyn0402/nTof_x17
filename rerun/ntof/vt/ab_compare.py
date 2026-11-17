#!/usr/bin/env python3
"""ab_compare.py -- compare the v(t) A/B test (make_vt_ab.py, condor 4677861) with the main pass.

    ab_compare.py --dir DIR        # DIR/A = main-pass outputs, DIR/B = test outputs, DIR/ab.json from make_vt_ab.py

A = main pass (frozen v_drift), B = the same acquisition with v_drift = the measured sub-run v.  r = v_A / v_B.
The tan of A converted to B's v is tan_A * r (exact for w; the bow tilt term is subtracted in tan units after the scale,
so ~1e-3 differences remain).  Questions:

 1. Same tracks?  Per (event, plane): the chosen fit's p0 equal within 0.05 mm and t0 within 0.5 ns -> "same".
    For same tracks: w_B / w_A - 1 (should be 0) and tan_B / (tan_A r) - 1.
 2. Where they differ, where are they in true |tan| (inside / outside 0.5), and which side has the better dchi2?
 3. Acceptance: chosen fits with true |tan| < 0.5 (and 0.5-0.6), quality_ok, A vs B.
 4. All tracks (candidates with track_id >= 0), matched by p0: tracks only in A, only in B.

Writes DIR/ab_summary.json and prints it.
"""
import argparse, glob, json, os

import numpy as np
import pandas as pd


def load(path):
    t = pd.read_parquet(path)
    c = pd.read_parquet(path.replace('.parquet', '.candidates.parquet'))
    return t, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    a = ap.parse_args()
    ab = json.load(open(f'{a.dir}/ab.json'))
    per = []
    out = {}
    for fb in sorted(glob.glob(f'{a.dir}/B/run_*/*/*/run_*_?.parquet')):
        rel = os.path.relpath(fb, f'{a.dir}/B')
        fa = f'{a.dir}/A/{rel}'
        if not os.path.exists(fa):
            continue
        run, sub, arm = rel.split('/')[0][4:], rel.split('/')[1], rel.split('/')[2]
        m = ab[f'{arm}_{run}_{sub}']
        r = m['v_frozen'] / m['v_test']
        tA, cA = load(fa)
        tB, cB = load(fb)
        for p in 'xy':
            A = tA[['event_id', f'{p}_ok', f'{p}_p0', f'{p}_w', f'{p}_t0', f'{p}_tan_theta', f'{p}_quality_ok']]
            B = tB[['event_id', f'{p}_ok', f'{p}_p0', f'{p}_w', f'{p}_t0', f'{p}_tan_theta', f'{p}_quality_ok']]
            M = A.merge(B, on='event_id', how='outer', suffixes=('_A', '_B'), indicator=True)
            okA = M[f'{p}_ok_A'].fillna(False).astype(bool)
            okB = M[f'{p}_ok_B'].fillna(False).astype(bool)
            both = okA & okB
            same = both & ((M[f'{p}_p0_A'] - M[f'{p}_p0_B']).abs() < 0.05) & ((M[f'{p}_t0_A'] - M[f'{p}_t0_B']).abs() < 0.5)
            tAt = M[f'{p}_tan_theta_A'] * r          # A's tan in B's (measured) v
            tBt = M[f'{p}_tan_theta_B']
            qA = okA & M[f'{p}_quality_ok_A'].fillna(False).astype(bool)
            qB = okB & M[f'{p}_quality_ok_B'].fillna(False).astype(bool)
            d = dict(arm=arm, run=run, sub=sub, tag=os.path.basename(fb).split('_')[4] if False else rel, plane=p, r=r,
                     n_events=len(M), only_A_event=int((M._merge == 'left_only').sum()),
                     only_B_event=int((M._merge == 'right_only').sum()),
                     okA=int(okA.sum()), okB=int(okB.sum()), both=int(both.sum()), same=int(same.sum()),
                     w_rel_max=float(((M[f'{p}_w_B'] / M[f'{p}_w_A'] - 1)[same]).abs().max()) if same.any() else np.nan,
                     tan_rel_med=float(((tBt / tAt - 1)[same & (tAt.abs() > 0.05)]).median()) if same.any() else np.nan,
                     tan_rel_68=float(np.subtract(*np.percentile((tBt / tAt - 1)[same & (tAt.abs() > 0.05)], [84, 16])) / 2)
                     if (same & (tAt.abs() > 0.05)).sum() > 10 else np.nan,
                     acc05_A=int((qA & (tAt.abs() < 0.5)).sum()), acc05_B=int((qB & (tBt.abs() < 0.5)).sum()),
                     acc0506_A=int((qA & (tAt.abs() >= 0.5) & (tAt.abs() < 0.6)).sum()),
                     acc0506_B=int((qB & (tBt.abs() >= 0.5) & (tBt.abs() < 0.6)).sum()),
                     acc06_A=int((qA & (tAt.abs() < 0.6)).sum()), acc06_B=int((qB & (tBt.abs() < 0.6)).sum()))
            diff = both & ~same
            d['diff'] = int(diff.sum())
            d['diff_in_in'] = int((diff & (tAt.abs() < 0.5) & (tBt.abs() < 0.5)).sum())
            d['diff_in_out'] = int((diff & (tAt.abs() < 0.5) & (tBt.abs() >= 0.5)).sum())
            d['diff_out_in'] = int((diff & (tAt.abs() >= 0.5) & (tBt.abs() < 0.5)).sum())
            d['diff_out_out'] = int((diff & (tAt.abs() >= 0.5) & (tBt.abs() >= 0.5)).sum())
            # all tracks (track_id >= 0) per (event, plane), matched by p0 within 0.3 mm
            TA = cA[(cA.plane == p) & (cA.track_id >= 0)][['event_id', 'p0', 'tan_theta']]
            TB = cB[(cB.plane == p) & (cB.track_id >= 0)][['event_id', 'p0', 'tan_theta']]
            nA = nB = nm = nm05 = nA05 = nB05 = 0
            gb = dict(tuple(TB.groupby('event_id')))
            for eid, ga in TA.groupby('event_id'):
                pa = ga.p0.to_numpy()
                ta = ga.tan_theta.to_numpy() * r
                nA += len(pa); nA05 += int((np.abs(ta) < 0.5).sum())
                g2 = gb.get(eid)
                if g2 is None:
                    continue
                pb = g2.p0.to_numpy()
                for x, tt in zip(pa, ta):
                    if np.any(np.abs(pb - x) < 0.3):
                        nm += 1
                        nm05 += int(abs(tt) < 0.5)
            nB = len(TB)
            nB05 = int((TB.tan_theta.abs() < 0.5).sum())
            d.update(tracks_A=nA, tracks_B=nB, tracks_matched=nm, tracks05_A=nA05, tracks05_B=nB05, tracks05_matched=nm05)
            per.append(d)
    P = pd.DataFrame(per)
    P.to_csv(f'{a.dir}/ab_per_acq.csv', index=False)
    for arm, g in P.groupby('arm'):
        s = g.select_dtypes('number').sum()
        out[arm] = dict(acq_planes=len(g), r_range=[float(g.r.min()), float(g.r.max())],
                        events=int(s.n_events), only_A_event=int(s.only_A_event), only_B_event=int(s.only_B_event),
                        planes_ok_A=int(s.okA), planes_ok_B=int(s.okB), same_frac=float(s.same / s.both),
                        w_rel_max=float(g.w_rel_max.max()), tan_rel_med=float(g.tan_rel_med.median()),
                        tan_rel_68=float(g.tan_rel_68.median()),
                        diff=int(s['diff']), diff_in_in=int(s.diff_in_in), diff_in_out=int(s.diff_in_out),
                        diff_out_in=int(s.diff_out_in), diff_out_out=int(s.diff_out_out),
                        acc05=[int(s.acc05_A), int(s.acc05_B)], acc0506=[int(s.acc0506_A), int(s.acc0506_B)],
                        acc06=[int(s.acc06_A), int(s.acc06_B)],
                        tracks=[int(s.tracks_A), int(s.tracks_B), int(s.tracks_matched)],
                        tracks05=[int(s.tracks05_A), int(s.tracks05_B), int(s.tracks05_matched)])
    json.dump(out, open(f'{a.dir}/ab_summary.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
