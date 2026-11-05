#!/usr/bin/env python3
"""dpin_score.py -- D pinned t0 vs free (plan §6 11-05): the two checks before the D bundle freezes.

    dpin_score.py cosmic     # kw_D under the pin: B-D / A-D centroid lines (line_kw.measure) on d2 (free) vs d2pin
    dpin_score.py beam       # run_145: fit quality, t0 spread, capsule pointing; d2 vs d2pinb (beam lat.) vs d2pin

Bundles (d_b/bundles/, all ntof_bundle.py D --v 31.0 --kw 0.880,0.881 --dp 0.0157 --mask-era E2 --tan-max 0.6, select
cost + split_corrob charge): d2 t0 free; d2pin t0 pinned on latency_D_r149 (cosmic, measured); d2pinb pinned on
latency_D_r145beam (D cosmic + the mean A/C beam-minus-cosmic offset, NOT measured on D).  Reco: dpin_run.sh.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..')), os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))]
W = '/media/ucla/work/ntof_pass1'
O = f'{W}/d_b/reco_pin'
RC145 = f'{W}/runs/run_145/run_config.json'
STACK = '/media/dylan/data/x17/scint_stack/tracks/stack_run_145.parquet'
LAT = {'d2pin': f'{W}/bundles/latency_D_r149.json', 'd2pinb': f'{W}/bundles/latency_D_r145beam.json', 'd2': None}
rng = np.random.default_rng(5)


def boot_med(d, n=2000):
    d = np.asarray(d, float)
    return float(np.std([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]))


def cosmic():
    import line_kw
    out = {}
    for lab in ('d2', 'd2pin'):
        if not os.path.exists(f'{line_kw.OUT}/lines_D_{lab}.parquet'):
            line_kw.measure(f'{O}/r149_*_D_{lab}.parquet', 'D', lab)
        R = pd.read_csv(f'{line_kw.OUT}/measure_D_{lab}.csv')
        h = R[(R.partner == 'B') & R.partner_strict & (R.point == 'D fitted p0') & (R.lever_min == 0) & (R.miss_max == 60)]
        a = R[(R.partner == 'A') & R.partner_strict & (R.point == 'D fitted p0') & (R.depth == 'time')
              & (R.lever_min == 250) & (R.miss_max == 60)]
        out[lab] = {f'{p}_{v}': {k: float(r[k]) for k in ('n', 'ratio', 'err', 'res_tan_68')}
                    for p, S in (('BD', h), ('AD', a)) for v in 'xy' for _, r in S[S.view == v].iterrows()}
    # paired: the same lines (subrun, event, partner) in both
    L = {lab: pd.read_parquet(f'{line_kw.OUT}/lines_D_{lab}.parquet') for lab in ('d2', 'd2pin')}
    print(L['d2'].columns.tolist())
    for lab in L:
        for v in 'xy':
            L[lab][f'res_tan_{v}'] = L[lab][f'tan{v}'] - L[lab][f'true_tan{v}']
    key = [k for k in ('subrun', 'event_id', 'partner') if k in L['d2']]
    m = L['d2'].merge(L['d2pin'], on=key, suffixes=('_f', '_p'))
    pr = {}
    for v in 'xy':
        rf, rp = f'res_tan_{v}_f', f'res_tan_{v}_p'
        if rf not in m:
            cand = [c for c in m.columns if c.startswith(f'res') and c.endswith('_f')]
            print('no', rf, 'have', cand)
            continue
        for part in ('B', 'A'):
            mm = m[(m.partner == part) & np.isfinite(m[rf]) & np.isfinite(m[rp]) & (m.miss_mm_f < 60) & (m.miss_mm_p < 60)
                   & (m[f'true_tan{v}_f'].abs().between(0.0, 0.6))]
            if len(mm) < 10:
                continue
            af, ap = np.abs(mm[rf]), np.abs(mm[rp])
            pr[f'{part}D_{v}'] = dict(n=int(len(mm)), res68_free=float(np.percentile(af, 68)),
                                      res68_pin=float(np.percentile(ap, 68)),
                                      paired_med_abs_diff=float(np.median(ap - af)), err=boot_med(ap - af))
    out['paired'] = pr
    print(json.dumps(out, indent=1))
    json.dump(out, open(f'{O}/cosmic_score.json', 'w'), indent=1)


def plane_stats(d, lat):
    s = {}
    for p in 'xy':
        q = d[(d[f'{p}_ok'] == True) & (d[f'{p}_quality_ok'] == True) & (d[f'{p}_ftst'] >= 0)]
        t = (q[f'{p}_t0'] + 10 * q[f'{p}_ftst']).to_numpy(float)
        c = lat[p]['C'] if lat else np.median(t)
        r = t - c
        lo, hi = np.percentile(r, [16, 84]) if len(r) else (np.nan, np.nan)
        chi = (d[f'{p}_chi2'] / d[f'{p}_dof'].clip(lower=1))[d[f'{p}_ok'] == True]
        s[p] = dict(ok=float((d[f'{p}_ok'] == True).mean()), quality=float((d[f'{p}_quality_ok'] == True).mean()),
                    chi2dof_med=float(chi.median()), t0_minus_lat_med=float(np.median(r)) if len(r) else np.nan,
                    t0_w68=float(0.5 * (hi - lo)))
    return s


def beam():
    import capsule_point
    st = pd.read_parquet(STACK, columns=['subrun', 'event_id', 'hw_D', 't_since_flash_ns'])
    st = st[st.subrun == 'stat090_0000'].drop(columns='subrun').drop_duplicates('event_id')
    res, T = {}, {}
    for lab in ('d2', 'd2pinb', 'd2pin'):
        f = f'{O}/r145_s0_D_{lab}.parquet'
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f).merge(st, on='event_id', how='left')
        lat = json.load(open(LAT[lab])) if LAT[lab] else None
        late = d.t_since_flash_ns > 10e6
        own = late & (d.hw_D == True)
        res[lab] = dict(n_events=int(len(d)), all=plane_stats(d, lat), late=plane_stats(d[late], lat),
                        own_late=plane_stats(d[own], lat), n_own_late=int(own.sum()))
        s, t = capsule_point.score(f, 'D', RC145, lab)
        res[lab]['pointing'] = {k: v for k, v in s.items() if k != 'bins'}
        T[lab] = t.merge(st, on='event_id', how='left')
    # paired DCA on the same tracks (event + mesh positions within 2 mm), as tests/t0pin/beam_common.py
    for cfg in ('d2pinb', 'd2pin'):
        if cfg not in T or 'd2' not in T:
            continue
        b, c = T['d2'], T[cfg]
        cols = ['event_id', 'x_p0', 'y_p0', 'dca', 'tanx', 'tany', 'hw_D']
        m = b[cols].merge(c[cols[:-1]], on='event_id', suffixes=('_b', '_c'))
        keep = (np.abs(m.x_p0_b - m.x_p0_c) < 2) & (np.abs(m.y_p0_b - m.y_p0_c) < 2)
        m = m[keep].drop_duplicates('event_id', keep=False)
        near = m[(m.dca_b < 25) | (m.dca_c < 25)]
        dd = (near.dca_c - near.dca_b).to_numpy()
        res[f'paired_{cfg}_vs_d2'] = dict(
            matched=int(len(m)), of=[int(len(b)), int(len(c))], near=int(len(near)),
            dca_med=[float(near.dca_b.median()), float(near.dca_c.median())],
            dca_68=[float(np.percentile(near.dca_b, 68)), float(np.percentile(near.dca_c, 68))],
            paired_med_diff=float(np.median(dd)) if len(dd) else np.nan, err=boot_med(dd) if len(dd) > 5 else np.nan,
            n_lt10=[int((near.dca_b < 10).sum()), int((near.dca_c < 10).sum())],
            dtan_med=[float(np.median(np.abs(m.tanx_c - m.tanx_b))), float(np.median(np.abs(m.tany_c - m.tany_b)))])
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open(f'{O}/beam_score.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    {'cosmic': cosmic, 'beam': beam}[sys.argv[1]]()
