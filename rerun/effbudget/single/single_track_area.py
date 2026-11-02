#!/usr/bin/env python3
"""single_track_area.py [--root R] -- Paper I headline: det3 efficiency over the FULL active area (Dylan 2026-11-02:
the fiducial cut changes nothing, so the headline is the whole area and the edge section studies the difference),
with the single-track / multi-particle split built from the candidate tables alone, so it is defined everywhere.

Per M3 ray (one clean ray, decoder-complete, footprint inside the measured active area; det3 has no hardware mask):
  detected      both views fitted, <= 2 competing strips, |r| <= 5 mm (budget cat_xy_5)
  credit        detected, or some candidate within 5 mm of the M3 ray in BOTH views (det_any_5): the M3 track is
                one of several reconstructed tracks
  two-particle  (i)  >= 2 candidates in both views, the extra pair charge-matched (|ln qx/qy - med| < 5 sigma) and
                     detached from the muon (not within 3 mm anywhere over the gap, both views);
                (ii) >= 2 candidates in one view and, in the other, one candidate merged in projection: >= 15 strips
                     [single tracks 12], or its charge = the sum and not the muon alone, or 5-15 mm off the muon;
                (iii) spark-vetoed with a clean muon + separate deposit (sparks.py 'muon+other'; interior only, the
                     waveform class exists only there: edge spark rays stay in the single sample, i.e. losses)
  single        every other ray: everything we cannot show to be a second particle counts against the chamber.
Nothing leaves the denominator (Dylan 2026-11-02: what we cannot pin on something else counts as a loss). A1's
POS_TAIL is only about half explained by muons bent between the M3 doublets (kinkvtx: 53 % within 2 mm of the
doublet-vertex prediction; sixpt puts the rest on the chamber), and the other-coordinate EDGE class is edge response,
part of the full-area number. DROP_CLASSES restores the old exclusion for comparison.
Writes results/single_track_area.json.
"""
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
R_MM, GAP, ATTACH, NSIG = 5.0, 28.0, 3.0, 5.0
DROP_CLASSES = [c for c in os.environ.get('DROP_CLASSES', '').split(',') if c]   # e.g. far:POS_TAIL,far:EDGE


def seg_dist(a, b):
    d0 = a.p0 - b.p0; d1 = (a.p0 + a.tan_theta * GAP) - (b.p0 + b.tan_theta * GAP)
    return 0.0 if d0 * d1 <= 0 else min(abs(d0), abs(d1))


def classify(root, rel):
    B = pd.read_parquet(os.path.join(root, 'x17rerun/paper1_eff/budget', rel, 'rays_det3.parquet'))
    B = B.drop_duplicates('event_id').set_index('event_id')
    B = B[(~B.not_measured) & (B.region != 'outside')].copy()
    C = pd.read_parquet(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
    C = C[(C['rank'] >= 0) & C.event_id.isin(B.index)].copy()
    C['mesh'] = np.where(C.plane == 'x', B.mesh_x.reindex(C.event_id).to_numpy(), B.mesh_y.reindex(C.event_id).to_numpy())
    C['d'] = (C.p0 - C.mesh).abs()
    n = C.groupby(['event_id', 'plane']).size().unstack(fill_value=0).reindex(B.index, fill_value=0)
    one = n.index[(n.x == 1) & (n.y == 1)]
    S = C[C.event_id.isin(one)].pivot(index='event_id', columns='plane')
    S = S[(S['d']['x'] < R_MM) & (S['d']['y'] < R_MM)]
    lq1 = np.log(S['q_sum']['x'].clip(lower=1) / S['q_sum']['y'].clip(lower=1))
    med = float(lq1.median()); q = np.percentile(lq1, [16, 84]); rs = 0.5 * (q[1] - q[0])
    cls = pd.Series('single', index=B.index, dtype=object)
    for e in n.index[(n.x >= 2) | (n.y >= 2)]:
        g = C[C.event_id == e]; gx = g[g.plane == 'x'].sort_values('d'); gy = g[g.plane == 'y'].sort_values('d')
        if len(gx) >= 2 and len(gy) >= 2:
            mx, my = gx.iloc[0], gy.iloc[0]
            z, u, v = min(((abs(np.log(max(u.q_sum, 1) / max(v.q_sum, 1)) - med) / rs, u, v)
                           for _, u in gx.iloc[1:].iterrows() for _, v in gy.iloc[1:].iterrows()), key=lambda t: t[0])
            if z < NSIG and not (seg_dist(mx, u) < ATTACH and seg_dist(my, v) < ATTACH):
                cls[e] = 'two: separate in both views'
        elif len(gx) >= 1 and len(gy) >= 1:
            gm, go, vm = (gx, gy, 'x') if len(gx) >= 2 else (gy, gx, 'y')
            mu, sec, oth = gm.iloc[0], gm.iloc[1], go.iloc[0]
            lq_mu = np.log(mu.q_sum / oth.q_sum) if vm == 'x' else np.log(oth.q_sum / mu.q_sum)
            sm = mu.q_sum + sec.q_sum
            lq_mg = np.log(sm / oth.q_sum) if vm == 'x' else np.log(oth.q_sum / sm)
            z_mu, z_mg = (lq_mu - med) / rs, (lq_mg - med) / rs
            if oth.n_strips >= 15 or (abs(z_mg) < 3 and abs(z_mu) >= 3) or (5 <= oth.d < 15):
                cls[e] = 'two: merged in one projection'
    return B, cls, dict(lq_med=med, lq_rsig=rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    a = ap.parse_args()
    res, acc = {}, {}
    for key, rel in ROWS.items():
        B, cls, ref = classify(a.root, rel)
        R = pd.read_parquet(os.path.join(a.root, 'x17rerun/paper1_eff/a1', rel, 'recat.parquet'))
        recat = R.recat.reindex(B.index)
        cls[recat.eq('spark:muon+other').to_numpy()] = 'two: muon + separate deposit (vetoed)'
        drop = recat.isin(DROP_CLASSES).to_numpy()
        det = (B.cat_xy_5 == 'detected').to_numpy()
        credit = det | B.det_any_5.astype(bool).to_numpy()
        interior = (B.region == 'interior').to_numpy()
        ev = pd.DataFrame(dict(cls=cls, recat=recat, region=B.region, cat=B.cat_xy_5, detected=det, credit=credit,
                               dropped=drop), index=B.index)
        ev.to_parquet(os.path.join(RES, f'single_track_area_events_{key}.parquet'))
        out = {'ref': ref}
        for area, m_area in (('active area', np.ones(len(B), bool)), ('interior 30x/8y', interior)):
            m = m_area & ~drop
            sel = {'all rays: production pick': (m, det), 'all rays: credit': (m, credit),
                   'single-track': (m & cls.eq('single').to_numpy(), credit)}
            for c in sorted(set(cls) - {'single'}):
                sel[c] = (m & cls.eq(c).to_numpy(), credit)
            sel['all two-particle events'] = (m & ~cls.eq('single').to_numpy(), credit)
            for nm, (mm, d) in sel.items():
                nn, k = int(mm.sum()), int((d & mm).sum())
                out[f'{area} | {nm}'] = dict(n=nn, k=k, eff=100 * k / nn if nn else None)
                acc.setdefault(f'{area} | {nm}', [0, 0]); acc[f'{area} | {nm}'][0] += nn; acc[f'{area} | {nm}'][1] += k
        res[key] = out
    res['pooled'] = {nm: dict(n=nn, eff=100 * k / nn, err=100 * np.sqrt(k / nn * (1 - k / nn) / nn)) for nm, (nn, k) in acc.items() if nn}
    json.dump(res, open(os.path.join(RES, 'single_track_area.json'), 'w'), indent=1)
    for nm, v in res['pooled'].items():
        print(f"{nm:70s} n={v['n']:6d}  {v['eff']:6.2f} ± {v['err']:.2f}")


if __name__ == '__main__':
    main()
