#!/usr/bin/env python3
"""kinkvtx.py -- are the POS_TAIL events bent tracks?  MX17 against the M3 doublet-vertex prediction.

A muon that scatters between the doublets leaves the straight 4-layer M3 fit off at the DUT while barely raising
its chi2 (a 12 mrad kink at the DUT: 3.5 mm off, chi2 0.5 mm^2). The upper (1302/1185) and lower (144/24)
doublet lines, each extrapolated to the DUT, then both pass near the true point if the kink is at the DUT.
Per event and raw view: r_full = MX17 - straight fit, r_vtx = MX17 - mean(up, dn) at the DUT, d_ud = up - dn,
and the M3 kink. If POS_TAIL is scattering, r_vtx is small where r_full is large, and r_full follows the kink.
"""
import json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from olib import BASE, load  # noqa: E402


def per_view(det):
    x = pd.read_parquet(os.path.join(BASE, f'classified_{det}.parquet'))
    d = load(det)
    d = d.merge(x[['event_id', 'cls', 'outlier']], on='event_id', how='left')
    M = pd.read_parquet(os.path.join(BASE, f'm3feat_{det}.parquet'))
    d = d.merge(M[['event_id', 'm3_up_x', 'm3_dn_x', 'm3_up_y', 'm3_dn_y', 'm3_kink_x', 'm3_kink_y']], on='event_id',
                suffixes=('', '_m'))
    rows = []
    for r in d.to_dict('records'):
        P = {}
        for mv in 'xy':
            if not (isinstance(r[f'm3_up_{mv}'], str) and isinstance(r[f'm3_dn_{mv}'], str)):
                continue
            P[mv] = (np.array(json.loads(r[f'm3_up_{mv}'])), np.array(json.loads(r[f'm3_dn_{mv}'])))
        if len(P) < 2:
            continue
        full = np.array([r['ref_mesh_x'], r['ref_mesh_y']])
        for k, v in enumerate('xy'):
            if not r[f'use_{v}']:
                continue
            # the M3 view whose doublets move this raw coordinate
            mv = max(P, key=lambda m: abs(P[m][0][k] - P[m][1][k]) + abs(P[m][0][k] + P[m][1][k] - 2 * full[k]))
            up, dn = P[mv][0][k], P[mv][1][k]
            mx = r[f'{v}_p0']
            rows.append(dict(event_id=r['event_id'], view=v, cls=r['cls'] if isinstance(r['cls'], str) else 'GOOD',
                             r_full=mx - full[k], r_vtx=mx - 0.5 * (up + dn), r_up=mx - up, r_dn=mx - dn,
                             d_ud=up - dn, kink=r[f'm3_kink_{mv}'], dl=r[f'dl_{v}'], tan=r[f'ref_tan_{v}']))
    return pd.DataFrame(rows)


def main():
    out = {}
    for det in ('det3', 'det2', 'det7', 'det6', 'det4'):
        V = per_view(det); V.to_parquet(os.path.join(BASE, f'kinkvtx_{det}.parquet'))
        good = V[(V.cls == 'GOOD') & (V.dl.abs() < 3)]
        pt = V[(V.cls == 'POS_TAIL') & (V.r_full.abs() > 3)]
        f = lambda a: float(np.percentile(np.abs(a), 68))
        cf = np.polyfit(good.kink, good.r_full, 1)[0] if len(good) > 50 else np.nan
        out[det] = dict(n_good=len(good), s68_full_good=f(good.r_full), s68_vtx_good=f(good.r_vtx),
                        n_pos_tail=len(pt), med_full_pt=float(np.median(np.abs(pt.r_full))),
                        med_vtx_pt=float(np.median(np.abs(pt.r_vtx))),
                        med_best_doublet_pt=float(np.median(np.minimum(np.abs(pt.r_up), np.abs(pt.r_dn)))),
                        frac_vtx_lt2_pt=float(np.mean(np.abs(pt.r_vtx) < 2)),
                        slope_rfull_vs_kink_good_mm_per_mrad=float(cf),
                        corr_rfull_kink_pt=float(np.corrcoef(pt.r_full, pt.kink)[0, 1]) if len(pt) > 5 else np.nan)
        print(det, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[det].items()})
    json.dump(out, open(os.path.join(BASE, 'an', 'kinkvtx.json'), 'w'), indent=1)


if __name__ == '__main__' and not os.environ.get('BENT_ONLY'):
    main()


def bent_fraction(det, nboot=200, seed=0):
    """Share of POS_TAIL views that are bent tracks: mixture fit of |r_vtx| with two empirical templates,
    bent = r_vtx of good events (the vertex predictor's own spread), MX17 = r_full of the POS_TAIL view
    + (r_vtx - r_full) of a random good event (an MX17 error seen through the vertex predictor)."""
    rng = np.random.default_rng(seed)
    V = pd.read_parquet(os.path.join(BASE, f'kinkvtx_{det}.parquet'))
    g = V[(V.cls == 'GOOD') & (V.dl.abs() < 3)]
    p = V[(V.cls == 'POS_TAIL') & (V.r_full.abs() > 3)]
    bins = np.linspace(0, 8, 17)
    hb = np.histogram(np.abs(g.r_vtx), bins)[0] + 0.5; hb = hb / hb.sum()
    noise = (g.r_vtx - g.r_full).to_numpy()

    def fit(pp):
        sm = np.abs(np.repeat(pp.r_full.to_numpy(), 50) + rng.choice(noise, 50 * len(pp)))
        hm = np.histogram(sm, bins)[0] + 0.5; hm = hm / hm.sum()
        ho = np.histogram(np.abs(pp.r_vtx), bins)[0]
        fs = np.linspace(0, 1, 101)
        ll = [np.sum(ho * np.log(f * hb + (1 - f) * hm)) for f in fs]
        return fs[int(np.argmax(ll))]
    f0 = fit(p)
    bs = [fit(p.iloc[rng.integers(0, len(p), len(p))]) for _ in range(nboot)]
    return f0, float(np.std(bs)), len(p)


if __name__ == '__main__' and os.environ.get('BENT_ONLY'):
    res = {d: bent_fraction(d) for d in ('det3', 'det2', 'det7', 'det4', 'det6')}
    for d, (f, e, n) in res.items():
        print(f'{d}: POS_TAIL views {n}, bent-track share {f:.2f} +- {e:.2f}')
    json.dump({d: dict(f=f, err=e, n=n) for d, (f, e, n) in res.items()},
              open(os.path.join(BASE, 'an', 'bent_fraction.json'), 'w'), indent=1)
