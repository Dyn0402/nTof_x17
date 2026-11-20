#!/usr/bin/env python3
"""eff_vs_angle.py [--root R] -- chamber A single-track efficiency against the reference-track angle (Paper I Sec. 7,
the uniformity-over-angle requirement of Sec. 2).

Same sample, classes and "detected" as single_track_area.py (the 94.8 % headline: single-track rays, credit, full
active area, Runs 1 and 2, two-track re-pass tables), binned in the angle of the M3 reference track in each view (rays tx, ty = reference
tan) and in the polar angle. Also the interior, where the edge zone does not mix position with angle.
Writes results/eff_vs_angle.json.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import side_tree
from single_track_area import ROWS, RES, classify


def binned(ang, ok, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ang >= lo) & (ang < hi); n, k = int(m.sum()), int(ok[m].sum())
        if n:
            e = k / n
            out.append(dict(lo=float(lo), hi=float(hi), n=n, k=k, eff=round(100 * e, 2),
                            err=round(100 * np.sqrt(e * (1 - e) / n), 2)))
    return out


def summary(rows):
    r = [b for b in rows if b['n'] >= 200]
    e = np.array([b['eff'] for b in r]); s = np.array([b['err'] for b in r]); w = 1 / s**2
    mean = float((w * e).sum() / w.sum())
    return dict(bins_n_ge_200=len(r), min=float(e.min()), max=float(e.max()), mean=round(mean, 2),
                chi2=round(float((w * (e - mean)**2).sum()), 1), ndf=len(r) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--analysis', default='Analysis_pass1_twotrack', help='candidate tree (the headline uses the two-track re-pass)')
    ap.add_argument('--eff', default='x17rerun/paper1_eff_twotrack', help='efficiency-chain dir under --root')
    ap.add_argument('--out', default=RES)
    a = ap.parse_args()
    parts = []
    for key, rel in ROWS.items():
        B, cls, _ = classify(a.root, rel, a.analysis, a.eff)
        R = pd.read_parquet(os.path.join(side_tree.eff_dir(a.root, a.eff), 'a1', rel, 'recat.parquet'))
        cls[R.recat.reindex(B.index).eq('spark:muon+other').to_numpy()] = 'two'
        credit = (B.cat_xy_5 == 'detected').to_numpy() | B.det_any_5.astype(bool).to_numpy()
        parts.append(pd.DataFrame(dict(single=cls.eq('single').to_numpy(), ok=credit, interior=(B.region == 'interior').to_numpy(),
                                       ax=np.degrees(np.arctan(B.tx.to_numpy())), ay=np.degrees(np.arctan(B.ty.to_numpy())),
                                       pol=np.degrees(np.arctan(np.hypot(B.tx.to_numpy(), B.ty.to_numpy()))))))
    E = pd.concat(parts); E = E[E.single]
    res = dict(note='single-track rays, credit, Runs 1 and 2 (single_track_area.py); angle = M3 reference, degrees',
               n=len(E), eff=round(100 * E.ok.mean(), 2))
    signed, absb, pol = np.arange(-25, 26, 5.0), np.arange(0, 26, 5.0), np.arange(0, 31, 5.0)
    for area, S in (('active area', E), ('interior', E[E.interior])):
        ok = S.ok.to_numpy()
        d = dict(n=len(S), eff=round(100 * ok.mean(), 2),
                 x_signed=binned(S.ax.to_numpy(), ok, signed), y_signed=binned(S.ay.to_numpy(), ok, signed),
                 x_abs=binned(S.ax.abs().to_numpy(), ok, absb), y_abs=binned(S.ay.abs().to_numpy(), ok, absb),
                 polar=binned(S.pol.to_numpy(), ok, pol))
        d['summary'] = {k: summary(d[k]) for k in ('x_signed', 'y_signed', 'x_abs', 'y_abs', 'polar')}
        res[area] = d
        print(area, d['n'], d['eff'])
        for k in ('x_abs', 'y_abs', 'polar', 'x_signed', 'y_signed'):
            print(' ', k, ' '.join(f"[{b['lo']:.0f},{b['hi']:.0f}) {b['eff']:.1f}±{b['err']:.1f} n={b['n']}" for b in d[k]))
            print('    ', d['summary'][k])
    json.dump(res, open(os.path.join(a.out, 'eff_vs_angle.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
